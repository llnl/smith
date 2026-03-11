#include <gtest/gtest.h>
#include <mpi.h>
#include "mfem.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/infrastructure/application_manager.hpp"

using namespace mfem;
using namespace smith;

// ----------- Parameter Structs ------------

struct BlockPrecTestParams {
  enum class BlockPattern
  {
    Diagonal2x2,
    Lower2x2
  };
  enum class PrecKind
  {
    Diagonal,
    Triangular
  };
  enum class SolverBackend
  {
    HypreBoomerAMG,
    Strumpack
  };

  BlockPattern pattern;
  PrecKind prec_kind;
  smith::BlockTriangularType tri_type;  // Only for triangular
  std::string name;
  double rel_tol;
  SolverBackend backend;
};

// ----------- Test Fixture ------------

class BlockPreconditionerParamTest : public ::testing::TestWithParam<BlockPrecTestParams> {
 protected:
  MPI_Comm comm_;
  int rank_;
  void SetUp() override
  {
    comm_ = MPI_COMM_WORLD;
    MPI_Comm_rank(comm_, &rank_);
  }
};

// ----------- Test Body ------------

TEST_P(BlockPreconditionerParamTest, SolvesBlockSystemApproximately)
{
  const auto& params = GetParam();

  // Build FE problem
  const int dim = 2, ne = 2, order = 1;
  Mesh mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
  ParMesh pmesh(comm_, mesh);
  mesh.Clear();
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fes(&pmesh, &fec);

  Array<int> ess_bdr(pmesh.bdr_attributes.Max());
  ess_bdr[0] = 1;
  Array<int> ess_tdof_list;
  fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  ConstantCoefficient one(1.0);
  ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new DiffusionIntegrator(one));
  a.Assemble();
  a.Finalize();

  ParLinearForm b_(&fes);
  b_ = 0.0;
  ParGridFunction x_(&fes);
  x_ = 0.0;

  OperatorPtr A;
  Vector X, B;
  a.FormLinearSystem(ess_tdof_list, x_, b_, A, X, B);

  // Block setup
  HypreParMatrix* A_hypre = A.As<HypreParMatrix>();
  ASSERT_TRUE(A_hypre != nullptr);

  int N = A_hypre->NumRows();
  Array<int> block_offsets(3);
  block_offsets[0] = 0;
  block_offsets[1] = N;
  block_offsets[2] = 2 * N;

  BlockOperator J(block_offsets);

  J.SetBlock(0, 0, A_hypre);
  auto A11_copy = std::make_unique<HypreParMatrix>(*A_hypre);
  J.SetBlock(1, 1, A11_copy.get());

  HypreParMatrix* C = nullptr;
  if (params.pattern == BlockPrecTestParams::BlockPattern::Lower2x2) {
    // Build mass matrix for off-diagonal
    ParBilinearForm c(&fes);
    c.AddDomainIntegrator(new MassIntegrator(one));
    c.Assemble();
    c.Finalize();
    C = c.ParallelAssemble();
    J.SetBlock(1, 0, C);
  }

  // Build solver array
  std::vector<std::unique_ptr<Solver>> block_solvers;
  block_solvers.reserve(2);
  for (int i = 0; i < 2; ++i) {
    if (params.backend == BlockPrecTestParams::SolverBackend::HypreBoomerAMG) {
      auto solver = std::make_unique<mfem::HypreBoomerAMG>();
      solver->SetPrintLevel(0);
      block_solvers.push_back(std::move(solver));
    } else {
#ifdef MFEM_USE_STRUMPACK
      block_solvers.push_back(std::make_unique<smith::StrumpackSolver>(0, comm_));
#else
      MFEM_ABORT("Strumpack backend requested but MFEM_USE_STRUMPACK not enabled.");
#endif
    }
  }

  // Instantiate preconditioner
  std::unique_ptr<Solver> P;
  if (params.prec_kind == BlockPrecTestParams::PrecKind::Diagonal) {
    P = std::make_unique<smith::BlockDiagonalPreconditioner>(block_offsets, std::move(block_solvers));
  } else {
    P = std::make_unique<smith::BlockTriangularPreconditioner>(block_offsets, std::move(block_solvers),
                                                               params.tri_type);
  }
  P->SetOperator(J);

  BlockVector r(block_offsets), x(block_offsets), b(block_offsets);
  b.Randomize(1);

  P->Mult(b, x);
  J.Mult(x, r);
  r -= b;
  double resid_err = r.Norml2();
  double rel_err = resid_err / b.Norml2();

  if (rank_ == 0) {
    std::cout << "Test " << params.name << ", rel_err = " << rel_err << std::endl;
  }
  ASSERT_LT(rel_err, params.rel_tol);

  if (C) delete C;
}

// ----------- Instantiate Test Cases ------------

INSTANTIATE_TEST_SUITE_P(
    BlockPreconditionerTests, BlockPreconditionerParamTest,
    ::testing::Values(
        // BlockDiagonalPreconditioner + HypreBoomerAMG on [A 0; 0 A]
        BlockPrecTestParams{BlockPrecTestParams::BlockPattern::Diagonal2x2, BlockPrecTestParams::PrecKind::Diagonal,
                            smith::BlockTriangularType::Lower,  // unused for diagonal
                            "Diag_HypreBoomerAMG", 1e-1, BlockPrecTestParams::SolverBackend::HypreBoomerAMG},
        // BlockTriangularPreconditioner + HypreBoomerAMG on [A 0; C A]
        BlockPrecTestParams{BlockPrecTestParams::BlockPattern::Lower2x2, BlockPrecTestParams::PrecKind::Triangular,
                            smith::BlockTriangularType::Lower, "TriLower_HypreBoomerAMG", 1e-1,
                            BlockPrecTestParams::SolverBackend::HypreBoomerAMG}),
    [](const ::testing::TestParamInfo<BlockPrecTestParams>& param_info) { return param_info.param.name; });

#ifdef MFEM_USE_STRUMPACK
INSTANTIATE_TEST_SUITE_P(
    BlockPreconditionerStrumpackTests, BlockPreconditionerParamTest,
    ::testing::Values(
        // BlockDiagonalPreconditioner + Strumpack on [A 0; 0 A]
        BlockPrecTestParams{BlockPrecTestParams::BlockPattern::Diagonal2x2, BlockPrecTestParams::PrecKind::Diagonal,
                            smith::BlockTriangularType::Lower, "Diag_Strumpack", 1e-10,
                            BlockPrecTestParams::SolverBackend::Strumpack},
        // BlockTriangularPreconditioner + Strumpack on [A 0; C A]
        BlockPrecTestParams{BlockPrecTestParams::BlockPattern::Lower2x2, BlockPrecTestParams::PrecKind::Triangular,
                            smith::BlockTriangularType::Lower, "TriLower_Strumpack", 1e-10,
                            BlockPrecTestParams::SolverBackend::Strumpack}),
    [](const ::testing::TestParamInfo<BlockPrecTestParams>& param_info) { return param_info.param.name; });
#endif
// ----------- Google Test Main ------------

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}