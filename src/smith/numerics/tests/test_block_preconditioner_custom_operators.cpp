#include <gtest/gtest.h>
#include <mpi.h>

#include "mfem.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/infrastructure/application_manager.hpp"

using namespace mfem;

/* ============================================================
   Helper utilities
   ============================================================ */

namespace
{

// Own both the local CSR (used to build the hypre matrix) and the HypreParMatrix.
// This avoids dangling pointers because some HypreParMatrix constructors wrap
// CSR arrays without taking ownership.
struct OwnedHypreParMatrix
{
  std::unique_ptr<SparseMatrix> diag;
  std::unique_ptr<HypreParMatrix> A;
};

// Build a HypreParMatrix from a local CSR matrix, assuming serial execution.
// Uses a block-diagonal ParCSR matrix whose local diagonal block is 'diag'.
OwnedHypreParMatrix makeHypreFromLocalDiag(std::unique_ptr<SparseMatrix> diag,
                                          HYPRE_BigInt global_num_rows,
                                          HYPRE_BigInt global_num_cols,
                                          MPI_Comm comm)
{
  MFEM_VERIFY(diag, "diag must be non-null");
  diag->Finalize();

  auto row_starts = std::array<HYPRE_BigInt, 2>{static_cast<HYPRE_BigInt>(0), global_num_rows};
  auto col_starts = std::array<HYPRE_BigInt, 2>{static_cast<HYPRE_BigInt>(0), global_num_cols};

  auto A = std::make_unique<HypreParMatrix>(comm,
                                            global_num_rows,
                                            global_num_cols,
                                            row_starts.data(),
                                            col_starts.data(),
                                            diag.get());

  // Ensure partition arrays are owned by HypreParMatrix (so std::array can die)
  A->CopyRowStarts();
  A->CopyColStarts();

  OwnedHypreParMatrix out;
  out.diag = std::move(diag);
  out.A = std::move(A);
  return out;
}

// Build c * I (square)
OwnedHypreParMatrix makeHypreScaledIdentity(int n, double c, MPI_Comm comm = MPI_COMM_WORLD)
{
  auto diag = std::make_unique<SparseMatrix>(n);
  for (int i = 0; i < n; i++) { diag->Add(i, i, c); }

  const auto N = static_cast<HYPRE_BigInt>(n);
  return makeHypreFromLocalDiag(std::move(diag), N, N, comm);
}

class IdentitySolver : public mfem::Solver
{
public:
  void SetOperator(const mfem::Operator& op) override
  {
    height = op.Height();
    width = op.Width();
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override { y = x; }
};

// Exact diagonal inverse solver for HypreParMatrix
class HypreExactDiagonalSolver : public mfem::Solver
{
public:
  void SetOperator(const mfem::Operator& op) override
  {
    const auto* A = dynamic_cast<const mfem::HypreParMatrix*>(&op);
    MFEM_VERIFY(A, "HypreExactDiagonalSolver requires HypreParMatrix");

    A_ = A;
    height = A_->Height();
    width = A_->Width();

    diag_.SetSize(height);
    A_->GetDiag(diag_);
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    MFEM_ASSERT(A_, "Operator not set");
    y.SetSize(x.Size());
    for (int i = 0; i < x.Size(); i++) { y[i] = x[i] / diag_[i]; }
  }

private:
  const mfem::HypreParMatrix* A_ = nullptr;
  mfem::Vector diag_;
};

}  // namespace
/* ============================================================
   Tests
   ============================================================ */

// Makes sure an error is thrown when length of solvers does not match the
// number of blocks

// If the solver for each block is identity, the block solver is identity
TEST(BlockDiagonalPreconditionerCustom, IdentityActsAsIdentity)
{
  Array<int> offsets({0, 2, 5});

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(3, 2.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());

  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreScaledIdentity(2, 1.0));  // M1
  override_mats.push_back(makeHypreScaledIdentity(3, 1.0));  // M2

  std::vector<std::pair<int, std::unique_ptr<const mfem::Operator>>> overrides;
  overrides.emplace_back(
    0, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));
  overrides.emplace_back(
    1, std::unique_ptr<const mfem::Operator>(std::move(override_mats[1].A)));

  smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers), std::move(overrides));

  P.SetOperator(A);  // This actually doesn't use A, it's overridden by M1, M2

  Vector x(5), y(5);
  x.Randomize();

  P.Mult(x, y);

  mfem::Vector diff(x);
  diff -= y;

  EXPECT_NEAR(diff.Norml2(), 0.0, 1e-14);
}

// Same but for single blocks
TEST(BlockDiagonalPreconditionerCustom, IdentityActsAsIdentity2)
{
  Array<int> offsets({0, 2, 5});

  auto A11o = makeHypreScaledIdentity(2, 1.0);
  auto A22o = makeHypreScaledIdentity(3, 2.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());

  // Override only block 1 with identity
  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreScaledIdentity(3, 1.0));  // M2

  std::vector<std::pair<int, std::unique_ptr<const mfem::Operator>>> overrides;
  overrides.emplace_back(
    1, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));

  smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers), std::move(overrides));

  P.SetOperator(A);

  Vector x(5), y(5);
  x.Randomize();

  P.Mult(x, y);

  mfem::Vector diff(x);
  diff -= y;

  EXPECT_NEAR(diff.Norml2(), 0.0, 1e-14);
}

/* ============================================================
   main
   ============================================================ */

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
