#include <gtest/gtest.h>

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/functional_weak_form.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/smith_config.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/numerics/block_preconditioner.hpp"

#include "gretl/data_store.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"

using namespace smith;

using ShapeDispSpace = H1<1, 2>;
// Taylor-Hood elements
using Space = H1<1>;
using VectorSpace = H1<2, 2>;

enum class BlockSolverType
{
  Direct,
  Iterative,
  BoomerAMG
};
enum class BlockPrecondType
{
  SchurLower,
  SchurUpper,
  SchurDiag,
  SchurFull,
  SchurFullCustom
};

struct BlockTestParams {
  BlockSolverType solver_type;
  BlockPrecondType precond_type;
};

std::string BlockParamNameGenerator(const ::testing::TestParamInfo<BlockTestParams>& info)
{
  auto solver_to_str = [](BlockSolverType t) {
    switch (t) {
      case BlockSolverType::Direct:
        return "Direct";
      case BlockSolverType::Iterative:
        return "Iterative";
      case BlockSolverType::BoomerAMG:
        return "BoomerAMG";
    }
    return "Unknown";
  };
  auto precond_to_str = [](BlockPrecondType t) {
    switch (t) {
      case BlockPrecondType::SchurLower:
        return "SchurLower";
      case BlockPrecondType::SchurUpper:
        return "SchurUpper";
      case BlockPrecondType::SchurDiag:
        return "SchurDiag";
      case BlockPrecondType::SchurFull:
        return "SchurFull";
      case BlockPrecondType::SchurFullCustom:
        return "SchurFullCustom";
    }
    return "Unknown";
  };
  return std::string(solver_to_str(info.param.solver_type)) + "_" + precond_to_str(info.param.precond_type);
}

class MeshFixture : public testing::Test {
 protected:
  double length = 1.0;
  double width = 1.0;
  int num_elements_x = 32;
  int num_elements_y = 32;
  double elem_size = length / num_elements_x;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;

  void SetUp() override
  {
    smith::StateManager::initialize(datastore, "mixed_poisson");

    MPI_Barrier(MPI_COMM_WORLD);
    int serial_refinement = 4;
    int parallel_refinement = 0;

    std::string filename = SMITH_REPO_DIR "/data/meshes/square_attribute.mesh";

    const std::string meshtag = "mesh";
    mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(filename), meshtag, serial_refinement,
                                         parallel_refinement);
  }
};

class BlockPreconditionerTest : public MeshFixture, public ::testing::WithParamInterface<BlockTestParams> {};

TEST_P(BlockPreconditionerTest, BlockSolve)
{
  const auto& test_params = GetParam();

  std::string physics_name = "mixed_poisson";
  auto graph = std::make_shared<gretl::DataStore>(std::make_unique<gretl::WangCheckpointStrategy>(100));
  auto shape_disp = createFieldState(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
  auto flux = createFieldState(*graph, VectorSpace{}, physics_name + "_flux", mesh->tag());
  auto potential = createFieldState(*graph, Space{}, physics_name + "_potential", mesh->tag());
  smith::FunctionalWeakForm<2, VectorSpace, smith::Parameters<VectorSpace, Space>> con_form("constitutive_eqn", mesh, space(flux),
                                                                                           spaces({flux, potential}));
  smith::FunctionalWeakForm<2, Space, smith::Parameters<VectorSpace, Space>> bal_form("balance_eqn", mesh, space(potential),
                                                                                           spaces({flux, potential}));

  con_form.addBodyIntegral(DependsOn<0, 1>{}, mesh->entireBodyName(),
                          [](double /* t */, auto /* x */, auto SIGMA, auto U) {
                            auto sigma = get<VALUE>(SIGMA);
                            auto u = get<VALUE>(U);
                            // Need to wrap u in a tensor to convert grad(test_function) to . div(test_function)
                            using Scalar = decltype(u);
                            smith::tensor<Scalar, 2, 2> u_{};
                            u_[0][0] = u;
                            u_[1][1] = u;
                            return smith::tuple{sigma, -u_};
                          });

  bal_form.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(),
                          [](double /* t */, auto X, auto SIGMA) {
                            const auto& x = get<VALUE>(X);
                            auto div_sigma = smith::tr(get<DERIVATIVE>(SIGMA));
                            double pi = M_PI;
                            auto f = 2.0 * pi * pi * sin(pi * x[0]) * sin(pi * x[1]);
                            return smith::tuple{-f + div_sigma, smith::zero{}};
                          });
  // u_exact = sin(M_PI * x(0)) * sin(M_PI * x(1));
  // sigma_exact
  // pi * cos(pi * x(0)) * sin(pi * x(1))
  // pi * sin(pi * x(0)) * cos(pi * x(1));

  auto flux_bc_manager = std::make_shared<smith::BoundaryConditionManager>(mesh->mfemParMesh());
  auto potential_bc_manager = std::make_shared<smith::BoundaryConditionManager>(mesh->mfemParMesh());

  auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
  potential_bc_manager->addEssential(std::set<int>{1,2,3,4}, zero_bcs, space(potential), 0);

  // Block Preconditioner Options
  smith::LinearSolverOptions linear_options;
  linear_options.linear_solver = smith::LinearSolver::GMRES;
  linear_options.relative_tol = 1.0e-10;
  linear_options.absolute_tol = 1.0e-14;
  linear_options.max_iterations = 100;
  linear_options.print_level = 1;

  // Parameter sweep: construct solvers according to test parameters
  if (test_params.solver_type == BlockSolverType::Direct) {
    smith::LinearSolverOptions direct_solver_options{.linear_solver = smith::LinearSolver::SuperLU};
    linear_options.sub_block_linear_solver_options.push_back(direct_solver_options);
    linear_options.sub_block_linear_solver_options.push_back(direct_solver_options);
  } else if (test_params.solver_type == BlockSolverType::Iterative) {
    smith::LinearSolverOptions iter_solver_options = {.linear_solver = smith::LinearSolver::GMRES,
                                                      .preconditioner = smith::Preconditioner::HypreAMG,
                                                      .relative_tol = 1.0e-3,
                                                      .absolute_tol = 1.0e-6,
                                                      .max_iterations = 100,
                                                      .print_level = 1};
    linear_options.sub_block_linear_solver_options.push_back(iter_solver_options);
    linear_options.sub_block_linear_solver_options.push_back(iter_solver_options);
  } else if (test_params.solver_type == BlockSolverType::BoomerAMG) {
    smith::LinearSolverOptions amg_solver_options;
    amg_solver_options.linear_solver = smith::LinearSolver::GMRES;
    amg_solver_options.preconditioner = smith::Preconditioner::HypreAMG;
    amg_solver_options.max_iterations = 1;  // Since it's a preconditioner-only analog
    amg_solver_options.relative_tol = 0.99;  // Since it's a preconditioner-only analog
    linear_options.sub_block_linear_solver_options.push_back(amg_solver_options);
    linear_options.sub_block_linear_solver_options.push_back(amg_solver_options);
  }

  // Need these here for the custom operator
  auto time = graph->create_state<double, double>(0.0);
  auto dt = graph->create_state<double, double>(0.025);
  size_t cycle = 0;
  std::vector<smith::FieldState> params;
  auto& flux_params = params;
  auto& potential_params = params;
  std::vector<FieldState> con_arguments{flux, potential};
  std::vector<FieldState> bal_arguments{flux, potential};

  switch (test_params.precond_type) {
    case BlockPrecondType::SchurLower:
      linear_options.preconditioner = smith::Preconditioner::BlockSchur;
      linear_options.block_schur_type = smith::BlockSchurType::Lower;
      break;
    case BlockPrecondType::SchurUpper:
      linear_options.preconditioner = smith::Preconditioner::BlockSchur;
      linear_options.block_schur_type = smith::BlockSchurType::Upper;
      break;
    case BlockPrecondType::SchurDiag:
      linear_options.preconditioner = smith::Preconditioner::BlockSchur;
      linear_options.block_schur_type = smith::BlockSchurType::Diagonal;
      break;
    case BlockPrecondType::SchurFull:
      linear_options.preconditioner = smith::Preconditioner::BlockSchur;
      linear_options.block_schur_type = smith::BlockSchurType::Full;
      break;
    case BlockPrecondType::SchurFullCustom:
      linear_options.preconditioner = smith::Preconditioner::BlockSchur;
      linear_options.block_schur_type = smith::BlockSchurType::Full;
      /// linear_options.schur_approx_type = smith::BlockSchurType::Custom;
      break;
  }

  smith::NonlinearSolverOptions nonlin_opts;
  nonlin_opts.nonlin_solver = smith::NonlinearSolver::Newton;
  nonlin_opts.relative_tol = linear_options.relative_tol;
  nonlin_opts.absolute_tol = linear_options.absolute_tol;
  nonlin_opts.max_iterations = 1;
  nonlin_opts.print_level = linear_options.print_level;

  auto nonlinear_block_solver = smith::buildNonlinearBlockSolver(nonlin_opts, linear_options, *mesh);

  auto sols = block_solve({&con_form, &bal_form}, {{0, 1}, {0, 1}}, shape_disp, {con_arguments, bal_arguments},
                          {flux_params, potential_params}, smith::TimeInfo(time.get(), dt.get(), cycle),
                          nonlinear_block_solver.get(), {flux_bc_manager.get(), potential_bc_manager.get()});

  auto pv_writer = smith::createParaviewWriter(*mesh, sols, physics_name);
  pv_writer.write(0, 0.0, sols);

  // Calculate error versus exact solution
  auto u_exact_fun = [](const mfem::Vector& X) -> double {
    double x = X(0), y = X(1);
    return std::sin(M_PI * x) * std::sin(M_PI * y);
  };
  mfem::FunctionCoefficient u_exact_coef(u_exact_fun);

  auto sigma_exact_fun = [](const mfem::Vector& X, mfem::Vector& S) {
    double x = X(0), y = X(1);
    S.SetSize(2);
    S(0) = -M_PI * std::cos(M_PI * x) * std::sin(M_PI * y);
    S(1) = -M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
  };
  mfem::VectorFunctionCoefficient sigma_exact_coef(2, sigma_exact_fun);
  double u_err = smith::computeL2Error(*sols[1].get(), u_exact_coef);
  double sigma_err = smith::computeL2Error(*sols[0].get(), sigma_exact_coef);

  EXPECT_LT(u_err, 5e-3);
  EXPECT_LT(sigma_err, 5e-2);
  SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(BlockPrecondSweep, BlockPreconditionerTest,
                         ::testing::Values(
                             // Direct solvers
                             BlockTestParams{BlockSolverType::Direct, BlockPrecondType::SchurLower},
                             BlockTestParams{BlockSolverType::Direct, BlockPrecondType::SchurUpper},
                             BlockTestParams{BlockSolverType::Direct, BlockPrecondType::SchurDiag},
                             BlockTestParams{BlockSolverType::Direct, BlockPrecondType::SchurFull},

                             // Iterative solvers
                             BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::SchurLower},
                             BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::SchurUpper},
                             BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::SchurDiag},
                             BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::SchurFull},

                             // BoomerAMG solvers
                             BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurLower},
                             BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurUpper},
                             BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurDiag},
                             BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurFull},
                             BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurFullCustom}),
                         BlockParamNameGenerator);

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
