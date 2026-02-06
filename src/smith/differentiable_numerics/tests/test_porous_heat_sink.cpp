#include <gtest/gtest.h>

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"

#include "smith/mesh_utils/mesh_utils.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/materials/thermal_material.hpp"
#include "smith/physics/functional_weak_form.hpp"

#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"

#include "smith/smith_config.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/numerics/block_preconditioner.hpp"

#include "gretl/data_store.hpp"

using namespace smith;

using ShapeDispSpace = H1<1, 2>;
using Space = H1<1>;
using GammaSpace = L2<1>;
 
struct MeshFixture : public testing::Test {
  double length = 1.0;
  double width = 1.0;
  int num_elements_x = 64;
  int num_elements_y = 64;
  double elem_size = length / num_elements_x;
 
  void SetUp()
  {
    smith::StateManager::initialize(datastore, "porous_heat");

    MPI_Barrier(MPI_COMM_WORLD);
    int serial_refinement   = 0;
    int parallel_refinement = 0;

    std::string filename = SMITH_REPO_DIR "/data/meshes/square_attribute.mesh";

    const std::string meshtag = "mesh";
    mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(filename), meshtag, serial_refinement,
                                              parallel_refinement);
  }  // Construct the appropriate dimension mesh and give it to the data store
 
  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
};

struct HeatSinkOptions {
  /// \f$ \kappa_0 \f$ : reference fluid conductivity, \f$ W/m/K \f$
  double kappa_0 = 0.5;

  /// \f$ \sigma_0 \f$ :reference solid conductivity, \f$ W/m/K \f$
  double sigma_0 = 5.0;

  /// \f$ \eta \f$ : bruggeman correlation exponent
  double eta = 1.5;

  /// \f$ \epsilon_m \f$ : liquid porosity
  double epsilon_m = 1.0;

  /// \f$ \epsilon_n \f$ : heat sink porosity
  double epsilon_n = 0.5;

  /// \f$ a_m \f$ : liquid specific surface area, \f$ m^2/m^3 \f$
  double a_m = 0.0;

  /// \f$ a_n \f$ : heat sink specific surface area, \f$ m^2/m^3 \f$
  double a_n = 5e+6;

  /// \f$ h \f$ : heat transfer coefficient \f$ W/m^2/K \f$ \f$
  double h = 0.01;

  /// \f$ q_{app} \f$ : applied heat flux, \f$ W/m^2 \f$ \f$
  double q_app = 0.0;

  /// \f$ T_{app} \f$ : applied temperature, \f$ K \f$
  double T_app = 0.0;

  /// \f$ f_{mb} \f$ : modified Bruggeman correlation scaling coefficient
  double f_mb = 1.0;
};

TEST_F(MeshFixture,B)
{

    std::string physics_name = "heatsink";
    auto graph = std::make_shared<gretl::DataStore>(100);
    auto shape_disp = createFieldState(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
    auto T1 = createFieldState(*graph, Space{}, physics_name + "_T1", mesh->tag());
    auto T2 = createFieldState(*graph, Space{}, physics_name + "_T2", mesh->tag());
    auto gamma = createFieldState(*graph, GammaSpace{}, physics_name + "_gamma", mesh->tag());
    smith::FunctionalWeakForm<2, Space, smith::Parameters<Space, Space, GammaSpace>> T1_form("T1_eqn", mesh, space(T1), spaces({T1, T2, gamma}));
    smith::FunctionalWeakForm<2, Space, smith::Parameters<Space, Space, GammaSpace>> T2_form("T2_eqn", mesh, space(T2), spaces({T1, T2, gamma}));

    HeatSinkOptions heatsink_options{
        .kappa_0 = 0.5, .sigma_0 = 5., .a_n = 1000.0, .h = 0.01, .q_app = -10.0};
    // lambda helper functions for the underlying PDEs. Note that these must be generic lambdas (i.e. using auto)
    // as this function will be called both with doubles (for residual evaluation) or dual numbers (auto diff-enabled
    // tangents).
    auto epsilon = [heatsink_options = heatsink_options](auto gamma_) {
      return (1.0 - gamma_) * heatsink_options.epsilon_m + heatsink_options.f_mb * gamma_ * heatsink_options.epsilon_n;
    };

    auto a = [heatsink_options = heatsink_options](auto gamma_) {
      return (1.0 - gamma_) * heatsink_options.a_m + gamma_ * heatsink_options.a_n; };

    auto sigma = [heatsink_options = heatsink_options, epsilon](auto gamma_) {
      using std::pow;
      return pow(1.0 - epsilon(gamma_), heatsink_options.eta) * heatsink_options.sigma_0;
    };

    auto kappa = [heatsink_options = heatsink_options, epsilon](auto gamma_) {
      using std::pow;
      return pow(epsilon(gamma_), heatsink_options.eta) * heatsink_options.kappa_0;
    };

    auto q_n = [heatsink_options = heatsink_options](auto T_1, auto T_2) {
      return heatsink_options.h * (T_1 - T_2);
    };

    // Gamma field function for Full-cell heatsink
    auto gamma_fun = [](const mfem::Vector& x) -> double {
      if (x[0] >= 4.0 / 16.0 && x[0] <= 7.0 / 16.0 && x[1] >= 5.0 / 16.0) return 1.0;
      else if (x[0] >= 14.0 / 16.0 && x[1] >= 5.0 / 16.0) return 1.0;
      else if (x[1] >= 13.0 / 16.0) return 1.0;

      return 1e-8;
    };

    auto gamma_coef = std::make_shared<mfem::FunctionCoefficient>(gamma_fun);
    gamma.get()->project(gamma_coef);

    T1_form.addBodyIntegral(DependsOn<0, 1, 2>{}, mesh->entireBodyName(),
      [sigma, a, q_n](double /* t */, auto /* x */, auto T_1, auto T_2, auto gamma_) {
        // Get the value and the gradient from the input tuple
        auto [T_1_val, dT_1_dX] = T_1;
        auto [T_2_val, dT_2_dX] = T_2;
        auto [gamma_val, dgamma_dX] = gamma_;

        // The first element is the "source" term which acts on the scalar part of the test function.
        // The second element is the "flux" term which acts on the gradient part of the test function.
        return smith::tuple{a(gamma_val) * q_n(T_1_val, T_2_val),
                            sigma(gamma_val) * dT_1_dX};
        }
    );
    T2_form.addBodyIntegral(DependsOn<0, 1, 2>{}, mesh->entireBodyName(),
      [kappa, a, q_n](double /* t */, auto /* x */, auto T_1, auto T_2, auto gamma_) {
        // Get the value and the gradient from the input tuple
        auto [T_1_val, dT_1_dX] = T_1;
        auto [T_2_val, dT_2_dX] = T_2;
        auto [gamma_val, dgamma_dX] = gamma_;

        // The first element is the "source" term which acts on the scalar part of the test function.
        // The second element is the "flux" term which acts on the gradient part of the test function.
        return smith::tuple{-1.0 * a(gamma_val) * q_n(T_1_val, T_2_val),
                            kappa(gamma_val) * dT_2_dX};
        }
    );

    auto T1_bc_manager = std::make_shared<smith::BoundaryConditionManager>(mesh->mfemParMesh());
    auto T2_bc_manager = std::make_shared<smith::BoundaryConditionManager>(mesh->mfemParMesh());

    // Apply Dirichlet BC T_2=0 at bulk
    auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
    T2_bc_manager->addEssential(std::set<int>{1}, zero_bcs, space(T2), 0);

    // Apply Neumann BC DT_1.n = q_app
    mesh->addDomainOfBoundaryElements("heat_spreader", by_attr<2>(2));
    T1_form.addBoundaryIntegral(DependsOn<>{}, "heat_spreader", [heatsink_options = heatsink_options](double, auto) { return heatsink_options.q_app; });



    // Block Diagonal Preconditioner
    smith::LinearSolverOptions default_linear_options = {.linear_solver = smith::LinearSolver::GMRES,
                                                      .preconditioner = smith::Preconditioner::HypreAMG,
                                                      .relative_tol = 1.0e-8,
                                                      .absolute_tol = 1.0e-12,
                                                      .max_iterations = 200,
                                                      .print_level = 1};
    // Create an offset representing a block vector view of the combined solution vector
    mfem::Array<int> block_offsets_;
    block_offsets_.SetSize(3);
    block_offsets_[0] = 0;
    block_offsets_[1] = T1.get()->space().TrueVSize();
    block_offsets_[2] = T2.get()->space().TrueVSize();
    block_offsets_.PartialSum();
    std::vector<std::unique_ptr<mfem::Solver>> solvers;
    // BoomerAMG solvers for the blocks as before
    // solvers.push_back(std::make_unique<mfem::HypreBoomerAMG>());
    // solvers.push_back(std::make_unique<mfem::HypreBoomerAMG>());
    // LU direct solvers
    // smith::LinearSolverOptions direct_solver_options{.linear_solver = smith::LinearSolver::Strumpack};
    // auto[solver1, precond1] = smith::buildLinearSolverAndPreconditioner(direct_solver_options, mesh->getComm());
    // auto[solver2, precond2] = smith::buildLinearSolverAndPreconditioner(direct_solver_options, mesh->getComm());
    // Iterative solver for blocks
    smith::LinearSolverOptions iter_solver_options = {.linear_solver = smith::LinearSolver::GMRES,
                                                      .preconditioner = smith::Preconditioner::HypreJacobi,
                                                      .relative_tol = 1.0e-3,
                                                      .absolute_tol = 1.0e-6,
                                                      .max_iterations = 100,
                                                      .print_level = 1};
    auto[solver1, precond1] = smith::buildLinearSolverAndPreconditioner(iter_solver_options, mesh->getComm());
    auto[solver2, precond2] = smith::buildLinearSolverAndPreconditioner(iter_solver_options, mesh->getComm());
    solvers.push_back(std::move(solver1));
    solvers.push_back(std::move(solver2));
    // std::unique_ptr<mfem::Solver> diff_precond = std::make_unique<smith::BlockDiagonalPreconditioner>(block_offsets_, std::move(solvers));
    std::unique_ptr<mfem::Solver> diff_precond = std::make_unique<smith::BlockTriangularPreconditioner>(block_offsets_, std::move(solvers), smith::BlockTriangularType::Symmetric);
    std::unique_ptr<mfem::Solver> linear_solver = std::make_unique<mfem::GMRESSolver>(mesh->getComm());
    mfem::GMRESSolver* iter_lin_solver = dynamic_cast<mfem::GMRESSolver*>(linear_solver.get());

    // Set up linear solver with custom preconditioner
    iter_lin_solver->iterative_mode = false;
    iter_lin_solver->SetRelTol(default_linear_options.relative_tol);
    iter_lin_solver->SetAbsTol(default_linear_options.absolute_tol);
    iter_lin_solver->SetMaxIter(default_linear_options.max_iterations);
    iter_lin_solver->SetPrintLevel(default_linear_options.print_level);
    iter_lin_solver->SetPreconditioner(*diff_precond);
    std::shared_ptr<smith::DifferentiableBlockSolver> d_linear_solver =
        std::make_shared<smith::LinearDifferentiableBlockSolver>(std::move(linear_solver), std::move(diff_precond));
    // Block Solve
    auto time = graph->create_state<double,double>(0.0);
    auto dt = graph->create_state<double,double>(0.025);
    int cycle = 0;
    std::vector<smith::FieldState> params;
    auto& T1_params = params;
    auto& T2_params = params;
    std::vector<FieldState> T1_arguments{T1, T2, gamma};
    std::vector<FieldState> T2_arguments{T1, T2, gamma};
    auto sols = block_solve({&T1_form, &T2_form},
                            {{0,1}, {0,1}},
                            shape_disp, {T1_arguments, T2_arguments},  // states
                            {T1_params, T2_params},                    // params
                            smith::TimeInfo(time.get(), dt.get(), cycle), d_linear_solver.get(), {T1_bc_manager.get(), T2_bc_manager.get()});
    
    auto pv_writer = smith::createParaviewWriter(*mesh, sols, physics_name);
    pv_writer.write(0, 0.0, sols);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
