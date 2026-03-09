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
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/numerics/block_preconditioner.hpp"

#include "gretl/data_store.hpp"

using namespace smith;

using ShapeDispSpace = H1<1, 2>;
using Space = H1<1>;
using GammaSpace = L2<1>;

enum class BlockSolverType { Direct, Iterative, BoomerAMG };
enum class BlockPrecondType {
    Diagonal,
    TriLower,
    TriUpper,
    TriSym,
    SchurLower,
    SchurUpper,
    SchurDiag,
    SchurFull,
    SchurFullA22,
    SchurFullCustom
};

struct BlockTestParams {
    BlockSolverType solver_type;
    BlockPrecondType precond_type;
};

std::string BlockParamNameGenerator(const ::testing::TestParamInfo<BlockTestParams>& info) {
    auto solver_to_str = [](BlockSolverType t) {
        switch (t) {
            case BlockSolverType::Direct:    return "Direct";
            case BlockSolverType::Iterative: return "Iterative";
            case BlockSolverType::BoomerAMG: return "BoomerAMG";
        }
        return "Unknown";
    };
    auto precond_to_str = [](BlockPrecondType t) {
        switch (t) {
            case BlockPrecondType::Diagonal:    return "Diag";
            case BlockPrecondType::TriLower:    return "TriLower";
            case BlockPrecondType::TriUpper:    return "TriUpper";
            case BlockPrecondType::TriSym:      return "TriSym";
            case BlockPrecondType::SchurLower:  return "SchurLower";
            case BlockPrecondType::SchurUpper:  return "SchurUpper";
            case BlockPrecondType::SchurDiag:   return "SchurDiag";
            case BlockPrecondType::SchurFull:   return "SchurFull";
            case BlockPrecondType::SchurFullA22:   return "SchurFullA22";
            case BlockPrecondType::SchurFullCustom:   return "SchurFullCustom";
        }
        return "Unknown";
    };
    return std::string(solver_to_str(info.param.solver_type)) + "_" + precond_to_str(info.param.precond_type);
}

struct HeatSinkOptions {
    double kappa_0 = 0.5;
    double sigma_0 = 5.0;
    double eta = 1.5;
    double epsilon_m = 1.0;
    double epsilon_n = 0.5;
    double a_m = 0.0;
    double a_n = 5e+6;
    double h = 0.01;
    double q_app = 0.0;
    double T_app = 0.0;
    double f_mb = 1.0;
};

class MeshFixture : public testing::Test {
protected:
    double length = 1.0;
    double width = 1.0;
    int num_elements_x = 32;
    int num_elements_y = 32;
    double elem_size = length / num_elements_x;

    axom::sidre::DataStore datastore;
    std::shared_ptr<smith::Mesh> mesh;

    void SetUp() override {
        smith::StateManager::initialize(datastore, "porous_heat");

        MPI_Barrier(MPI_COMM_WORLD);
        int serial_refinement   = 0;
        int parallel_refinement = 0;

        std::string filename = SMITH_REPO_DIR "/data/meshes/square_attribute.mesh";

        const std::string meshtag = "mesh";
        mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(filename), meshtag, serial_refinement,
                                                  parallel_refinement);
    }
};

class BlockPreconditionerTest : public MeshFixture, public ::testing::WithParamInterface<BlockTestParams> {};

TEST_P(BlockPreconditionerTest, BlockSolve) {
    const auto& test_params = GetParam();

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
        auto [T_1_val, dT_1_dX] = T_1;
        auto [T_2_val, dT_2_dX] = T_2;
        auto [gamma_val, dgamma_dX] = gamma_;
        return smith::tuple{a(gamma_val) * q_n(T_1_val, T_2_val),
                            sigma(gamma_val) * dT_1_dX};
        }
    );
    T2_form.addBodyIntegral(DependsOn<0, 1, 2>{}, mesh->entireBodyName(),
      [kappa, a, q_n](double /* t */, auto /* x */, auto T_1, auto T_2, auto gamma_) {
        auto [T_1_val, dT_1_dX] = T_1;
        auto [T_2_val, dT_2_dX] = T_2;
        auto [gamma_val, dgamma_dX] = gamma_;
        return smith::tuple{-1.0 * a(gamma_val) * q_n(T_1_val, T_2_val),
                            kappa(gamma_val) * dT_2_dX};
        }
    );

    auto T1_bc_manager = std::make_shared<smith::BoundaryConditionManager>(mesh->mfemParMesh());
    auto T2_bc_manager = std::make_shared<smith::BoundaryConditionManager>(mesh->mfemParMesh());

    auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
    T2_bc_manager->addEssential(std::set<int>{1}, zero_bcs, space(T2), 0);

    mesh->addDomainOfBoundaryElements("heat_spreader", by_attr<2>(2));
    T1_form.addBoundaryIntegral(DependsOn<>{}, "heat_spreader", [heatsink_options = heatsink_options](double, auto) { return heatsink_options.q_app; });

    // Block Diagonal Preconditioner
    smith::LinearSolverOptions default_linear_options = {.linear_solver = smith::LinearSolver::GMRES,
                                                      .preconditioner = smith::Preconditioner::HypreAMG,
                                                      .relative_tol = 1.0e-8,
                                                      .absolute_tol = 1.0e-12,
                                                      .max_iterations = 200,
                                                      .print_level = 1};

    mfem::Array<int> block_offsets_;
    block_offsets_.SetSize(3);
    block_offsets_[0] = 0;
    block_offsets_[1] = T1.get()->space().TrueVSize();
    block_offsets_[2] = T2.get()->space().TrueVSize();
    block_offsets_.PartialSum();

    std::vector<std::unique_ptr<mfem::Solver>> solvers;
    std::vector<std::unique_ptr<mfem::Solver>> preconds;

    // Parameter sweep: construct solvers according to test parameters
    if (test_params.solver_type == BlockSolverType::Direct) {
        smith::LinearSolverOptions direct_solver_options{.linear_solver = smith::LinearSolver::Strumpack};
        auto [solver1, precond1] = smith::buildLinearSolverAndPreconditioner(direct_solver_options, mesh->getComm());
        auto [solver2, precond2] = smith::buildLinearSolverAndPreconditioner(direct_solver_options, mesh->getComm());
        solvers.push_back(std::move(solver1));
        solvers.push_back(std::move(solver2));
    } else if (test_params.solver_type == BlockSolverType::Iterative) {
        smith::LinearSolverOptions iter_solver_options = {.linear_solver = smith::LinearSolver::GMRES,
                                                          .preconditioner = smith::Preconditioner::HypreAMG,
                                                          .relative_tol = 1.0e-3,
                                                          .absolute_tol = 1.0e-6,
                                                          .max_iterations = 100,
                                                          .print_level = 1};
        auto [solver1, precond1] = smith::buildLinearSolverAndPreconditioner(iter_solver_options, mesh->getComm());
        auto [solver2, precond2] = smith::buildLinearSolverAndPreconditioner(iter_solver_options, mesh->getComm());
        solvers.push_back(std::move(solver1));
        solvers.push_back(std::move(solver2));
        // So that preconds don't go out of scope
        preconds.push_back(std::move(precond1));
        preconds.push_back(std::move(precond2));
    } else if (test_params.solver_type == BlockSolverType::BoomerAMG) {
        auto solver1 = std::make_unique<mfem::HypreBoomerAMG>();
        auto solver2 = std::make_unique<mfem::HypreBoomerAMG>();
        solvers.push_back(std::move(solver1));
        solvers.push_back(std::move(solver2));
    }

    // Solver inputs (might need here for custom solvers)
    auto time = graph->create_state<double,double>(0.0);
    auto dt = graph->create_state<double,double>(0.025);
    int cycle = 0;
    std::vector<smith::FieldState> params;
    auto& T1_params = params;
    auto& T2_params = params;
    std::vector<FieldState> T1_arguments{T1, T2, gamma};
    std::vector<FieldState> T2_arguments{T1, T2, gamma};
    std::unique_ptr<mfem::Solver> diff_precond;

    switch (test_params.precond_type) {
        case BlockPrecondType::Diagonal:
            diff_precond = std::make_unique<smith::BlockDiagonalPreconditioner>(block_offsets_, std::move(solvers));
            break;
        case BlockPrecondType::TriLower:
            diff_precond = std::make_unique<smith::BlockTriangularPreconditioner>(block_offsets_, std::move(solvers), smith::BlockTriangularType::Lower);
            break;
        case BlockPrecondType::TriUpper:
            diff_precond = std::make_unique<smith::BlockTriangularPreconditioner>(block_offsets_, std::move(solvers), smith::BlockTriangularType::Upper);
            break;
        case BlockPrecondType::TriSym:
            diff_precond = std::make_unique<smith::BlockTriangularPreconditioner>(block_offsets_, std::move(solvers), smith::BlockTriangularType::Symmetric);
            break;
        case BlockPrecondType::SchurLower:
            diff_precond = std::make_unique<smith::BlockSchurPreconditioner>(block_offsets_, std::move(solvers), smith::BlockSchurType::Lower);
            break;
        case BlockPrecondType::SchurUpper:
            diff_precond = std::make_unique<smith::BlockSchurPreconditioner>(block_offsets_, std::move(solvers), smith::BlockSchurType::Upper);
            break;
        case BlockPrecondType::SchurDiag:
            diff_precond = std::make_unique<smith::BlockSchurPreconditioner>(block_offsets_, std::move(solvers), smith::BlockSchurType::Diagonal);
            break;
        case BlockPrecondType::SchurFull:
            diff_precond = std::make_unique<smith::BlockSchurPreconditioner>(block_offsets_, std::move(solvers), smith::BlockSchurType::Full);
            break;
        case BlockPrecondType::SchurFullA22:
            diff_precond = std::make_unique<smith::BlockSchurPreconditioner>(block_offsets_, std::move(solvers), smith::BlockSchurType::Full,
                                                                                                             smith::SchurApproxType::A22Only);
            break;
        case BlockPrecondType::SchurFullCustom:
            std::vector<double> jacobian_weights{0.0, 1.0, 0.0};
            std::vector<smith::ConstFieldPtr> T2_field_ptrs;
            T2_field_ptrs.reserve(T2_arguments.size());
            for (const auto& f : T2_arguments) {
            T2_field_ptrs.push_back(f.get().get());  // FieldState -> shared_ptr -> raw ptr
            }
            auto S_approx = T2_form.jacobian(smith::TimeInfo(time.get(), dt.get(), cycle), shape_disp.get().get(), T2_field_ptrs, jacobian_weights);
            std::vector<BlockOverride> overrides;
            overrides.emplace_back(
            1,
            std::shared_ptr<const mfem::Operator>(std::move(S_approx)) // transfer ownership
            );
            diff_precond = std::make_unique<smith::BlockSchurPreconditioner>(block_offsets_, std::move(solvers), smith::BlockSchurType::Full,
                                                                             smith::SchurApproxType::Custom, overrides);
            break;
    }

    std::unique_ptr<mfem::Solver> linear_solver = std::make_unique<mfem::GMRESSolver>(mesh->getComm());
    mfem::GMRESSolver* iter_lin_solver = dynamic_cast<mfem::GMRESSolver*>(linear_solver.get());

    iter_lin_solver->iterative_mode = false;
    iter_lin_solver->SetRelTol(default_linear_options.relative_tol);
    iter_lin_solver->SetAbsTol(default_linear_options.absolute_tol);
    iter_lin_solver->SetMaxIter(default_linear_options.max_iterations);
    iter_lin_solver->SetPrintLevel(default_linear_options.print_level);
    iter_lin_solver->SetPreconditioner(*diff_precond);

    std::shared_ptr<smith::DifferentiableBlockSolver> d_linear_solver =
        std::make_shared<smith::LinearDifferentiableBlockSolver>(std::move(linear_solver), std::move(diff_precond));

    auto sols = block_solve({&T1_form, &T2_form},
                            {{0,1}, {0,1}},
                            shape_disp, {T1_arguments, T2_arguments},
                            {T1_params, T2_params},
                            smith::TimeInfo(time.get(), dt.get(), cycle), d_linear_solver.get(), {T1_bc_manager.get(), T2_bc_manager.get()});

    auto pv_writer = smith::createParaviewWriter(*mesh, sols, physics_name);
    pv_writer.write(0, 0.0, sols);

    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(
    BlockPrecondSweep,
    BlockPreconditionerTest,
    ::testing::Values(
        // Direct solvers
        BlockTestParams{BlockSolverType::Direct, BlockPrecondType::Diagonal},
        BlockTestParams{BlockSolverType::Direct, BlockPrecondType::TriLower},
        BlockTestParams{BlockSolverType::Direct, BlockPrecondType::TriUpper},
        BlockTestParams{BlockSolverType::Direct, BlockPrecondType::TriSym},
        BlockTestParams{BlockSolverType::Direct, BlockPrecondType::SchurLower},
        BlockTestParams{BlockSolverType::Direct, BlockPrecondType::SchurUpper},
        BlockTestParams{BlockSolverType::Direct, BlockPrecondType::SchurDiag},
        BlockTestParams{BlockSolverType::Direct, BlockPrecondType::SchurFull},

        // Iterative solvers
        BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::Diagonal},
        BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::TriLower},
        BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::TriUpper},
        BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::TriSym},
        BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::SchurLower},
        BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::SchurUpper},
        BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::SchurDiag},
        BlockTestParams{BlockSolverType::Iterative, BlockPrecondType::SchurFull},

        // BoomerAMG solvers
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::Diagonal},
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::TriLower},
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::TriUpper},
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::TriSym},
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurLower},
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurUpper},
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurDiag},
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurFull},
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurFullA22},
        BlockTestParams{BlockSolverType::BoomerAMG, BlockPrecondType::SchurFullCustom}
    ),
    BlockParamNameGenerator);

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    smith::ApplicationManager applicationManager(argc, argv);
    return RUN_ALL_TESTS();
}