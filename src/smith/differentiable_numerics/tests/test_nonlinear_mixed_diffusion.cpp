// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_nonlinear_mixed_diffusion.cpp
 *
 * @brief Test field-dependent Schur complement preconditioning.
 *
 * Problem: Nonlinear thermal diffusion with temperature-dependent conductivity
 *
 *   -∇·(k(T)∇T) = f    in Ω
 *   T = 0              on ∂Ω
 *
 * where k(T) = k₀(1 + α·T) is temperature-dependent conductivity.
 *
 * Mixed formulation (introduce flux q = -k(T)∇T):
 *
 *   q + k(T)∇T = 0     (constitutive equation)
 *   ∇·q = f            (balance equation)
 *
 * This gives a 2×2 saddle-point system:
 *
 *   [K(T)   B^T] [q]   [0]
 *   [B      0  ] [T] = [f]
 *
 * where K(T) = k(T)⁻¹ I and B = ∇· operator.
 *
 * The Schur complement S = -B K(T)⁻¹ B^T ≈ -∇·(k(T)∇) depends on the
 * temperature field T. At each Newton iteration, we need to update the
 * Schur complement approximation with the current temperature iterate.
 *
 * This example demonstrates:
 * 1. Mixed formulation with field-dependent coefficients
 * 2. Custom Schur complement operator that depends on solution field
 * 3. State update plumbing in the Newton loop
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "smith/smith_config.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/weak_form_block_operator.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/functional_weak_form.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "gretl/data_store.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"

using namespace smith;

// Taylor-Hood-style elements for stability
using ShapeDispSpace = H1<1, 2>;
using FluxSpace = H1<2, 2>;      // Vector flux q
using TemperatureSpace = H1<1>;  // Scalar temperature T
using TemperatureSchurForm = smith::FunctionalWeakForm<2, TemperatureSpace, smith::Parameters<TemperatureSpace>>;

/**
 * @brief Temperature-dependent conductivity k(T) = k₀(1 + α·T)
 */
struct ThermalConductivity {
  double k0 = 1.0;     // Base conductivity
  double alpha = 0.5;  // Temperature dependence parameter

  // Support both double and dual numbers for automatic differentiation
  template <typename T>
  auto operator()(const T& temp) const
  {
    return k0 * (1.0 + alpha * temp);
  }
};

enum class PreconditionerCase
{
  StateDependentCustomFullSchur,
  StateDependentCustomActionSchur
};

const char* preconditionerCaseName(PreconditionerCase preconditioner)
{
  switch (preconditioner) {
    case PreconditionerCase::StateDependentCustomFullSchur:
      return "StateDependentCustomFullSchur";
    case PreconditionerCase::StateDependentCustomActionSchur:
      return "StateDependentCustomActionSchur";
  }
  return "Unknown";
}

std::string preconditionerCaseNameGenerator(const ::testing::TestParamInfo<PreconditionerCase>& info)
{
  return preconditionerCaseName(info.param);
}

class MeshFixture : public testing::Test {
 protected:
  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  ThermalConductivity k_func;

  void SetUp() override
  {
    smith::StateManager::initialize(datastore, "nonlinear_mixed_diffusion");

    MPI_Barrier(MPI_COMM_WORLD);
    int serial_refinement = 3;
    int parallel_refinement = 0;

    std::string filename = SMITH_REPO_DIR "/data/meshes/square_attribute.mesh";

    const std::string meshtag = "mesh";
    mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(filename), meshtag, serial_refinement,
                                         parallel_refinement);

    // Set up nonlinear conductivity
    k_func.k0 = 1.0;
    k_func.alpha = 0.5;  // Moderate nonlinearity
  }
};

class NonlinearMixedDiffusionPreconditionerTest : public MeshFixture,
                                                  public ::testing::WithParamInterface<PreconditionerCase> {};

/**
 * @brief Custom Schur action solver that applies an approximate inverse of
 *        the temperature diffusion Schur approximation.
 *
 * This exercises the CustomAction path: the block Schur preconditioner does not
 * receive a custom Schur operator. Instead, the block-1 solver owns the
 * state-dependent operator and applies an internal AMG-preconditioned GMRES solve.
 */
class TemperatureDiffusionSchurActionSolver : public smith::SchurComplementActionSolver {
 public:
  TemperatureDiffusionSchurActionSolver(std::unique_ptr<mfem::HypreParMatrix> initial_schur_operator,
                                        smith::StateDependentWeakFormOperator schur_operator_update)
      : schur_operator_update_(std::move(schur_operator_update)), schur_operator_(std::move(initial_schur_operator))
  {
    configureLinearSolver();
  }

  void updateForState(const mfem::Vector& state, const mfem::Array<int>& block_offsets) override
  {
    schur_operator_ = schur_operator_update_(state, block_offsets);
    configureLinearSolver();
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    y.SetSize(x.Size());
    y = 0.0;
    linear_solver_->Mult(x, y);
  }

 private:
  void configureLinearSolver()
  {
    amg_solver_ = std::make_unique<mfem::HypreBoomerAMG>();
    amg_solver_->SetPrintLevel(0);
    amg_solver_->SetOperator(*schur_operator_);

    linear_solver_ = std::make_unique<mfem::GMRESSolver>(schur_operator_->GetComm());
    linear_solver_->SetRelTol(1.0e-8);
    linear_solver_->SetAbsTol(0.0);
    linear_solver_->SetMaxIter(100);
    linear_solver_->SetPrintLevel(0);
    linear_solver_->SetPreconditioner(*amg_solver_);
    linear_solver_->SetOperator(*schur_operator_);
    linear_solver_->iterative_mode = false;
  }

  smith::StateDependentWeakFormOperator schur_operator_update_;
  std::unique_ptr<mfem::HypreParMatrix> schur_operator_;
  std::unique_ptr<mfem::HypreBoomerAMG> amg_solver_;
  std::unique_ptr<mfem::GMRESSolver> linear_solver_;
};

// Exercises state-dependent Schur preconditioners on nonlinear mixed diffusion.
TEST_P(NonlinearMixedDiffusionPreconditionerTest, BlockSolve)
{
  const auto preconditioner_case = GetParam();

  std::string physics_name = std::string("nonlinear_thermal_") + preconditionerCaseName(preconditioner_case);
  auto graph = std::make_shared<gretl::DataStore>(std::make_unique<gretl::WangCheckpointStrategy>(100));

  // Create field states
  auto shape_disp = createFieldState(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
  auto flux = createFieldState(*graph, FluxSpace{}, physics_name + "_flux", mesh->tag());
  auto temperature = createFieldState(*graph, TemperatureSpace{}, physics_name + "_temperature", mesh->tag());

  // Set up weak forms
  smith::FunctionalWeakForm<2, FluxSpace, smith::Parameters<FluxSpace, TemperatureSpace>> constitutive_form(
      "constitutive", mesh, space(flux), spaces({flux, temperature}));

  smith::FunctionalWeakForm<2, TemperatureSpace, smith::Parameters<FluxSpace, TemperatureSpace>> balance_form(
      "balance", mesh, space(temperature), spaces({flux, temperature}));

  TemperatureSchurForm temperature_schur_form("temperature_schur", mesh, space(temperature), spaces({temperature}));

  // Constitutive equation: q + k(T)∇T = 0
  constitutive_form.addBodyIntegral(DependsOn<0, 1>{}, mesh->entireBodyName(),
                                    [this](auto /* time_info */, auto /* x */, auto Q, auto T) {
                                      auto q = get<VALUE>(Q);
                                      auto temp = get<VALUE>(T);
                                      auto grad_temp = get<DERIVATIVE>(T);

                                      // k(T) = k₀(1 + α·T)
                                      auto k_val = k_func(temp);

                                      // Residual: q + k(T)∇T
                                      auto residual = q + k_val * grad_temp;

                                      return smith::tuple{residual, smith::zero{}};
                                    });

  // Balance equation: ∇·q = f
  balance_form.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [](auto /* time_info */, auto /* x */, auto Q) {
    auto div_q = smith::tr(get<DERIVATIVE>(Q));

    double f = 1.0;

    return smith::tuple{-f + div_q, smith::zero{}};
  });

  temperature_schur_form.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(),
                                         [this](auto /* time_info */, auto /* x */, auto T) {
                                           auto temp = get<VALUE>(T);
                                           auto grad_temp = get<DERIVATIVE>(T);

                                           return smith::tuple{smith::zero{}, k_func(temp) * grad_temp};
                                         });

  // Boundary conditions: T = 0 on all boundaries
  auto flux_bc_manager = std::make_shared<smith::BoundaryConditionManager>(mesh->mfemParMesh());
  auto temp_bc_manager = std::make_shared<smith::BoundaryConditionManager>(mesh->mfemParMesh());

  auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
  temp_bc_manager->addEssential(std::set<int>{1, 2, 3, 4}, zero_bcs, space(temperature), 0);

  // Linear solver options
  smith::LinearSolverOptions linear_options;
  linear_options.linear_solver = smith::LinearSolver::GMRES;
  linear_options.relative_tol = 1.0e-10;
  linear_options.absolute_tol = 1.0e-14;
  linear_options.max_iterations = 500;
  linear_options.print_level = 0;

  smith::LinearSolverOptions amg_options;
  amg_options.linear_solver = smith::LinearSolver::PrecondOnly;
  amg_options.preconditioner = smith::Preconditioner::HypreAMG;
  const smith::LinearSolverOptions flux_solver_options = amg_options;
  const smith::LinearSolverOptions schur_solver_options = amg_options;

  linear_options.preconditioner = smith::Preconditioner::BlockSchur;
  linear_options.block_schur_type = smith::BlockSchurType::Full;

  // Nonlinear solver options
  smith::NonlinearSolverOptions nonlin_opts;
  nonlin_opts.nonlin_solver = smith::NonlinearSolver::Newton;
  nonlin_opts.relative_tol = 1.0e-10;
  nonlin_opts.absolute_tol = 1.0e-12;
  nonlin_opts.max_iterations = 12;
  nonlin_opts.print_level = 0;

  std::shared_ptr<smith::NonlinearBlockSolver> nonlinear_solver;
  auto time = graph->create_state<double, double>(0.0);
  auto dt = graph->create_state<double, double>(1.0);
  size_t cycle = 0;
  const auto time_info = smith::TimeInfo(time.get(), dt.get(), cycle);

  if (preconditioner_case == PreconditionerCase::StateDependentCustomFullSchur) {
    linear_options.schur_approx_type = smith::SchurApproxType::Custom;
    const std::vector<smith::LinearSolverOptions> sub_solver_options{flux_solver_options, schur_solver_options};
    auto sub_solvers = smith::buildBlockPreconditionerSubSolvers(sub_solver_options, mesh->getComm());

    std::vector<smith::BlockProviderOverride> overrides;
    overrides.push_back(smith::makeStateDependentWeakFormBlockProviderOverride(
        1, temperature_schur_form, shape_disp, {temperature}, {1.0}, time_info, temp_bc_manager.get(),
        {smith::StateBlockBinding{1, 0}}));

    auto preconditioner =
        std::make_unique<smith::BlockSchurPreconditioner>(std::move(sub_solvers), linear_options.block_schur_type,
                                                          linear_options.schur_approx_type, std::move(overrides));
    nonlinear_solver = smith::buildNonlinearBlockSolver(nonlin_opts, linear_options, *mesh, std::move(preconditioner));
  } else if (preconditioner_case == PreconditionerCase::StateDependentCustomActionSchur) {
    linear_options.schur_approx_type = smith::SchurApproxType::CustomAction;

    const std::vector<smith::LinearSolverOptions> sub_solver_options{flux_solver_options};
    auto sub_solvers = smith::buildBlockPreconditionerSubSolvers(sub_solver_options, mesh->getComm());

    auto action_solver = std::make_unique<TemperatureDiffusionSchurActionSolver>(
        smith::buildWeakFormOperator(temperature_schur_form, shape_disp, {temperature}, {1.0}, time_info,
                                     temp_bc_manager.get()),
        smith::makeStateDependentWeakFormOperator(temperature_schur_form, shape_disp, {temperature}, {1.0}, time_info,
                                                  temp_bc_manager.get(), {smith::StateBlockBinding{1, 0}}));
    sub_solvers.push_back(std::move(action_solver));

    auto preconditioner = std::make_unique<smith::BlockSchurPreconditioner>(
        std::move(sub_solvers), linear_options.block_schur_type, linear_options.schur_approx_type);
    nonlinear_solver = smith::buildNonlinearBlockSolver(nonlin_opts, linear_options, *mesh, std::move(preconditioner));
  }

  ASSERT_TRUE(nonlinear_solver != nullptr);

  // Solve
  std::vector<smith::FieldState> params;

  auto sols = block_solve({&constitutive_form, &balance_form}, {{0, 1}, {0, 1}}, shape_disp,
                          {{flux, temperature}, {flux, temperature}}, {params, params}, time_info,
                          nonlinear_solver.get(), {flux_bc_manager.get(), temp_bc_manager.get()});

  // Verify convergence
  EXPECT_EQ(sols.size(), 2);

  double temp_norm = sols[1].get()->Norml2();
  double flux_norm = sols[0].get()->Norml2();
  auto& newton_solver = nonlinear_solver->nonlinear_solver_->nonlinearSolver();
  EXPECT_TRUE(newton_solver.GetConverged());
  EXPECT_TRUE(std::isfinite(temp_norm));
  EXPECT_TRUE(std::isfinite(flux_norm));
  EXPECT_GT(temp_norm, 1e-6);
  EXPECT_GT(flux_norm, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(StateDependentSchur, NonlinearMixedDiffusionPreconditionerTest,
                         ::testing::Values(PreconditionerCase::StateDependentCustomFullSchur,
                                           PreconditionerCase::StateDependentCustomActionSchur),
                         preconditionerCaseNameGenerator);

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
