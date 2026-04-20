// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_thermo_mechanics_three_system.cpp
 * @brief Tests 3-system coupling: SolidMechanicsSystem + ThermalSystem + StateVariableSystem.
 *
 * Validates N>2 system coupling end-to-end via combineSystems.
 *
 * Layout:
 *  - Solid: receives temperature coupling (1-way: temperature affects elastic modulus via expansion).
 *  - Thermal: standalone heat diffusion with a heat source.
 *  - State (damage): receives solid displacement coupling; damage evolves with strain norm.
 */

#include <memory>
#include "gtest/gtest.h"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"
#include "smith/differentiable_numerics/state_variable_system.hpp"
#include "smith/differentiable_numerics/combined_system.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"

namespace smith {

static constexpr int dim3 = 3;
static constexpr int disp_ord = 1;
static constexpr int temp_ord = 1;
static constexpr int state_ord = 0;
using ThreeSystemStateSpace = L2<state_ord>;

struct ThreeSystemMeshFixture : public testing::Test {
  void SetUp() override
  {
    datastore_ = std::make_unique<axom::sidre::DataStore>();
    smith::StateManager::initialize(*datastore_, "three_system");
    mesh_ = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(4, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 0.1, 0.1), "mesh", 0, 0);
    mesh_->addDomainOfBoundaryElements("left", smith::by_attr<dim3>(3));
    mesh_->addDomainOfBoundaryElements("right", smith::by_attr<dim3>(5));
  }

  std::unique_ptr<axom::sidre::DataStore> datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
};

// 3-system staggered: solid (with thermal coupling) + thermal + state (with displacement coupling).
TEST_F(ThreeSystemMeshFixture, StaggeredThreeSystems)
{
  smith::LinearSolverOptions lin_opts{.linear_solver = smith::LinearSolver::SuperLU,
                                      .relative_tol = 1e-8,
                                      .absolute_tol = 1e-10,
                                      .max_iterations = 200,
                                      .print_level = 0};
  smith::NonlinearSolverOptions nonlin_opts{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                            .relative_tol = 1e-7,
                                            .absolute_tol = 1e-8,
                                            .max_iterations = 20,
                                            .max_line_search_iterations = 6,
                                            .print_level = 0};

  auto solid_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);
  auto thermal_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);
  auto state_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);

  auto field_store = std::make_shared<FieldStore>(mesh_, 100, "three");

  using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;
  using StateRule = BackwardEulerFirstOrderTimeIntegrationRule;

  DispRule disp_rule;
  TempRule temp_rule;
  StateRule state_rule;

  // Phase 1: register all fields.
  // registerSolidMechanicsFields must come before registerThermalFields to get
  // the displacement field tokens available for thermal coupling.
  auto solid_exported = registerSolidMechanicsFields<dim3, disp_ord, DispRule>(field_store);
  auto thermal_exported = registerThermalFields<dim3, temp_ord>(field_store, temp_rule);
  registerStateVariableFields<ThreeSystemStateSpace>(field_store, state_rule);

  // Phase 2: build each system.

  // Solid receives thermal coupling (temperature_solve_state, temperature).
  auto [solid, solid_cz, solid_end] = buildSolidMechanicsSystem<dim3, disp_ord, DispRule>(
      field_store, thermal_exported, std::make_shared<SystemSolver>(solid_block_solver), SolidMechanicsOptions{});

  // Thermal is standalone (no coupling back from solid for this test).
  auto [thermal, thermal_cz, thermal_end] = buildThermalSystem<dim3, temp_ord, TempRule>(
      field_store, CouplingParams<>{}, std::make_shared<SystemSolver>(thermal_block_solver), ThermalOptions{});

  // StateVariable receives solid displacement coupling (4 fields: disp_ss, displacement, velocity, acceleration).
  auto [state_sys, state_cz, state_end] = buildStateVariableSystem<dim3, ThreeSystemStateSpace>(
      field_store, state_rule, solid_exported, std::make_shared<SystemSolver>(state_block_solver),
      StateVariableOptions{});

  // Phase 3: register material integrands.

  // Solid: thermoelastic — temperature drives isotropic expansion.
  // Lambda args: (t_info, X, u, u_old, v_old, a_old, temperature_ss, temperature_old)
  {
    auto captured_disp_rule = solid->disp_time_rule;
    auto captured_temp_rule = thermal->temperature_time_rule;
    solid->solid_weak_form->addBodyIntegral(
        mesh_->entireBodyName(), [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old, auto a_old,
                                     auto temperature_ss, auto temperature_old) {
          auto [u_c, v_c, a_c] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
          auto T = captured_temp_rule->value(t_info, temperature_ss, temperature_old);

          double E = 100.0, nu = 0.25, alpha_th = 0.001, density = 1.0;
          double lam = E * nu / ((1 + nu) * (1 - 2 * nu));
          double mu = E / (2 * (1 + nu));
          static constexpr auto I = Identity<dim3>();
          auto eps = sym(get<DERIVATIVE>(u_c));
          auto T_val = get<VALUE>(T);
          auto sigma = lam * (tr(eps) - 3.0 * alpha_th * T_val) * I + 2.0 * mu * eps;
          return smith::tuple{get<VALUE>(a_c) * density, sigma};
        });
  }

  // Thermal: simple conduction with constant heat source.
  // Lambda args: (t_info, X, T, T_old)
  {
    auto captured_temp_rule = thermal->temperature_time_rule;
    const double kappa = 0.05, C_v = 1.0, heat_source = 0.5;
    thermal->thermal_weak_form->addBodyIntegral(mesh_->entireBodyName(),
                                                [=](auto t_info, auto /*X*/, auto T, auto T_old) {
                                                  auto [T_c, T_dot] = captured_temp_rule->interpolate(t_info, T, T_old);
                                                  auto q = kappa * get<DERIVATIVE>(T_c);
                                                  return smith::tuple{C_v * get<VALUE>(T_dot) - heat_source, -q};
                                                });
  }

  // State (damage): evolves as d_dot = (1 - d) * eps_norm.
  // addStateEvolution lambda args: (t_info, alpha_val, alpha_dot, disp_ss, displacement, velocity, acceleration)
  {
    auto captured_disp_rule = solid->disp_time_rule;
    state_sys->addStateEvolution(mesh_->entireBodyName(),
                                 [=](auto t_info, auto alpha_val, auto alpha_dot, auto u_ss, auto u, auto v, auto a) {
                                   auto [u_c, v_c, a_c] = captured_disp_rule->interpolate(t_info, u_ss, u, v, a);
                                   auto eps = sym(get<DERIVATIVE>(u_c));
                                   using std::sqrt;
                                   auto eps_norm = sqrt(inner(eps, eps) + 1e-16);
                                   return alpha_dot - (1.0 - alpha_val) * eps_norm;
                                 });
  }

  // Phase 4: boundary conditions.
  solid->setDisplacementBC(mesh_->domain("left"));
  thermal->setTemperatureBC(mesh_->domain("left"));

  // Compressive traction on right face.
  // Lambda args from addTraction: (t, X, n, u, v, a, temp_ss, temp_old)
  // — 6 self fields + 2 thermal coupling fields forwarded as trailing args.
  solid->addTraction("right", [](double, auto X, auto /*n*/, auto /*u*/, auto /*v*/, auto /*a*/, auto /*temp_ss*/,
                                 auto /*temp_old*/) {
    auto t = 0.0 * X;
    t[0] = -0.005;
    return t;
  });

  // Phase 5: combine and solve.
  auto [combined, combined_cz] = combineSystems(solid, thermal, state_sys);

  double dt = 1.0, time = 0.0;
  auto shape_disp = field_store->getShapeDisp();
  auto states = field_store->getStateFields();
  auto params = field_store->getParameterFields();
  std::vector<ReactionState> reactions;

  for (size_t step = 0; step < 2; ++step) {
    std::tie(states, reactions) =
        makeAdvancer(combined, combined_cz)->advanceState(smith::TimeInfo(time, dt, step), shape_disp, states, params);
    time += dt;
  }

  // All subsystems should converge and produce non-trivial solutions.
  EXPECT_TRUE(solid_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged())
      << "Solid solver did not converge";
  EXPECT_TRUE(thermal_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged())
      << "Thermal solver did not converge";
  EXPECT_TRUE(state_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged())
      << "State solver did not converge";

  // Displacement should be non-zero (compressive traction).
  mfem::Vector final_disp(*states[field_store->getFieldIndex("three_displacement_solve_state")].get());
  double max_disp = final_disp.Normlinf();
  EXPECT_GT(max_disp, 1e-8) << "Displacement should be non-zero under compressive traction";

  // Temperature should be non-zero (heat source applied).
  mfem::Vector final_temp(*states[field_store->getFieldIndex("three_temperature_solve_state")].get());
  double max_temp = final_temp.Normlinf();
  EXPECT_GT(max_temp, 1e-10) << "Temperature should be non-zero under heat source";

  // Damage should be non-zero (driven by strain norm).
  mfem::Vector final_state(*states[field_store->getFieldIndex("three_state_solve_state")].get());
  double max_state = final_state.Normlinf();
  EXPECT_GT(max_state, 1e-10) << "Damage state should grow under deformation";
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
