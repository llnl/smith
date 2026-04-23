// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_thermo_mechanics_three_system.cpp
 * @brief Tests 3-system coupling: SolidMechanicsSystem + ThermalSystem + InternalVariableSystem.
 *
 * Validates N>2 system coupling end-to-end via combineSystems.
 *
 * Layout:
 *  - Solid: receives temperature coupling (1-way: temperature affects elastic modulus via expansion).
 *  - Thermal: standalone heat diffusion with a heat source.
 *  - Internal variable (damage): receives solid displacement coupling; damage evolves with strain norm.
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
#include "smith/differentiable_numerics/solid_mechanics_with_internal_vars_system.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"
#include "smith/differentiable_numerics/thermo_mechanical_system.hpp"
#include "smith/differentiable_numerics/state_variable_system.hpp"
#include "smith/differentiable_numerics/combined_system.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"

namespace smith {

static constexpr int dim3 = 3;
static constexpr int disp_ord = 1;
static constexpr int temp_ord = 1;
static constexpr int state_ord = 0;
using StateSpace = L2<state_ord>;

struct SimpleThermoelasticMaterial {
  using State = smith::QOI;

  double density = 1.0;
  double E = 100.0;
  double nu = 0.25;
  double alpha_th = 0.001;
  double kappa = 0.05;
  double heat_capacity = 1.0;
  double heat_source = 0.5;

  template <typename DT, typename StateType, typename GradUType, typename GradVType, typename ThetaType,
            typename GradThetaType, typename... Params>
  SMITH_HOST_DEVICE auto operator()(DT /*dt*/, StateType /*state*/, GradUType grad_u, GradVType /*grad_v*/,
                                    ThetaType theta, GradThetaType grad_theta, Params... /*params*/) const
  {
    double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));
    static constexpr auto I = Identity<dim3>();
    auto eps = sym(grad_u);
    auto pk = lam * (tr(eps) - 3.0 * alpha_th * theta) * I + 2.0 * mu * eps;
    auto q0 = kappa * grad_theta;
    return smith::tuple{pk, heat_capacity, heat_source, q0};
  }
};

struct StrainDrivenInternalVariableMaterial {
  template <typename TimeInfo, typename AlphaType, typename AlphaDotType, typename DerivType, typename... Params>
  SMITH_HOST_DEVICE auto operator()(TimeInfo /*t_info*/, AlphaType alpha, AlphaDotType alpha_dot, DerivType grad_u,
                                    Params... /*params*/) const
  {
    using std::sqrt;
    auto eps = sym(grad_u);
    auto eps_norm = sqrt(inner(eps, eps) + 1e-16);
    return alpha_dot - (1.0 - alpha) * eps_norm;
  }
};

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

// 3-system staggered: solid + thermal + internal variable.
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
  auto internal_variable_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);

  auto field_store = std::make_shared<FieldStore>(mesh_, 100, "three");

  using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;
  using InternalVariableRule = BackwardEulerFirstOrderTimeIntegrationRule;

  // Phase 1: register all fields.
  // registerSolidMechanicsFields must come before registerThermalFields to get
  // the displacement field tokens available for thermal coupling.
  auto solid_fields = registerSolidMechanicsFields<dim3, disp_ord, DispRule>(field_store);
  auto thermal_fields = registerThermalFields<dim3, temp_ord, TempRule>(field_store);
  auto internal_variable_fields = registerInternalVariableFields<StateSpace, InternalVariableRule>(field_store);

  // Phase 2: build each system.

  // Solid receives thermal coupling (temperature_solve_state, temperature).
  auto solid = buildSolidMechanicsSystem<dim3, disp_ord, DispRule>(
      std::make_shared<SystemSolver>(solid_block_solver), SolidMechanicsOptions{}, solid_fields, thermal_fields);

  auto thermal = buildThermalSystem<dim3, temp_ord, TempRule>(std::make_shared<SystemSolver>(thermal_block_solver),
                                                              ThermalOptions{}, thermal_fields, solid_fields);

  auto internal_variables = buildInternalVariableSystem<dim3, StateSpace, InternalVariableRule>(
      std::make_shared<SystemSolver>(internal_variable_block_solver), InternalVariableOptions{},
      internal_variable_fields, solid_fields);

  // Phase 3: register material integrands.
  setCoupledThermoMechanicsMaterial(solid, thermal, SimpleThermoelasticMaterial{}, mesh_->entireBodyName());
  setCoupledInternalVariableMaterial(internal_variables, solid, StrainDrivenInternalVariableMaterial{},
                                     mesh_->entireBodyName());

  // Phase 4: boundary conditions.
  solid->setDisplacementBC(mesh_->domain("left"));
  thermal->setTemperatureBC(mesh_->domain("left"));

  // Compressive traction on right face.
  // Lambda args from addTraction: (t, X, n, u, v, a, temp_ss, temp_old)
  // — 6 self fields + 2 thermal coupling fields forwarded as trailing args.
  solid->addTraction(
      "right", [](double, auto X, auto /*n*/, auto /*u*/, auto /*v*/, auto /*a*/, auto /*temp_ss*/, auto /*temp_old*/) {
        auto t = 0.0 * X;
        t[0] = -0.005;
        return t;
      });

  // Phase 5: combine and solve.
  auto combined = combineSystems(solid, thermal, internal_variables);

  double dt = 1.0, time = 0.0;
  auto shape_disp = field_store->getShapeDisp();
  auto states = field_store->getStateFields();
  auto params = field_store->getParameterFields();
  std::vector<ReactionState> reactions;

  for (size_t step = 0; step < 2; ++step) {
    std::tie(states, reactions) =
        makeAdvancer(combined)->advanceState(smith::TimeInfo(time, dt, step), shape_disp, states, params);
    time += dt;
  }

  // All subsystems should converge and produce non-trivial solutions.
  EXPECT_TRUE(solid_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged())
      << "Solid solver did not converge";
  EXPECT_TRUE(thermal_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged())
      << "Thermal solver did not converge";
  EXPECT_TRUE(internal_variable_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged())
      << "Internal-variable solver did not converge";

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
