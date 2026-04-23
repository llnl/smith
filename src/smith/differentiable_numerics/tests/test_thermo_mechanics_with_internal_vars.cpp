// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_thermo_mechanics_with_internal_vars.cpp
 * @brief Tests thermo-mechanics with internal variables using three composed systems.
 *
 * Validates N>2 system coupling end-to-end via combineSystems.
 *
 * Layout:
 *  - Solid: receives temperature and alpha coupling.
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

static constexpr int dim = 3;
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

  template <typename AlphaType>
  SMITH_HOST_DEVICE auto effectiveYoungsModulus(AlphaType alpha) const
  {
    return E * (1.0 - 0.8 * alpha);
  }

  template <typename StateType, typename GradUType, typename GradVType, typename ThetaType, typename GradThetaType,
            typename AlphaType, typename... Params>
  SMITH_HOST_DEVICE auto operator()(const TimeInfo& /*t_info*/, StateType /*state*/, GradUType grad_u,
                                    GradVType /*grad_v*/, ThetaType theta, GradThetaType grad_theta, AlphaType alpha,
                                    Params... /*params*/) const
  {
    auto E_eff = effectiveYoungsModulus(alpha);
    auto lam = E_eff * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    auto mu = E_eff / (2.0 * (1.0 + nu));
    static constexpr auto I = Identity<dim>();
    auto eps = sym(grad_u);
    auto pk = lam * (tr(eps) - 3.0 * alpha_th * theta) * I + 2.0 * mu * eps;
    auto q0 = kappa * grad_theta;
    return smith::tuple{pk, heat_capacity, heat_source, q0};
  }

  template <typename StateType, typename GradUType, typename GradVType, typename ThetaType, typename GradThetaType,
            typename... Params>
  SMITH_HOST_DEVICE auto operator()(const TimeInfo& t_info, StateType state, GradUType grad_u, GradVType grad_v,
                                    ThetaType theta, GradThetaType grad_theta, Params... params) const
  {
    return (*this)(t_info, state, grad_u, grad_v, theta, grad_theta, 0.0, params...);
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
    smith::StateManager::initialize(*datastore_, "thermo_mechanics_with_internal_vars");
    mesh_ = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(4, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 0.1, 0.1), "mesh", 0, 0);
    mesh_->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
    mesh_->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  }

  std::unique_ptr<axom::sidre::DataStore> datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
};

TEST_F(ThreeSystemMeshFixture, StronglyCoupledThreeSystems)
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

  auto field_store = std::make_shared<FieldStore>(mesh_, 100);

  using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;
  using InternalVariableRule = BackwardEulerFirstOrderTimeIntegrationRule;

  // Phase 1: register all fields.
  // registerSolidMechanicsFields must come before registerThermalFields to get
  // the displacement field tokens available for thermal coupling.
  auto solid_fields = registerSolidMechanicsFields<dim, disp_ord, DispRule>(field_store);
  auto thermal_fields = registerThermalFields<dim, temp_ord, TempRule>(field_store);
  auto internal_variable_fields = registerInternalVariableFields<StateSpace, InternalVariableRule>(field_store);

  // Phase 2: build each system.
  // Solid receives thermal and alpha coupling.
  auto solid = buildSolidMechanicsSystem<dim, disp_ord>(std::make_shared<SystemSolver>(solid_block_solver),
                                                        SolidMechanicsOptions{}, solid_fields, thermal_fields,
                                                        internal_variable_fields);

  auto thermal = buildThermalSystem<dim, temp_ord>(std::make_shared<SystemSolver>(thermal_block_solver),
                                                   ThermalOptions{}, thermal_fields, solid_fields);

  auto internal_variables =
      buildInternalVariableSystem<dim, StateSpace>(std::make_shared<SystemSolver>(internal_variable_block_solver),
                                                   InternalVariableOptions{}, internal_variable_fields, solid_fields);

  // Phase 3: register material integrands.
  auto material = SimpleThermoelasticMaterial{};
  auto disp_rule = solid->disp_time_rule;
  auto temp_rule = thermal->temperature_time_rule;
  auto alpha_rule = internal_variables->internal_variable_time_rule;

  solid->solid_weak_form->addBodyIntegral(
      mesh_->entireBodyName(), [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old, auto a_old,
                                   auto temperature, auto temperature_old, auto alpha, auto alpha_old) {
        auto [u_current, v_current, a_current] = disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
        auto T = temp_rule->value(t_info, temperature, temperature_old);
        auto alpha_current = alpha_rule->value(t_info, alpha, alpha_old);

        SimpleThermoelasticMaterial::State state{};
        auto response = material(t_info, state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current), get<VALUE>(T),
                                 get<DERIVATIVE>(T), get<VALUE>(alpha_current));
        auto pk = get<0>(response);
        return smith::tuple{get<VALUE>(a_current) * material.density, pk};
      });

  thermal->thermal_weak_form->addBodyIntegral(
      mesh_->entireBodyName(),
      [=](auto t_info, auto /*X*/, auto T, auto T_old, auto disp, auto disp_old, auto v_old, auto a_old) {
        auto [T_current, T_dot] = temp_rule->interpolate(t_info, T, T_old);
        auto [u_current, v_current, a_current] = disp_rule->interpolate(t_info, disp, disp_old, v_old, a_old);

        SimpleThermoelasticMaterial::State state{};
        auto response = material(t_info, state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current),
                                 get<VALUE>(T_current), get<DERIVATIVE>(T_current));
        auto C_v = get<1>(response);
        auto s0 = get<2>(response);
        auto q0 = get<3>(response);
        return smith::tuple{C_v * get<VALUE>(T_dot) - s0, -q0};
      });

  setCoupledInternalVariableMaterial(internal_variables, solid, StrainDrivenInternalVariableMaterial{},
                                     mesh_->entireBodyName());

  // Phase 4: boundary conditions.
  solid->setDisplacementBC(mesh_->domain("left"));
  thermal->setTemperatureBC(mesh_->domain("left"));

  // Compressive traction on right face.
  // Lambda args from addTraction: (t, X, n, u, v, a, temp_ss, temp_old, alpha_ss, alpha_old)
  solid->addTraction("right", [](double, auto X, auto /*n*/, auto /*u*/, auto /*v*/, auto /*a*/, auto /*temp_ss*/,
                                 auto /*temp_old*/, auto /*alpha_ss*/, auto /*alpha_old*/) {
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
  auto final_disp = states[field_store->getFieldIndex("displacement")].get();
  EXPECT_GT(final_disp->Normlinf(), 1e-8) << "Displacement should be non-zero under compressive traction";

  // Temperature should be non-zero (heat source applied).
  auto final_temp = states[field_store->getFieldIndex("temperature")].get();
  EXPECT_GT(final_temp->Normlinf(), 1e-10) << "Temperature should be non-zero under heat source";

  // Damage should be non-zero (driven by strain norm).
  auto final_state = states[field_store->getFieldIndex("state")].get();
  EXPECT_GT(final_state->Normlinf(), 1e-10) << "Damage state should grow under deformation";
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
