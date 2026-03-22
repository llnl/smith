// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_with_internal_vars_system.hpp
 * @brief Defines the SolidMechanicsWithInternalVarsSystem struct and its factory function
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/system_base.hpp"

namespace smith {

/**
 * @brief System struct for solid statics with an additional L2 state variable.
 *
 * Uses 2-state layout for each field (predicted + history), for a total of 4 state fields.
 *
 * @tparam dim Spatial dimension.
 * @tparam disp_order Polynomial order for displacement field.
 * @tparam StateSpace Finite element space for the state variable (e.g., L2<order>).
 * @tparam DisplacementTimeRule Time integration rule for displacement (must have num_states == 2).
 * @tparam InternalVarTimeRule Time integration rule for the state variable (must have num_states == 2).
 * @tparam parameter_space Parameter spaces for material properties.
 */
template <int dim, int disp_order, typename StateSpace,
          typename DisplacementTimeRule = QuasiStaticFirstOrderTimeIntegrationRule,
          typename InternalVarTimeRule = BackwardEulerFirstOrderTimeIntegrationRule, typename... parameter_space>
struct SolidMechanicsWithInternalVarsSystem : public SystemBase {
  static_assert(DisplacementTimeRule::num_states == 2, "SolidMechanicsWithInternalVarsSystem requires a 2-state displacement rule");
  static_assert(InternalVarTimeRule::num_states == 2, "SolidMechanicsWithInternalVarsSystem requires a 2-state state rule");

  // Primary weak form: residual for displacement (u).
  // Inputs: u, u_old, alpha, alpha_old, params...
  /// using
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      TimeRuleParams2<DisplacementTimeRule, H1<disp_order, dim>, InternalVarTimeRule, StateSpace, parameter_space...>>;

  // State weak form: residual for state variable (alpha).
  // Inputs: alpha, alpha_old, u, u_old, params...
  /// using
  using StateWeakFormType = TimeDiscretizedWeakForm<
      dim, StateSpace,
      TimeRuleParams2<InternalVarTimeRule, StateSpace, DisplacementTimeRule, H1<disp_order, dim>, parameter_space...>>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;                  ///< Solid mechanics weak form.
  std::shared_ptr<StateWeakFormType> state_weak_form;                  ///< State variable weak form.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;                ///< Displacement boundary conditions.
  std::shared_ptr<DirichletBoundaryConditions> state_bc;               ///< State variable boundary conditions.
  std::shared_ptr<DisplacementTimeRule> disp_time_rule;                            ///< Time integration for displacement.
  std::shared_ptr<InternalVarTimeRule> state_time_rule;                          ///< Time integration for state.

  /**
   * @brief Get the list of all state fields.
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    return {field_store->getField(prefix("displacement_predicted")), field_store->getField(prefix("displacement")),
            field_store->getField(prefix("state_predicted")), field_store->getField(prefix("state"))};
  }

  /**
   * @brief Create a DifferentiablePhysics object for this system.
   * @param physics_name The name of the physics.
   * @return std::shared_ptr<DifferentiablePhysics> The differentiable physics object.
   */
  std::shared_ptr<DifferentiablePhysics> createDifferentiablePhysics(std::string physics_name)
  {
    return std::make_shared<DifferentiablePhysics>(
        field_store->getMesh(), field_store->graph(), field_store->getShapeDisp(), getStateFields(),
        getParameterFields(), advancer, physics_name,
        std::vector<std::string>{prefix("solid_residual"), prefix("state_residual")});
  }

  /**
   * @brief Set the material model for the solid mechanics part.
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto alpha, auto alpha_old, auto... params) {
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);

          typename MaterialType::State state;
          auto pk_stress = material(state, get<DERIVATIVE>(u_current), get<VALUE>(alpha_current), params...);

          tensor<double, dim> source{};
          return smith::tuple{source, pk_stress};
        });
  }

  /**
   * @brief Add the evolution law for the state variable.
   * @tparam EvolutionType The evolution law function type.
   * @param domain_name The name of the domain.
   * @param evolution_law Function returning the residual of the ODE.
   */
  template <typename EvolutionType>
  void addStateEvolution(const std::string& domain_name, EvolutionType evolution_law)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_state_rule = state_time_rule;

    state_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto alpha, auto alpha_old, auto u, auto u_old, auto... params) {
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto [alpha_current, alpha_dot] = captured_state_rule->interpolate(t_info, alpha, alpha_old);

          auto residual_val = evolution_law(t_info, get<VALUE>(alpha_current), get<VALUE>(alpha_dot),
                                            get<DERIVATIVE>(u_current), params...);

          tensor<double, dim> flux{};
          return smith::tuple{residual_val, flux};
        });
  }
};

/**
 * @brief Factory function to build a solid statics system with L2 state variable.
 */
template <int dim, int disp_order, typename StateSpace, typename DisplacementTimeRule, typename InternalVarTimeRule,
          typename... parameter_space>
SolidMechanicsWithInternalVarsSystem<dim, disp_order, StateSpace, DisplacementTimeRule, InternalVarTimeRule, parameter_space...>
buildSolidMechanicsWithInternalVarsSystem(std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> solver,
                                   DisplacementTimeRule disp_rule, InternalVarTimeRule state_rule, std::string prepend_name = "",
                                   FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<FieldStore>(mesh, 100);

  auto prefix = [&](const std::string& name) {
    if (prepend_name.empty()) {
      return name;
    }
    return prepend_name + "_" + name;
  };

  // Add shape displacement
  FieldType<H1<1, dim>> shape_disp_type(prefix("shape_displacement"));
  field_store->addShapeDisp(shape_disp_type);

  // 1. Displacement fields
  auto disp_time_rule_ptr = std::make_shared<DisplacementTimeRule>(disp_rule);
  FieldType<H1<disp_order, dim>> disp_type(prefix("displacement_predicted"));
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule_ptr);
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VAL, prefix("displacement"));

  // 2. State variable fields
  auto state_time_rule_ptr = std::make_shared<InternalVarTimeRule>(state_rule);
  FieldType<StateSpace> state_type(prefix("state_predicted"));
  auto state_bc = field_store->addIndependent(state_type, state_time_rule_ptr);
  auto state_old_type = field_store->addDependent(state_type, FieldStore::TimeDerivative::VAL, prefix("state"));

  // 3. Parameters
  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  using SystemType =
      SolidMechanicsWithInternalVarsSystem<dim, disp_order, StateSpace, DisplacementTimeRule, InternalVarTimeRule, parameter_space...>;

  // 4. Solid Weak Form (Residual for u)
  std::string solid_res_name = prefix("solid_residual");
  auto solid_weak_form = std::make_shared<typename SystemType::SolidWeakFormType>(
      solid_res_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
      field_store->createSpaces(solid_res_name, disp_type.name, disp_type, disp_old_type, state_type, state_old_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // 5. State Weak Form (Residual for alpha)
  std::string state_res_name = prefix("state_residual");
  auto state_weak_form = std::make_shared<typename SystemType::StateWeakFormType>(
      state_res_name, field_store->getMesh(), field_store->getField(state_type.name).get()->space(),
      field_store->createSpaces(state_res_name, state_type.name, state_type, state_old_type, disp_type, disp_old_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // 6. Solver and Advancer
  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, state_weak_form};
  auto advancer = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, solver);

  return SystemType{{field_store, solver, advancer, parameter_fields, prepend_name},
                    solid_weak_form,
                    state_weak_form,
                    disp_bc,
                    state_bc,
                    disp_time_rule_ptr,
                    state_time_rule_ptr};
}

/**
 * @brief Factory function with default rules (backward compatible).
 */
template <int dim, int disp_order, typename StateSpace, typename... parameter_space>
auto buildSolidMechanicsWithInternalVarsSystem(std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> solver,
                                        std::string prepend_name = "",
                                        FieldType<parameter_space>... parameter_types)
{
  return buildSolidMechanicsWithInternalVarsSystem<dim, disp_order, StateSpace>(
      mesh, solver, QuasiStaticFirstOrderTimeIntegrationRule{}, BackwardEulerFirstOrderTimeIntegrationRule{},
      prepend_name, parameter_types...);
}

}  // namespace smith
