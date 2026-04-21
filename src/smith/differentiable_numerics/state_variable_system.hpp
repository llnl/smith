// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state_variable_system.hpp
 * @brief Standalone composable system for a first-order internal variable (damage, plasticity, etc.).
 *
 * Two-phase factory:
 *   auto state_coupling_fields = registerStateVariableFields<StateSpace>(
 *       field_store, state_rule, params...);
 *
 *   auto [state_sys, cycle_zero, ends] = buildStateVariableSystem<dim, StateSpace>(
 *       field_store, state_rule, [solid_coupling,] solver, opts, params...);
 *
 * The returned CouplingParams from registerStateVariableFields carries field tokens
 * (state_solve_state, state) that can be injected into another physics system
 * (e.g. SolidMechanicsSystem) as coupling input.
 *
 * The system's addStateEvolution registers an ODE residual of the form:
 *   evolution_law(t_info, alpha_val, alpha_dot, coupling_fields..., params...) == 0
 *
 * With Coupling = CouplingParams<H1<order,dim>, H1<order,dim>, H1<order,dim>, H1<order,dim>>
 * (the solid displacement coupling fields), the user accesses the displacement gradient via
 * get<DERIVATIVE>(u_solve_state_arg) as the first coupling field argument.
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/coupling_params.hpp"

namespace smith {

/**
 * @brief System for a single internal variable using a two-state first-order rule.
 *
 * Field layout: (state_solve_state, state) — 2 fields.
 * With a non-empty Coupling, coupling fields appear after the two state fields,
 * before user parameter fields.
 *
 * @tparam dim Spatial dimension (needed for the weak form and zero-flux tensor).
 * @tparam StateSpace FE space for the internal variable (e.g., L2<0>).
 * @tparam InternalVarTimeRule Time integration rule (must have num_states == 2).
 * @tparam Coupling CouplingParams listing fields borrowed from other physics (default: none).
 */
template <int dim, typename StateSpace, typename InternalVarTimeRule = BackwardEulerFirstOrderTimeIntegrationRule,
          typename Coupling = CouplingParams<>>
struct StateVariableSystem : public SystemBase {
  using SystemBase::SystemBase;

  static_assert(InternalVarTimeRule::num_states == 2, "StateVariableSystem requires a 2-state time integration rule");

  /// State weak form: (alpha, alpha_old, coupling_fields..., params...)
  using StateWeakFormType = TimeDiscretizedWeakForm<
      dim, StateSpace, typename detail::TimeRuleParamsWithCoupling<InternalVarTimeRule, StateSpace, Coupling>::type>;

  std::shared_ptr<StateWeakFormType> state_weak_form;     ///< Internal variable weak form.
  std::shared_ptr<DirichletBoundaryConditions> state_bc;  ///< Internal variable boundary conditions.
  std::shared_ptr<InternalVarTimeRule> state_time_rule;   ///< Time integration rule.

  /**
   * @brief Register an ODE evolution law for the internal variable.
   *
   * The evolution_law is called as:
   *   evolution_law(t_info, alpha_val, alpha_dot, coupling_fields..., params...)
   * and must return a scalar residual (zero when the ODE is satisfied).
   *
   * When Coupling carries solid displacement fields (u_ss, u, v, a), access the
   * displacement gradient via get<DERIVATIVE>(u_ss_arg) on the first coupling field.
   *
   * @param domain_name Domain to apply the evolution on.
   * @param evolution_law Callable returning the ODE residual.
   */
  template <typename EvolutionType>
  void addStateEvolution(const std::string& domain_name, EvolutionType evolution_law)
  {
    auto captured_state_rule = state_time_rule;
    state_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto alpha, auto alpha_old, auto... coupling_and_params) {
          auto [alpha_current, alpha_dot] = captured_state_rule->interpolate(t_info, alpha, alpha_old);
          auto residual_val =
              evolution_law(t_info, get<VALUE>(alpha_current), get<VALUE>(alpha_dot), coupling_and_params...);
          tensor<double, dim> flux{};
          return smith::tuple{residual_val, flux};
        });
  }
};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct StateVariableOptions {};

// ---------------------------------------------------------------------------
// Phase 1: registerStateVariableFields
// ---------------------------------------------------------------------------

/**
 * @brief Register state variable fields into a FieldStore.
 *
 * Adds a 2-field layout: (state_solve_state, state).
 * Pass an instance of the desired time integration rule so its type is deduced;
 * only `<StateSpace>` needs to be specified explicitly.
 *
 * @return CouplingParams carrying (state_solve_state, state[, params...]) field tokens
 *         suitable for injection into another physics system.
 */
template <typename StateSpace, typename InternalVarTimeRule, typename... parameter_space>
auto registerStateVariableFields(std::shared_ptr<FieldStore> field_store, InternalVarTimeRule /*rule*/,
                                 FieldType<parameter_space>... parameter_types)
{
  auto state_time_rule_ptr = std::make_shared<InternalVarTimeRule>();
  FieldType<StateSpace> state_type("state_solve_state");
  field_store->addIndependent(state_type, state_time_rule_ptr);
  field_store->addDependent(state_type, FieldStore::TimeDerivative::VAL, "state");

  auto prefix_param = [&](auto& pt) {
    pt.name = "param_" + pt.name;
    field_store->addParameter(pt);
  };
  (prefix_param(parameter_types), ...);

  return CouplingParams{FieldType<StateSpace>(field_store->prefix("state_solve_state")),
                        FieldType<StateSpace>(field_store->prefix("state")), parameter_types...};
}

// ---------------------------------------------------------------------------
// Phase 2: buildStateVariableSystem
// ---------------------------------------------------------------------------

/**
 * @brief Build a StateVariableSystem with coupling, assuming fields are already registered.
 *
 * Pass the same rule instance used in registerStateVariableFields so its type is deduced;
 * only `<dim, StateSpace>` need be specified.
 *
 * Returns `{system, nullptr, {}}` as a tuple (no cycle-zero or end-step systems).
 */
template <int dim, typename StateSpace, typename InternalVarTimeRule, typename Coupling>
  requires detail::is_coupling_params_v<Coupling>
auto buildStateVariableSystem(std::shared_ptr<FieldStore> field_store, InternalVarTimeRule /*rule*/,
                              const Coupling& coupling, std::shared_ptr<SystemSolver> solver,
                              const StateVariableOptions& /*options*/)
{
  auto state_time_rule_ptr = std::make_shared<InternalVarTimeRule>();

  FieldType<StateSpace> state_type(field_store->prefix("state_solve_state"), true);
  FieldType<StateSpace> state_old_type(field_store->prefix("state"));

  auto state_bc = field_store->getBoundaryConditions(state_type.name);

  using SystemType = StateVariableSystem<dim, StateSpace, InternalVarTimeRule, Coupling>;

  std::string state_res_name = field_store->prefix("state_residual");
  auto state_weak_form = std::apply(
      [&](auto&... cfs) {
        return std::make_shared<typename SystemType::StateWeakFormType>(
            state_res_name, field_store->getMesh(), field_store->getField(state_type.name).get()->space(),
            field_store->createSpaces(state_res_name, state_type.name, state_type, state_old_type, cfs...));
      },
      coupling.fields);

  auto sys = std::make_shared<SystemType>(field_store, solver, std::vector<std::shared_ptr<WeakForm>>{state_weak_form});
  sys->state_bc = state_bc;
  sys->state_time_rule = state_time_rule_ptr;
  sys->state_weak_form = state_weak_form;

  std::shared_ptr<SystemBase> cycle_zero_system;
  std::vector<std::shared_ptr<SystemBase>> end_step_systems;
  return std::make_tuple(sys, cycle_zero_system, end_step_systems);
}

/**
 * @brief Build a StateVariableSystem without coupling.
 *
 * Overload for the common case of no inter-physics coupling.
 * Parameters (if any) are wrapped into a CouplingParams so the system sees them.
 */
template <int dim, typename StateSpace, typename InternalVarTimeRule, typename... parameter_space>
auto buildStateVariableSystem(std::shared_ptr<FieldStore> field_store, InternalVarTimeRule rule,
                              std::shared_ptr<SystemSolver> solver, const StateVariableOptions& options,
                              FieldType<parameter_space>... parameter_types)
{
  auto prefix_param = [&](auto& pt) {
    pt.name = "param_" + pt.name;
    field_store->addParameter(pt);
  };
  (prefix_param(parameter_types), ...);
  return buildStateVariableSystem<dim, StateSpace>(field_store, rule, CouplingParams{parameter_types...}, solver,
                                                   options);
}

}  // namespace smith
