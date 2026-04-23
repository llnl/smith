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
 *   auto internal_variable_fields = registerInternalVariableFields<StateSpace, StateRule>(
 *       field_store, params...);
 *
 *   auto internal_variables = buildInternalVariableSystem<dim, StateSpace, StateRule>(
 *       solver, opts, internal_variable_fields, params..., solid_fields);
 *
 * The returned PhysicsFields from registerInternalVariableFields carries field tokens
 * (state_solve_state, state) that can be injected into another physics system
 * (e.g. SolidMechanicsSystem) as coupling input.
 *
 * The system's setMaterial registers an ODE residual of the form:
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
 * Field layout: (state_solve_state, state) - 2 fields.
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
struct InternalVariableSystem : public SystemBase {
  using SystemBase::SystemBase;

  static_assert(InternalVarTimeRule::num_states == 2,
                "InternalVariableSystem requires a 2-state time integration rule");

  /// State weak form: (alpha, alpha_old, coupling_fields..., params...)
  using InternalVariableWeakFormType = TimeDiscretizedWeakForm<
      dim, StateSpace, typename detail::TimeRuleParamsWithCoupling<InternalVarTimeRule, StateSpace, Coupling>::type>;

  std::shared_ptr<InternalVariableWeakFormType> internal_variable_weak_form;  ///< Internal variable weak form.
  std::shared_ptr<DirichletBoundaryConditions> internal_variable_bc;          ///< Internal variable BCs.
  std::shared_ptr<InternalVarTimeRule> internal_variable_time_rule;           ///< Time integration rule.

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
  void addEvolution(const std::string& domain_name, EvolutionType evolution_law)
  {
    auto captured_time_rule = internal_variable_time_rule;
    internal_variable_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto alpha, auto alpha_old, auto... coupling_and_params) {
          auto [alpha_current, alpha_dot] = captured_time_rule->interpolate(t_info, alpha, alpha_old);
          auto residual_val =
              evolution_law(t_info, get<VALUE>(alpha_current), get<VALUE>(alpha_dot), coupling_and_params...);
          tensor<double, dim> flux{};
          return smith::tuple{residual_val, flux};
        });
  }

  template <typename MaterialType>
  /// @brief Register an internal-variable material or evolution law on a domain.
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    addEvolution(domain_name, material);
  }

  template <typename EvolutionType>
  /// @brief Backward-compatible alias for `addEvolution`.
  void addStateEvolution(const std::string& domain_name, EvolutionType evolution_law)
  {
    addEvolution(domain_name, evolution_law);
  }
};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// @brief Options for building an `InternalVariableSystem`.
struct InternalVariableOptions {};
/// @brief Backward-compatible alias for `InternalVariableOptions`.
using StateVariableOptions = InternalVariableOptions;

// ---------------------------------------------------------------------------
// Phase 1: registerInternalVariableFields
// ---------------------------------------------------------------------------

/**
 * @brief Register state variable fields into a FieldStore.
 *
 * Adds a 2-field layout: (state_solve_state, state).
 *
 * @return PhysicsFields carrying (state_solve_state, state) field tokens
 *         suitable for injection into another physics system.
 */
template <typename StateSpace, typename InternalVarTimeRule, typename... parameter_space>
auto registerInternalVariableFields(std::shared_ptr<FieldStore> field_store,
                                    FieldType<parameter_space>... parameter_types)
{
  auto internal_variable_time_rule = std::make_shared<InternalVarTimeRule>();
  FieldType<StateSpace> state_type("state_solve_state");
  field_store->addIndependent(state_type, internal_variable_time_rule);
  field_store->addDependent(state_type, FieldStore::TimeDerivative::VAL, "state");

  if constexpr (sizeof...(parameter_space) > 0) {
    auto prefix_param = [&](auto& pt) {
      pt.name = "param_" + pt.name;
      field_store->addParameter(pt);
    };
    (prefix_param(parameter_types), ...);
  }

  return PhysicsFields<InternalVarTimeRule, StateSpace, StateSpace>{
      field_store, FieldType<StateSpace>(field_store->prefix("state_solve_state")),
      FieldType<StateSpace>(field_store->prefix("state"))};
}

template <typename StateSpace, typename InternalVarTimeRule, typename... parameter_space>
/// @brief Backward-compatible alias for `registerInternalVariableFields`.
auto registerStateVariableFields(std::shared_ptr<FieldStore> field_store, FieldType<parameter_space>... parameter_types)
{
  return registerInternalVariableFields<StateSpace, InternalVarTimeRule>(field_store, parameter_types...);
}

// ---------------------------------------------------------------------------
// Phase 2: buildInternalVariableSystem
// ---------------------------------------------------------------------------

/**
 * @brief Build an InternalVariableSystem with coupling, assuming fields are already registered.
 */
namespace detail {

/**
 * @brief Internal builder for an internal-variable system after public registration and coupling collection.
 */
template <int dim, typename StateSpace, typename InternalVarTimeRule, typename Coupling>
  requires detail::is_coupling_params_v<Coupling>
auto buildInternalVariableSystemImpl(std::shared_ptr<FieldStore> field_store, const Coupling& coupling,
                                     std::shared_ptr<SystemSolver> solver, const InternalVariableOptions& /*options*/)
{
  auto internal_variable_time_rule = std::make_shared<InternalVarTimeRule>();

  FieldType<StateSpace> state_type(field_store->prefix("state_solve_state"), true);
  FieldType<StateSpace> state_old_type(field_store->prefix("state"));

  auto internal_variable_bc = field_store->getBoundaryConditions(state_type.name);

  using SystemType = InternalVariableSystem<dim, StateSpace, InternalVarTimeRule, Coupling>;

  std::string internal_variable_residual_name = field_store->prefix("state_residual");
  auto internal_variable_weak_form = std::apply(
      [&](auto&... coupling_fields) {
        return std::make_shared<typename SystemType::InternalVariableWeakFormType>(
            internal_variable_residual_name, field_store->getMesh(),
            field_store->getField(state_type.name).get()->space(),
            field_store->createSpaces(internal_variable_residual_name, state_type.name, state_type, state_old_type,
                                      coupling_fields...));
      },
      coupling.fields);

  auto sys = std::make_shared<SystemType>(field_store, solver,
                                          std::vector<std::shared_ptr<WeakForm>>{internal_variable_weak_form});
  sys->internal_variable_bc = internal_variable_bc;
  sys->internal_variable_time_rule = internal_variable_time_rule;
  sys->internal_variable_weak_form = internal_variable_weak_form;

  return sys;
}

}  // namespace detail

/**
 * @brief Build an internal-variable system from explicitly typed field packs.
 */
template <int dim, typename StateSpace, typename InternalVarTimeRule, typename SelfFields, typename... OtherPacks>
  requires(detail::is_physics_fields_v<SelfFields> &&
           std::is_same_v<typename std::decay_t<SelfFields>::time_rule_type, InternalVarTimeRule> &&
           (detail::is_coupling_params_v<OtherPacks> && ...))
auto buildInternalVariableSystem(std::shared_ptr<SystemSolver> solver, const InternalVariableOptions& options,
                                 const SelfFields& self_fields, const OtherPacks&... other_packs)
{
  auto field_store = self_fields.field_store;
  (detail::registerParamsIfNeeded(field_store, other_packs), ...);
  auto coupling = detail::collectCouplingFields<InternalVarTimeRule>(field_store, self_fields, other_packs...);
  return detail::buildInternalVariableSystemImpl<dim, StateSpace, InternalVarTimeRule>(field_store, coupling, solver,
                                                                                       options);
}

/**
 * @brief Backward-compatible alias for `buildInternalVariableSystem`.
 */
template <int dim, typename StateSpace, typename InternalVarTimeRule, typename SelfFields, typename... OtherPacks>
  requires(detail::is_physics_fields_v<SelfFields> &&
           std::is_same_v<typename std::decay_t<SelfFields>::time_rule_type, InternalVarTimeRule> &&
           (detail::is_coupling_params_v<OtherPacks> && ...))
auto buildStateVariableSystem(std::shared_ptr<SystemSolver> solver, const StateVariableOptions& options,
                              const SelfFields& self_fields, const OtherPacks&... other_packs)
{
  return buildInternalVariableSystem<dim, StateSpace, InternalVarTimeRule>(solver, options, self_fields,
                                                                           other_packs...);
}

/**
 * @brief Build an internal-variable system from registered field packs.
 */
template <int dim, typename StateSpace, typename SelfFields, typename... OtherPacks>
  requires(detail::has_time_rule_v<SelfFields> && (detail::is_coupling_params_v<OtherPacks> && ...))
auto buildInternalVariableSystem(std::shared_ptr<SystemSolver> solver, const InternalVariableOptions& options,
                                 const SelfFields& self_fields, const OtherPacks&... other_packs)
{
  using InternalVarTimeRule = typename std::decay_t<SelfFields>::time_rule_type;
  return buildInternalVariableSystem<dim, StateSpace, InternalVarTimeRule>(solver, options, self_fields,
                                                                           other_packs...);
}

/**
 * @brief Backward-compatible alias for `buildInternalVariableSystem`.
 */
template <int dim, typename StateSpace, typename SelfFields, typename... OtherPacks>
  requires(detail::has_time_rule_v<SelfFields> && (detail::is_coupling_params_v<OtherPacks> && ...))
auto buildStateVariableSystem(std::shared_ptr<SystemSolver> solver, const StateVariableOptions& options,
                              const SelfFields& self_fields, const OtherPacks&... other_packs)
{
  using InternalVarTimeRule = typename std::decay_t<SelfFields>::time_rule_type;
  return buildInternalVariableSystem<dim, StateSpace, InternalVarTimeRule>(solver, options, self_fields,
                                                                           other_packs...);
}

template <int dim, typename StateSpace, typename InternalVarTimeRule = BackwardEulerFirstOrderTimeIntegrationRule,
          typename Coupling = CouplingParams<>>
/// @brief Backward-compatible alias for `InternalVariableSystem`.
using StateVariableSystem = InternalVariableSystem<dim, StateSpace, InternalVarTimeRule, Coupling>;

}  // namespace smith
