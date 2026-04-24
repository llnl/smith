// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_system.hpp
 * @brief Defines the ThermalSystem struct and its factory function
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/coupling_params.hpp"

namespace smith {

/**
 * @brief Container for a thermal system with configurable time integration.
 *
 * Always uses a 2-state field layout (temperature_solve_state, temperature).
 * Use QuasiStaticFirstOrderTimeIntegrationRule for steady-state problems,
 * or BackwardEulerFirstOrderTimeIntegrationRule for transient problems.
 *
 * @tparam dim Spatial dimension.
 * @tparam temp_order Order of the temperature basis.
 * @tparam TemperatureTimeRule Time integration rule type (must have num_states == 2).
 * @tparam Coupling CouplingParams listing fields borrowed from other physics (default: none).
 *         Coupling fields occupy leading positions in the tail after the 2 time-rule state fields,
 *         before user parameter_space fields.
 */
template <int dim, int temp_order, typename TemperatureTimeRule = QuasiStaticFirstOrderTimeIntegrationRule,
          typename Coupling = CouplingParams<>>
struct ThermalSystem : public SystemBase {
  using SystemBase::SystemBase;

  static_assert(TemperatureTimeRule::num_states == 2, "ThermalSystem requires a 2-state time integration rule");

  /// Thermal weak form: (temp, temp_old, coupling_fields..., params...)
  using ThermalWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<temp_order>,
      typename detail::TimeRuleParamsWithCoupling<TemperatureTimeRule, H1<temp_order>, Coupling>::type>;

  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;       ///< Thermal weak form.
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;  ///< Temperature boundary conditions.
  std::shared_ptr<TemperatureTimeRule> temperature_time_rule;   ///< Time integration for temperature.

  /**
   * @brief Set the thermal material model for a domain.
   *
   * Material is called as `material(t_info, temperature, grad_temperature, params...)` and must return
   * `smith::tuple{heat_capacity, heat_flux}`. Consistent with heat_transfer.hpp convention.
   *
   * The system forms the residual as: heat_capacity * dT/dt for the source term, and -heat_flux
   * for the flux term.
   *
   * @tparam MaterialType The thermal material type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto temperature, auto temperature_old,
                                                        auto... params) {
      auto [T_current, T_dot] = captured_temp_rule->interpolate(t_info, temperature, temperature_old);
      auto [heat_capacity, heat_flux] = material(t_info, get<VALUE>(T_current), get<DERIVATIVE>(T_current), params...);
      return smith::tuple{heat_capacity * get<VALUE>(T_dot), -heat_flux};
    });
  }

  /**
   * @brief Add a body heat source to the thermal system (with DependsOn).
   * @param depends_on Selects which primal and parameter fields the contribution depends on.
   * @param domain_name The name of the domain where the heat source is applied.
   * @param source_function (t_info, X, T, params...) -> heat_source.
   */
  template <int... active_parameters, typename HeatSourceType>
  void addHeatSource(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                     HeatSourceType source_function)
  {
    (void)depends_on;
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->template addBodySource<0, 1, (2 + active_parameters)...>(
        DependsOn<0, 1, (2 + active_parameters)...>{}, domain_name,
        [=](auto t_info, auto X, auto temperature, auto temperature_old, auto... params) {
          auto T = captured_temp_rule->value(t_info, temperature, temperature_old);
          return source_function(t_info, X, T, params...);
        });
  }

  /**
   * @brief Add a body heat source that depends on all state and parameter fields.
   * @param domain_name The name of the domain where the heat source is applied.
   * @param source_function (t_info, X, T, params...) -> heat_source.
   */
  template <typename HeatSourceType>
  void addHeatSource(const std::string& domain_name, HeatSourceType source_function)
  {
    addHeatSourceAllParams(domain_name, source_function, std::make_index_sequence<Coupling::num_coupling_fields>{});
  }

  /**
   * @brief Add a boundary heat flux to the thermal system (with DependsOn).
   * @param depends_on Selects which primal and parameter fields the contribution depends on.
   * @param boundary_name The name of the boundary where the heat flux is applied.
   * @param flux_function (t_info, X, n, T, params...) -> heat_flux.
   */
  template <int... active_parameters, typename HeatFluxType>
  void addHeatFlux(DependsOn<active_parameters...> depends_on, const std::string& boundary_name,
                   HeatFluxType flux_function)
  {
    (void)depends_on;
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->template addBoundaryFlux<0, 1, (2 + active_parameters)...>(
        DependsOn<0, 1, (2 + active_parameters)...>{}, boundary_name,
        [=](auto t_info, auto X, auto n, auto temperature, auto temperature_old, auto... params) {
          auto T = captured_temp_rule->value(t_info, temperature, temperature_old);
          return -flux_function(t_info, X, n, T, params...);
        });
  }

  /**
   * @brief Add a boundary heat flux that depends on all state and parameter fields.
   * @param boundary_name The name of the boundary where the heat flux is applied.
   * @param flux_function (t_info, X, n, T, params...) -> heat_flux.
   */
  template <typename HeatFluxType>
  void addHeatFlux(const std::string& boundary_name, HeatFluxType flux_function)
  {
    addHeatFluxAllParams(boundary_name, flux_function, std::make_index_sequence<Coupling::num_coupling_fields>{});
  }

  /// Set zero-temperature Dirichlet BC.
  void setTemperatureBC(const Domain& domain) { temperature_bc->template setFixedScalarBCs<dim>(domain); }

  /// Set temperature BC with a prescribed function.
  template <typename AppliedTemperatureFunction>
  void setTemperatureBC(const Domain& domain, AppliedTemperatureFunction f)
  {
    temperature_bc->template setScalarBCs<dim>(domain, f);
  }

 private:
  template <typename HeatSourceType, std::size_t... Is>
  void addHeatSourceAllParams(const std::string& domain_name, HeatSourceType f, std::index_sequence<Is...>)
  {
    addHeatSource(DependsOn<static_cast<int>(Is)...>{}, domain_name, f);
  }

  template <typename HeatFluxType, std::size_t... Is>
  void addHeatFluxAllParams(const std::string& boundary_name, HeatFluxType f, std::index_sequence<Is...>)
  {
    addHeatFlux(DependsOn<static_cast<int>(Is)...>{}, boundary_name, f);
  }
};

struct ThermalOptions {};

/**
 * @brief Register all thermal fields into a FieldStore.
 *
 * Phase 1 of the two-phase initialization.
 *
 * @return PhysicsFields carrying the exported field tokens and time rule type.
 */
template <int dim, int temp_order, typename TemperatureTimeRule>
auto registerThermalFields(std::shared_ptr<FieldStore> field_store,
                           const ThermalOptions& /*options*/ = ThermalOptions{})
{
  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  if (!field_store->hasField(field_store->prefix(shape_disp_type.name))) {
    field_store->addShapeDisp(shape_disp_type);
  }

  auto temperature_time_rule_ptr = std::make_shared<TemperatureTimeRule>();
  FieldType<H1<temp_order>> temperature_type("temperature_solve_state");
  field_store->addIndependent(temperature_type, temperature_time_rule_ptr);
  field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VAL, "temperature");

  return PhysicsFields<TemperatureTimeRule, H1<temp_order>, H1<temp_order>>{
      field_store, FieldType<H1<temp_order>>(field_store->prefix("temperature_solve_state")),
      FieldType<H1<temp_order>>(field_store->prefix("temperature"))};
}

/**
 * @brief Internal thermal builder after coupling fields are assembled.
 *
 * Phase 2 of the two-phase initialization.
 */
namespace detail {

template <int dim, int temp_order, typename TemperatureTimeRule, typename Coupling>
  requires detail::is_coupling_params_v<Coupling>
auto buildThermalSystemImpl(std::shared_ptr<FieldStore> field_store, const Coupling& coupling,
                            std::shared_ptr<SystemSolver> solver, const ThermalOptions& /*options*/)
{
  auto temperature_time_rule_ptr = std::make_shared<TemperatureTimeRule>();

  FieldType<H1<1, dim>> shape_disp_type(field_store->prefix("shape_displacement"));
  FieldType<H1<temp_order>> temperature_type(field_store->prefix("temperature_solve_state"), true);
  FieldType<H1<temp_order>> temperature_old_type(field_store->prefix("temperature"));

  auto temperature_bc = field_store->getBoundaryConditions(temperature_type.name);

  using SystemType = ThermalSystem<dim, temp_order, TemperatureTimeRule, Coupling>;

  std::string thermal_flux_name = field_store->prefix("thermal_flux");
  auto thermal_weak_form = std::apply(
      [&](auto&... coupling_fields) {
        return std::make_shared<typename SystemType::ThermalWeakFormType>(
            thermal_flux_name, field_store->getMesh(), field_store->getField(temperature_type.name).get()->space(),
            field_store->createSpaces(thermal_flux_name, temperature_type.name, temperature_type, temperature_old_type,
                                      coupling_fields...));
      },
      coupling.fields);

  auto sys =
      std::make_shared<SystemType>(field_store, solver, std::vector<std::shared_ptr<WeakForm>>{thermal_weak_form});
  sys->temperature_bc = temperature_bc;
  sys->temperature_time_rule = temperature_time_rule_ptr;
  sys->thermal_weak_form = thermal_weak_form;

  return sys;
}

}  // namespace detail

/**
 * @brief Build a ThermalSystem from already-registered field packs.
 *
 * Explicit-rule API: rule is given as template param.
 * Additional parameter packs are registered as parameters. Coupling packs are taken from
 * the trailing field-pack arguments.
 *
 * Usage:
 * @code
 *   auto thermal = buildThermalSystem<dim, order, TempRule>(
 *       solver, opts, thermal_fields, solid_fields);
 * @endcode
 */
template <int dim, int temp_order, typename TemperatureTimeRule, typename SelfFields, typename... OtherPacks>
  requires(detail::is_physics_fields_v<SelfFields> &&
           std::is_same_v<typename std::decay_t<SelfFields>::time_rule_type, TemperatureTimeRule> &&
           (detail::is_coupling_params_v<OtherPacks> && ...))
auto buildThermalSystem(std::shared_ptr<SystemSolver> solver, const ThermalOptions& options,
                        const SelfFields& self_fields, const OtherPacks&... other_packs)
{
  auto field_store = self_fields.field_store;
  (detail::registerParamsIfNeeded(field_store, other_packs), ...);
  auto coupling = detail::collectCouplingFields<TemperatureTimeRule>(field_store, self_fields, other_packs...);
  return detail::buildThermalSystemImpl<dim, temp_order, TemperatureTimeRule>(field_store, coupling, solver, options);
}

/**
 * @brief Build a ThermalSystem from already-registered field packs.
 *
 * Preferred API: deduce rule from `self_fields`.
 *
 * Usage:
 * @code
 *   auto thermal = buildThermalSystem<dim, order>(
 *       solver, opts, thermal_fields, solid_fields);
 * @endcode
 */
template <int dim, int temp_order, typename SelfFields, typename... OtherPacks>
  requires(detail::has_time_rule_v<SelfFields> && (detail::is_coupling_params_v<OtherPacks> && ...))
auto buildThermalSystem(std::shared_ptr<SystemSolver> solver, const ThermalOptions& options,
                        const SelfFields& self_fields, const OtherPacks&... other_packs)
{
  using TemperatureTimeRule = typename std::decay_t<SelfFields>::time_rule_type;
  return buildThermalSystem<dim, temp_order, TemperatureTimeRule>(solver, options, self_fields, other_packs...);
}

/**
 * @brief Build a ThermalSystem from solver options and a FieldStore.
 *
 * Registers the thermal field pack, builds a nonlinear block solver from the supplied options,
 * then forwards to the existing field-pack overload.
 *
 * Usage:
 * @code
 *   auto thermal = buildThermalSystem<dim, order, TempRule>(
 *       nonlin_opts, lin_opts, field_store, opts, param_fields, solid_fields);
 * @endcode
 */
template <int dim, int temp_order, typename TemperatureTimeRule, typename... OtherPacks>
  requires(detail::is_coupling_params_v<OtherPacks> && ...)
auto buildThermalSystem(const NonlinearSolverOptions& nonlinear_options, const LinearSolverOptions& linear_options,
                        const ThermalOptions& options, std::shared_ptr<FieldStore> field_store,
                        const OtherPacks&... other_packs)
{
  auto self_fields = registerThermalFields<dim, temp_order, TemperatureTimeRule>(field_store, options);
  auto solver = std::make_shared<SystemSolver>(
      buildNonlinearBlockSolver(nonlinear_options, linear_options, *field_store->getMesh()));
  return buildThermalSystem<dim, temp_order, TemperatureTimeRule>(solver, options, self_fields, other_packs...);
}

}  // namespace smith
