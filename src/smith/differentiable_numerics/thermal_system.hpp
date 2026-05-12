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
#include "smith/physics/functional_weak_form.hpp"
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
          typename Coupling = std::tuple<>>
struct ThermalSystem : public SystemBase {
  using SystemBase::SystemBase;

  static_assert(TemperatureTimeRule::num_states == 2, "ThermalSystem requires a 2-state time integration rule");

  /// Thermal weak form: (temp, temp_old, coupling_fields..., params...)
  using ThermalWeakFormType =
      FunctionalWeakForm<dim, H1<temp_order>, detail::TimeRuleParams<TemperatureTimeRule, H1<temp_order>, Coupling>>;

  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;       ///< Thermal weak form.
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;  ///< Temperature boundary conditions.
  std::shared_ptr<TemperatureTimeRule> temperature_time_rule;   ///< Time integration for temperature.
  std::shared_ptr<const Coupling> coupling;                     ///< Coupling metadata for callback interpolation.

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
    auto captured_coupling = coupling;

    thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto... raw_args) {
      return detail::applyTimeRuleAndCoupling(
          *captured_temp_rule, *captured_coupling, t_info,
          [&](auto T_current, auto T_dot, auto... interpolated_params) {
            auto [heat_capacity, heat_flux] =
                material(t_info, get<VALUE>(T_current), get<DERIVATIVE>(T_current), interpolated_params...);
            return smith::tuple{heat_capacity * get<VALUE>(T_dot), -heat_flux};
          },
          raw_args...);
    });
  }

  /**
   * @brief Set thermal material and a coincident body heat source from a single callable.
   *
   * The callable is invoked once per quadrature point and must return
   * `smith::tuple{heat_capacity, heat_flux, heat_source}`. Used by coupled physics (e.g.
   * thermo-mechanics) where one material evaluation produces all three contributions and we
   * want to avoid re-evaluating the material for each piece.
   *
   * Residual contribution: `(heat_capacity * dT/dt - heat_source, -heat_flux)`.
   *
   * @tparam MaterialType The thermal material type.
   * @param material The material model instance returning {C_v, q, s}.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterialAndHeatSource(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_temp_rule = temperature_time_rule;
    auto captured_coupling = coupling;

    thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto... raw_args) {
      return detail::applyTimeRuleAndCoupling(
          *captured_temp_rule, *captured_coupling, t_info,
          [&](auto T_current, auto T_dot, auto... interpolated_params) {
            auto [heat_capacity, heat_flux, heat_source] =
                material(t_info, get<VALUE>(T_current), get<DERIVATIVE>(T_current), interpolated_params...);
            return smith::tuple{heat_capacity * get<VALUE>(T_dot) - heat_source, -heat_flux};
          },
          raw_args...);
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
    auto captured_rule = temperature_time_rule;
    auto captured_coupling = coupling;
    thermal_weak_form->addBodySource(domain_name, [=](auto t_info, auto X, auto... raw_args) {
      return detail::applyTimeRuleAndCoupling(
          *captured_rule, *captured_coupling, t_info,
          [&](auto T_current, auto /*T_dot*/, auto... interpolated_params) {
            return source_function(t_info.time(), X, T_current, interpolated_params...);
          },
          raw_args...);
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
    auto captured_rule = temperature_time_rule;
    auto captured_coupling = coupling;
    thermal_weak_form->addBoundaryFlux(boundary_name, [=](auto t_info, auto X, auto n, auto... raw_args) {
      return detail::applyTimeRuleAndCoupling(
          *captured_rule, *captured_coupling, t_info,
          [&](auto T_current, auto /*T_dot*/, auto... interpolated_params) {
            return flux_function(t_info.time(), X, n, T_current, interpolated_params...);
          },
          raw_args...);
    });
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

  return PhysicsFields<dim, temp_order, TemperatureTimeRule, H1<temp_order>, H1<temp_order>>{
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
  requires detail::is_coupling_packs_v<Coupling>
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
  auto thermal_weak_form = detail::buildWeakFormWithCoupling<typename SystemType::ThermalWeakFormType>(
      field_store, thermal_flux_name, temperature_type.name, temperature_type, temperature_old_type,
      detail::flattenCouplingFields(coupling));

  auto sys =
      std::make_shared<SystemType>(field_store, solver, std::vector<std::shared_ptr<WeakForm>>{thermal_weak_form});
  sys->temperature_bc = temperature_bc;
  sys->temperature_time_rule = temperature_time_rule_ptr;
  sys->coupling = std::make_shared<Coupling>(coupling);
  sys->thermal_weak_form = thermal_weak_form;

  return sys;
}

}  // namespace detail

/**
 * @brief Build a ThermalSystem from already-registered field packs.
 *
 * Deduces the temperature time rule from `self_fields`.
 *
 * Usage:
 * @code
 *   auto thermal_system = buildThermalSystem(
 *       solver, opts, thermal_fields, couplingFields(solid_fields));
 * @endcode
 */
template <typename SelfFields, typename... Trailing>
  requires(detail::has_time_rule_v<SelfFields> && detail::trailing_coupling_args_valid_v<Trailing...>)
auto buildThermalSystem(std::shared_ptr<SystemSolver> solver, const ThermalOptions& options,
                        const SelfFields& self_fields, const Trailing&... trailing)
{
  constexpr int dim = SelfFields::dim;
  constexpr int temp_order = SelfFields::order;
  using TemperatureTimeRule = typename std::decay_t<SelfFields>::time_rule_type;
  auto field_store = self_fields.field_store;
  detail::registerParamsIfNeeded(field_store, trailing...);
  auto coupling = detail::collectCouplingFields(field_store, trailing...);
  return detail::buildThermalSystemImpl<dim, temp_order, TemperatureTimeRule>(field_store, coupling, solver, options);
}

}  // namespace smith
