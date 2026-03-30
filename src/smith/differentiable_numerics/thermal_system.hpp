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
 * @tparam parameter_space Finite element spaces for optional parameters.
 */
template <int dim, int temp_order, typename TemperatureTimeRule = QuasiStaticFirstOrderTimeIntegrationRule,
          typename... parameter_space>
struct ThermalSystem : public SystemBase {
  static_assert(TemperatureTimeRule::num_states == 2, "ThermalSystem requires a 2-state time integration rule");

  /// @brief using for ThermalWeakFormType
  using ThermalWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<temp_order>,
                              TimeRuleParams<TemperatureTimeRule, H1<temp_order>, parameter_space...>>;

  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;       ///< Thermal weak form.
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;  ///< Temperature boundary conditions.
  std::shared_ptr<TemperatureTimeRule> temperature_time_rule;   ///< Time integration for temperature.

  /**
   * @brief Get the list of all state fields (temperature_solve_state, temperature).
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    return {field_store->getField(prefix("temperature_solve_state")), field_store->getField(prefix("temperature"))};
  }

  /**
   * @brief Get the list of physical, non-solve state fields.
   * @return std::vector<FieldState> List of physical fields suitable for output.
   */
  std::vector<FieldState> getOutputFieldStates() const { return {field_store->getField(prefix("temperature"))}; }

  /**
   * @brief Get information about reaction fields for this system.
   * @return List of ReactionInfo structures.
   */
  std::vector<ReactionInfo> getReactionInfos() const
  {
    return {{prefix("thermal_flux"), &field_store->getField(prefix("temperature")).get()->space()}};
  }

  /**
   * @brief Create a DifferentiablePhysics object for this system.
   * @param physics_name The name of the physics.
   * @return std::unique_ptr<DifferentiablePhysics> The differentiable physics object.
   */
  std::unique_ptr<DifferentiablePhysics> createDifferentiablePhysics(std::string physics_name)
  {
    return std::make_unique<DifferentiablePhysics>(field_store->getMesh(), field_store->graph(),
                                                   field_store->getShapeDisp(), getStateFields(), getParameterFields(),
                                                   advancer, physics_name, getReactionInfos());
  }

  /**
   * @brief Set the thermal material model for a domain.
   *
   * Material is called as `material(x, temperature, grad_temperature, params...)` and must return
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

    thermal_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto temperature, auto temperature_old, auto... params) {
          auto [T_current, T_dot] = captured_temp_rule->interpolate(t_info, temperature, temperature_old);
          auto [heat_capacity, heat_flux] = material(get<VALUE>(T_current), get<DERIVATIVE>(T_current), params...);
          return smith::tuple{heat_capacity * get<VALUE>(T_dot), -heat_flux};
        });
  }

  /**
   * @brief Add a body heat source to the thermal system (with DependsOn).
   * @param depends_on Selects which primal and parameter fields the contribution depends on.
   * @param domain_name The name of the domain where the heat source is applied.
   * @param source_function (t, X, T, params...) -> heat_source.
   */
  template <int... active_parameters, typename HeatSourceType>
  void addHeatSource(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                     HeatSourceType source_function)
  {
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBodySource(depends_on, domain_name,
                                     [=](auto t_info, auto X, auto temperature, auto temperature_old, auto... params) {
                                       auto T = captured_temp_rule->value(t_info, temperature, temperature_old);
                                       return source_function(t_info.time(), X, T, params...);
                                     });
  }

  /**
   * @brief Add a body heat source that depends on all state and parameter fields.
   * @param domain_name The name of the domain where the heat source is applied.
   * @param source_function (t, X, T, params...) -> heat_source.
   */
  template <typename HeatSourceType>
  void addHeatSource(const std::string& domain_name, HeatSourceType source_function)
  {
    addHeatSourceAllParams(domain_name, source_function, std::make_index_sequence<2 + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a boundary heat flux to the thermal system (with DependsOn).
   * @param depends_on Selects which primal and parameter fields the contribution depends on.
   * @param boundary_name The name of the boundary where the heat flux is applied.
   * @param flux_function (t, X, n, T, params...) -> heat_flux.
   */
  template <int... active_parameters, typename HeatFluxType>
  void addHeatFlux(DependsOn<active_parameters...> depends_on, const std::string& boundary_name,
                   HeatFluxType flux_function)
  {
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBoundaryFlux(
        depends_on, boundary_name,
        [=](auto t_info, auto X, auto n, auto temperature, auto temperature_old, auto... params) {
          auto T = captured_temp_rule->value(t_info, temperature, temperature_old);
          return -flux_function(t_info.time(), X, n, T, params...);
        });
  }

  /**
   * @brief Add a boundary heat flux that depends on all state and parameter fields.
   * @param boundary_name The name of the boundary where the heat flux is applied.
   * @param flux_function (t, X, n, T, params...) -> heat_flux.
   */
  template <typename HeatFluxType>
  void addHeatFlux(const std::string& boundary_name, HeatFluxType flux_function)
  {
    addHeatFluxAllParams(boundary_name, flux_function, std::make_index_sequence<2 + sizeof...(parameter_space)>{});
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

/**
 * @brief Factory function to build a thermal system.
 * @tparam dim Spatial dimension.
 * @tparam temp_order Order of the temperature basis.
 * @tparam TemperatureTimeRule Time integration rule type (must have num_states == 2).
 * @tparam parameter_space Finite element spaces for optional parameters.
 * @param mesh The mesh.
 * @param solver The coupled system solver.
 * @param temp_rule The time integration rule for temperature.
 * @param prepend_name The name of the physics (used as field prefix).
 * @param parameter_types Parameter field types.
 * @return ThermalSystem with all components initialized.
 */
template <int dim, int temp_order, typename TemperatureTimeRule, typename... parameter_space>
ThermalSystem<dim, temp_order, TemperatureTimeRule, parameter_space...> buildThermalSystem(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> solver, TemperatureTimeRule temp_rule,
    std::string prepend_name = "", FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<FieldStore>(mesh, 100);

  auto prefix = [&](const std::string& name) {
    if (prepend_name.empty()) {
      return name;
    }
    return prepend_name + "_" + name;
  };

  FieldType<H1<1, dim>> shape_disp_type(prefix("shape_displacement"));
  field_store->addShapeDisp(shape_disp_type);

  auto temperature_time_rule = std::make_shared<TemperatureTimeRule>(temp_rule);
  FieldType<H1<temp_order>> temperature_type(prefix("temperature_solve_state"));
  auto temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule);
  auto temperature_old_type =
      field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VAL, prefix("temperature"));

  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  std::string thermal_flux_name = prefix("thermal_flux");
  auto thermal_weak_form = std::make_shared<
      typename ThermalSystem<dim, temp_order, TemperatureTimeRule, parameter_space...>::ThermalWeakFormType>(
      thermal_flux_name, field_store->getMesh(), field_store->getField(temperature_type.name).get()->space(),
      field_store->createSpaces(thermal_flux_name, temperature_type.name, temperature_type, temperature_old_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  std::vector<std::shared_ptr<WeakForm>> weak_forms{thermal_weak_form};
  auto advancer = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, solver);

  return ThermalSystem<dim, temp_order, TemperatureTimeRule, parameter_space...>{
      {field_store, solver, advancer, parameter_fields, prepend_name},
      thermal_weak_form,
      temperature_bc,
      temperature_time_rule};
}

/**
 * @brief Factory function to build a thermal system with default quasi-static rule (backward compatible).
 */
template <int dim, int temp_order, typename... parameter_space>
auto buildThermalSystem(std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> solver,
                        std::string prepend_name = "", FieldType<parameter_space>... parameter_types)
{
  return buildThermalSystem<dim, temp_order>(mesh, solver, QuasiStaticFirstOrderTimeIntegrationRule{}, prepend_name,
                                             parameter_types...);
}

}  // namespace smith
