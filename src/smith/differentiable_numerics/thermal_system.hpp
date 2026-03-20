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
 * @brief Container for a thermal system.
 * @tparam dim Spatial dimension.
 * @tparam temp_order Order of the temperature basis.
 * @tparam parameter_space Finite element spaces for optional parameters.
 */
template <int dim, int temp_order, typename... parameter_space>
struct ThermalSystem : public SystemBase {
  /// @brief using for ThermalWeakFormType
  using ThermalWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<temp_order>, Parameters<H1<temp_order>, parameter_space...>>;

  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;       ///< Thermal weak form.
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;  ///< Temperature boundary conditions.
  std::shared_ptr<QuasiStaticRule> temperature_time_rule;       ///< Time integration for temperature.

  /**
   * @brief Get the list of all state fields.
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    std::vector<FieldState> states;
    states.push_back(field_store->getField(prefix("temperature")));
    return states;
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
        getParameterFields(), advancer, physics_name, std::vector<std::string>{prefix("thermal_flux")});
  }

  /**
   * @brief Set the material model for a domain.
   * @tparam ThermalIntegrandType Function with signature (TimeInfo, X, T, params...) -> {residual, flux}.
   * @param domain_name The name of the domain to apply the material to.
   * @param integrand The thermal integrand function.
   */
  template <typename ThermalIntegrandType>
  void setThermalIntegrand(const std::string& domain_name, ThermalIntegrandType integrand)
  {
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto X, auto temperature, auto... params) {
      // Apply time integration to get current state
      auto T = captured_temp_rule->value(t_info, temperature);
      return integrand(t_info, X, T, params...);
    });
  }

  /**
   * @brief Add a body heat source to the thermal system.
   * @tparam HeatSourceType Function with signature (t, X, T, params...) -> heat_source.
   * @param domain_name The name of the domain to apply the source to.
   * @param source_function The heat source function.
   */
  template <typename HeatSourceType>
  void addBodyHeatSource(const std::string& domain_name, HeatSourceType source_function)
  {
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBodySource(domain_name, [=](auto t_info, auto X, auto temperature, auto... params) {
      auto T = captured_temp_rule->value(t_info, temperature);
      return source_function(t_info.time(), X, T, params...);
    });
  }

  /**
   * @brief Add a boundary heat flux to the thermal system.
   * @tparam HeatFluxType Function with signature (t, X, n, T, params...) -> heat_flux.
   * @param boundary_name The name of the boundary to apply the flux to.
   * @param flux_function The heat flux function.
   */
  template <typename HeatFluxType>
  void addBoundaryHeatFlux(const std::string& boundary_name, HeatFluxType flux_function)
  {
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBoundaryFlux(boundary_name,
                                       [=](auto t_info, auto X, auto n, auto temperature, auto... params) {
                                         auto T = captured_temp_rule->value(t_info, temperature);
                                         return -flux_function(t_info.time(), X, n, T, params...);
                                       });
  }
};

/**
 * @brief Factory function to build a thermal system.
 */
template <int dim, int temp_order, typename... parameter_space>
ThermalSystem<dim, temp_order, parameter_space...> buildThermalSystem(std::shared_ptr<Mesh> mesh,
                                                                      std::shared_ptr<CoupledSystemSolver> solver,
                                                                      std::string prepend_name = "",
                                                                      FieldType<parameter_space>... parameter_types)
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

  // Temperature field with quasi-static rule (1 state)
  auto temperature_time_rule = std::make_shared<QuasiStaticRule>();
  FieldType<H1<temp_order>> temperature_type(prefix("temperature"));
  auto temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule);

  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  // Thermal weak form
  std::string thermal_flux_name = prefix("thermal_flux");
  auto thermal_weak_form =
      std::make_shared<typename ThermalSystem<dim, temp_order, parameter_space...>::ThermalWeakFormType>(
          thermal_flux_name, field_store->getMesh(), field_store->getField(temperature_type.name).get()->space(),
          field_store->createSpaces(thermal_flux_name, temperature_type.name, temperature_type,
                                    FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // Build solver and advancer
  std::vector<std::shared_ptr<WeakForm>> weak_forms{thermal_weak_form};
  auto advancer = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, solver);

  return ThermalSystem<dim, temp_order, parameter_space...>{
      {field_store, solver, advancer, parameter_fields, prepend_name},
      thermal_weak_form,
      temperature_bc,
      temperature_time_rule};
}

}  // namespace smith
