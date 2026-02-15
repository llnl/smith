// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermo_mechanics_system.hpp
 * @brief Defines the ThermoMechanicsSystem struct and its factory function
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"

namespace smith {

/**
 * @brief Container for a coupled thermo-mechanical system.
 * @tparam dim Spatial dimension.
 * @tparam disp_order Order of the displacement basis.
 * @tparam temp_order Order of the temperature basis.
 * @tparam parameter_space Finite element spaces for optional parameters.
 */
template <int dim, int disp_order, int temp_order, typename... parameter_space>
struct ThermoMechanicsSystem {
  /// @brief using for SolidWeakFormType
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<temp_order>, H1<temp_order>, parameter_space...>>;

  /// @brief using for ThermalWeakFormType
  using ThermalWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<temp_order>,
      Parameters<H1<temp_order>, H1<temp_order>, H1<disp_order, dim>, H1<disp_order, dim>, parameter_space...>>;

  std::shared_ptr<FieldStore> field_store;                      ///< Field store managing the system's fields.
  std::shared_ptr<SolidWeakFormType> solid_weak_form;           ///< Solid mechanics weak form.
  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;       ///< Thermal weak form.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;         ///< Displacement boundary conditions.
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;  ///< Temperature boundary conditions.
  std::shared_ptr<DifferentiableBlockSolver> solver;            ///< The solver for the coupled system.
  std::shared_ptr<StateAdvancer> advancer;                      ///< The multiphysics state advancer.
  std::shared_ptr<QuasiStaticFirstOrderTimeIntegrationRule> disp_time_rule;  ///< Time integration for displacement.
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule>
      temperature_time_rule;                 ///< Time integration for temperature.
  std::vector<FieldState> parameter_fields;  ///< Optional parameter fields.
  std::string physics_name_;

  /**
   * @brief Get the list of all state fields (current and old).
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    std::vector<FieldState> states;
    states.push_back(field_store->getField(physics_name_ + "_displacement"));
    states.push_back(field_store->getField(physics_name_ + "_displacement_old"));
    states.push_back(field_store->getField(physics_name_ + "_temperature"));
    states.push_back(field_store->getField(physics_name_ + "_temperature_old"));
    return states;
  }

  /**
   * @brief Get the list of all parameter fields.
   * @return const std::vector<FieldState>& List of parameter fields.
   */
  const std::vector<FieldState>& getParameterFields() const { return parameter_fields; }

  /**
   * @brief Create a DifferentiablePhysics object for this system.
   * @param physics_name The name of the physics.
   * @return std::shared_ptr<DifferentiablePhysics> The differentiable physics object.
   */
  std::shared_ptr<DifferentiablePhysics> createDifferentiablePhysics(std::string physics_name)
  {
    return std::make_shared<DifferentiablePhysics>(field_store->getMesh(), field_store->graph(),
                                                    field_store->getShapeDisp(), getStateFields(),
                                                    getParameterFields(), advancer, physics_name,
                                                    std::vector<std::string>{physics_name_ + "_solid_force",
                                                                             physics_name_ + "_thermal_flux"});
  }

  /**
   * @brief Set the material model for a domain, defining integrals for solid and thermal weak forms.
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    // Solid weak form: inputs are (u, u_old, temperature, temperature_old, params...)
    // Manually apply time integration rules to get current state
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    solid_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old,
                                                      auto temperature, auto temperature_old, auto... params) {
      // Apply time integration to get current state
      auto u_current = captured_disp_rule->value(t_info, u, u_old);
      auto v_current = captured_disp_rule->dot(t_info, u, u_old);
      auto T = captured_temp_rule->value(t_info, temperature, temperature_old);

      typename MaterialType::State state;
      auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current),
                                        get<VALUE>(T), get<DERIVATIVE>(T), params...);
      return smith::tuple{zero{}, pk};
    });

    // Thermal weak form: inputs are (T, T_old, u, u_old, params...)
    // Manually apply time integration rules to get current state
    thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto T, auto T_old,
                                                        auto disp, auto disp_old, auto... params) {
      // Apply time integration to get current state
      auto T_current = captured_temp_rule->value(t_info, T, T_old);
      auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
      auto u = captured_disp_rule->value(t_info, disp, disp_old);
      auto v = captured_disp_rule->dot(t_info, disp, disp_old);

      typename MaterialType::State state;
      auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v),
                                        get<VALUE>(T_current), get<DERIVATIVE>(T_current), params...);
      auto dT_dt = get<VALUE>(T_dot);
      return smith::tuple{C_v * dT_dt - s0, -q0};
    });
  }

  /**
   * @brief Add a body force to the solid mechanics part of the system.
   * @tparam BodyForceType The body force function type.
   * @param domain_name The name of the domain to apply the force to.
   * @param force_function The force function (t, X, u, v, T, params...).
   */
  template <typename BodyForceType>
  void addSolidBodyForce(const std::string& domain_name, BodyForceType force_function)
  {
      auto captured_disp_rule = disp_time_rule;
      auto captured_temp_rule = temperature_time_rule;

      solid_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto X, auto u, auto u_old,
                                                        auto temperature, auto temperature_old, auto... params) {
          // Apply time integration to get current state
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto v_current = captured_disp_rule->dot(t_info, u, u_old);
          auto current_T = captured_temp_rule->value(t_info, temperature, temperature_old);

          return smith::tuple{-force_function(t_info.time(), get<VALUE>(X), get<VALUE>(u_current),
                                              get<VALUE>(v_current), get<VALUE>(current_T),
                                              get<VALUE>(params)...), zero{}};
      });
  }

  /**
   * @brief Add a surface flux (traction) to the solid mechanics part of the system.
   * @tparam SurfaceFluxType The surface flux function type.
   * @param domain_name The name of the boundary domain to apply the flux to.
   * @param flux_function The flux function (t, X, n, u, v, T, params...).
   */
  template <typename SurfaceFluxType>
  void addSolidSurfaceFlux(const std::string& domain_name, SurfaceFluxType flux_function)
  {
      auto captured_disp_rule = disp_time_rule;
      auto captured_temp_rule = temperature_time_rule;

      solid_weak_form->addSurfaceFlux(domain_name, [=](auto t_info, auto X, auto n, auto u, auto u_old,
                                                        auto temperature, auto temperature_old, auto... params) {
          // Apply time integration to get current state
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto v_current = captured_disp_rule->dot(t_info, u, u_old);
          auto current_T = captured_temp_rule->value(t_info, temperature, temperature_old);

          return -flux_function(t_info.time(), get<VALUE>(X), n, get<VALUE>(u_current), get<VALUE>(v_current),
                                get<VALUE>(current_T), get<VALUE>(params)...);
      });
  }

  /**
   * @brief Add a body source (heat source) to the thermal part of the system.
   * @tparam BodySourceType The body source function type.
   * @param domain_name The name of the domain to apply the source to.
   * @param source_function The source function (t, X, T, T_dot, u, params...).
   */
  template <typename BodySourceType>
  void addThermalBodySource(const std::string& domain_name, BodySourceType source_function)
  {
      auto captured_disp_rule = disp_time_rule;
      auto captured_temp_rule = temperature_time_rule;

      thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto X, auto T, auto T_old,
                                                          auto disp, auto disp_old, auto... params) {
          // Apply time integration to get current state
          auto T_current = captured_temp_rule->value(t_info, T, T_old);
          auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
          auto current_u = captured_disp_rule->value(t_info, disp, disp_old);

          return smith::tuple{-source_function(t_info.time(), get<VALUE>(X), get<VALUE>(T_current),
                                               get<VALUE>(T_dot), get<VALUE>(current_u),
                                               get<VALUE>(params)...), zero{}};
      });
  }

  /**
   * @brief Add a surface flux (heat flux) to the thermal part of the system.
   * @tparam SurfaceFluxType The surface flux function type.
   * @param domain_name The name of the boundary domain to apply the flux to.
   * @param flux_function The flux function (t, X, n, T, T_dot, u, params...).
   */
  template <typename SurfaceFluxType>
  void addThermalSurfaceFlux(const std::string& domain_name, SurfaceFluxType flux_function)
  {
      auto captured_disp_rule = disp_time_rule;
      auto captured_temp_rule = temperature_time_rule;

      thermal_weak_form->addSurfaceFlux(domain_name, [=](auto t_info, auto X, auto n, auto T, auto T_old,
                                                          auto disp, auto disp_old, auto... params) {
          // Apply time integration to get current state
          auto T_current = captured_temp_rule->value(t_info, T, T_old);
          auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
          auto current_u = captured_disp_rule->value(t_info, disp, disp_old);

          return -flux_function(t_info.time(), get<VALUE>(X), n, get<VALUE>(T_current), get<VALUE>(T_dot),
                                get<VALUE>(current_u), get<VALUE>(params)...);
      });
  }
};

/**
 * @brief Factory function to build a thermo-mechanical system.
 * @tparam dim Spatial dimension.
 * @tparam disp_order Order of the displacement basis.
 * @tparam temp_order Order of the temperature basis.
 * @tparam parameter_space Finite element spaces for optional parameters.
 * @param mesh The mesh.
 * @param solver The differentiable block solver.
 * @param physics_name The name of the physics (used as field prefix).
 * @param parameter_types Parameter field types.
 * @return ThermoMechanicsSystem with all components initialized.
 */
template <int dim, int disp_order, int temp_order, typename... parameter_space>
ThermoMechanicsSystem<dim, disp_order, temp_order, parameter_space...> buildThermoMechanicsSystem(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<DifferentiableBlockSolver> solver, std::string physics_name,
    FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<FieldStore>(mesh, 100);

  FieldType<H1<1, dim>> shape_disp_type(physics_name + "_shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  // Displacement field with quasi-static time integration
  auto disp_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
  FieldType<H1<disp_order, dim>> disp_type(physics_name + "_displacement");
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE, physics_name + "_displacement_old");

  // Temperature field with backward Euler time integration
  auto temperature_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  FieldType<H1<temp_order>> temperature_type(physics_name + "_temperature");
  auto temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule);
  auto temperature_old_type = field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VALUE, physics_name + "_temperature_old");

  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(physics_name + "_param_" + parameter_types.name)), ...);
  (parameter_fields.push_back(field_store->getField(physics_name + "_param_" + parameter_types.name)), ...);

  // Solid mechanics weak form
  std::string solid_force_name = physics_name + "_solid_force";
  field_store->addWeakFormTestField(solid_force_name, disp_type.name);
  const mfem::ParFiniteElementSpace& disp_test_space = field_store->getField(disp_type.name).get()->space();
  std::vector<const mfem::ParFiniteElementSpace*> disp_input_spaces;
  createSpaces(solid_force_name, *field_store, disp_input_spaces, 0, disp_type, disp_old_type,
               temperature_type, temperature_old_type, FieldType<parameter_space>(physics_name + "_param_" + parameter_types.name)...);

  auto solid_weak_form = std::make_shared<typename ThermoMechanicsSystem<dim, disp_order, temp_order, parameter_space...>::SolidWeakFormType>(
      solid_force_name, field_store->getMesh(), disp_test_space, disp_input_spaces);

  // Thermal weak form
  std::string thermal_flux_name = physics_name + "_thermal_flux";
  field_store->addWeakFormTestField(thermal_flux_name, temperature_type.name);
  const mfem::ParFiniteElementSpace& temp_test_space = field_store->getField(temperature_type.name).get()->space();
  std::vector<const mfem::ParFiniteElementSpace*> temp_input_spaces;
  createSpaces(thermal_flux_name, *field_store, temp_input_spaces, 0, temperature_type,
               temperature_old_type, disp_type, disp_old_type, FieldType<parameter_space>(physics_name + "_param_" + parameter_types.name)...);

  auto thermal_weak_form = std::make_shared<typename ThermoMechanicsSystem<dim, disp_order, temp_order, parameter_space...>::ThermalWeakFormType>(
      thermal_flux_name, field_store->getMesh(), temp_test_space, temp_input_spaces);

  // Build solver and advancer
  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, thermal_weak_form};
  auto advancer = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, solver);

  return ThermoMechanicsSystem<dim, disp_order, temp_order, parameter_space...>{
      field_store, solid_weak_form, thermal_weak_form,     disp_bc,         temperature_bc, solver,
      advancer,    disp_time_rule,  temperature_time_rule, parameter_fields, physics_name};
}

}  // namespace smith
