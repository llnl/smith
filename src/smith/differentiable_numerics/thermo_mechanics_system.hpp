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
 * @brief Container for a coupled thermo-mechanical system with configurable time integration.
 *
 * Displacement uses a 4-state second-order layout (displacement_solve_state, displacement, velocity, acceleration).
 * Temperature uses a 2-state first-order layout (temperature_solve_state, temperature).
 * Total: 6 state fields.
 *
 * @tparam dim Spatial dimension.
 * @tparam disp_order Order of the displacement basis.
 * @tparam temp_order Order of the temperature basis.
 * @tparam DisplacementTimeRule Time integration rule type for displacement (must have num_states == 4).
 * @tparam TemperatureTimeRule Time integration rule type for temperature (must have num_states == 2).
 * @tparam parameter_space Finite element spaces for optional parameters.
 */
template <int dim, int disp_order, int temp_order,
          typename DisplacementTimeRule = QuasiStaticSecondOrderTimeIntegrationRule,
          typename TemperatureTimeRule = BackwardEulerFirstOrderTimeIntegrationRule, typename... parameter_space>
struct ThermoMechanicsSystem : public SystemBase {
  static_assert(DisplacementTimeRule::num_states == 4,
                "ThermoMechanicsSystem requires a 4-state displacement time rule");
  static_assert(TemperatureTimeRule::num_states == 2, "ThermoMechanicsSystem requires a 2-state temperature time rule");

  /// @brief using for SolidWeakFormType
  using SolidWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<disp_order, dim>,
                              Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<disp_order, dim>,
                                         H1<disp_order, dim>, H1<temp_order>, H1<temp_order>, parameter_space...>>;

  /// @brief using for ThermalWeakFormType
  using ThermalWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<temp_order>,
                              Parameters<H1<temp_order>, H1<temp_order>, H1<disp_order, dim>, H1<disp_order, dim>,
                                         H1<disp_order, dim>, H1<disp_order, dim>, parameter_space...>>;

  // Cycle-zero weak form: test field = acceleration, inputs: u, v, a, temp, temp_old, params...
  /// @brief using for CycleZeroWeakFormType
  using CycleZeroWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<disp_order, dim>,
                              Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<disp_order, dim>, H1<temp_order>,
                                         H1<temp_order>, parameter_space...>>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;           ///< Solid mechanics weak form.
  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;       ///< Thermal weak form.
  std::shared_ptr<CycleZeroWeakFormType> cycle_zero_weak_form;  ///< Cycle-zero weak form.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;         ///< Displacement boundary conditions.
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;  ///< Temperature boundary conditions.
  std::shared_ptr<DisplacementTimeRule> disp_time_rule;         ///< Time integration for displacement.
  std::shared_ptr<TemperatureTimeRule> temperature_time_rule;   ///< Time integration for temperature.

  /**
   * @brief Get the list of all state fields (disp_pred, disp, vel, accel, temp_pred, temp).
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    return {field_store->getField(prefix("displacement_solve_state")),
            field_store->getField(prefix("displacement")),
            field_store->getField(prefix("velocity")),
            field_store->getField(prefix("acceleration")),
            field_store->getField(prefix("temperature_solve_state")),
            field_store->getField(prefix("temperature"))};
  }

  /**
   * @brief Get information about reaction fields for this system.
   * @return List of ReactionInfo structures.
   */
  std::vector<ReactionInfo> getReactionInfos() const
  {
    return {{prefix("solid_force"), &field_store->getField(prefix("displacement")).get()->space()},
            {prefix("thermal_flux"), &field_store->getField(prefix("temperature")).get()->space()}};
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
   * @brief Set the material model for a domain, defining integrals for solid and thermal weak forms.
   *
   * The material is called as material(dt, state, grad_u, grad_v, T, grad_T, params...) and
   * must expose a `density` member for the cycle-zero acceleration solve.
   *
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    solid_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old, auto a_old, auto temperature,
                         auto temperature_old, auto... params) {
          auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
          auto T = captured_temp_rule->value(t_info, temperature, temperature_old);

          typename MaterialType::State state;
          auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current),
                                            get<VALUE>(T), get<DERIVATIVE>(T), params...);
          return smith::tuple{zero{}, pk};
        });

    thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto T, auto T_old, auto disp,
                                                        auto disp_old, auto v_old, auto a_old, auto... params) {
      auto [T_current, T_dot] = captured_temp_rule->interpolate(t_info, T, T_old);
      auto [u, v, a] = captured_disp_rule->interpolate(t_info, disp, disp_old, v_old, a_old);

      typename MaterialType::State state;
      auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v),
                                        get<VALUE>(T_current), get<DERIVATIVE>(T_current), params...);
      auto dT_dt = get<VALUE>(T_dot);
      return smith::tuple{C_v * dT_dt - s0, -q0};
    });

    // Cycle-zero: u and v are given, solve for a; temperature at initial condition
    cycle_zero_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto u, auto v, auto a,
                                                           auto temperature, auto temperature_old, auto... params) {
      auto T = captured_temp_rule->value(t_info, temperature, temperature_old);

      typename MaterialType::State state;
      auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T),
                                        get<DERIVATIVE>(T), params...);
      return smith::tuple{get<VALUE>(a) * material.density, pk};
    });
  }

  /**
   * @brief Add a body force to the solid mechanics part of the system (with DependsOn).
   */
  template <int... active_parameters, typename BodyForceType>
  void addSolidBodyForce(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                         BodyForceType force_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    solid_weak_form->addBodySource(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto temperature, auto temperature_old,
            auto... params) {
          auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
          auto [current_T, T_dot] = captured_temp_rule->interpolate(t_info, temperature, temperature_old);
          return force_function(t_info.time(), X, u_current, v_current, a_current, current_T, T_dot, params...);
        });
  }

  /**
   * @brief Add a body force to the solid mechanics part of the system.
   */
  template <typename BodyForceType>
  void addSolidBodyForce(const std::string& domain_name, BodyForceType force_function)
  {
    addSolidBodyForceAllParams(domain_name, force_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a surface traction to the solid mechanics part (with DependsOn).
   */
  template <int... active_parameters, typename SurfaceFluxType>
  void addSolidTraction(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                        SurfaceFluxType flux_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    solid_weak_form->addBoundaryFlux(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto u_old, auto v_old, auto a_old, auto temperature,
            auto temperature_old, auto... params) {
          auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
          auto [current_T, T_dot] = captured_temp_rule->interpolate(t_info, temperature, temperature_old);
          return flux_function(t_info.time(), X, n, u_current, v_current, a_current, current_T, T_dot, params...);
        });

  }

  /**
   * @brief Add a surface traction to the solid mechanics part.
   */
  template <typename SurfaceFluxType>
  void addSolidTraction(const std::string& domain_name, SurfaceFluxType flux_function)
  {
    addSolidTractionAllParams(domain_name, flux_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{},
                              std::make_index_sequence<5 + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a body heat source to the thermal part (with DependsOn).
   */
  template <int... active_parameters, typename BodySourceType>
  void addHeatSource(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                     BodySourceType source_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBodySource(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto T, auto T_old, auto disp, auto disp_old, auto v_old, auto a_old, auto... params) {
          auto [current_u, v_current, a_current] =
              captured_disp_rule->interpolate(t_info, disp, disp_old, v_old, a_old);
          auto [T_current, T_dot] = captured_temp_rule->interpolate(t_info, T, T_old);
          return source_function(t_info.time(), X, current_u, v_current, a_current, T_current, T_dot, params...);
        });
  }

  /**
   * @brief Add a body heat source to the thermal part.
   */
  template <typename BodySourceType>
  void addHeatSource(const std::string& domain_name, BodySourceType source_function)
  {
    addHeatSourceAllParams(domain_name, source_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a boundary heat flux to the thermal part (with DependsOn).
   */
  template <int... active_parameters, typename SurfaceFluxType>
  void addHeatFlux(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   SurfaceFluxType flux_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBoundaryFlux(depends_on, domain_name,
                                       [=](auto t_info, auto X, auto n, auto T, auto T_old, auto disp, auto disp_old,
                                           auto v_old, auto a_old, auto... params) {
                                         auto [current_u, v_current, a_current] =
                                             captured_disp_rule->interpolate(t_info, disp, disp_old, v_old, a_old);
                                         auto [T_current, T_dot] = captured_temp_rule->interpolate(t_info, T, T_old);
                                         return -flux_function(t_info.time(), X, n, current_u, v_current, a_current,
                                                               T_current, T_dot, params...);
                                       });
  }

  /**
   * @brief Add a boundary heat flux to the thermal part.
   */
  template <typename SurfaceFluxType>
  void addHeatFlux(const std::string& domain_name, SurfaceFluxType flux_function)
  {
    addHeatFluxAllParams(domain_name, flux_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a pressure boundary condition (follower force) to the solid part (with DependsOn).
   */
  template <int... active_parameters, typename PressureType>
  void addPressure(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   PressureType pressure_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    solid_weak_form->addBoundaryIntegral(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto temperature, auto temperature_old,
            auto... params) {
          auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
          auto [T_current, T_dot] = captured_temp_rule->interpolate(t_info, temperature, temperature_old);

          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), u_current, v_current, a_current, T_current,
                                            T_dot, get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        });
  }

  /**
   * @brief Add a pressure boundary condition (follower force) to the solid part.
   */
  template <typename PressureType>
  void addPressure(const std::string& domain_name, PressureType pressure_function)
  {
    addPressureAllParams(domain_name, pressure_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

 private:
  template <typename BodyForceType, std::size_t... Is>
  void addSolidBodyForceAllParams(const std::string& domain_name, BodyForceType force_function,
                                  std::index_sequence<Is...>)
  {
    addSolidBodyForce(DependsOn<static_cast<int>(Is)...>{}, domain_name, force_function);
  }

  template <typename SurfaceFluxType, std::size_t... MainIs, std::size_t... CycleZeroIs>
  void addSolidTractionAllParams(const std::string& domain_name, SurfaceFluxType flux_function,
                                 std::index_sequence<MainIs...>, std::index_sequence<CycleZeroIs...>)
  {
    addSolidTraction(DependsOn<static_cast<int>(MainIs)...>{}, domain_name, flux_function);

    auto captured_temp_rule = temperature_time_rule;
    cycle_zero_weak_form->addBoundaryFlux(
        DependsOn<static_cast<int>(CycleZeroIs)...> {}, domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto v, auto a, auto temperature, auto temperature_old, auto... params) {
          auto [current_T, T_dot] = captured_temp_rule->interpolate(t_info, temperature, temperature_old);
          return flux_function(t_info.time(), X, n, u, v, a, current_T, T_dot, params...);
        });
  }

  template <typename PressureType, std::size_t... Is>
  void addPressureAllParams(const std::string& domain_name, PressureType pressure_function, std::index_sequence<Is...>)
  {
    addPressure(DependsOn<static_cast<int>(Is)...>{}, domain_name, pressure_function);
  }

  template <typename BodySourceType, std::size_t... Is>
  void addHeatSourceAllParams(const std::string& domain_name, BodySourceType source_function,
                              std::index_sequence<Is...>)
  {
    addHeatSource(DependsOn<static_cast<int>(Is)...>{}, domain_name, source_function);
  }

  template <typename SurfaceFluxType, std::size_t... Is>
  void addHeatFluxAllParams(const std::string& domain_name, SurfaceFluxType flux_function, std::index_sequence<Is...>)
  {
    addHeatFlux(DependsOn<static_cast<int>(Is)...>{}, domain_name, flux_function);
  }

};

/**
 * @brief Factory function to build a thermo-mechanical system.
 */
template <int dim, int disp_order, int temp_order, typename DisplacementTimeRule, typename TemperatureTimeRule,
          typename... parameter_space>
ThermoMechanicsSystem<dim, disp_order, temp_order, DisplacementTimeRule, TemperatureTimeRule, parameter_space...>
buildThermoMechanicsSystem(std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> solver,
                           DisplacementTimeRule disp_rule, TemperatureTimeRule temp_rule, std::string prepend_name = "",
                           std::shared_ptr<CoupledSystemSolver> cycle_zero_solver = nullptr,
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

  // Displacement fields (4-state second-order)
  auto disp_time_rule_ptr = std::make_shared<DisplacementTimeRule>(disp_rule);
  FieldType<H1<disp_order, dim>> disp_type(prefix("displacement_solve_state"));
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule_ptr);
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VAL, prefix("displacement"));
  auto velo_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DOT, prefix("velocity"));
  auto accel_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DDOT, prefix("acceleration"));

  // Temperature fields (2-state first-order)
  auto temperature_time_rule_ptr = std::make_shared<TemperatureTimeRule>(temp_rule);
  FieldType<H1<temp_order>> temperature_type(prefix("temperature_solve_state"));
  auto temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule_ptr);
  auto temperature_old_type =
      field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VAL, prefix("temperature"));

  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  using SystemType =
      ThermoMechanicsSystem<dim, disp_order, temp_order, DisplacementTimeRule, TemperatureTimeRule, parameter_space...>;

  // Solid mechanics weak form (u, u_old, v_old, a_old, temp, temp_old, params...)
  std::string solid_force_name = prefix("solid_force");
  auto solid_weak_form = std::make_shared<typename SystemType::SolidWeakFormType>(
      solid_force_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
      field_store->createSpaces(solid_force_name, disp_type.name, disp_type, disp_old_type, velo_old_type,
                                accel_old_type, temperature_type, temperature_old_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // Thermal weak form (temp, temp_old, u, u_old, v_old, a_old, params...)
  std::string thermal_flux_name = prefix("thermal_flux");
  auto thermal_weak_form = std::make_shared<typename SystemType::ThermalWeakFormType>(
      thermal_flux_name, field_store->getMesh(), field_store->getField(temperature_type.name).get()->space(),
      field_store->createSpaces(thermal_flux_name, temperature_type.name, temperature_type, temperature_old_type,
                                disp_type, disp_old_type, velo_old_type, accel_old_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // Cycle-zero weak form (u, v, a, temp, temp_old, params...)
  std::string cycle_zero_name = prefix("solid_reaction");
  auto cycle_zero_weak_form = std::make_shared<typename SystemType::CycleZeroWeakFormType>(
      cycle_zero_name, field_store->getMesh(), field_store->getField(accel_old_type.name).get()->space(),
      field_store->createSpaces(cycle_zero_name, accel_old_type.name, disp_type, velo_old_type, accel_old_type,
                                temperature_type, temperature_old_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  if (cycle_zero_solver == nullptr) {
    cycle_zero_solver = solver->singleBlockSolver(0);
  }
  SLIC_ERROR_IF(cycle_zero_solver == nullptr,
                "Could not derive a cycle-zero solver for block 0 from the provided thermo-mechanics solver.");

  // Build solver and advancer
  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, thermal_weak_form};
  auto advancer = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, solver, cycle_zero_weak_form,
                                                               cycle_zero_solver);

  return SystemType{{field_store, solver, advancer, parameter_fields, prepend_name},
                    solid_weak_form,
                    thermal_weak_form,
                    cycle_zero_weak_form,
                    disp_bc,
                    temperature_bc,
                    disp_time_rule_ptr,
                    temperature_time_rule_ptr};
}

/**
 * @brief Factory function to build a thermo-mechanical system with a physics name and parameter fields.
 */
template <int dim, int disp_order, int temp_order, typename DisplacementTimeRule, typename TemperatureTimeRule,
          typename... parameter_space>
auto buildThermoMechanicsSystem(std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> solver,
                                DisplacementTimeRule disp_rule, TemperatureTimeRule temp_rule,
                                std::string prepend_name, FieldType<parameter_space>... parameter_types)
{
  return buildThermoMechanicsSystem<dim, disp_order, temp_order>(
      mesh, solver, disp_rule, temp_rule, std::move(prepend_name), nullptr, parameter_types...);
}

/**
 * @brief Factory function to build a thermo-mechanical system (without physics name).
 */
template <int dim, int disp_order, int temp_order, typename DisplacementTimeRule, typename TemperatureTimeRule,
          typename... parameter_space>
auto buildThermoMechanicsSystem(std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> solver,
                                DisplacementTimeRule disp_rule, TemperatureTimeRule temp_rule,
                                std::shared_ptr<CoupledSystemSolver> cycle_zero_solver,
                                FieldType<parameter_space>... parameter_types)
{
  return buildThermoMechanicsSystem<dim, disp_order, temp_order>(
      mesh, solver, disp_rule, temp_rule, "", cycle_zero_solver, parameter_types...);
}

/**
 * @brief Factory function to build a thermo-mechanical system (without physics name).
 */
template <int dim, int disp_order, int temp_order, typename DisplacementTimeRule, typename TemperatureTimeRule,
          typename... parameter_space>
auto buildThermoMechanicsSystem(std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> solver,
                                DisplacementTimeRule disp_rule, TemperatureTimeRule temp_rule,
                                FieldType<parameter_space>... parameter_types)
{
  return buildThermoMechanicsSystem<dim, disp_order, temp_order>(
      mesh, solver, disp_rule, temp_rule, "", nullptr, parameter_types...);
}

}  // namespace smith
