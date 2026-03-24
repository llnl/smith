// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file extended_thermomechanics.hpp
 * @brief Defines an extended thermo-mechanics system with an additional L2 state variable.
 *
 * This is an example-facing "module" that mirrors `smith::ThermoMechanicsSystem` but adds an
 * additional scalar field (typically in an `L2<order>` space) with its own residual.
 *
 * The additional L2 field can be used for local internal variables (pointwise ODE evolution),
 * damage variables, phase fields (if treated in L2), etc.
 */

#pragma once

#include <type_traits>

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/physics/weak_form.hpp"

namespace smith {

/**
 * @brief Container for a coupled thermo-mechanical system with an additional L2 state variable.
 *
 * The system assembles and advances three coupled residuals:
 * - solid mechanics residual (test space: H1 displacement)
 * - thermal residual (test space: H1 temperature)
 * - state residual (test space: StateSpace, typically L2)
 *
 * @tparam dim Spatial dimension.
 * @tparam disp_order Order of the displacement basis.
 * @tparam temp_order Order of the temperature basis.
 * @tparam StateSpace FE space for the additional state field (e.g. L2<0>).
 * @tparam parameter_space FE spaces for optional parameters.
 */
template <int dim, int disp_order, int temp_order, typename StateSpace, typename... parameter_space>
struct ExtendedThermoMechanicsSystem : public SystemBase {
  /// @brief Solid mechanics weak form (residual for displacement).
  using SolidWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<disp_order, dim>,
                              Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<temp_order>, H1<temp_order>,
                                         StateSpace, StateSpace, parameter_space...>>;

  /// @brief Thermal weak form (residual for temperature).
  using ThermalWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<temp_order>,
                              Parameters<H1<temp_order>, H1<temp_order>, H1<disp_order, dim>, H1<disp_order, dim>,
                                         StateSpace, StateSpace, parameter_space...>>;

  /// @brief State weak form (residual for the additional state variable).
  using StateWeakFormType =
      TimeDiscretizedWeakForm<dim, StateSpace,
                              Parameters<StateSpace, StateSpace, H1<disp_order, dim>, H1<disp_order, dim>,
                                         H1<temp_order>, H1<temp_order>, parameter_space...>>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;
  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;
  std::shared_ptr<StateWeakFormType> state_weak_form;

  std::shared_ptr<DirichletBoundaryConditions> disp_bc;
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;
  std::shared_ptr<DirichletBoundaryConditions> state_bc;

  std::shared_ptr<QuasiStaticFirstOrderTimeIntegrationRule> disp_time_rule;
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule> temperature_time_rule;
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule> state_time_rule;

  std::vector<FieldState> getStateFields() const
  {
    return {field_store->getField(prefix("displacement_predicted")), field_store->getField(prefix("displacement")),
            field_store->getField(prefix("temperature_predicted")),  field_store->getField(prefix("temperature")),
            field_store->getField(prefix("state_predicted")),        field_store->getField(prefix("state"))};
  }

  std::shared_ptr<DifferentiablePhysics> createDifferentiablePhysics(std::string physics_name)
  {
    return std::make_shared<DifferentiablePhysics>(
        field_store->getMesh(), field_store->graph(), field_store->getShapeDisp(), getStateFields(),
        getParameterFields(), advancer, std::move(physics_name),
        std::vector<std::string>{prefix("solid_force"), prefix("thermal_flux"), prefix("state_residual")});
  }

  /**
   * @brief Set the thermo-mechanical material model for a domain.
   *
   * This wires the constitutive model into the solid and thermal weak forms. The material model
   * may optionally depend on the additional state variable (and its rate).
   *
   * Expected material signature:
   *   material(dt, state, grad_u, grad_v, theta, grad_theta, alpha, alpha_dot, params...) -> (pk, C_v, s0, q0)
   *
   * - `pk`   : first Piola stress-like flux for momentum balance
   * - `C_v`  : heat capacity-like coefficient
   * - `s0`   : thermo-mechanical source term
   * - `q0`   : heat flux
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto temperature, auto temperature_old,
                         auto /*alpha*/, auto alpha_old, auto... params) {
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto v_current = captured_disp_rule->dot(t_info, u, u_old);
          auto T = captured_temp_rule->value(t_info, temperature, temperature_old);

          typename MaterialType::State state;
          auto [pk, C_v, s0, q0, alpha_new] =
              material(t_info.dt(), state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current), get<VALUE>(T),
                       get<DERIVATIVE>(T), get<VALUE>(alpha_old), params...);

          // auto delta_alpha = alpha - alpha_old;

          return smith::tuple{zero{}, pk};
        });

    thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto T, auto T_old, auto disp,
                                                        auto disp_old, auto /*alpha*/, auto alpha_old, auto... params) {
      auto T_current = captured_temp_rule->value(t_info, T, T_old);
      auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
      auto u = captured_disp_rule->value(t_info, disp, disp_old);
      auto v = captured_disp_rule->dot(t_info, disp, disp_old);

      typename MaterialType::State state;
      auto [pk, C_v, s0, q0, alpha_new] =
          material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T_current),
                   get<DERIVATIVE>(T_current), get<VALUE>(alpha_old), params...);

      auto dT_dt = get<VALUE>(T_dot);
      return smith::tuple{C_v * dT_dt - s0, -q0};
    });

    state_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto alpha, auto alpha_old, auto disp,
                                                      auto disp_old, auto T, auto T_old, auto... params) {
      auto T_current = captured_temp_rule->value(t_info, T, T_old);
      auto u = captured_disp_rule->value(t_info, disp, disp_old);
      auto v = captured_disp_rule->dot(t_info, disp, disp_old);
      // auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
      // auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

      typename MaterialType::State state;
      auto [pk, C_v, s0, q0, alpha_new] =
          material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T_current),
                   get<DERIVATIVE>(T_current), get<VALUE>(alpha_old), params...);

      using FluxType = std::decay_t<decltype(get<DERIVATIVE>(alpha))>;
      FluxType flux{};
      return smith::tuple{get<VALUE>(alpha) - alpha_new, flux};
    });
  }

  template <int... active_parameters, typename BodyForceType>
  void addSolidBodyForce(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                         BodyForceType force_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBodySource(depends_on, domain_name,
                                   [=](auto t_info, auto X, auto u, auto u_old, auto temperature, auto temperature_old,
                                       auto alpha, auto alpha_old, auto... params) {
                                     auto u_current = captured_disp_rule->value(t_info, u, u_old);
                                     auto v_current = captured_disp_rule->dot(t_info, u, u_old);
                                     auto T_current = captured_temp_rule->value(t_info, temperature, temperature_old);
                                     auto T_dot = captured_temp_rule->dot(t_info, temperature, temperature_old);
                                     auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
                                     auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

                                     return force_function(t_info.time(), X, u_current, v_current, T_current, T_dot,
                                                           alpha_current, alpha_dot, params...);
                                   });
  }

  template <typename BodyForceType>
  void addSolidBodyForce(const std::string& domain_name, BodyForceType force_function)
  {
    addSolidBodyForceAllParams(domain_name, force_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename SurfaceFluxType>
  void addSolidTraction(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                        SurfaceFluxType flux_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBoundaryFlux(depends_on, domain_name,
                                     [=](auto t_info, auto X, auto n, auto u, auto u_old, auto temperature,
                                         auto temperature_old, auto alpha, auto alpha_old, auto... params) {
                                       auto u_current = captured_disp_rule->value(t_info, u, u_old);
                                       auto v_current = captured_disp_rule->dot(t_info, u, u_old);
                                       auto T_current = captured_temp_rule->value(t_info, temperature, temperature_old);
                                       auto T_dot = captured_temp_rule->dot(t_info, temperature, temperature_old);
                                       auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
                                       auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

                                       return flux_function(t_info.time(), X, n, u_current, v_current, T_current, T_dot,
                                                            alpha_current, alpha_dot, params...);
                                     });
  }

  template <typename SurfaceFluxType>
  void addSolidTraction(const std::string& domain_name, SurfaceFluxType flux_function)
  {
    addSolidTractionAllParams(domain_name, flux_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename BodySourceType>
  void addThermalHeatSource(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                            BodySourceType source_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    thermal_weak_form->addBodySource(depends_on, domain_name,
                                     [=](auto t_info, auto X, auto T, auto T_old, auto disp, auto disp_old, auto alpha,
                                         auto alpha_old, auto... params) {
                                       auto u_current = captured_disp_rule->value(t_info, disp, disp_old);
                                       auto v_current = captured_disp_rule->dot(t_info, disp, disp_old);
                                       auto T_current = captured_temp_rule->value(t_info, T, T_old);
                                       auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
                                       auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
                                       auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

                                       return source_function(t_info.time(), X, u_current, v_current, T_current, T_dot,
                                                              alpha_current, alpha_dot, params...);
                                     });
  }

  template <typename BodySourceType>
  void addThermalHeatSource(const std::string& domain_name, BodySourceType source_function)
  {
    addThermalHeatSourceAllParams(domain_name, source_function,
                                  std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename SurfaceFluxType>
  void addThermalHeatFlux(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                          SurfaceFluxType flux_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    thermal_weak_form->addBoundaryFlux(depends_on, domain_name,
                                       [=](auto t_info, auto X, auto n, auto T, auto T_old, auto disp, auto disp_old,
                                           auto alpha, auto alpha_old, auto... params) {
                                         auto u_current = captured_disp_rule->value(t_info, disp, disp_old);
                                         auto v_current = captured_disp_rule->dot(t_info, disp, disp_old);
                                         auto T_current = captured_temp_rule->value(t_info, T, T_old);
                                         auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
                                         auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
                                         auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

                                         return -flux_function(t_info.time(), X, n, u_current, v_current, T_current,
                                                               T_dot, alpha_current, alpha_dot, params...);
                                       });
  }

  template <typename SurfaceFluxType>
  void addThermalHeatFlux(const std::string& domain_name, SurfaceFluxType flux_function)
  {
    addThermalHeatFluxAllParams(domain_name, flux_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename PressureType>
  void addPressure(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   PressureType pressure_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBoundaryIntegral(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto temperature, auto temperature_old, auto alpha, auto alpha_old,
            auto... params) {
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto v_current = captured_disp_rule->dot(t_info, u, u_old);
          auto T_current = captured_temp_rule->value(t_info, temperature, temperature_old);
          auto T_dot = captured_temp_rule->dot(t_info, temperature, temperature_old);
          auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
          auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), u_current, v_current, T_current, T_dot,
                                            alpha_current, alpha_dot, get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        });
  }

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

  template <typename SurfaceFluxType, std::size_t... Is>
  void addSolidTractionAllParams(const std::string& domain_name, SurfaceFluxType flux_function,
                                 std::index_sequence<Is...>)
  {
    addSolidTraction(DependsOn<static_cast<int>(Is)...>{}, domain_name, flux_function);
  }

  template <typename PressureType, std::size_t... Is>
  void addPressureAllParams(const std::string& domain_name, PressureType pressure_function, std::index_sequence<Is...>)
  {
    addPressure(DependsOn<static_cast<int>(Is)...>{}, domain_name, pressure_function);
  }

  template <typename BodySourceType, std::size_t... Is>
  void addThermalHeatSourceAllParams(const std::string& domain_name, BodySourceType source_function,
                                     std::index_sequence<Is...>)
  {
    addThermalHeatSource(DependsOn<static_cast<int>(Is)...>{}, domain_name, source_function);
  }

  template <typename SurfaceFluxType, std::size_t... Is>
  void addThermalHeatFluxAllParams(const std::string& domain_name, SurfaceFluxType flux_function,
                                   std::index_sequence<Is...>)
  {
    addThermalHeatFlux(DependsOn<static_cast<int>(Is)...>{}, domain_name, flux_function);
  }
};

/**
 * @brief Factory function to build an extended thermo-mechanics system with an L2 state variable.
 *
 * Field naming convention (with optional `prepend_name` prefix):
 * - `displacement_predicted`, `displacement`
 * - `temperature_predicted`, `temperature`
 * - `state_predicted`, `state`
 * - parameter fields: `param_<name>`
 * - residual names: `solid_force`, `thermal_flux`, `state_residual`
 */
template <int dim, int disp_order, int temp_order, typename StateSpace, typename... parameter_space>
ExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace, parameter_space...>
buildExtendedThermoMechanicsSystem(std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> coupled_solver,
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

  // Displacement: quasi-static
  auto disp_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
  FieldType<H1<disp_order, dim>> disp_type(prefix("displacement_predicted"));
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VAL, prefix("displacement"));

  // Temperature: backward Euler
  auto temperature_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  FieldType<H1<temp_order>> temperature_type(prefix("temperature_predicted"));
  auto temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule);
  auto temperature_old_type =
      field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VAL, prefix("temperature"));

  // State (L2): backward Euler by default
  auto state_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  FieldType<StateSpace> state_type(prefix("state_predicted"));
  auto state_bc = field_store->addIndependent(state_type, state_time_rule);
  auto state_old_type = field_store->addDependent(state_type, FieldStore::TimeDerivative::VAL, prefix("state"));

  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  // Solid weak form: u residual depends on (u, u_old, T, T_old, alpha, alpha_old, params...)
  std::string solid_force_name = prefix("solid_force");
  auto solid_weak_form =
      std::make_shared<typename ExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace,
                                                              parameter_space...>::SolidWeakFormType>(
          solid_force_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
          field_store->createSpaces(solid_force_name, disp_type.name, disp_type, disp_old_type, temperature_type,
                                    temperature_old_type, state_type, state_old_type,
                                    FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // Thermal weak form: T residual depends on (T, T_old, u, u_old, alpha, alpha_old, params...)
  std::string thermal_flux_name = prefix("thermal_flux");
  auto thermal_weak_form =
      std::make_shared<typename ExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace,
                                                              parameter_space...>::ThermalWeakFormType>(
          thermal_flux_name, field_store->getMesh(), field_store->getField(temperature_type.name).get()->space(),
          field_store->createSpaces(thermal_flux_name, temperature_type.name, temperature_type, temperature_old_type,
                                    disp_type, disp_old_type, state_type, state_old_type,
                                    FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // State weak form: alpha residual depends on (alpha, alpha_old, u, u_old, T, T_old, params...)
  std::string state_residual_name = prefix("state_residual");
  auto state_weak_form =
      std::make_shared<typename ExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace,
                                                              parameter_space...>::StateWeakFormType>(
          state_residual_name, field_store->getMesh(), field_store->getField(state_type.name).get()->space(),
          field_store->createSpaces(state_residual_name, state_type.name, state_type, state_old_type, disp_type,
                                    disp_old_type, temperature_type, temperature_old_type,
                                    FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, thermal_weak_form, state_weak_form};
  auto advancer = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, coupled_solver);

  return ExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace, parameter_space...>{
      {field_store, coupled_solver, advancer, parameter_fields, prepend_name},
      solid_weak_form,
      thermal_weak_form,
      state_weak_form,
      disp_bc,
      temperature_bc,
      state_bc,
      disp_time_rule,
      temperature_time_rule,
      state_time_rule};
}

}  // namespace smith
