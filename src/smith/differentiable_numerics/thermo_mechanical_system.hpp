// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermo_mechanical_system.hpp
 * @brief Helper to wire a coupled thermo-mechanical material to a SolidMechanicsSystem and ThermalSystem.
 */

#pragma once

#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"

namespace smith {

namespace detail {

/**
 * @brief Dispatch to either coupled-rate or established thermoelastic material signatures.
 *
 * Supports materials that accept `(dt, state, grad_u, grad_v, theta, grad_theta, params...)`
 * and materials that accept `(state, grad_u, theta, grad_theta, params...)`.
 */
template <typename MaterialType, typename DT, typename StateType, typename GradUType, typename GradVType,
          typename ThetaType, typename GradThetaType, typename... ParamTypes>
auto evaluateCoupledThermoMechanicsMaterial(const MaterialType& material, DT dt, StateType& state,
                                            const GradUType& grad_u, const GradVType& grad_v, ThetaType theta,
                                            const GradThetaType& grad_theta, ParamTypes&&... params)
{
  if constexpr (requires {
                  material(dt, state, grad_u, grad_v, theta, grad_theta, std::forward<ParamTypes>(params)...);
                }) {
    return material(dt, state, grad_u, grad_v, theta, grad_theta, std::forward<ParamTypes>(params)...);
  } else {
    return material(state, grad_u, theta, grad_theta, std::forward<ParamTypes>(params)...);
  }
}

}  // namespace detail

/**
 * @brief Register a coupled thermo-mechanical material integrand on a SolidMechanicsSystem
 *        and ThermalSystem that were built with mutual coupling fields.
 *
 * Assumes:
 *  - solid was built with thermal fields as the leading coupling fields
 *    (first 2 coupling positions: temperature_solve_state, temperature).
 *  - thermal was built with solid displacement fields as the leading coupling fields
 *    (first 4 coupling positions: displacement_solve_state, displacement, velocity, acceleration).
 *
 * The solid integrand lambda receives:
 *   (t_info, X, u, u_old, v_old, a_old, temperature_ss, temperature_old, ...params)
 * The thermal integrand lambda receives:
 *   (t_info, X, T, T_old, disp_ss, displacement, velocity, acceleration, ...params)
 *
 * The material callable must satisfy:
 *   material(dt, state, grad_u, grad_v, T_value, grad_T, params...) -> tuple{PK1, C_v, s0, q0}
 *
 * @tparam dim            Spatial dimension (deduced from SolidMechanicsSystem template arg).
 * @tparam disp_order_    Displacement polynomial order.
 * @tparam temp_order_    Temperature polynomial order.
 * @tparam DispRule       Displacement time integration rule.
 * @tparam TempRule       Temperature time integration rule.
 * @tparam SolidCoupling  Coupling type on the solid system.
 * @tparam ThermalCoupling Coupling type on the thermal system.
 * @tparam MaterialType   Material model type.
 * @param solid       Solid mechanics system with thermal coupling.
 * @param thermal     Thermal system with solid displacement coupling.
 * @param material    Material model instance.
 * @param domain_name Domain on which to apply the material.
 */
template <int dim, int disp_order_, int temp_order_, typename DispRule, typename TempRule, typename SolidCoupling,
          typename ThermalCoupling, typename MaterialType>
void setCoupledThermoMechanicsMaterial(
    std::shared_ptr<SolidMechanicsSystem<dim, disp_order_, DispRule, SolidCoupling>> solid,
    std::shared_ptr<ThermalSystem<dim, temp_order_, TempRule, ThermalCoupling>> thermal, const MaterialType& material,
    const std::string& domain_name)
{
  auto captured_disp_rule = solid->disp_time_rule;
  auto captured_temp_rule = thermal->temperature_time_rule;

  // Solid contribution: inertia + PK1 stress
  solid->solid_weak_form->addBodyIntegral(
      domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old, auto a_old, auto temperature,
                       auto temperature_old, auto... params) {
        auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
        auto T = captured_temp_rule->value(t_info, temperature, temperature_old);

        typename MaterialType::State state{};
        auto [pk, C_v, s0, q0] = detail::evaluateCoupledThermoMechanicsMaterial(
            material, t_info.dt(), state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current), get<VALUE>(T),
            get<DERIVATIVE>(T), params...);
        return smith::tuple{get<VALUE>(a_current) * material.density, pk};
      });

  // Cycle-zero: (u, v, a, temperature, temperature_old, ...params)
  if (solid->cycle_zero_solid_weak_form) {
    solid->cycle_zero_solid_weak_form->addBodyIntegral(
        domain_name,
        [=](auto t_info, auto /*X*/, auto u, auto v, auto a, auto temperature, auto temperature_old, auto... params) {
          auto T = captured_temp_rule->value(t_info, temperature, temperature_old);
          typename MaterialType::State state{};
          auto [pk, C_v, s0, q0] = detail::evaluateCoupledThermoMechanicsMaterial(
              material, t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T), get<DERIVATIVE>(T),
              params...);
          return detail::makeScaledCycleZeroResidual(t_info, get<VALUE>(a) * material.density, pk);
        });
  }

  // Thermal contribution: (T, T_old, disp_ss, displacement, velocity, acceleration, ...params)
  thermal->thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto T, auto T_old, auto disp,
                                                               auto disp_old, auto v_old, auto a_old, auto... params) {
    auto [T_current, T_dot] = captured_temp_rule->interpolate(t_info, T, T_old);
    auto [u, v, a] = captured_disp_rule->interpolate(t_info, disp, disp_old, v_old, a_old);

    typename MaterialType::State state{};
    auto [pk, C_v, s0, q0] = detail::evaluateCoupledThermoMechanicsMaterial(
        material, t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T_current),
        get<DERIVATIVE>(T_current), params...);
    return smith::tuple{C_v * get<VALUE>(T_dot) - s0, -q0};
  });
}

}  // namespace smith
