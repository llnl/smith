// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermo_mechanics_system.hpp
 * @brief Helper to wire a coupled thermo-mechanical material to a SolidMechanicsSystem and ThermalSystem.
 */

#pragma once

#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"

namespace smith {

namespace detail {

/**
 * @brief Evaluate coupled thermo-mechanical material using TimeInfo-aware signature.
 */
template <typename MaterialType, typename StateType, typename GradUType, typename GradVType, typename ThetaType,
          typename GradThetaType, typename... ParamTypes>
auto evaluateCoupledThermoMechanicsMaterial(const MaterialType& material, const TimeInfo& t_info, StateType& state,
                                            const GradUType& grad_u, const GradVType& grad_v, ThetaType theta,
                                            const GradThetaType& grad_theta, ParamTypes&&... params)
{
  return material(t_info, state, grad_u, grad_v, theta, grad_theta, std::forward<ParamTypes>(params)...);
}

template <typename MaterialType, typename TemperatureRulePtr>
/// @brief Adapts coupled thermo-mechanical material to solid-system material interface.
struct CoupledSolidThermoMechanicsMaterialAdapter {
  /// Material state type forwarded to solid system.
  using State = typename MaterialType::State;

  MaterialType material;                     ///< Wrapped thermo-mechanical material.
  TemperatureRulePtr temperature_time_rule;  ///< Time rule used to recover current temperature value.
  double density;                            ///< Material density exposed for solid residual.

  template <typename StateType, typename GradUType, typename GradVType, typename TemperatureType,
            typename TemperatureOldType, typename... ParamTypes>
  /// @brief Evaluate wrapped material and return solid PK1 contribution.
  auto operator()(const TimeInfo& t_info, StateType& state, GradUType grad_u, GradVType grad_v,
                  TemperatureType temperature, TemperatureOldType temperature_old, ParamTypes&&... params) const
  {
    auto T = temperature_time_rule->value(t_info, temperature, temperature_old);
    auto [pk, C_v, s0, q0] =
        evaluateCoupledThermoMechanicsMaterial(material, t_info, state, grad_u, grad_v, get<VALUE>(T),
                                               get<DERIVATIVE>(T), std::forward<ParamTypes>(params)...);
    return pk;
  }
};

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
 * The material callable must satisfy:
 *   material(t_info, state, grad_u, grad_v, T_value, grad_T, params...) -> tuple{PK1, C_v, s0, q0}
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

  auto solid_material = detail::CoupledSolidThermoMechanicsMaterialAdapter<MaterialType, decltype(captured_temp_rule)>{
      material, captured_temp_rule, material.density};

  solid->setMaterial(solid_material, domain_name);

  thermal->setMaterialAndHeatSource(
      [=](const TimeInfo& t_info, auto temperature, auto grad_temperature, auto disp, auto disp_old, auto v_old,
          auto a_old, auto... params) {
        auto [u, v, a] = captured_disp_rule->interpolate(t_info, disp, disp_old, v_old, a_old);
        typename MaterialType::State state{};
        auto [pk, C_v, s0, q0] = detail::evaluateCoupledThermoMechanicsMaterial(
            material, t_info, state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), temperature, grad_temperature, params...);
        // Material's s0 sits on the LHS (residual = C_v*T_dot + s0 - q·∇v); negate to fit the
        // physical-heat-source convention used by setMaterialAndHeatSource.
        return smith::tuple{C_v, q0, -s0};
      },
      domain_name);
}

}  // namespace smith
