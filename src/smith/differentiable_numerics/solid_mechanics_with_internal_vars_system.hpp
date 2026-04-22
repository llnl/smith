// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_with_internal_vars_system.hpp
 * @brief Helpers to wire internal-variable materials to composed solid and internal-variable systems.
 */

#pragma once

#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/state_variable_system.hpp"

namespace smith {

namespace detail {

template <typename MaterialType, typename TimeInfoType, typename AlphaType, typename AlphaDotType, typename DerivType,
          typename... ParamTypes>
auto evaluateCoupledInternalVariableMaterial(const MaterialType& material, const TimeInfoType& t_info,
                                             AlphaType alpha, AlphaDotType alpha_dot, DerivType deriv_u,
                                             ParamTypes&&... params)
{
  if constexpr (requires { material(t_info, alpha, alpha_dot, deriv_u, std::forward<ParamTypes>(params)...); }) {
    return material(t_info, alpha, alpha_dot, deriv_u, std::forward<ParamTypes>(params)...);
  } else {
    return material(alpha, alpha_dot, deriv_u, std::forward<ParamTypes>(params)...);
  }
}

}  // namespace detail

/**
 * @brief Register a solid material integrand on a SolidMechanicsSystem that is coupled to
 *        an InternalVariableSystem carrying internal variables.
 *
 * Assumes:
 *  - solid was built with state fields as leading coupling fields
 *    (first 2 coupling positions: state_solve_state, state).
 *  - state was built with solid displacement fields as leading coupling fields
 *    (first 4 coupling positions: displacement_solve_state, displacement, velocity, acceleration).
 *
 * Solid material callable must satisfy:
 *   material(state, grad_u, alpha_value, params...) -> PK1
 */
template <int dim, int disp_order, typename DispRule, typename SolidCoupling, typename StateSpace, typename StateRule,
          typename StateCoupling, typename MaterialType>
void setCoupledSolidMechanicsInternalVariableMaterial(
    std::shared_ptr<SolidMechanicsSystem<dim, disp_order, DispRule, SolidCoupling>> solid,
    std::shared_ptr<InternalVariableSystem<dim, StateSpace, StateRule, StateCoupling>> internal_variables,
    const MaterialType& material, const std::string& domain_name)
{
  auto captured_disp_rule = solid->disp_time_rule;
  auto captured_internal_variable_rule = internal_variables->internal_variable_time_rule;

  solid->solid_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old,
                                                           auto a_old, auto alpha, auto alpha_old, auto... params) {
    auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
    auto alpha_current = captured_internal_variable_rule->value(t_info, alpha, alpha_old);

    typename MaterialType::State material_state;
    auto pk_stress = material(material_state, get<DERIVATIVE>(u_current), get<VALUE>(alpha_current), params...);

    return smith::tuple{get<VALUE>(a_current) * material.density, pk_stress};
  });

  if (solid->cycle_zero_solid_weak_form) {
    solid->cycle_zero_solid_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto u, auto /*v*/, auto a, auto alpha, auto... params) {
          typename MaterialType::State material_state;
          auto pk_stress = material(material_state, get<DERIVATIVE>(u), get<VALUE>(alpha), params...);
          return detail::makeScaledCycleZeroResidual(t_info, get<VALUE>(a) * material.density, pk_stress);
        });
  }
}

/**
 * @brief Register an internal-variable evolution law using solid displacement coupling.
 *
 * Evolution callable may satisfy either:
 *   material(t_info, alpha, alpha_dot, grad_u, params...) -> residual
 * or
 *   material(alpha, alpha_dot, grad_u, params...) -> residual
 */
template <int dim, typename StateSpace, typename StateRule, typename StateCoupling, int disp_order, typename DispRule,
          typename SolidCoupling, typename MaterialType>
void setCoupledInternalVariableMaterial(
    std::shared_ptr<InternalVariableSystem<dim, StateSpace, StateRule, StateCoupling>> internal_variables,
    std::shared_ptr<SolidMechanicsSystem<dim, disp_order, DispRule, SolidCoupling>> solid, const MaterialType& material,
    const std::string& domain_name)
{
  auto captured_disp_rule = solid->disp_time_rule;
  internal_variables->setMaterial(
      [=](auto t_info, auto alpha, auto alpha_dot, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
        auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
        return detail::evaluateCoupledInternalVariableMaterial(material, t_info, alpha, alpha_dot,
                                                               get<DERIVATIVE>(u_current), params...);
      },
      domain_name);
}

template <int dim, int disp_order, typename DispRule, typename SolidCoupling, typename StateSpace, typename StateRule,
          typename StateCoupling, typename MaterialType>
void setCoupledSolidMechanicsInternalVarsMaterial(
    std::shared_ptr<SolidMechanicsSystem<dim, disp_order, DispRule, SolidCoupling>> solid,
    std::shared_ptr<StateVariableSystem<dim, StateSpace, StateRule, StateCoupling>> state, const MaterialType& material,
    const std::string& domain_name)
{
  setCoupledSolidMechanicsInternalVariableMaterial(solid, state, material, domain_name);
}

}  // namespace smith
