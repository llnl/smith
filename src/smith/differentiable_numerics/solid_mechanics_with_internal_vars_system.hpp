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

template <typename MaterialType, typename AlphaType, typename AlphaDotType, typename DerivType, typename... ParamTypes>
/// @brief Dispatch internal-variable material calls with or without explicit `TimeInfo`.
auto evaluateCoupledInternalVariableMaterial(const MaterialType& material, const TimeInfo& t_info, AlphaType alpha,
                                             AlphaDotType alpha_dot, DerivType deriv_u, ParamTypes&&... params)
{
  if constexpr (requires { material(t_info, alpha, alpha_dot, deriv_u, std::forward<ParamTypes>(params)...); }) {
    return material(t_info, alpha, alpha_dot, deriv_u, std::forward<ParamTypes>(params)...);
  } else {
    return material(alpha, alpha_dot, deriv_u, std::forward<ParamTypes>(params)...);
  }
}

template <typename MaterialType, typename StateType, typename GradUType, typename GradVType, typename AlphaType,
          typename... ParamTypes>
/// @brief Evaluate solid/internal-variable material using TimeInfo-aware signature.
auto evaluateSolidInternalVariableMaterial(const MaterialType& material, const TimeInfo& t_info, StateType& state,
                                           const GradUType& grad_u, const GradVType& grad_v, AlphaType alpha,
                                           ParamTypes&&... params)
{
  return material(t_info, state, grad_u, grad_v, alpha, std::forward<ParamTypes>(params)...);
}

template <typename MaterialType>
/// @brief Adapts coupled internal-variable solids to solid-system material interface.
struct CoupledSolidInternalVariableMaterialAdapter {
  /// Material state type forwarded to solid system.
  using State = typename MaterialType::State;

  MaterialType material;  ///< Wrapped constitutive model.
  double density;         ///< Material density exposed for solid residual.

  template <typename StateType, typename GradUType, typename GradVType, typename AlphaType, typename AlphaDotType,
            typename... ParamTypes>
  /// @brief Evaluate wrapped material with current internal-variable value.
  auto operator()(const TimeInfo& t_info, StateType& state, GradUType grad_u, GradVType grad_v, AlphaType alpha,
                  AlphaDotType /*alpha_dot*/, ParamTypes&&... params) const
  {
    return evaluateSolidInternalVariableMaterial(material, t_info, state, grad_u, grad_v, get<VALUE>(alpha),
                                                 std::forward<ParamTypes>(params)...);
  }
};

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
 *   material(t_info, state, grad_u, grad_v, alpha_value, params...) -> PK1
 */
template <int dim, int disp_order, typename DispRule, typename SolidCoupling, typename StateSpace, typename StateRule,
          typename StateCoupling, typename MaterialType>
void setCoupledSolidMechanicsInternalVariableMaterial(
    std::shared_ptr<SolidMechanicsSystem<dim, disp_order, DispRule, SolidCoupling>> solid,
    std::shared_ptr<InternalVariableSystem<dim, StateSpace, StateRule, StateCoupling>> /*internal_variables*/,
    const MaterialType& material, const std::string& domain_name)
{
  auto solid_material = detail::CoupledSolidInternalVariableMaterialAdapter<MaterialType>{material, material.density};

  solid->setMaterial(solid_material, domain_name);
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
    std::shared_ptr<SolidMechanicsSystem<dim, disp_order, DispRule, SolidCoupling>> /*solid*/,
    const MaterialType& material, const std::string& domain_name)
{
  internal_variables->addEvolution(
      domain_name, [=](auto t_info, auto alpha, auto alpha_dot, auto u, auto /*v*/, auto /*a*/, auto... params) {
        return detail::evaluateCoupledInternalVariableMaterial(material, t_info, alpha, alpha_dot, get<DERIVATIVE>(u),
                                                               params...);
      });
}

template <int dim, int disp_order, typename DispRule, typename SolidCoupling, typename StateSpace, typename StateRule,
          typename StateCoupling, typename MaterialType>
/**
 * @brief Backward-compatible alias for `setCoupledSolidMechanicsInternalVariableMaterial`.
 * @param solid Solid system receiving internal-variable coupling.
 * @param state Backward-compatible internal-variable system alias.
 * @param material Material model.
 * @param domain_name Domain on which to apply the material.
 */
void setCoupledSolidMechanicsInternalVarsMaterial(
    std::shared_ptr<SolidMechanicsSystem<dim, disp_order, DispRule, SolidCoupling>> solid,
    std::shared_ptr<InternalVariableSystem<dim, StateSpace, StateRule, StateCoupling>> state,
    const MaterialType& material, const std::string& domain_name)
{
  setCoupledSolidMechanicsInternalVariableMaterial(solid, state, material, domain_name);
}

}  // namespace smith
