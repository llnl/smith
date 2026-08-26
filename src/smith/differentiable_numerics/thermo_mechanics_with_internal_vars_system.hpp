// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"
#include "smith/differentiable_numerics/state_variable_system.hpp"
#include "smith/differentiable_numerics/solid_mechanics_with_internal_vars_system.hpp"
#include "smith/differentiable_numerics/thermo_mechanics_system.hpp"

namespace smith {

/**
 * @brief Wire a coupled thermo-mechanical internal-variable material across solid, thermal, and internal-variable
 *        systems by composing the existing two-system helpers.
 *
 * Preconditions on the systems passed in:
 *  - @p solid_system was built with @c (thermal_fields, internal_variable_fields) coupling, in that order. The
 *    leading 4 tail arguments are then @c (T_solve_state, T_old, alpha_solve_state, alpha_old). The thermo-mech
 *    material must accept @c (t_info, state, grad_u, grad_v, theta, grad_theta, alpha, ...) — alpha is forwarded
 *    from the trailing variadic params.
 *  - @p thermal_system was built with @c (solid_fields, internal_variable_fields) coupling. The thermo-mech material
 *    must accept @c (t_info, state, grad_u, grad_v, theta, grad_theta, alpha, ...).
 *  - @p internal_variable_system was built with @c solid_fields coupling. The evolution callable must accept
 *    @c (t_info, alpha, alpha_dot, grad_u, ...).
 *  - The internal-variable time rule's @c value() must equal its current state (true for backward Euler), so the
 *    raw @c alpha solve-state may be forwarded into the thermo-mech material in place of an interpolated alpha.
 *
 * Routing through @c solid->setMaterial and @c thermal->setMaterialAndHeatSource means cycle-zero and
 * stress-output paths automatically receive the coupled stress contribution.
 *
 * @tparam SolidSystem Type of the solid mechanics system.
 * @tparam ThermalSystem Type of the thermal system.
 * @tparam InternalVariableSystem Type of the internal variable system.
 * @tparam ThermoMechMaterial Type of the thermo-mechanical material integrand.
 * @tparam InternalVarEvolution Type of the internal variable evolution integrand.
 *
 * @param solid_system The solid mechanics system.
 * @param thermal_system The thermal system.
 * @param internal_variable_system The internal variable system.
 * @param thermo_mech_material The material model for stress, heat capacity, heat source, and heat flux.
 * @param internal_var_evolution The ODE residual for the internal variable.
 * @param domain_name The domain name to apply the material to.
 */
template <typename SolidSystem, typename ThermalSystem, typename InternalVariableSystem, typename ThermoMechMaterial,
          typename InternalVarEvolution>
void setCoupledThermoMechanicsInternalVariableMaterial(std::shared_ptr<SolidSystem> solid_system,
                                                       std::shared_ptr<ThermalSystem> thermal_system,
                                                       std::shared_ptr<InternalVariableSystem> internal_variable_system,
                                                       ThermoMechMaterial thermo_mech_material,
                                                       InternalVarEvolution internal_var_evolution,
                                                       const std::string& domain_name)
{
  setCoupledThermoMechanicsMaterial(solid_system, thermal_system, thermo_mech_material, domain_name);
  setCoupledInternalVariableMaterial(internal_variable_system, solid_system, internal_var_evolution, domain_name);
}

}  // namespace smith
