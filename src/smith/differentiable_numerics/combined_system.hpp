// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file combined_system.hpp
 * @brief CombinedSystem and combineSystems for composing independent physics into a coupled system.
 *
 * Individual physics sub-systems (SolidMechanicsSystem, ThermalSystem, ...) remain authoritative
 * for their own configuration APIs.  combineSystems wires them together via a shared FieldStore
 * and provides a coupled setMaterial that registers integrands on both sub-system weak forms.
 *
 * Usage:
 * @code
 *   auto field_store = std::make_shared<FieldStore>(mesh, 100, "coupled");
 *
 *   auto solid_info = registerSolidMechanicsFields<dim, disp_order>(*field_store, disp_rule, params...);
 *   auto thermal_info = registerThermalFields<dim, temp_order>(*field_store, temp_rule);
 *
 *   CouplingSpec solid_coupling{FieldType<H1<temp_order>>("temperature"),
 *                               FieldType<H1<temp_order>>("temperature_old")};
 *   CouplingSpec thermal_coupling{FieldType<H1<disp_order,dim>>("displacement_solve_state"),
 *                                 FieldType<H1<disp_order,dim>>("displacement")};
 *
 *   auto solid   = buildSolidMechanicsSystemFromStore<dim, disp_order, DispRule>(
 *       solid_info, solid_solver, solid_opts, solid_coupling, params...);
 *   auto thermal = buildThermalSystemFromStore<dim, temp_order, TempRule>(
 *       thermal_info, thermal_solver, thermal_opts, thermal_coupling);
 *
 *   auto coupled = combineSystems(solid, thermal);
 *   coupled->setMaterial(thermo_mech_material, domain);   // tight coupling
 *   solid->addTraction(right, traction_fn);               // loose via sub-system
 *   thermal->addHeatFlux(top, flux_fn);
 *
 *   auto advancer = makeAdvancer(coupled);
 * @endcode
 */

#pragma once

#include <vector>
#include <memory>
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/field_store.hpp"

namespace smith {

/**
 * @brief A non-templated system wrapper that combines multiple sub-systems sharing one FieldStore.
 *
 * The combined solve does staggered iterations: in each sweep every sub-system is solved in
 * order, writing its updated unknowns back to the shared FieldStore so subsequent sub-systems see
 * the updated fields.  The base class weak_forms member is the concatenation of sub-system weak
 * forms, which allows makeAdvancer / MultiphysicsTimeIntegrator to integrate the combined system
 * transparently.
 *
 * For tight coupling (setMaterial on the combined system), sub-classes or concrete callers can
 * down-cast subsystems[i] to the typed sub-system and call its own addBodyIntegral.  This is done
 * via the template setMaterial below.
 */
struct CombinedSystem : public SystemBase {
  std::vector<std::shared_ptr<SystemBase>> subsystems;
  std::vector<std::shared_ptr<SystemBase>> cycle_zero_systems;  ///< Cycle-zero systems from sub-systems, in order.
  int max_stagger_iters = 10;
  double stagger_tolerance = 1e-8;

  /// @brief Construct a CombinedSystem.  weak_forms is populated by combineSystems.
  using SystemBase::SystemBase;

  /**
   * @brief Staggered solve: iterate over sub-systems, writing each result to the shared
   * FieldStore before the next sub-system reads it.
   *
   * Returns one FieldState per combined weak_form (same contract as SystemBase::solve).
   */
  std::vector<FieldState> solve(const TimeInfo& time_info) const override;

  /**
   * @brief Set a coupled material on all sub-systems.
   *
   * Calls setMaterial on each sub-system if that sub-system is of the given typed pointer
   * (down-cast via dynamic_pointer_cast).  Intended for tight-coupling materials that register
   * integrands on both sub-systems' weak forms.
   *
   * @tparam SubSystemType The concrete type that exposes setMaterial.
   * @tparam MaterialType  The material type.
   */
  template <typename SubSystemType, typename MaterialType>
  void setMaterialOn(std::shared_ptr<SubSystemType> sub, const MaterialType& material,
                     const std::string& domain_name)
  {
    sub->setMaterial(material, domain_name);
  }
};

/**
 * @brief Combine two or more independently-built sub-systems into a CombinedSystem.
 *
 * Preconditions:
 *  - All sub-systems share the same FieldStore (built via registerXxxFields + buildXxxFromStore).
 *  - Sub-system weak_forms are already populated (registerXxx was called before buildXxx).
 *
 * The returned CombinedSystem:
 *  - Holds shared_ptrs to each sub-system (accessible as combined->subsystems[i]).
 *  - Its weak_forms is the concatenation of sub-system weak forms in argument order.
 *  - Its field_store is the shared FieldStore from the first sub-system.
 *  - Its solver member is null (CombinedSystem::solve() drives sub-system solvers directly).
 *  - Its cycle_zero_systems is populated with each sub-system's cycle_zero_system (if non-null).
 *
 * @param subs  Two or more sub-systems that share a FieldStore.
 */
template <typename... SubSystems>
std::shared_ptr<CombinedSystem> combineSystems(std::shared_ptr<SubSystems>... subs)
{
  static_assert(sizeof...(subs) >= 2, "combineSystems requires at least two sub-systems");

  auto combined = std::make_shared<CombinedSystem>();

  // All sub-systems must share the same FieldStore — use the first one.
  combined->field_store = std::get<0>(std::forward_as_tuple(subs...))->field_store;

  // Concatenate weak_forms, collect subsystems, and gather any cycle-zero systems.
  (
      [&](auto& sub) {
        combined->subsystems.push_back(sub);
        for (auto& wf : sub->weak_forms) {
          combined->weak_forms.push_back(wf);
        }
        if constexpr (requires { sub->cycle_zero_system; }) {
          if (sub->cycle_zero_system) {
            combined->cycle_zero_systems.push_back(sub->cycle_zero_system);
          }
        }
      }(subs),
      ...);

  return combined;
}

}  // namespace smith
