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
 *   CouplingParams solid_coupling{FieldType<H1<temp_order>>("temperature"),
 *                               FieldType<H1<temp_order>>("temperature_old")};
 *   CouplingParams thermal_coupling{FieldType<H1<disp_order,dim>>("displacement_solve_state"),
 *                                 FieldType<H1<disp_order,dim>>("displacement")};
 *
 *   auto solid_res = buildSolidMechanicsSystem<dim, disp_order, DispRule>(
 *       field_store, disp_rule, solid_coupling, solid_solver, solid_opts, params...);
 *   auto solid = solid_res.system;
 *
 *   auto thermal_res = buildThermalSystem<dim, temp_order, TempRule>(
 *       field_store, temp_rule, thermal_coupling, thermal_solver, thermal_opts);
 *   auto thermal = thermal_res.system;
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
#include <tuple>
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
  void setMaterialOn(std::shared_ptr<SubSystemType> sub, const MaterialType& material, const std::string& domain_name)
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
 * The returned pair contains:
 *  - first: The combined main system (staggered).
 *  - second: The combined cycle-zero system (staggered, max_stagger_iters=1), or nullptr if none.
 *
 * @param subs  Two or more sub-systems that share a FieldStore.
 */
template <typename... SubSystems>
std::pair<std::shared_ptr<CombinedSystem>, std::shared_ptr<SystemBase>> combineSystems(
    std::shared_ptr<SubSystems>... subs)
{
  static_assert(sizeof...(subs) >= 2, "combineSystems requires at least two sub-systems");

  auto first_sub = std::get<0>(std::forward_as_tuple(subs...));
  auto field_store = first_sub->field_store;

  auto combined = std::make_shared<CombinedSystem>();
  combined->field_store = field_store;

  std::vector<std::shared_ptr<SystemBase>> cycle_zero_subs;

  (
      [&](auto& sub) {
        combined->subsystems.push_back(sub);
        for (auto& wf : sub->weak_forms) {
          combined->weak_forms.push_back(wf);
        }
        if constexpr (requires { sub->cycle_zero_system; }) {
          if (sub->cycle_zero_system) {
            cycle_zero_subs.push_back(sub->cycle_zero_system);
          }
        }
      }(subs),
      ...);

  std::shared_ptr<SystemBase> cycle_zero_combined = nullptr;
  if (!cycle_zero_subs.empty()) {
    auto cz = std::make_shared<CombinedSystem>();
    cz->field_store = field_store;
    cz->max_stagger_iters = 1;  // Cycle-zero solves are one-shot
    for (auto& sub : cycle_zero_subs) {
      cz->subsystems.push_back(sub);
      for (auto& wf : sub->weak_forms) {
        cz->weak_forms.push_back(wf);
      }
    }
    cycle_zero_combined = cz;
  }

  return {combined, cycle_zero_combined};
}

/**
 * @brief A generic wrapper that combines multiple sub-systems into a single monolithic block system.
 *
 * Unlike CombinedSystem (which performs staggered solver iterations), MonolithicCombinedSystem
 * concatenates all weak forms and solves them simultaneously using a single global SystemSolver.
 */
struct MonolithicCombinedSystem : public SystemBase {
  using SystemBase::SystemBase;
};

/**
 * @brief Combine two or more independently-built sub-systems into a MonolithicCombinedSystem.
 *
 * Preconditions:
 *  - All sub-systems share the same FieldStore.
 *  - Sub-system weak_forms are already populated.
 *
 * The returned pair contains:
 *  - first: The combined main system (monolithic).
 *  - second: The combined cycle-zero system (monolithic), or nullptr if none.
 *
 * @param solver  The monolithic SystemSolver that will solve the combined block system,
 *                including the aggregated cycle-zero system if any sub-systems have one.
 * @param subs    Two or more sub-systems that share a FieldStore.
 */
template <typename... SubSystems>
std::pair<std::shared_ptr<MonolithicCombinedSystem>, std::shared_ptr<SystemBase>> combineSystems(
    std::shared_ptr<SystemSolver> solver, std::shared_ptr<SubSystems>... subs)
{
  static_assert(sizeof...(subs) >= 2, "combineSystems requires at least two sub-systems");

  auto field_store = std::get<0>(std::forward_as_tuple(subs...))->field_store;

  std::vector<std::shared_ptr<WeakForm>> wfs;
  std::vector<std::shared_ptr<WeakForm>> cycle_zero_wfs;

  (
      [&](auto& sub) {
        for (auto& wf : sub->weak_forms) {
          wfs.push_back(wf);
        }
        if constexpr (requires { sub->cycle_zero_system; }) {
          if (sub->cycle_zero_system) {
            for (auto& cz_wf : sub->cycle_zero_system->weak_forms) {
              cycle_zero_wfs.push_back(cz_wf);
            }
          }
        }
      }(subs),
      ...);

  auto combined = std::make_shared<MonolithicCombinedSystem>(field_store, solver, wfs);
  std::shared_ptr<SystemBase> cycle_zero_combined = nullptr;
  if (!cycle_zero_wfs.empty()) {
    cycle_zero_combined = std::make_shared<SystemBase>(field_store, solver, cycle_zero_wfs);
  }

  return {combined, cycle_zero_combined};
}

/**
 * @brief Concatenate multiple vectors of systems into a single vector.
 *
 * Primarily used to gather end-step systems (like stress output) from multiple physics systems.
 */
template <typename... Vectors>
std::vector<std::shared_ptr<SystemBase>> mergeSystems(Vectors&&... vectors)
{
  std::vector<std::shared_ptr<SystemBase>> result;
  (result.insert(result.end(), vectors.begin(), vectors.end()), ...);
  return result;
}

}  // namespace smith
