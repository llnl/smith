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
 *   auto solid_fields = registerSolidMechanicsFields<dim, disp_order, DispRule>(field_store);
 *   auto thermal_fields = registerThermalFields<dim, temp_order, TempRule>(field_store);
 *
 *   auto solid = buildSolidMechanicsSystem<dim, disp_order, DispRule>(
 *       solid_solver, solid_opts, solid_fields, params..., thermal_fields);
 *
 *   auto thermal = buildThermalSystem<dim, temp_order, TempRule>(
 *       thermal_solver, thermal_opts, thermal_fields, solid_fields);
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
  std::vector<std::shared_ptr<SystemBase>> subsystems;  ///< Ordered sub-systems solved during each staggered sweep.

  /// @brief Construct a CombinedSystem.  weak_forms is populated by combineSystems.
  using SystemBase::SystemBase;

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
 * @param subs  Two or more sub-systems that share a FieldStore.
 */
template <typename... SubSystems>
auto combineSystems(std::shared_ptr<SubSystems>... subs)
{
  static_assert(sizeof...(subs) >= 2, "combineSystems requires at least two sub-systems");

  auto first_sub = std::get<0>(std::forward_as_tuple(subs...));
  auto field_store = first_sub->field_store;

  auto combined = std::make_shared<CombinedSystem>();
  combined->field_store = field_store;

  int max_stagger_iters = 1;
  bool exact_staggered_steps = false;

  std::vector<std::shared_ptr<SystemBase>> cycle_zero_subs;
  std::vector<std::shared_ptr<SystemBase>> post_solve_systems;
  std::vector<std::vector<size_t>> subsystem_global_block_indices;

  (
      [&](auto& sub) {
        std::vector<size_t> global_block_indices;
        combined->subsystems.push_back(sub);
        for (auto& wf : sub->weak_forms) {
          global_block_indices.push_back(combined->weak_forms.size());
          combined->weak_forms.push_back(wf);
        }
        subsystem_global_block_indices.push_back(global_block_indices);
        if (sub->solver) {
          max_stagger_iters = std::max(max_stagger_iters, sub->solver->maxStaggeredIterations());
          exact_staggered_steps = exact_staggered_steps || sub->solver->exactStaggeredSteps();
        }
        if constexpr (requires { sub->cycle_zero_system; }) {
          if (sub->cycle_zero_system) {
            cycle_zero_subs.push_back(sub->cycle_zero_system);
          }
        }
        if constexpr (requires { sub->post_solve_systems; }) {
          post_solve_systems.insert(post_solve_systems.end(), sub->post_solve_systems.begin(),
                                    sub->post_solve_systems.end());
        }
      }(subs),
      ...);

  combined->solver = std::make_shared<SystemSolver>(max_stagger_iters, exact_staggered_steps);
  for (size_t i = 0; i < combined->subsystems.size(); ++i) {
    const auto& sub = combined->subsystems[i];
    SLIC_ERROR_IF(!sub->solver, "Combined subsystem must have a solver");
    combined->solver->appendStagesWithBlockMapping(*sub->solver, subsystem_global_block_indices[i]);
  }

  std::shared_ptr<SystemBase> cycle_zero_combined = nullptr;
  if (!cycle_zero_subs.empty()) {
    auto cycle_zero = std::make_shared<CombinedSystem>();
    cycle_zero->field_store = field_store;
    cycle_zero->solver = std::make_shared<SystemSolver>(1, true);
    for (auto& sub : cycle_zero_subs) {
      std::vector<size_t> global_block_indices;
      cycle_zero->subsystems.push_back(sub);
      for (auto& wf : sub->weak_forms) {
        global_block_indices.push_back(cycle_zero->weak_forms.size());
        cycle_zero->weak_forms.push_back(wf);
      }
      SLIC_ERROR_IF(!sub->solver, "Combined cycle-zero subsystem must have a solver");
      cycle_zero->solver->appendStagesWithBlockMapping(*sub->solver, global_block_indices);
    }
    cycle_zero_combined = cycle_zero;
  }

  combined->cycle_zero_system = cycle_zero_combined;
  combined->post_solve_systems = post_solve_systems;

  return combined;
}

/**
 * @brief A generic wrapper that combines multiple sub-systems into a single monolithic block system.
 *
 * Unlike CombinedSystem (which performs staggered solver iterations), MonolithicCombinedSystem
 * concatenates all weak forms and solves them simultaneously using a single global SystemSolver.
 */
struct MonolithicCombinedSystem : public SystemBase {
  /// @brief Inherit `SystemBase` constructors for monolithic wrappers.
  using SystemBase::SystemBase;
};

/**
 * @brief Combine two or more independently-built sub-systems into a MonolithicCombinedSystem.
 *
 * Preconditions:
 *  - All sub-systems share the same FieldStore.
 *  - Sub-system weak_forms are already populated.
 *
 * @param solver  The monolithic SystemSolver that will solve the combined block system,
 *                including the aggregated cycle-zero system if any sub-systems have one.
 * @param subs    Two or more sub-systems that share a FieldStore.
 */
template <typename... SubSystems>
auto combineSystems(std::shared_ptr<SystemSolver> solver, std::shared_ptr<SubSystems>... subs)
{
  static_assert(sizeof...(subs) >= 2, "combineSystems requires at least two sub-systems");

  auto field_store = std::get<0>(std::forward_as_tuple(subs...))->field_store;

  std::vector<std::shared_ptr<WeakForm>> wfs;
  std::vector<std::shared_ptr<WeakForm>> cycle_zero_wfs;
  std::vector<std::shared_ptr<SystemBase>> post_solve_systems;

  (
      [&](auto& sub) {
        for (auto& wf : sub->weak_forms) {
          wfs.push_back(wf);
        }
        if constexpr (requires { sub->cycle_zero_system; }) {
          if (sub->cycle_zero_system) {
            for (auto& cycle_zero_wf : sub->cycle_zero_system->weak_forms) {
              cycle_zero_wfs.push_back(cycle_zero_wf);
            }
          }
        }
        if constexpr (requires { sub->post_solve_systems; }) {
          post_solve_systems.insert(post_solve_systems.end(), sub->post_solve_systems.begin(),
                                    sub->post_solve_systems.end());
        }
      }(subs),
      ...);

  auto combined = std::make_shared<MonolithicCombinedSystem>(field_store, solver, wfs);
  std::shared_ptr<SystemBase> cycle_zero_combined = nullptr;
  if (!cycle_zero_wfs.empty()) {
    cycle_zero_combined = std::make_shared<SystemBase>(field_store, solver, cycle_zero_wfs);
  }

  combined->cycle_zero_system = cycle_zero_combined;
  combined->post_solve_systems = post_solve_systems;

  return combined;
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
