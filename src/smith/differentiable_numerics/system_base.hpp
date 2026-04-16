// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file system_base.hpp
 * @brief Defines the SystemBase struct for common system functionality.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "field_state.hpp"
#include "field_store.hpp"
#include "system_solver.hpp"
#include "state_advancer.hpp"
#include "smith/physics/common.hpp"
#include "mfem.hpp"

namespace smith {

template <typename... Args>
struct Parameters;

namespace detail {

/// @brief Helper: given an index and a type, always produces the type (used to repeat a type N times via pack
/// expansion)
template <std::size_t, typename T>
using always_t = T;

/// @brief Implementation for TimeRuleParams: repeat Space N times, then append Tail...
template <typename Space, typename... Tail, std::size_t... Is>
auto time_rule_params_impl(std::index_sequence<Is...>) -> Parameters<always_t<Is, Space>..., Tail...>;

}  // namespace detail

/// @brief Generate a Parameters<...> type with Rule::num_states copies of Space followed by additional Tail types.
/// Used to build weak form parameter lists that adapt to the time integration rule's arity.
template <typename Rule, typename Space, typename... Tail>
using TimeRuleParams =
    decltype(detail::time_rule_params_impl<Space, Tail...>(std::make_index_sequence<Rule::num_states>{}));

/**
 * @brief Base struct for physics systems containing common members and helper functions.
 */
struct SystemBase {
  std::vector<std::shared_ptr<WeakForm>> weak_forms;
  std::shared_ptr<FieldStore> field_store;  ///< Field store managing the system's fields.
  std::shared_ptr<SystemSolver> solver;     ///< The solver for the system.

  SystemBase() = default;
  explicit SystemBase(std::shared_ptr<FieldStore> fs, std::shared_ptr<SystemSolver> sol = nullptr,
                      std::vector<std::shared_ptr<WeakForm>> wfs = {})
      : weak_forms(std::move(wfs)), field_store(std::move(fs)), solver(std::move(sol))
  {
  }
  virtual ~SystemBase() = default;

  /**
   * @brief Solve the system using the internal weak_forms and solver.
   * @param time_info Current time information.
   * @return std::vector<FieldState> The updated state fields from the solver.
   */
  virtual std::vector<FieldState> solve(const TimeInfo& time_info) const;

  /**
   * @brief Compute reactions after solving the main state.
   * @param time_info Current time information.
   * @param states_for_reactions The fields configured for reaction computation.
   * @return std::vector<ReactionState> Computed reactions across all weak_forms.
   */
  virtual std::vector<ReactionState> computeReactions(const TimeInfo& time_info,
                                                      const std::vector<FieldState>& states_for_reactions) const; // MRT, why is states_for_reactions needed
};

// MRT: rename to makeSystem
inline std::shared_ptr<SystemBase> makeSubSystem(std::shared_ptr<FieldStore> field_store,
                                                 std::shared_ptr<SystemSolver> solver,
                                                 std::vector<std::shared_ptr<WeakForm>> weak_forms)
{
  return std::make_shared<SystemBase>(std::move(field_store), std::move(solver), std::move(weak_forms));
}

}  // namespace smith
