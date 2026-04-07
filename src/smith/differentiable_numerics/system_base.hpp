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
#include "coupled_system_solver.hpp"
#include "state_advancer.hpp"
#include "smith/physics/common.hpp"
#include "mfem.hpp"

namespace smith {

template <typename... Args>
struct Parameters;

/**
 * @brief Information about a dual field.
 */
struct ReactionInfo {
  std::string name;                                    ///< The name of the dual field.
  const mfem::ParFiniteElementSpace* space = nullptr;  ///< The finite element space of the dual field.
};

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
  std::shared_ptr<FieldStore> field_store;      ///< Field store managing the system's fields.
  std::shared_ptr<CoupledSystemSolver> solver;  ///< The solver for the system.
  std::shared_ptr<StateAdvancer> advancer;      ///< The state advancer.
  std::vector<FieldState> parameter_fields;     ///< Optional parameter fields.
  std::string prepend_name;                     ///< Optional prepended name for all fields.

  /**
   * @brief Get the list of all parameter fields.
   * @return const std::vector<FieldState>& List of parameter fields.
   */
  const std::vector<FieldState>& getParameterFields() const { return parameter_fields; }

  /**
   * @brief Helper function to prepend the physics name to a string.
   * @param name The name to prepend to.
   * @return std::string The prepended name.
   */
  std::string prefix(const std::string& name) const
  {
    if (prepend_name.empty()) {
      return name;
    }
    return prepend_name + "_" + name;
  }

  /// @brief Metadata for dual outputs exported by this system.
  std::vector<ReactionInfo> getReactionInfos() const { return {}; }
};

}  // namespace smith
