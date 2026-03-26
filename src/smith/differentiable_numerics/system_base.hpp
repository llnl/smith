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
#include "nonlinear_block_solver.hpp"
#include "state_advancer.hpp"
#include "smith/physics/common.hpp"
#include "mfem.hpp"

namespace smith {

struct DualInfo {
  std::string name;
  const mfem::ParFiniteElementSpace* space;
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
 * @brief Create a dedicated monolithic solver for cycle-zero acceleration solves.
 *
 * This solver is independent of the (possibly staggered/multi-block) main solver,
 * avoiding SuperLU factorization size mismatches between cycle-zero and main solves.
 */
inline std::shared_ptr<CoupledSystemSolver> buildCycleZeroSolver(const Mesh& mesh)
{
  LinearSolverOptions cz_lin_opts{.linear_solver = LinearSolver::SuperLU};
  NonlinearSolverOptions cz_nonlin_opts{
      .nonlin_solver = NonlinearSolver::Newton, .relative_tol = 1e-10, .absolute_tol = 1e-10, .max_iterations = 10};
  auto cz_block_solver = buildNonlinearBlockSolver(cz_nonlin_opts, cz_lin_opts, mesh);
  return std::make_shared<CoupledSystemSolver>(cz_block_solver);
}

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
  std::vector<DualInfo> getDualInfos() const { return {}; }
};

}  // namespace smith
