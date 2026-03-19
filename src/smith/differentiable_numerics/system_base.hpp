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
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/coupled_system_solver.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"

namespace smith {

/**
 * @brief Base struct for physics systems containing common members and helper functions.
 */
struct SystemBase {
  std::shared_ptr<FieldStore> field_store;      ///< Field store managing the system's fields.
  std::shared_ptr<CoupledSystemSolver> solver;  ///< The solver for the system.
  std::shared_ptr<StateAdvancer> advancer;      ///< The state advancer.
  std::vector<FieldState> parameter_fields;     ///< Optional parameter fields.
  std::string prepend_name_;                    ///< Optional prepended name for all fields.

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
    if (prepend_name_.empty()) {
      return name;
    }
    return prepend_name_ + "_" + name;
  }
};

}  // namespace smith
