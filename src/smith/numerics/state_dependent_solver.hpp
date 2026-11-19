// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state_dependent_solver.hpp
 *
 * @brief Interface for solvers that refresh data from a nonlinear state.
 */

#pragma once

#include "mfem.hpp"

namespace smith {

/**
 * @brief Interface for solvers that can refresh state-dependent internals.
 */
class StateDependentSolver : public mfem::Solver {
 public:
  virtual ~StateDependentSolver() = default;

  /**
   * @brief Refresh solver-owned data for the current nonlinear state.
   * @param state Current nonlinear state, including its block layout.
   */
  virtual void updateForState(const mfem::BlockVector& state) = 0;
};

}  // namespace smith
