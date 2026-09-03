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
class StateDependentSolver {
 public:
  virtual ~StateDependentSolver() = default;

  /**
   * @brief Refresh solver-owned data for the current nonlinear state.
   * @param state Monolithic state vector at the current nonlinear iterate.
   * @param block_offsets Offsets describing the block layout of @a state.
   */
  virtual void updateForState(const mfem::Vector& state, const mfem::Array<int>& block_offsets) = 0;
};

}  // namespace smith
