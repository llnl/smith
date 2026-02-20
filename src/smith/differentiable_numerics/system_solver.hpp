// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>
#include <memory>
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/physics/common.hpp"

namespace smith {

class WeakForm;
class DifferentiableBlockSolver;
class BoundaryConditionManager;

class SystemSolver {
public:
  struct Stage {
    std::vector<size_t> block_indices;  // which blocks (residuals) to solve in this stage
    std::shared_ptr<DifferentiableBlockSolver> solver;
  };

  /// @brief Construct a SystemSolver for staggered iteration.
  /// @param max_staggered_iterations Maximum number of staggered sweeps across all stages.  When
  ///        @p exact_staggered_steps is false, the solver may exit early once all stage solvers
  ///        report convergence.
  /// @param exact_staggered_steps If true, always perform exactly @p max_staggered_iterations
  ///        sweeps with no early-exit convergence check.  Useful when a fixed number of
  ///        partitioned-stagger steps is required regardless of residual level.
  SystemSolver(int max_staggered_iterations, bool exact_staggered_steps = false);

  void addStage(const Stage& stage);
  void addStage(const std::vector<size_t>& block_indices, std::shared_ptr<DifferentiableBlockSolver> solver);

  std::vector<FieldState> solve(
      const std::vector<WeakForm*>& residual_evals,
      const std::vector<std::vector<size_t>>& block_indices,
      const FieldState& shape_disp,
      const std::vector<std::vector<FieldState>>& states,
      const std::vector<std::vector<FieldState>>& params,
      const TimeInfo& time_info,
      const std::vector<const BoundaryConditionManager*>& bc_managers) const;

private:
  int max_staggered_iterations_;
  bool exact_staggered_steps_;
  std::vector<Stage> stages_;
};

} // namespace smith
