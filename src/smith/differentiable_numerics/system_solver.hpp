// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>
#include <memory>
#include <mpi.h>
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/common.hpp"

namespace smith {

class WeakForm;
class NonlinearBlockSolverBase;
class BoundaryConditionManager;

/// @brief Orchestrates staggered solution for multiphysics systems.
class SystemSolver {
 public:
  /// @brief Represents a single stage in a staggered iteration.
  struct Stage {
    std::vector<size_t> block_indices;                 ///< Which blocks (residuals) to solve in this stage.
    std::shared_ptr<NonlinearBlockSolverBase> solver;  ///< Solver to use for this stage.
    double relaxation_factor = 1.0;                    ///< Per-stage relaxation factor. Values in (0, 1) under-relax
                                                       ///< the update: x_new = omega * x_solved + (1 - omega) * x_old.
                                                       ///< A value of 1.0 (default) means no relaxation (full update).
  };

  /// @brief Construct a monolithic SystemSolver from a single block solver.
  /// @param single_solver The solver to use for all blocks simultaneously.
  SystemSolver(std::shared_ptr<NonlinearBlockSolverBase> single_solver);

  /// @brief Construct a SystemSolver for staggered iteration.
  /// @param max_staggered_iterations Maximum number of staggered sweeps across all stages.  When
  ///        @p exact_staggered_steps is false, the solver may exit early once all stage solvers
  ///        report convergence.
  /// @param exact_staggered_steps If true, always perform exactly @p max_staggered_iterations
  ///        sweeps with no early-exit convergence check.  Useful when a fixed number of
  ///        partitioned-stagger steps is required regardless of residual level.
  SystemSolver(int max_staggered_iterations, bool exact_staggered_steps = false);

  /// @brief Convenience method to add a solver stage.
  /// @param block_indices Indices of the blocks to solve.
  /// @param solver Nonlinear block solver for this stage.
  /// @param relaxation_factor Per-stage relaxation factor in `(0, 1]`.
  void addSubsystemSolver(const std::vector<size_t>& block_indices, std::shared_ptr<NonlinearBlockSolverBase> solver,
                          double relaxation_factor = 1.0);

  /// @brief Solves the multiphysics system using staggered iterations.
  /// @param residual_evals Vector of WeakForm evaluations for each block.
  /// @param block_indices Block indices for each residual evaluation.
  /// @param shape_disp Current shape displacement.
  /// @param states Nested vector of field states.
  /// @param params Nested vector of parameters.
  /// @param time_info Current time information.
  /// @param bc_managers Managers for boundary conditions.
  /// @return Updated field states.
  std::vector<FieldState> solve(const std::vector<WeakForm*>& residual_evals,
                                const std::vector<std::vector<size_t>>& block_indices, const FieldState& shape_disp,
                                const std::vector<std::vector<FieldState>>& states,
                                const std::vector<std::vector<FieldState>>& params, const TimeInfo& time_info,
                                const std::vector<const BoundaryConditionManager*>& bc_managers) const;

  /// @brief Build a single-block solver from the stage responsible for @p block_index.
  /// Prefers constructing a fresh solver instance when the underlying stage solver retains rebuildable config.
  std::shared_ptr<SystemSolver> singleBlockSolver(size_t block_index) const;

 private:
  int max_staggered_iterations_;  ///< Maximum number of staggered iterations.
  bool exact_staggered_steps_;    ///< If true, no early-exit convergence check.
  std::vector<Stage> stages_;     ///< Solver stages for the staggered iterations.
};

}  // namespace smith
