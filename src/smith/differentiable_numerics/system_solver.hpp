// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <mpi.h>
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/common.hpp"

namespace smith {

class WeakForm;
class NonlinearBlockSolverBase;
class BoundaryConditionManager;

/// @brief Options controlling adaptive cutback of prescribed BC updates.
///
/// When @c enabled is true, @ref SystemSolver::solve interpolates between the
/// previously applied BC tdof values and the new target across one or more
/// internal cutbacks before handing off to the full nonlinear solve.
struct BcRampOptions {
  bool enabled = false;                        ///< Master switch. Default off → behavior unchanged.
  double shrink_factor = 0.5;                  ///< Multiplier on alpha after a failed predictor step.
  double j_floor = 0.0;                        ///< Reserved threshold for optional Jacobian-based cutback checks.
  int max_cutbacks = 20;                       ///< Hard cap on cutback iterations per outer solve.
  int intermediate_max_iterations = 10;        ///< Nonlinear iteration cap for accepted intermediate ramp solves.
  double intermediate_relative_tol = 0.05;     ///< Relative tolerance for accepted intermediate ramp solves.
  double intermediate_absolute_tol_fac = 1e3;  ///< Absolute tolerance multiplier for intermediate ramp solves.
};

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

  /// @brief Append stages from another solver using a local-to-global block mapping.
  /// @param subsystem_solver Source solver whose stages operate on subsystem-local block indices.
  /// @param global_block_indices Mapping from subsystem-local block index to global block index.
  void appendStagesWithBlockMapping(const SystemSolver& subsystem_solver,
                                    const std::vector<size_t>& global_block_indices);

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

  /// @brief Maximum number of staggered sweeps allowed for this solver.
  int maxStaggeredIterations() const { return max_staggered_iterations_; }

  /// @brief Whether solver always performs exactly `maxStaggeredIterations()` sweeps.
  bool exactStaggeredSteps() const { return exact_staggered_steps_; }

  /// @brief Configure BC cutback behavior. See @ref BcRampOptions.
  /// Default-constructed options leave behavior unchanged.
  void setBcRampOptions(const BcRampOptions& options) { bc_ramp_options_ = options; }

  /// @brief Read current BC ramp options.
  const BcRampOptions& bcRampOptions() const { return bc_ramp_options_; }

  /// @brief Clear the per-block prev-BC snapshot cache. Call this whenever
  /// BoundaryConditionManagers are replaced between solves; the cache is keyed
  /// by raw pointer and will silently use stale data if the pointer is reused.
  void clearBcRampCache() { prev_bc_cache_.clear(); }

  /// @brief Number of hidden (intermediate) Newton solves performed during the
  /// last @ref solve call as part of BC ramp cutbacks. Zero when ramp disabled
  /// or accepted at alpha=1. Useful for tests that bound cost.
  int lastHiddenSolveCount() const { return last_hidden_solve_count_; }

 private:
  int max_staggered_iterations_;  ///< Maximum number of staggered iterations.
  bool exact_staggered_steps_;    ///< If true, no early-exit convergence check.
  std::vector<Stage> stages_;     ///< Solver stages for the staggered iterations.

  /// @brief Inner staggered solve. When @p bc_field_overrides is non-empty its
  /// entries are forwarded to block_solve as the BC values written into
  /// constrained tdofs (in place of evaluating BC coefficients at time).
  /// Returns {solutions, all_stages_converged}. The convergence flag is used
  /// by the BC cutback loop as a failure predicate alongside NaN detection.
  std::pair<std::vector<FieldState>, bool> solveInner(
      const std::vector<WeakForm*>& residual_evals, const std::vector<std::vector<size_t>>& block_indices,
      const FieldState& shape_disp, const std::vector<std::vector<FieldState>>& states,
      const std::vector<std::vector<FieldState>>& params, const TimeInfo& time_info,
      const std::vector<const BoundaryConditionManager*>& bc_managers,
      const std::vector<FEFieldPtr>& bc_field_overrides, bool use_intermediate_tolerances) const;

  BcRampOptions bc_ramp_options_{};  ///< BC ramp configuration (default: disabled).

  /// @brief Per-block cache of previously-applied BC tdof values, keyed by
  /// BoundaryConditionManager identity. Snapshot at end of every successful
  /// solve() when ramp is enabled; used as the lerp basepoint on the next call.
  /// First call (cache miss): pre-BC diagonal state acts as prev.
  /// mutable: solve() is logically const from the caller's perspective.
  mutable std::unordered_map<const BoundaryConditionManager*, FEFieldPtr> prev_bc_cache_;

  mutable int last_hidden_solve_count_ = 0;  ///< Diagnostic counter; written by ramp loop.
};

}  // namespace smith
