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
#include "smith/physics/common.hpp"

namespace smith {

class WeakForm;
class DifferentiableBlockSolver;
class BoundaryConditionManager;

/// @brief Orchestrates staggered solution for multiphysics systems.
class SystemSolver {
 public:
  /// @brief Represents a single stage in a staggered iteration.
  struct Stage {
    std::vector<size_t> block_indices;                  ///< Which blocks (residuals) to solve in this stage.
    std::shared_ptr<DifferentiableBlockSolver> solver;  ///< Solver to use for this stage.
  };

  /// @brief Construct a SystemSolver for staggered iteration.
  /// @param comm MPI communicator for parallel norm computation and diagnostic output.
  /// @param max_staggered_iterations Maximum number of staggered sweeps across all stages.  When
  ///        @p exact_staggered_steps is false, the solver may exit early once all stage solvers
  ///        report convergence.
  /// @param exact_staggered_steps If true, always perform exactly @p max_staggered_iterations
  ///        sweeps with no early-exit convergence check.  Useful when a fixed number of
  ///        partitioned-stagger steps is required regardless of residual level.
  SystemSolver(MPI_Comm comm, int max_staggered_iterations, bool exact_staggered_steps = false);

  /// @brief Adds a solver stage defined by a Stage struct.
  /// @param stage Stage configuration.
  void addStage(const Stage& stage);

  /// @brief Convenience method to add a solver stage.
  /// @param block_indices Indices of the blocks to solve.
  /// @param solver Differentiable solver for this stage.
  void addStage(const std::vector<size_t>& block_indices, std::shared_ptr<DifferentiableBlockSolver> solver);

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

 private:
  MPI_Comm comm_;
  int max_staggered_iterations_;
  bool exact_staggered_steps_;
  std::vector<Stage> stages_;
};

}  // namespace smith
