// Copyright (c) Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"

#include <axom/slic.hpp>
#include <axom/fmt.hpp>

namespace smith {

SystemSolver::SystemSolver(int max_staggered_iterations, bool exact_staggered_steps)
    : max_staggered_iterations_(max_staggered_iterations), exact_staggered_steps_(exact_staggered_steps)
{
}

void SystemSolver::addStage(const Stage& stage)
{
  stages_.push_back(stage);
}

void SystemSolver::addStage(const std::vector<size_t>& block_indices, std::shared_ptr<DifferentiableBlockSolver> solver)
{
  stages_.push_back({block_indices, std::move(solver)});
}

std::vector<FieldState> SystemSolver::solve(
    const std::vector<WeakForm*>& residual_evals,
    const std::vector<std::vector<size_t>>& block_indices,
    const FieldState& shape_disp,
    const std::vector<std::vector<FieldState>>& states,
    const std::vector<std::vector<FieldState>>& params,
    const TimeInfo& time_info,
    const std::vector<const BoundaryConditionManager*>& bc_managers) const
{
  SLIC_ERROR_IF(stages_.empty(), "SystemSolver has no stages defined.");

  // Reset each stage solver's convergence tracking (e.g. initial residual norm for rel-tol)
  for (const auto& stage : stages_) {
    stage.solver->resetConvergenceState();
  }

  // Working copy of states, updated in-place as stages solve
  std::vector<std::vector<FieldState>> current_states = states;
  size_t num_residuals = residual_evals.size();

  for (int iter = 0; iter < max_staggered_iterations_; ++iter) {
    // --- Run each stage ---
    for (size_t stage_idx = 0; stage_idx < stages_.size(); ++stage_idx) {
      const auto& stage = stages_[stage_idx];
      size_t num_stage_blocks = stage.block_indices.size();

      std::vector<WeakForm*> stage_residuals;
      std::vector<std::vector<size_t>> stage_block_indices;
      std::vector<std::vector<FieldState>> stage_states;
      std::vector<std::vector<FieldState>> stage_params;
      std::vector<const BoundaryConditionManager*> stage_bc_managers;

      for (size_t i = 0; i < num_stage_blocks; ++i) {
        size_t global_row = stage.block_indices[i];
        stage_residuals.push_back(residual_evals[global_row]);
        stage_bc_managers.push_back(bc_managers[global_row]);
        stage_states.push_back(current_states[global_row]);
        stage_params.push_back(params[global_row]);

        std::vector<size_t> row_indices(num_stage_blocks, invalid_block_index);
        for (size_t col_idx = 0; col_idx < num_stage_blocks; ++col_idx) {
          size_t global_col = stage.block_indices[col_idx];
          row_indices[col_idx] = block_indices[global_row][global_col];
        }
        stage_block_indices.push_back(row_indices);
      }

      std::vector<FieldState> stage_solutions = block_solve(
          stage_residuals, stage_block_indices, shape_disp, stage_states, stage_params, time_info,
          stage.solver.get(), stage_bc_managers);

      // Propagate updated fields to all residuals that reference them
      for (size_t i = 0; i < num_stage_blocks; ++i) {
        size_t global_col = stage.block_indices[i];
        FieldState new_state = stage_solutions[i];
        for (size_t r = 0; r < num_residuals; ++r) {
          size_t c = block_indices[r][global_col];
          if (c != invalid_block_index) {
            current_states[r][c] = new_state;
          }
        }
      }
    }

    // --- Convergence check (skipped in exact-steps mode or if only one iteration) ---
    if (!exact_staggered_steps_ && max_staggered_iterations_ > 1) {
      bool all_converged = true;
      for (const auto& stage : stages_) {
        size_t num_stage_blocks = stage.block_indices.size();
        std::vector<mfem::Vector> stage_residuals;
        for (size_t i = 0; i < num_stage_blocks; ++i) {
          size_t global_row = stage.block_indices[i];
          std::vector<const FiniteElementState*> input_ptrs;
          for (const auto& field_state : current_states[global_row]) {
            input_ptrs.push_back(field_state.get().get());
          }
          mfem::Vector res = residual_evals[global_row]->residual(time_info, shape_disp.get().get(), input_ptrs);
          if (bc_managers[global_row]) {
            res.SetSubVector(bc_managers[global_row]->allEssentialTrueDofs(), 0.0);
          }
          stage_residuals.push_back(std::move(res));
        }
        if (!stage.solver->checkConvergence(1.0, stage_residuals)) {
          all_converged = false;
          break;
        }
      }
      if (all_converged) {
        break;
      }
    }
  }

  // Return the diagonal (unknown) states as the final solution
  std::vector<FieldState> final_solutions;
  final_solutions.reserve(num_residuals);
  for (size_t r = 0; r < num_residuals; ++r) {
    size_t s_idx = block_indices[r][r];
    final_solutions.push_back(current_states[r][s_idx]);
  }

  return final_solutions;
}

} // namespace smith
