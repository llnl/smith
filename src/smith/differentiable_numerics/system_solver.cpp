// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"

#include <iostream>
#include <axom/slic.hpp>
#include <axom/fmt.hpp>

namespace smith {

SystemSolver::SystemSolver(double global_rel_tol, int global_max_iterations)
    : global_rel_tol_(global_rel_tol), global_max_iterations_(global_max_iterations)
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
  // Check if there are no stages. If not, maybe create a monolithic stage?
  // Let's enforce that stages_ is not empty.
  SLIC_ERROR_IF(stages_.empty(), "SystemSolver has no stages defined.");

  // Make a working copy of the states that we will update iteratively
  std::vector<std::vector<FieldState>> current_states = states;

  size_t num_residuals = residual_evals.size();
  
  // Create a loop over global max iterations
  for (int iter = 0; iter < global_max_iterations_; ++iter) {
    // bool all_stages_converged = true; // In a real implementation we would check the global residual here

    // Instead of checking global residual right now, let's just do a single loop if global_max_iterations_ == 1
    // or we can just rely on the sub-solvers for now.
    
    // For each stage in the staggered solve
    for (size_t stage_idx = 0; stage_idx < stages_.size(); ++stage_idx) {
      const auto& stage = stages_[stage_idx];
      
      // Extract the subset of equations and variables for this stage
      std::vector<WeakForm*> stage_residuals;
      std::vector<std::vector<size_t>> stage_block_indices;
      std::vector<std::vector<FieldState>> stage_states;
      std::vector<std::vector<FieldState>> stage_params;
      std::vector<const BoundaryConditionManager*> stage_bc_managers;

      size_t num_stage_blocks = stage.block_indices.size();

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

      // Solve this stage!
      std::vector<FieldState> stage_solutions = block_solve(
          stage_residuals, stage_block_indices, shape_disp, stage_states, stage_params, time_info, stage.solver.get(), stage_bc_managers);

      // Now, update the current_states with the solutions from this stage
      // stage_solutions contains the updated fields for the diagonal blocks of this stage
      for (size_t i = 0; i < num_stage_blocks; ++i) {
        size_t global_row = stage.block_indices[i];
        size_t global_col = global_row; // The solution corresponds to the diagonal block
        size_t s_idx = block_indices[global_row][global_col]; // the index of the unknown in the states array

        // We must update the state globally
        // Wait, current_states[some_row] might also contain this field as an input.
        // We should just update ALL references to this state across current_states.
        // The easiest way is to match by field name or identity. Since we cloned it, 
        // we can just replace it everywhere it appears.
        
        // Actually, in `SystemSolver`, it's much simpler if the physics system (e.g., MultiphysicsTimeIntegrator)
        // just extracts the diagonal fields.
        // Let's replace the occurrences of the field in `current_states` with `stage_solutions[i]`.
        // To do this, we know that original `states[r][s_idx]` was the unknown.
        // We can just find all occurrences of the OLD state in `current_states` and replace with `stage_solutions[i]`.
        
        // Let's just update it:
        FieldState new_state = stage_solutions[i];
        FieldState old_state = current_states[global_row][s_idx]; // The one we just solved for
        
        for (size_t r = 0; r < num_residuals; ++r) {
          for (size_t c = 0; c < current_states[r].size(); ++c) {
            // Check identity (e.g. name or pointer). Name is unique enough for states.
            // For now, let's just compare the `get<FEFieldPtr>()` pointers or names.
            if (current_states[r][c].get()->name() == old_state.get()->name()) {
              current_states[r][c] = new_state;
            }
          }
        }
      }
    }

    // TODO: Evaluate global residual to break early if max_iterations > 1
    // if (global_max_iterations_ > 1 && evaluateGlobalResidual(...) < global_rel_tol_) break;
    // For now, we just rely on max_iters
  }

  (void)global_rel_tol_; // unused for now, will be used when evaluating global residual

  // At the end, return the DIAGONAL states as the final solution
  std::vector<FieldState> final_solutions;
  final_solutions.reserve(num_residuals);
  for (size_t r = 0; r < num_residuals; ++r) {
    size_t s_idx = block_indices[r][r];
    final_solutions.push_back(current_states[r][s_idx]);
  }

  return final_solutions;
}

} // namespace smith