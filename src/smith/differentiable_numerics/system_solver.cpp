// Copyright (c) Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <string>
#include <axom/slic.hpp>
#include <axom/fmt.hpp>

namespace smith {

SystemSolver::SystemSolver(std::shared_ptr<NonlinearBlockSolverBase> single_solver)
    : max_staggered_iterations_(1), exact_staggered_steps_(false)
{
  addSubsystemSolver({}, std::move(single_solver));
}

SystemSolver::SystemSolver(int max_staggered_iterations, bool exact_staggered_steps)
    : max_staggered_iterations_(max_staggered_iterations), exact_staggered_steps_(exact_staggered_steps)
{
  SLIC_ERROR_IF(max_staggered_iterations <= 0, "max_staggered_iterations must be > 0");
}

void SystemSolver::addSubsystemSolver(const std::vector<size_t>& block_indices,
                                      std::shared_ptr<NonlinearBlockSolverBase> solver, double relaxation_factor)
{
  SLIC_ERROR_IF(!solver, "SystemSolver stage solver must be non-null");
  SLIC_ERROR_IF(relaxation_factor <= 0.0 || relaxation_factor > 1.0,
                axom::fmt::format("Stage relaxation_factor {} must be in (0, 1]", relaxation_factor));

  stages_.push_back(Stage{block_indices, std::move(solver), relaxation_factor});
}

void SystemSolver::appendStagesWithBlockMapping(const SystemSolver& subsystem_solver,
                                                const std::vector<size_t>& global_block_indices)
{
  SLIC_ERROR_IF(global_block_indices.empty(), "Global block index map must be non-empty");

  for (const auto& stage : subsystem_solver.stages_) {
    std::vector<size_t> remapped_block_indices;
    if (stage.block_indices.empty()) {
      remapped_block_indices = global_block_indices;
    } else {
      remapped_block_indices.reserve(stage.block_indices.size());
      for (size_t local_block_index : stage.block_indices) {
        SLIC_ERROR_IF(local_block_index >= global_block_indices.size(),
                      axom::fmt::format("Local block index {} exceeds subsystem size {}", local_block_index,
                                        global_block_indices.size()));
        remapped_block_indices.push_back(global_block_indices[local_block_index]);
      }
    }
    addSubsystemSolver(remapped_block_indices, stage.solver, stage.relaxation_factor);
  }
}

std::vector<FieldState> SystemSolver::solve(const std::vector<WeakForm*>& residual_evals,
                                            const std::vector<std::vector<size_t>>& block_indices,
                                            const FieldState& shape_disp,
                                            const std::vector<std::vector<FieldState>>& states,
                                            const std::vector<std::vector<FieldState>>& params,
                                            const TimeInfo& time_info,
                                            const std::vector<const BoundaryConditionManager*>& bc_managers) const
{
  SLIC_ERROR_IF(stages_.empty(), "SystemSolver has no stages defined.");

  size_t num_residuals = residual_evals.size();
  std::vector<Stage> active_stages = stages_;
  for (auto& stage : active_stages) {
    if (stage.block_indices.empty()) {
      stage.block_indices.resize(num_residuals);
      std::iota(stage.block_indices.begin(), stage.block_indices.end(), 0);
    }
    for (size_t block_index : stage.block_indices) {
      SLIC_ERROR_IF(block_index >= num_residuals,
                    axom::fmt::format("Stage block index {} exceeds residual count {}", block_index, num_residuals));
    }
  }
  // Set the inner tolerance factor based on the number of stages.  For single-stage
  // solves, we don't want to reduce the tolerances as that's pointless and
  // unintuitive.  For multi-stage solves, we want a tighter inner solve to
  // ensure outer staggered convergence.
  const double inner_tol_factor = (active_stages.size() == 1) ? 1.0 : 0.6;
  for (auto& stage : active_stages) {
    stage.solver->setInnerToleranceMultiplier(inner_tol_factor);
  }

  // Reset each stage solver's convergence tracking (e.g. initial residual norm for rel-tol)
  for (const auto& stage : active_stages) {
    stage.solver->resetConvergenceState();
  }
  std::vector<NonlinearConvergenceContext> stage_convergence_contexts(active_stages.size());

  // Working copy of states, updated in-place as stages solve
  std::vector<std::vector<FieldState>> current_states = states;

  // Pre-compute name -> (row, slot) routing so the propagation loop avoids O(N*M) string compares
  // on every staggered iteration. Field-name identity within current_states is invariant across
  // the iteration loop: only values are replaced, never the underlying name.
  std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>> field_routing;
  for (size_t r = 0; r < num_residuals; ++r) {
    for (size_t slot = 0; slot < current_states[r].size(); ++slot) {
      field_routing[current_states[r][slot].get()->name()].emplace_back(r, slot);
    }
  }

  // Helper lambda to assemble input pointers, evaluate residual, and zero essential BCs
  auto eval_residual_and_zero_bcs = [&](size_t global_row) {
    std::vector<const FiniteElementState*> input_ptrs;
    for (const auto& field_state : current_states[global_row]) {
      input_ptrs.push_back(field_state.get().get());
    }
    for (const auto& param_state : params[global_row]) {
      input_ptrs.push_back(param_state.get().get());
    }
    mfem::Vector res = residual_evals[global_row]->residual(time_info, shape_disp.get().get(), input_ptrs);
    if (bc_managers[global_row]) {
      res.SetSubVector(bc_managers[global_row]->allEssentialTrueDofs(), 0.0);
    }
    return res;
  };

  // Evaluate and register true initial residuals before block sweeps mutate the state.
  for (size_t stage_idx = 0; stage_idx < active_stages.size(); ++stage_idx) {
    const auto& stage = active_stages[stage_idx];
    size_t num_stage_blocks = stage.block_indices.size();
    std::vector<mfem::Vector> stage_init_residuals;
    for (size_t i = 0; i < num_stage_blocks; ++i) {
      stage_init_residuals.push_back(eval_residual_and_zero_bcs(stage.block_indices[i]));
    }
    stage.solver->primeConvergenceContext(stage_init_residuals, stage_convergence_contexts[stage_idx]);
  }

  for (int iter = 0; iter < max_staggered_iterations_; ++iter) {
    // --- Run each stage ---
    for (size_t stage_idx = 0; stage_idx < active_stages.size(); ++stage_idx) {
      const auto& stage = active_stages[stage_idx];
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

      std::vector<FieldState> stage_solutions =
          block_solve(stage_residuals, stage_block_indices, shape_disp, stage_states, stage_params, time_info,
                      stage.solver.get(), stage_bc_managers);

      // Propagate updated fields to every residual input that references the solved field.
      // Match by field name (looked up via the pre-computed routing map): coupling fields appear
      // as fixed inputs in other rows and therefore do not have a valid unknown-block entry there.
      // Apply relaxation: x_new = omega * x_solved + (1 - omega) * x_k.
      for (size_t i = 0; i < num_stage_blocks; ++i) {
        size_t global_col = stage.block_indices[i];
        FieldState new_state = stage_solutions[i];

        if (stage.relaxation_factor != 1.0) {
          FieldState old_state = current_states[global_col][block_indices[global_col][global_col]];
          new_state = weighted_average(new_state, old_state, stage.relaxation_factor);
        }

        auto it = field_routing.find(new_state.get()->name());
        if (it != field_routing.end()) {
          for (const auto& [r, slot] : it->second) {
            current_states[r][slot] = new_state;
          }
        }
      }
    }

    // --- Convergence check (skipped in exact-steps mode, single-iteration mode,
    //     or on the last iteration where a break has no effect) ---
    if (!exact_staggered_steps_ && max_staggered_iterations_ > 1 && iter < max_staggered_iterations_ - 1) {
      bool all_converged = true;
      for (size_t s = 0; s < active_stages.size(); ++s) {
        const auto& stage = active_stages[s];
        size_t num_stage_blocks = stage.block_indices.size();
        std::vector<mfem::Vector> stage_residuals;
        for (size_t i = 0; i < num_stage_blocks; ++i) {
          stage_residuals.push_back(eval_residual_and_zero_bcs(stage.block_indices[i]));
        }
        auto stage_status = stage.solver->convergenceStatus(1.0, stage_residuals, stage_convergence_contexts[s]);

        if (!stage_status.converged) {
          all_converged = false;
          break;
        }
      }
      if (all_converged) {
        SLIC_INFO_ROOT(axom::fmt::format("Staggered iteration converged after {} iteration(s)", iter + 1));
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

std::shared_ptr<SystemSolver> SystemSolver::singleBlockSolver(size_t block_index) const
{
  constexpr bool exact_staggered_steps = true;
  for (const auto& stage : stages_) {
    if (stage.block_indices.empty()) {
      auto result = std::make_shared<SystemSolver>(1, exact_staggered_steps);
      std::shared_ptr<NonlinearBlockSolverBase> stage_solver = stage.solver;
      if (const auto* equation_solver = dynamic_cast<const NonlinearBlockSolver*>(stage.solver.get())) {
        if (auto cloned_solver = equation_solver->cloneFresh()) {
          stage_solver = cloned_solver;
        }
      }
      Stage single_stage{{0}, stage_solver, stage.relaxation_factor};
      result->addSubsystemSolver(single_stage.block_indices, single_stage.solver, single_stage.relaxation_factor);
      return result;
    }

    auto found = std::find(stage.block_indices.begin(), stage.block_indices.end(), block_index);
    if (found != stage.block_indices.end()) {
      auto result = std::make_shared<SystemSolver>(1, exact_staggered_steps);
      std::shared_ptr<NonlinearBlockSolverBase> stage_solver = stage.solver;
      if (const auto* equation_solver = dynamic_cast<const NonlinearBlockSolver*>(stage.solver.get())) {
        if (auto cloned_solver = equation_solver->cloneFresh()) {
          stage_solver = cloned_solver;
        }
      }
      Stage single_stage{{0}, stage_solver, stage.relaxation_factor};
      result->addSubsystemSolver(single_stage.block_indices, single_stage.solver, single_stage.relaxation_factor);
      return result;
    }
  }

  return nullptr;
}

}  // namespace smith
