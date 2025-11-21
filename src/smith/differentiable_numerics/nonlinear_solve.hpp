// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_system.hpp
 *
 * @brief Specifies parametrized residuals and various linearized evaluations for arbitrary nonlinear systems of
 * equations
 */

#pragma once

#include <vector>
#include "smith/differentiable_numerics/field_state.hpp"

namespace smith {

class WeakForm;
class DifferentiableSolver;
// class DifferentiableBlockSolver;
class BoundaryConditionManager;
class DirichletBoundaryConditions;

FieldState nonlinearSolve(const WeakForm* residual_eval, const FieldState& shape_disp,
                          const std::vector<FieldState>& states, const std::vector<FieldState>& params,
                          const std::vector<double>& state_update_weights, size_t primal_solve_state_index,
                          size_t to_dirichlet_state_index, const TimeInfo& time_info,
                          const DifferentiableSolver* solver, const BoundaryConditionManager* bc_manager,
                          const FieldState* bc_field = nullptr);

FieldState solve(const FieldState& x_guess, const FieldState& shape_disp, const std::vector<FieldState>& params,
                 const TimeInfo& time_info, const WeakForm& weak_form, const DifferentiableSolver& solver,
                 const DirichletBoundaryConditions& bcs, size_t unknown_index = 0);

inline FieldState solve(const WeakForm* residual_eval, const FieldState& shape_disp,
                        const std::vector<FieldState>& states, const std::vector<FieldState>& params,
                        const TimeInfo& time_info, const DifferentiableSolver* solver,
                        const BoundaryConditionManager* bc_manager)
{
  std::vector<double> state_update_weights(states.size(), 0.0);
  state_update_weights[0] = 1.0;
  return nonlinearSolve(residual_eval, shape_disp, states, params, state_update_weights, 0, 0, time_info, solver,
                        bc_manager);
}

inline FieldState solve(const WeakForm* residual_eval, const FieldState& shape_disp,
                        const std::vector<FieldState>& states, const std::vector<FieldState>& params,
                        const TimeInfo& time_info, const DifferentiableSolver* solver,
                        const BoundaryConditionManager* bc_manager, const FieldState& bc_field)
{
  std::vector<double> state_update_weights(states.size(), 0.0);
  state_update_weights[0] = 1.0;
  return nonlinearSolve(residual_eval, shape_disp, states, params, state_update_weights, 0, 0, time_info, solver,
                        bc_manager, &bc_field);
}

/*
std::vector<FieldState> block_solve(const std::vector<WeakForm*>& residual_evals,
                                    const std::vector<std::vector<size_t>> block_indices, const FieldState& shape_disp,
                                    const std::vector<std::vector<FieldState>>& states,
                                    const std::vector<std::vector<FieldState>>& params, const DoubleState& time,
                                    const DoubleState& dt, size_t cycle, const DifferentiableBlockSolver* solver,
                                    const std::vector<BoundaryConditionManager*> bc_managers);
*/

}  // namespace smith
