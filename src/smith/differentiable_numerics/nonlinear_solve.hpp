// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_solve.hpp
 *
 * @brief Methods for solving systems of equations as given by WeakForms.  Tracks these operations on the gretl graph
 * with a custom vjp.
 */

#pragma once

#include <vector>
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/physics/common.hpp"

namespace smith {

class WeakForm;
class NonlinearBlockSolverBase;
class BoundaryConditionManager;
class DirichletBoundaryConditions;

/// @brief magic number for representing a field which is not an argument of the weak form.
static constexpr size_t invalid_block_index = std::numeric_limits<size_t>::max() - 1;

/// @brief Apply boundary conditions to @p primal_field by evaluating BC coefficients at @p time.
void applyBoundaryConditions(const BoundaryConditionManager* bc_manager, double time, FEFieldPtr& primal_field);

/// @brief Apply boundary conditions to @p primal_field as @c alpha * bc(t_new) + (1 - alpha) * bc(t_old)
/// at each essential tdof. Evaluates BC coefficients at both endpoints (preserving any user-defined
/// non-smoothness in the BC ramp) and lerps in tdof space.
void applyBoundaryConditions(const BoundaryConditionManager* bc_manager, double t_new, double t_old, double alpha,
                             FEFieldPtr& primal_field);

/// @brief Solve a block nonlinear system of equations as defined by the vector of weak form
/// @param residual_evals Vector of weak forms which define the equations to be solved
/// @param block_indices Matrix of index arguments specifying where in each WeakForm the unknown fields are passed in.
/// Example: for a 2 weak-form system, with weak-forms, r1, r2
/// r1(a,b,c)
/// r2(b,d,e,a)
// with unknowns (with respect to the solver) being a, and b.
// r1 has unknowns a,b in the ‘slots’ 0, 1
// r2 has unknowns a,b, in the ‘slots’ 3,0
/// @param shape_disp The mesh-morphed shape displacement
/// @param states The time varying states as inputs to the weak form
/// @param params The fixed field parameters as inputs to the weak form
/// @param time_info Timestep information (time, dt, cycle)
/// @param solver The nonlinear block solver used to solve the system of equations
/// @param bc_managers Holds information about which degrees of freedom (DOFS)
/// @return Vector of field solutions satisfying the weak forms
std::vector<FieldState> block_solve(const std::vector<WeakForm*>& residual_evals,
                                    const std::vector<std::vector<size_t>> block_indices, const FieldState& shape_disp,
                                    const std::vector<std::vector<FieldState>>& states,
                                    const std::vector<std::vector<FieldState>>& params, const TimeInfo& time_info,
                                    const NonlinearBlockSolverBase* solver,
                                    const std::vector<const BoundaryConditionManager*>& bc_managers);

/// @brief Solve a single nonlinear system of equations as defined by one weak form.
/// @param residual_eval The weak form that defines the residual.
/// @param shape_disp The mesh-morphed shape displacement.
/// @param states The weak-form state inputs, including the primary unknown at @p unknown_state_index.
/// @param params The fixed field parameters passed to the weak form.
/// @param time_info Timestep information (time, dt, cycle).
/// @param solver The nonlinear block solver used to solve the system.
/// @param bcs Boundary conditions applied to the primary unknown field.
/// @param unknown_state_index Index of the primary unknown within @p states.
/// @return The solved primary field.
inline FieldState solve(const WeakForm& residual_eval, const FieldState& shape_disp,
                        const std::vector<FieldState>& states, const std::vector<FieldState>& params,
                        const TimeInfo& time_info, const NonlinearBlockSolverBase& solver,
                        const DirichletBoundaryConditions& bcs, size_t unknown_state_index = 0)
{
  std::vector<const BoundaryConditionManager*> bc_managers{&bcs.getBoundaryConditionManager()};
  auto solutions = block_solve({const_cast<WeakForm*>(&residual_eval)}, {{unknown_state_index}}, shape_disp, {states},
                               {params}, time_info, &solver, bc_managers);
  return solutions[0];
}

/// @brief Backward-compatible overload that accepts but ignores an explicit initial guess.
inline FieldState solve([[maybe_unused]] const FieldState& initial_guess, const FieldState& shape_disp,
                        const std::vector<FieldState>& weak_form_inputs, const TimeInfo& time_info,
                        const WeakForm& residual_eval, const NonlinearBlockSolverBase& solver,
                        const DirichletBoundaryConditions& bcs, size_t unknown_state_index = 0)
{
  return solve(residual_eval, shape_disp, weak_form_inputs, {}, time_info, solver, bcs, unknown_state_index);
}

}  // namespace smith
