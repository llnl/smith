// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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

namespace smith {

class WeakForm;
class DifferentiableSolver;
class DifferentiableBlockSolver;
class BoundaryConditionManager;
class DirichletBoundaryConditions;

/// @brief Solve a nonlinear system of equations as defined by the weak form, assuming that the field indexed by
/// unknown_index is the unknown field
/// @param x_guess Initial guess field which is being solved for
/// @param shape_disp The mesh-morphed shape displacement
/// @param params All fixed fields pass to the weak form
/// @param time_info Timestep information (time, dt, cycle)
/// @param residual_eval The weak form which defines the equations to be solved
/// @param solver The differentiable, potentially nonlinear, equation solver used to solve the system of equations
/// @param bcs Holds information about which degrees of freedom (DOFS), and has the information about the time and space
/// varying values for the boundary conditions
/// @param unknown_index
/// @return The field solution to the weak form
FieldState solve(const FieldState& x_guess, const FieldState& shape_disp, const std::vector<FieldState>& params,
                 const TimeInfo& time_info, const WeakForm& residual_eval, const DifferentiableSolver& solver,
                 const DirichletBoundaryConditions& bcs, size_t unknown_index = 0);

// /// @brief Solve a nonlinear system of equations as defined by the weak_form, assuming the first field in states is
// the
// /// unknown field
// /// @param residual_eval The weak form which defines the equations to be solved
// /// @param shape_disp The mesh-morphed shape displacement
// /// @param states The time varying states as inputs to the weak form
// /// @param params The fixed field parameters as inputs to the weak form
// /// @param time_info Timestep information (time, dt, cycle)
// /// @param solver The differentiable, potentially nonlinear, equation solver used to solve the system of equations
// /// @param bc_manager Holds information about which degrees of freedom (DOFS), and has the information about the time
// /// and space varying values for the boundary conditions
// /// @return The field solution to the weak form
// FieldState solve(const WeakForm* residual_eval, const FieldState& shape_disp, const std::vector<FieldState>& states,
//                  const std::vector<FieldState>& params, const TimeInfo& time_info, const DifferentiableSolver*
//                  solver, const BoundaryConditionManager* bc_manager);

// /// @brief Solve a nonlinear system of equations as defined by the weak_form, assuming the first field in states is
// the
// /// unknown field
// /// @param residual_eval The weak form which defines the equations to be solved
// /// @param shape_disp The mesh-morphed shape displacement
// /// @param states The time varying states as inputs to the weak form
// /// @param params The fixed field parameters as inputs to the weak form
// /// @param time_info Timestep information (time, dt, cycle)
// /// @param solver The differentiable, potentially nonlinear, equation solver used to solve the system of equations
// /// @param bc_manager Holds information about which degrees of freedom (DOFS)
// /// @param bc_field A field which holds to desired values for the fixed degrees of freedom as specified by the
// /// bc_manager
// /// @return The field solution to the weak form
// inline FieldState solve(const WeakForm* residual_eval, const FieldState& shape_disp,
//                         const std::vector<FieldState>& states, const std::vector<FieldState>& params,
//                         const TimeInfo& time_info, const DifferentiableSolver* solver,
//                         const BoundaryConditionManager* bc_manager, const FieldState& bc_field)
// {
//   std::vector<double> state_update_weights(states.size(), 0.0);
//   state_update_weights[0] = 1.0;
//   return nonlinearSolve(residual_eval, shape_disp, states, params, state_update_weights, 0, 0, time_info, solver,
//                         bc_manager, &bc_field);
// }

/// @brief Solve a block nonlinear system of equations as defined by the vector of weak form
/// @param residual_eval The weak forms which defines the equations to be solved
/// @param shape_disp The mesh-morphed shape displacement
/// @param states The time varying states as inputs to the weak form
/// @param params The fixed field parameters as inputs to the weak form
/// @param time_info Timestep information (time, dt, cycle)
/// @param solver The differentiable, potentially nonlinear, equation solver used to solve the system of equations
/// @param bc_managers Holds information about which degrees of freedom (DOFS)
/// @return The field solution to the weak form
std::vector<FieldState> block_solve(const std::vector<WeakForm*>& residual_evals,
                                    const std::vector<std::vector<size_t>> block_indices, const FieldState& shape_disp,
                                    const std::vector<std::vector<FieldState>>& states,
                                    const std::vector<std::vector<FieldState>>& params, const TimeInfo& time_info,
                                    const DifferentiableBlockSolver* solver,
                                    const std::vector<BoundaryConditionManager*> bc_managers);

}  // namespace smith
