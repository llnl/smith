// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_solve.hpp
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

/// @brief Solve a nonlinear system of equations as defined by the weak form
/// @param residual_eval The weak form which defines the equations to be solved
/// @param shape_disp The mesh-morphed shape displacement
/// @param states The time varying states as inputs to the weak form
/// @param params The fixed field parameters as inputs to the weak form
/// @param state_update_weights Specifies how to blend the arguments of the weak form into a unique unknown.  The
/// primary unknown, p, is states[primal_solve_state_index].  However, other arguments to the weak form may also depend
/// on this unknown. The state_update_weights specify how this is done.  For a nonzero weight, the field argument i is
/// assumed to change as states[i] + (p - p_initial) * state_update_weights[i].
/// @param primal_solve_state_index Index specifying which of the states is the primary unknown.
/// @param to_dirichlet_state_index Index specifying which field has the Dirichlet boundary conditions applied to it.
/// Typically this will be the same as the primal_solve_state_index, but it can be different.  An example is explicit
/// dynamics, where the unknown is the acceleration, and the boundary conditions are often applied directly to the
/// displacement field.
/// @param time_info Timestep information (time, dt, cycle)
/// @param solver The differentiable, potentially nonlinear, equation solver used to solve the system of equations
/// @param bc_manager Holds information about which degrees of freedom (DOFS)
/// @param bc_field A field which holds to desired values for the fixed degrees of freedom as specified by the
/// bc_manager
/// @return The field solution to the weak form
FieldState nonlinearSolve(const WeakForm* residual_eval, const FieldState& shape_disp,
                          const std::vector<FieldState>& states, const std::vector<FieldState>& params,
                          const std::vector<double>& state_update_weights, size_t primal_solve_state_index,
                          size_t to_dirichlet_state_index, const TimeInfo& time_info,
                          const DifferentiableSolver* solver, const BoundaryConditionManager* bc_manager,
                          const FieldState* bc_field = nullptr);

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

}  // namespace smith
