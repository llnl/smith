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
#include "smith/physics/common.hpp"

namespace smith {

class WeakForm;
class NonlinearBlockSolverBase;
class BoundaryConditionManager;
class DirichletBoundaryConditions;

/// @brief magic number for representing a field which is not an argument of the weak form.
static constexpr size_t invalid_block_index = std::numeric_limits<size_t>::max() - 1;

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

}  // namespace smith
