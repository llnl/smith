// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/field_store.hpp"

namespace smith {

class DifferentiableBlockSolver;
class DirichletBoundaryConditions;
class BoundaryConditionManager;

/**
 * @brief Solve a set of weak forms.
 * @param weak_forms List of weak forms to solve.
 * @param field_store Field store containing the fields.
 * @param solver The solver to use.
 * @param time_info Current time information.
 * @param params Optional parameter fields.
 * @return std::vector<FieldState> The updated state fields.
 */
std::vector<FieldState> solve(const std::vector<std::shared_ptr<WeakForm>>& weak_forms, const FieldStore& field_store,
                              const DifferentiableBlockSolver* solver, const TimeInfo& time_info,
                              const std::vector<FieldState>& params = {});

/**
 * @brief Time integrator for multiphysics problems, coordinating multiple weak forms.
 */
class MultiphysicsTimeIntegrator : public StateAdvancer {
 public:
  /**
   * @brief Construct a new MultiphysicsTimeIntegrator object.
   * @param field_store Field store containing the fields.
   * @param weak_forms List of weak forms to coordinate.
   * @param solver The block solver to use.
   */
  MultiphysicsTimeIntegrator(std::shared_ptr<FieldStore> field_store,
                             const std::vector<std::shared_ptr<WeakForm>>& weak_forms,
                             std::shared_ptr<smith::DifferentiableBlockSolver> solver);

  /**
   * @brief Advance the multiphysics state by one time step.
   * @param time_info Current time information.
   * @param shape_disp Shape displacement field.
   * @param states Current state fields.
   * @param params Parameter fields.
   * @return std::pair<std::vector<FieldState>, std::vector<ReactionState>> Updated states and reactions.
   */
  std::pair<std::vector<FieldState>, std::vector<ReactionState>> advanceState(
      const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
      const std::vector<FieldState>& params) const override;

 private:
  std::shared_ptr<FieldStore> field_store_;
  std::vector<std::shared_ptr<WeakForm>> weak_forms_;
  std::shared_ptr<smith::DifferentiableBlockSolver> solver_;
};

}  // namespace smith
