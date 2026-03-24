// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/field_store.hpp"

namespace smith {

class CoupledSystemSolver;
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
                              const CoupledSystemSolver* solver, const TimeInfo& time_info,
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
   * @param cycle_zero_weak_form Optional weak form for initial acceleration solve at cycle 0.
   * @param cycle_zero_solver Optional solver paired with `cycle_zero_weak_form` for the cycle-0 solve.
   */
  MultiphysicsTimeIntegrator(std::shared_ptr<FieldStore> field_store,
                             const std::vector<std::shared_ptr<WeakForm>>& weak_forms,
                             std::shared_ptr<smith::CoupledSystemSolver> solver,
                             std::shared_ptr<WeakForm> cycle_zero_weak_form = nullptr,
                             std::shared_ptr<smith::CoupledSystemSolver> cycle_zero_solver = nullptr);

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
  std::shared_ptr<smith::CoupledSystemSolver> solver_;
  std::shared_ptr<WeakForm> cycle_zero_weak_form_;
  std::shared_ptr<smith::CoupledSystemSolver> cycle_zero_solver_;
};

}  // namespace smith
