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
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/combined_system.hpp"

namespace smith {

class SystemSolver;
class DirichletBoundaryConditions;
class BoundaryConditionManager;

/**
 * @brief Time integrator for multiphysics problems, coordinating multiple weak forms.
 */
class MultiphysicsTimeIntegrator : public StateAdvancer {
 public:
  /**
   * @brief Construct a multiphysics advancer around main and auxiliary systems.
   * @param system Main system solved every timestep.
   * @param cycle_zero_systems Optional startup systems solved independently before first regular step.
   * @param post_solve_systems Optional systems solved after the main step.
   */
  MultiphysicsTimeIntegrator(std::shared_ptr<SystemBase> system,
                             std::vector<std::shared_ptr<SystemBase>> cycle_zero_systems = {},
                             std::vector<std::shared_ptr<SystemBase>> post_solve_systems = {});

  /// @brief Register a system to be solved after the main solve and reaction computation.
  void addPostSolveSystem(std::shared_ptr<SystemBase> system);

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
  std::shared_ptr<SystemBase> system_;
  std::vector<std::shared_ptr<SystemBase>> cycle_zero_systems_;
  std::vector<std::shared_ptr<SystemBase>> post_solve_systems_;

  std::map<std::string, size_t> main_unknown_name_to_local_idx_;
};

/**
 * @brief Build a `MultiphysicsTimeIntegrator` from system-owned or explicit auxiliary systems.
 *
 * Missing optional arguments fall back to `system->cycle_zero_systems` and
 * `system->post_solve_systems`.
 */
inline std::shared_ptr<MultiphysicsTimeIntegrator> makeAdvancer(
    std::shared_ptr<SystemBase> system, std::vector<std::shared_ptr<SystemBase>> cycle_zero_systems = {},
    std::vector<std::shared_ptr<SystemBase>> post_solve_systems = {})
{
  if (cycle_zero_systems.empty()) {
    cycle_zero_systems = system->cycle_zero_systems;
  }
  if (post_solve_systems.empty()) {
    post_solve_systems = system->post_solve_systems;
  }
  return std::make_shared<MultiphysicsTimeIntegrator>(std::move(system), std::move(cycle_zero_systems),
                                                      std::move(post_solve_systems));
}

}  // namespace smith
