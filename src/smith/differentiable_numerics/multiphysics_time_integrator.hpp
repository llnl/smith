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
  MultiphysicsTimeIntegrator(std::shared_ptr<SystemBase> system,
                             std::shared_ptr<SystemBase> cycle_zero_system = nullptr);

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
  std::shared_ptr<SystemBase> cycle_zero_system_;
  std::vector<std::shared_ptr<SystemBase>> post_solve_systems_;
};

inline std::shared_ptr<MultiphysicsTimeIntegrator> makeAdvancer(std::shared_ptr<SystemBase> system,
                                                                std::shared_ptr<SystemBase> cycle_zero_system = nullptr)
{
  return std::make_shared<MultiphysicsTimeIntegrator>(std::move(system), std::move(cycle_zero_system));
}

template <typename SystemType>
std::shared_ptr<MultiphysicsTimeIntegrator> makeAdvancer(std::shared_ptr<SystemType> system)
{
  if constexpr (requires { system->cycle_zero_systems; }) {
    // CombinedSystem: run each sub-system's cycle-zero solve in sequence (one pass, no stagger).
    std::shared_ptr<SystemBase> cz = nullptr;
    if (!system->cycle_zero_systems.empty()) {
      auto cz_combined = std::make_shared<CombinedSystem>();
      cz_combined->field_store = system->field_store;
      cz_combined->max_stagger_iters = 1;
      for (auto& czs : system->cycle_zero_systems) {
        cz_combined->subsystems.push_back(czs);
        for (auto& wf : czs->weak_forms) {
          cz_combined->weak_forms.push_back(wf);
        }
      }
      cz = cz_combined;
    }
    return makeAdvancer(std::static_pointer_cast<SystemBase>(system), cz);
  } else if constexpr (requires { system->cycle_zero_system; }) {
    return makeAdvancer(std::static_pointer_cast<SystemBase>(system), system->cycle_zero_system);
  } else {
    return makeAdvancer(std::static_pointer_cast<SystemBase>(system), nullptr);
  }
}

}  // namespace smith
