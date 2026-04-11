// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/reaction.hpp"

#include <algorithm>
#include <stdexcept>

namespace smith {

MultiphysicsTimeIntegrator::MultiphysicsTimeIntegrator(std::shared_ptr<SystemBase> system,
                                                       std::shared_ptr<SystemBase> cycle_zero_system)
    : system_(system), cycle_zero_system_(cycle_zero_system)
{
}

std::pair<std::vector<FieldState>, std::vector<ReactionState>> MultiphysicsTimeIntegrator::advanceState(
    const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
    const std::vector<FieldState>& params) const
{
  std::vector<FieldState> current_states = states;

  // Sync FieldStore with (possibly updated) states and params so they are current for solve
  system_->field_store->setField(system_->field_store->getShapeDisp().get()->name(), shape_disp);

  for (size_t i = 0; i < current_states.size(); ++i) {
    system_->field_store->setField(i, current_states[i]);
  }
  // Optional: update parameter fields as well? (assuming they are aligned)
  SLIC_ERROR_ROOT_IF(params.size() != system_->field_store->getParameterFields().size(),
                     "Parameter size mismatch in advanceState");
  for (size_t i = 0; i < params.size(); ++i) {
    system_->field_store->setField(system_->field_store->getParameterFields()[i].get()->name(), params[i]);
  }

  // Handle initial acceleration solve at cycle 0
  const bool requires_cycle_zero_solve =
      std::any_of(system_->field_store->getTimeIntegrationRules().begin(),
                  system_->field_store->getTimeIntegrationRules().end(), [](const auto& rule_and_mapping) {
                    return rule_and_mapping.first && rule_and_mapping.first->requiresInitialAccelerationSolve();
                  });

  if (time_info.cycle() == 0 && cycle_zero_system_ && requires_cycle_zero_solve) {
    auto cz_unknowns = cycle_zero_system_->solve(time_info);

    // Cycle zero system solves for initial acceleration which translates into test field.
    // Sync the solved unknowns back into current_states before main solve
    // Assuming cycle zero solve gives us the state for test field:
    std::string test_field_name =
        system_->field_store->getWeakFormReaction(cycle_zero_system_->weak_forms.front()->name());
    size_t test_field_state_idx = system_->field_store->getFieldIndex(test_field_name);
    size_t cz_unknown_idx = cycle_zero_system_->field_store->getUnknownIndex(test_field_name);
    current_states[test_field_state_idx] = cz_unknowns[cz_unknown_idx];
    system_->field_store->setField(test_field_state_idx, cz_unknowns[cz_unknown_idx]);
  }

  std::vector<FieldState> primary_unknowns = system_->solve(time_info);

  // Create states for reaction computation: newly solved primary unknowns + current states
  std::vector<FieldState> states_for_reactions = current_states;
  for (const auto& [rule, mapping] : system_->field_store->getTimeIntegrationRules()) {
    size_t u_idx = system_->field_store->getFieldIndex(mapping.primary_name);
    size_t unknown_idx = system_->field_store->getUnknownIndex(mapping.primary_name);
    FieldState u_new = primary_unknowns[unknown_idx];
    states_for_reactions[u_idx] = u_new;
  }

  // Compute reactions using newly solved unknowns but BEFORE time integration state updates
  std::vector<ReactionState> reactions = system_->computeReactions(time_info, states_for_reactions);

  // Now do time integration to compute corrected velocities/accelerations and update all states
  const auto& all_current_states = system_->field_store->getAllFields();
  std::vector<FieldState> new_states = current_states;
  for (size_t i = 0; i < current_states.size(); ++i) {
    new_states[i] = all_current_states[i];
  }

  for (const auto& [rule, mapping] : system_->field_store->getTimeIntegrationRules()) {
    size_t u_idx = system_->field_store->getFieldIndex(mapping.primary_name);
    size_t unknown_idx = system_->field_store->getUnknownIndex(mapping.primary_name);
    FieldState u_new = primary_unknowns[unknown_idx];
    new_states[u_idx] = u_new;

    std::vector<FieldState> rule_inputs;
    rule_inputs.push_back(u_new);  // u_{n+1}
    if (rule->num_args() >= 2) {
      rule_inputs.push_back(current_states[u_idx]);  // u_n
    }

    if (rule->num_args() >= 3 && !mapping.dot_name.empty()) {
      size_t v_idx = system_->field_store->getFieldIndex(mapping.dot_name);
      rule_inputs.push_back(current_states[v_idx]);
    }

    if (rule->num_args() >= 4 && !mapping.ddot_name.empty()) {
      size_t a_idx = system_->field_store->getFieldIndex(mapping.ddot_name);
      rule_inputs.push_back(current_states[a_idx]);
    }

    if (!mapping.dot_name.empty()) {
      size_t v_idx = system_->field_store->getFieldIndex(mapping.dot_name);
      new_states[v_idx] = rule->corrected_dot(time_info, rule_inputs);
    }

    if (!mapping.ddot_name.empty()) {
      size_t a_idx = system_->field_store->getFieldIndex(mapping.ddot_name);
      new_states[a_idx] = rule->corrected_ddot(time_info, rule_inputs);
    }

    if (!mapping.history_name.empty()) {
      size_t hist_idx = system_->field_store->getFieldIndex(mapping.history_name);
      new_states[hist_idx] = u_new;
    }
  }

  return {new_states, reactions};
}

}  // namespace smith
