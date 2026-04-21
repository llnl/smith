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
                                                       std::shared_ptr<SystemBase> cycle_zero_system,
                                                       std::vector<std::shared_ptr<SystemBase>> post_solve_systems)
    : system_(system), cycle_zero_system_(cycle_zero_system), post_solve_systems_(std::move(post_solve_systems))
{
}

void MultiphysicsTimeIntegrator::addPostSolveSystem(std::shared_ptr<SystemBase> system)
{
  post_solve_systems_.push_back(std::move(system));
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
    auto cycle_zero_unknowns = cycle_zero_system_->solve(time_info);

    // Cycle zero system solves for the initial acceleration, but by convention the solved value
    // is returned through the first (and only) block of the cycle-zero subsystem — the weak form
    // uses an aliased unknown trial space that matches the acceleration test space. Copy that
    // single result into the acceleration state slot for the main solve.
    SLIC_ERROR_ROOT_IF(cycle_zero_unknowns.size() != 1,
                       "Cycle zero system is expected to be a single-block solve producing one unknown");
    std::string test_field_name =
        system_->field_store->getWeakFormReaction(cycle_zero_system_->weak_forms.front()->name());
    size_t test_field_state_idx = system_->field_store->getFieldIndex(test_field_name);
    current_states[test_field_state_idx] = cycle_zero_unknowns[0];
    system_->field_store->setField(test_field_state_idx, cycle_zero_unknowns[0]);
  }

  std::vector<FieldState> primary_unknowns = system_->solve(time_info);

  // Build a map from the main system's unknown names to their position in primary_unknowns.
  // Entries in the shared FieldStore's time integration rules that belong to post-solve
  // subsystems (e.g. stress projection) are NOT present here and must be skipped by downstream
  // lookups that walk getTimeIntegrationRules().
  std::map<std::string, size_t> main_unknown_name_to_local_idx;
  for (size_t i = 0; i < system_->weak_forms.size(); ++i) {
    const std::string wf_name = system_->weak_forms[i]->name();
    const std::string reaction_name = system_->field_store->getWeakFormReaction(wf_name);
    main_unknown_name_to_local_idx[reaction_name] = i;
  }

  // Create states for reaction computation: newly solved primary unknowns + current states
  std::vector<FieldState> states_for_reactions = current_states;
  for (const auto& [rule, mapping] : system_->field_store->getTimeIntegrationRules()) {
    auto it = main_unknown_name_to_local_idx.find(mapping.primary_name);
    if (it == main_unknown_name_to_local_idx.end()) {
      continue;  // rule belongs to a post-solve subsystem, not the main solve
    }
    size_t u_idx = system_->field_store->getFieldIndex(mapping.primary_name);
    FieldState u_new = primary_unknowns[it->second];
    states_for_reactions[u_idx] = u_new;
  }

  // Compute reactions using newly solved unknowns but BEFORE time integration state updates
  std::vector<ReactionState> reactions = system_->computeReactions(time_info, states_for_reactions);

  // Sync field_store with newly solved primary unknowns so post-solve systems (e.g. stress
  // projection) read the current displacement rather than the pre-solve snapshot.
  for (const auto& [rule, mapping] : system_->field_store->getTimeIntegrationRules()) {
    auto it = main_unknown_name_to_local_idx.find(mapping.primary_name);
    if (it == main_unknown_name_to_local_idx.end()) {
      continue;
    }
    size_t u_idx = system_->field_store->getFieldIndex(mapping.primary_name);
    system_->field_store->setField(u_idx, primary_unknowns[it->second]);
  }

  // Solve post-solve systems (e.g. stress projection for output) and sync their results back
  // into the shared field_store so getAllFields() returns the updated values for new_states.
  for (const auto& ps : post_solve_systems_) {
    auto ps_unknowns = ps->solve(time_info);
    for (size_t i = 0; i < ps->weak_forms.size(); ++i) {
      const std::string reaction_name = ps->field_store->getWeakFormReaction(ps->weak_forms[i]->name());
      size_t u_idx = ps->field_store->getFieldIndex(reaction_name);
      ps->field_store->setField(u_idx, ps_unknowns[i]);
    }
  }

  // Now do time integration to compute corrected velocities/accelerations and update all states
  const auto& all_current_states = system_->field_store->getAllFields();
  std::vector<FieldState> new_states = current_states;
  for (size_t i = 0; i < current_states.size(); ++i) {
    new_states[i] = all_current_states[i];
  }

  for (const auto& [rule, mapping] : system_->field_store->getTimeIntegrationRules()) {
    auto it = main_unknown_name_to_local_idx.find(mapping.primary_name);
    if (it == main_unknown_name_to_local_idx.end()) {
      continue;  // rule belongs to a post-solve subsystem, not the main solve
    }
    size_t u_idx = system_->field_store->getFieldIndex(mapping.primary_name);
    FieldState u_new = primary_unknowns[it->second];
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

  // Copy solve-state → history for post-solve fields (e.g. stress_solve_state → stress).
  // The main loop skipped these rules; their primary fields are already correct in new_states
  // (populated from all_current_states above), so only the history field needs updating.
  for (const auto& [rule, mapping] : system_->field_store->getTimeIntegrationRules()) {
    if (main_unknown_name_to_local_idx.count(mapping.primary_name)) {
      continue;  // already handled by main time integration loop above
    }
    if (!mapping.history_name.empty()) {
      size_t primary_idx = system_->field_store->getFieldIndex(mapping.primary_name);
      size_t hist_idx = system_->field_store->getFieldIndex(mapping.history_name);
      new_states[hist_idx] = new_states[primary_idx];
    }
  }

  return {new_states, reactions};
}

}  // namespace smith
