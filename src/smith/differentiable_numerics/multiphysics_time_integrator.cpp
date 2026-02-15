// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/reaction.hpp"

namespace smith {

MultiphysicsTimeIntegrator::MultiphysicsTimeIntegrator(std::shared_ptr<FieldStore> field_store,
                                                       const std::vector<std::shared_ptr<WeakForm>>& weak_forms,
                                                       std::shared_ptr<smith::DifferentiableBlockSolver> solver)
    : field_store_(field_store), weak_forms_(weak_forms), solver_(solver)
{
}

std::pair<std::vector<FieldState>, std::vector<ReactionState>> MultiphysicsTimeIntegrator::advanceState(
    const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
    const std::vector<FieldState>& params) const
{
  // Sync FieldStore with input states
  for (size_t i = 0; i < states.size(); ++i) {
    field_store_->setField(i, states[i]);
  }

  std::vector<FieldState> primary_unknowns = solve(weak_forms_, *field_store_, solver_.get(), time_info, params);

  // Create states for reaction computation: newly solved primary unknowns + original input states
  std::vector<FieldState> states_for_reactions = states;
  for (const auto& [rule, mapping] : field_store_->getTimeIntegrationRules()) {
    size_t u_idx = field_store_->getFieldIndex(mapping.primary_name);
    size_t unknown_idx = field_store_->getUnknownIndex(mapping.primary_name);
    FieldState u_new = primary_unknowns[unknown_idx];
    states_for_reactions[u_idx] = u_new;
  }

  // Compute reactions using newly solved unknowns but BEFORE time integration state updates
  std::vector<ReactionState> reactions;
  for (const auto& wf : weak_forms_) {
    std::vector<FieldState> wf_fields = field_store_->getStatesFromVectors(wf->name(), states_for_reactions, params);
    std::string test_field_name = field_store_->getWeakFormTestField(wf->name());
    size_t test_field_idx = field_store_->getFieldIndex(test_field_name);
    FieldState test_field = states_for_reactions[test_field_idx];
    reactions.push_back(smith::evaluateWeakForm(wf, time_info, shape_disp, wf_fields, test_field));
  }

  // Now do time integration to compute corrected velocities/accelerations and update all states
  const auto& all_current_states = field_store_->getAllFields();
  std::vector<FieldState> new_states = states;
  for (size_t i = 0; i < states.size(); ++i) {
    new_states[i] = all_current_states[i];
  }

  for (const auto& [rule, mapping] : field_store_->getTimeIntegrationRules()) {
    size_t u_idx = field_store_->getFieldIndex(mapping.primary_name);
    size_t unknown_idx = field_store_->getUnknownIndex(mapping.primary_name);
    FieldState u_new = primary_unknowns[unknown_idx];
    new_states[u_idx] = u_new;

    std::vector<FieldState> rule_inputs;
    rule_inputs.push_back(u_new);          // u_{n+1}
    if (rule->num_args() >= 2) {
      rule_inputs.push_back(states[u_idx]);  // u_n
    }

    if (rule->num_args() >= 3 && !mapping.dot_name.empty()) {
      size_t v_idx = field_store_->getFieldIndex(mapping.dot_name);
      rule_inputs.push_back(states[v_idx]);
    }

    if (rule->num_args() >= 4 && !mapping.ddot_name.empty()) {
      size_t a_idx = field_store_->getFieldIndex(mapping.ddot_name);
      rule_inputs.push_back(states[a_idx]);
    }

    if (!mapping.dot_name.empty()) {
      size_t v_idx = field_store_->getFieldIndex(mapping.dot_name);
      new_states[v_idx] = rule->corrected_dot(time_info, rule_inputs);
    }

    if (!mapping.ddot_name.empty()) {
      size_t a_idx = field_store_->getFieldIndex(mapping.ddot_name);
      new_states[a_idx] = rule->corrected_ddot(time_info, rule_inputs);
    }

    if (!mapping.history_name.empty()) {
      size_t hist_idx = field_store_->getFieldIndex(mapping.history_name);
      new_states[hist_idx] = u_new;
    }
  }

  return {new_states, reactions};
}

std::vector<FieldState> solve(const std::vector<std::shared_ptr<WeakForm>>& weak_forms, const FieldStore& field_store,
                              const DifferentiableBlockSolver* solver, const TimeInfo& time_info,
                              const std::vector<FieldState>& params)
{
  std::vector<std::string> weak_form_names;
  for (const auto& wf : weak_forms) {
    weak_form_names.push_back(wf->name());
  }
  std::vector<std::vector<size_t>> index_map = field_store.indexMap(weak_form_names);

  std::vector<std::vector<FieldState>> inputs;
  for (size_t i = 0; i < weak_forms.size(); ++i) {
    std::string wf_name = weak_forms[i]->name();
    std::vector<FieldState> fields_for_wk = field_store.getStates(wf_name);
    inputs.push_back(fields_for_wk);
  }
  std::vector<std::vector<FieldState>> wk_params(weak_forms.size(), params);

  std::vector<WeakForm*> weak_form_ptrs;
  for (auto& p : weak_forms) {
    weak_form_ptrs.push_back(p.get());
  }
  return block_solve(weak_form_ptrs, index_map, field_store.getShapeDisp(), inputs, wk_params, time_info, solver,
                     field_store.getBoundaryConditionManagers());
}

}  // namespace smith
