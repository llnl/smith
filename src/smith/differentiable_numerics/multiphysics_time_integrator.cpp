// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/coupled_system_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/reaction.hpp"

#include <stdexcept>

namespace smith {

MultiphysicsTimeIntegrator::MultiphysicsTimeIntegrator(std::shared_ptr<FieldStore> field_store,
                                                       const std::vector<std::shared_ptr<WeakForm>>& weak_forms,
                                                       std::shared_ptr<smith::CoupledSystemSolver> solver,
                                                       std::shared_ptr<WeakForm> cycle_zero_weak_form,
                                                       std::shared_ptr<smith::CoupledSystemSolver> cycle_zero_solver)
    : field_store_(field_store),
      weak_forms_(weak_forms),
      solver_(solver),
      cycle_zero_weak_form_(cycle_zero_weak_form),
      cycle_zero_solver_(cycle_zero_solver)
{
}

std::pair<std::vector<FieldState>, std::vector<ReactionState>> MultiphysicsTimeIntegrator::advanceState(
    const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
    const std::vector<FieldState>& params) const
{
  std::vector<FieldState> current_states = states;

  // Handle initial acceleration solve at cycle 0
  if (time_info.cycle() == 0 && cycle_zero_weak_form_) {
    for (size_t i = 0; i < current_states.size(); ++i) {
      field_store_->setField(i, current_states[i]);
    }

    std::string test_field_name = field_store_->getWeakFormReaction(cycle_zero_weak_form_->name());
    std::vector<FieldState> wf_fields = field_store_->getStates(cycle_zero_weak_form_->name());

    FieldState test_field = field_store_->getField(test_field_name);
    size_t test_field_idx_in_wf = invalid_block_index;
    for (size_t j = 0; j < wf_fields.size(); ++j) {
      if (wf_fields[j].get() == test_field.get()) {
        test_field_idx_in_wf = j;
        break;
      }
    }
    SLIC_ERROR_IF(test_field_idx_in_wf == invalid_block_index, "Test field '" << test_field_name
                                                                              << "' not found in cycle-zero weak form '"
                                                                              << cycle_zero_weak_form_->name() << "'");

    std::vector<WeakForm*> wf_ptrs = {cycle_zero_weak_form_.get()};
    std::vector<std::vector<size_t>> block_indices = {{test_field_idx_in_wf}};

    std::vector<const BoundaryConditionManager*> bcs;
    auto all_bcs = field_store_->getBoundaryConditionManagers();
    size_t test_field_unknown_idx = invalid_block_index;
    try {
      test_field_unknown_idx = field_store_->getUnknownIndex(test_field_name);
    } catch (const std::out_of_range&) {
      for (const auto& [rule, mapping] : field_store_->getTimeIntegrationRules()) {
        static_cast<void>(rule);
        if (mapping.primary_name == test_field_name || mapping.history_name == test_field_name ||
            mapping.dot_name == test_field_name || mapping.ddot_name == test_field_name) {
          test_field_unknown_idx = field_store_->getUnknownIndex(mapping.primary_name);
          break;
        }
      }
    }
    SLIC_ERROR_IF(test_field_unknown_idx == invalid_block_index,
                  "Could not map cycle-zero test field '" << test_field_name << "' to an independent unknown.");
    SLIC_ERROR_IF(test_field_unknown_idx >= all_bcs.size(),
                  "Cycle-zero test field '" << test_field_name << "' has unknown index " << test_field_unknown_idx
                                            << ", but only " << all_bcs.size() << " BC managers are registered.");
    bcs.push_back(all_bcs[test_field_unknown_idx]);

    std::vector<std::vector<FieldState>> states_vec = {wf_fields};
    std::vector<std::vector<FieldState>> params_vec = {params};

    auto& cz_solver = cycle_zero_solver_ ? cycle_zero_solver_ : solver_;
    auto result = cz_solver->solve(wf_ptrs, block_indices, shape_disp, states_vec, params_vec, time_info, bcs);

    size_t test_field_state_idx = field_store_->getFieldIndex(test_field_name);
    current_states[test_field_state_idx] = result[0];
  }

  // Sync FieldStore with (possibly updated) states
  for (size_t i = 0; i < current_states.size(); ++i) {
    field_store_->setField(i, current_states[i]);
  }

  std::vector<FieldState> primary_unknowns = solve(weak_forms_, *field_store_, solver_.get(), time_info, params);

  // Create states for reaction computation: newly solved primary unknowns + current states
  std::vector<FieldState> states_for_reactions = current_states;
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
    std::string test_field_name = field_store_->getWeakFormReaction(wf->name());
    size_t test_field_idx = field_store_->getFieldIndex(test_field_name);
    FieldState test_field = states_for_reactions[test_field_idx];
    reactions.push_back(smith::evaluateWeakForm(wf, time_info, shape_disp, wf_fields, test_field));
  }

  // Now do time integration to compute corrected velocities/accelerations and update all states
  const auto& all_current_states = field_store_->getAllFields();
  std::vector<FieldState> new_states = current_states;
  for (size_t i = 0; i < current_states.size(); ++i) {
    new_states[i] = all_current_states[i];
  }

  for (const auto& [rule, mapping] : field_store_->getTimeIntegrationRules()) {
    size_t u_idx = field_store_->getFieldIndex(mapping.primary_name);
    size_t unknown_idx = field_store_->getUnknownIndex(mapping.primary_name);
    FieldState u_new = primary_unknowns[unknown_idx];
    new_states[u_idx] = u_new;

    std::vector<FieldState> rule_inputs;
    rule_inputs.push_back(u_new);  // u_{n+1}
    if (rule->num_args() >= 2) {
      rule_inputs.push_back(current_states[u_idx]);  // u_n
    }

    if (rule->num_args() >= 3 && !mapping.dot_name.empty()) {
      size_t v_idx = field_store_->getFieldIndex(mapping.dot_name);
      rule_inputs.push_back(current_states[v_idx]);
    }

    if (rule->num_args() >= 4 && !mapping.ddot_name.empty()) {
      size_t a_idx = field_store_->getFieldIndex(mapping.ddot_name);
      rule_inputs.push_back(current_states[a_idx]);
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
                              const CoupledSystemSolver* solver, const TimeInfo& time_info,
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
  return solver->solve(weak_form_ptrs, index_map, field_store.getShapeDisp(), inputs, wk_params, time_info,
                       field_store.getBoundaryConditionManagers());
}

}  // namespace smith
