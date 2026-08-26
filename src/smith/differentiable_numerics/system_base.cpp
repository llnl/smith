// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"

namespace smith {

std::vector<FieldState> SystemBase::solve(const TimeInfo& time_info) const
{
  std::vector<std::string> weak_form_names;
  for (const auto& wf : weak_forms) {
    weak_form_names.push_back(wf->name());
  }
  std::vector<std::vector<size_t>> index_map = field_store->indexMap(weak_form_names);

  std::vector<std::vector<FieldState>> inputs;
  if (!solve_input_field_names.empty()) {
    SLIC_ERROR_IF(solve_input_field_names.size() != weak_forms.size(),
                  "solve_input_field_names size must match weak_forms size");
  }
  for (size_t i = 0; i < weak_forms.size(); ++i) {
    std::vector<FieldState> fields_for_wk;
    if (solve_input_field_names.empty()) {
      fields_for_wk = field_store->getStates(weak_forms[i]->name());
    } else {
      for (const auto& field_name : solve_input_field_names[i]) {
        fields_for_wk.push_back(field_store->getField(field_name));
      }
    }
    inputs.push_back(fields_for_wk);
  }

  std::vector<std::string> bc_field_names;
  if (!solve_result_field_names.empty()) {
    SLIC_ERROR_IF(solve_result_field_names.size() != weak_forms.size(),
                  "solve_result_field_names size must match weak_forms size");
    bc_field_names = solve_result_field_names;
    for (size_t row = 0; row < weak_forms.size(); ++row) {
      for (size_t col = 0; col < weak_forms.size(); ++col) {
        index_map[row][col] = invalid_block_index;
        for (size_t arg = 0; arg < inputs[row].size(); ++arg) {
          if (inputs[row][arg].get()->name() == solve_result_field_names[col]) {
            index_map[row][col] = arg;
            break;
          }
        }
      }
      SLIC_ERROR_IF(index_map[row][row] == invalid_block_index, "Requested solve result field '"
                                                                    << solve_result_field_names[row]
                                                                    << "' is not an argument of weak form '"
                                                                    << weak_form_names[row] << "'");
    }
  }

  auto params = field_store->getParameterFields();
  std::vector<std::vector<FieldState>> wk_params(weak_forms.size(), params);

  std::vector<WeakForm*> weak_form_ptrs;
  for (auto& p : weak_forms) {
    weak_form_ptrs.push_back(p.get());
  }
  auto bc_managers = solve_result_field_names.empty()
                         ? field_store->getBoundaryConditionManagers(weak_form_names)
                         : field_store->getBoundaryConditionManagersForFields(bc_field_names);
  return solver->solve(weak_form_ptrs, index_map, field_store->getShapeDisp(), inputs, wk_params, time_info,
                       bc_managers);
}

std::vector<ReactionState> SystemBase::computeReactions(const TimeInfo& time_info,
                                                        const std::vector<FieldState>& states_for_reactions) const
{
  std::vector<ReactionState> reactions;
  auto params = field_store->getParameterFields();
  for (const auto& wf : weak_forms) {
    std::vector<FieldState> wf_fields = field_store->getStatesFromVectors(wf->name(), states_for_reactions, params);
    std::string test_field_name = field_store->getWeakFormReaction(wf->name());
    size_t test_field_idx = field_store->getFieldIndex(test_field_name);
    FieldState test_field = states_for_reactions[test_field_idx];
    reactions.push_back(smith::evaluateWeakForm(wf, time_info, field_store->getShapeDisp(), wf_fields, test_field));
  }
  return reactions;
}

}  // namespace smith
