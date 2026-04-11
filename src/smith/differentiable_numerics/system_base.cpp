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
  for (size_t i = 0; i < weak_forms.size(); ++i) {
    std::string wf_name = weak_forms[i]->name();
    std::vector<FieldState> fields_for_wk = field_store->getStates(wf_name);
    inputs.push_back(fields_for_wk);
  }
  auto params = field_store->getParameterFields();
  std::vector<std::vector<FieldState>> wk_params(weak_forms.size(), params);

  std::vector<WeakForm*> weak_form_ptrs;
  for (auto& p : weak_forms) {
    weak_form_ptrs.push_back(p.get());
  }
  return solver->solve(weak_form_ptrs, index_map, field_store->getShapeDisp(), inputs, wk_params, time_info,
                       field_store->getBoundaryConditionManagers());
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
