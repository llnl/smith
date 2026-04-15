// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/combined_system.hpp"

namespace smith {

std::vector<FieldState> CombinedSystem::solve(const TimeInfo& time_info) const
{
  // Snapshot the current solve-state unknowns for convergence checking.
  // One mfem::Vector copy per combined weak form (indexed same as weak_forms).
  std::vector<mfem::Vector> prev(weak_forms.size());
  for (size_t k = 0; k < weak_forms.size(); ++k) {
    const std::string reaction_name = field_store->getWeakFormReaction(weak_forms[k]->name());
    size_t u_idx = field_store->getFieldIndex(reaction_name);
    prev[k] = mfem::Vector(*field_store->getAllFields()[u_idx].get());
  }

  // Staggered iteration: each sub-system reads from the shared FieldStore and writes its
  // updated unknowns back before the next sub-system reads.
  for (int iter = 0; iter < max_stagger_iters; ++iter) {
    for (const auto& sub : subsystems) {
      auto sub_unknowns = sub->solve(time_info);

      for (size_t i = 0; i < sub->weak_forms.size(); ++i) {
        const std::string reaction_name =
            field_store->getWeakFormReaction(sub->weak_forms[i]->name());
        size_t u_idx = field_store->getFieldIndex(reaction_name);
        field_store->setField(u_idx, sub_unknowns[i]);
      }
    }

    // Convergence check: relative change in each unknown must be below stagger_tolerance.
    double max_change = 0.0;
    for (size_t k = 0; k < weak_forms.size(); ++k) {
      const std::string reaction_name = field_store->getWeakFormReaction(weak_forms[k]->name());
      size_t u_idx = field_store->getFieldIndex(reaction_name);
      mfem::Vector curr(*field_store->getAllFields()[u_idx].get());
      mfem::Vector diff(curr);
      diff -= prev[k];
      const double change = diff.Norml2() / (1.0 + curr.Norml2());
      if (change > max_change) max_change = change;
      prev[k] = curr;
    }
    if (max_change < stagger_tolerance) break;
  }

  // Return one FieldState per combined weak_form.
  std::vector<FieldState> result;
  result.reserve(weak_forms.size());
  for (const auto& wf : weak_forms) {
    const std::string reaction_name = field_store->getWeakFormReaction(wf->name());
    size_t u_idx = field_store->getFieldIndex(reaction_name);
    result.push_back(field_store->getAllFields()[u_idx]);
  }
  return result;
}

}  // namespace smith
