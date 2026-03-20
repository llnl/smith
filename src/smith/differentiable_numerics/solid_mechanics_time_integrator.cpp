// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/solid_mechanics_time_integrator.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"

#include "smith/differentiable_numerics/coupled_system_solver.hpp"

namespace smith {

SolidMechanicsTimeIntegrator::SolidMechanicsTimeIntegrator(std::shared_ptr<FieldStore> field_store,
                                                           std::shared_ptr<WeakForm> solid_weak_form,
                                                           std::shared_ptr<WeakForm> cycle_zero_weak_form,
                                                           std::shared_ptr<smith::CoupledSystemSolver> solver)
    : field_store_(field_store), cycle_zero_weak_form_(cycle_zero_weak_form), solver_(solver)
{
  std::vector<std::shared_ptr<WeakForm>> weak_forms = {solid_weak_form};
  integrator_ = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, solver);
}

std::pair<std::vector<FieldState>, std::vector<ReactionState>> SolidMechanicsTimeIntegrator::advanceState(
    const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
    const std::vector<FieldState>& params) const
{
  // Handle initial acceleration solve at cycle 0
  std::vector<FieldState> current_states = states;
  if (time_info.cycle() == 0 && cycle_zero_weak_form_) {
    // Sync FieldStore with input states
    for (size_t i = 0; i < states.size(); ++i) {
      field_store_->setField(i, states[i]);
    }

    // The test field of cycle_zero_weak_form is the field we need to solve for (acceleration)
    std::string test_field_name = field_store_->getWeakFormReaction(cycle_zero_weak_form_->name());

    // Get the weak form's state fields
    std::vector<FieldState> wf_fields = field_store_->getStates(cycle_zero_weak_form_->name());

    // Find which argument index corresponds to the test field (the unknown we're solving for)
    FieldState test_field = field_store_->getField(test_field_name);
    size_t test_field_idx_in_wf = invalid_block_index;
    for (size_t j = 0; j < wf_fields.size(); ++j) {
      if (wf_fields[j].get() == test_field.get()) {
        test_field_idx_in_wf = j;
        break;
      }
    }
    SLIC_ERROR_IF(test_field_idx_in_wf == invalid_block_index, "Test field '" << test_field_name
                                                                              << "' not found in weak form '"
                                                                              << cycle_zero_weak_form_->name() << "'");

    // Set up block solve for this single unknown
    std::vector<WeakForm*> wf_ptrs = {cycle_zero_weak_form_.get()};
    std::vector<std::vector<size_t>> block_indices = {{test_field_idx_in_wf}};

    // Get boundary conditions (assume first BC manager for now)
    std::vector<const BoundaryConditionManager*> bcs;
    auto all_bcs = field_store_->getBoundaryConditionManagers();
    if (!all_bcs.empty()) {
      bcs.push_back(all_bcs[0]);
    }

    std::vector<std::vector<FieldState>> states_vec = {wf_fields};
    std::vector<std::vector<FieldState>> params_vec = {params};

    auto result = solver_->solve(wf_ptrs, block_indices, shape_disp, states_vec, params_vec, time_info, bcs);

    // Update the acceleration field in our current states
    size_t test_field_state_idx = field_store_->getFieldIndex(test_field_name);
    current_states[test_field_state_idx] = result[0];
  }

  // Now perform the regular time step
  auto [new_states, reactions] = integrator_->advanceState(time_info, shape_disp, current_states, params);

  return {new_states, reactions};
}

}  // namespace smith
