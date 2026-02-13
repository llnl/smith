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
#include "gretl/wang_checkpoint_strategy.hpp"

namespace smith {

FieldStore::FieldStore(std::shared_ptr<Mesh> mesh, size_t storage_size)
    : mesh_(mesh),
      graph_(std::make_shared<gretl::DataStore>(std::make_unique<gretl::WangCheckpointStrategy>(storage_size)))
{
}

std::shared_ptr<DirichletBoundaryConditions> FieldStore::addBoundaryConditions(FEFieldPtr field)
{
  boundary_conditions_.push_back(std::make_shared<DirichletBoundaryConditions>(mesh_->mfemParMesh(), field->space()));
  return boundary_conditions_.back();
}

void FieldStore::addWeakFormUnknownArg(std::string weak_form_name, std::string argument_name, size_t argument_index)
{
  FieldLabel argument_name_and_index{.field_name = argument_name, .field_index_in_residual = argument_index};
  if (weak_form_name_to_unknown_name_index_.count(weak_form_name)) {
    weak_form_name_to_unknown_name_index_.at(weak_form_name).push_back(argument_name_and_index);
  } else {
    weak_form_name_to_unknown_name_index_[weak_form_name] = std::vector<FieldLabel>{argument_name_and_index};
  }
}

void FieldStore::addWeakFormArg(std::string weak_form_name, std::string argument_name, size_t argument_index)
{
  // Store the field name instead of index to avoid confusion between states_ and params_ indices
  if (weak_form_name_to_field_names_.count(weak_form_name)) {
    weak_form_name_to_field_names_.at(weak_form_name).push_back(argument_name);
  } else {
    weak_form_name_to_field_names_[weak_form_name] = std::vector<std::string>{argument_name};
  }
  SLIC_ERROR_IF(argument_index + 1 != weak_form_name_to_field_names_.at(weak_form_name).size(),
                "Invalid order for adding weak form arguments.");
}

void FieldStore::printMap()
{
  for (auto& keyval : weak_form_name_to_unknown_name_index_) {
    std::cout << "for residual: " << keyval.first << " ";
    for (auto& name_index : keyval.second) {
      std::cout << "arg " << name_index.field_name << " at " << name_index.field_index_in_residual << ", ";
    }
    std::cout << std::endl;
  }
}

std::vector<std::vector<size_t>> FieldStore::indexMap(const std::vector<std::string>& residual_names) const
{
  std::vector<std::vector<size_t>> block_indices(residual_names.size());

  for (size_t res_i = 0; res_i < residual_names.size(); ++res_i) {
    std::vector<size_t>& res_indices = block_indices[res_i];
    res_indices = std::vector<size_t>(num_unknowns_, invalid_block_index);
    const std::string& res_name = residual_names[res_i];
    const auto& arg_info = weak_form_name_to_unknown_name_index_.at(res_name);

    for (const auto& field_name_and_arg_index : arg_info) {
      const std::string field_name = field_name_and_arg_index.field_name;
      size_t unknown_index = to_unknown_index_.at(field_name);
      SLIC_ASSERT(unknown_index < num_unknowns_);
      res_indices[unknown_index] = field_name_and_arg_index.field_index_in_residual;
    }
  }

  return block_indices;
}

std::vector<const BoundaryConditionManager*> FieldStore::getBoundaryConditionManagers() const
{
  std::vector<const BoundaryConditionManager*> bcs;
  for (auto& bc : boundary_conditions_) {
    bcs.push_back(&bc->getBoundaryConditionManager());
  }
  return bcs;
}

size_t FieldStore::getFieldIndex(const std::string& field_name) const
{
  if (to_states_index_.count(field_name)) {
    return to_states_index_.at(field_name);
  }
  if (to_params_index_.count(field_name)) {
    return to_params_index_.at(field_name);
  }
  SLIC_ERROR("Field or parameter '" << field_name << "' not found in getFieldIndex");
  return 0;  // unreachable
}

FieldState FieldStore::getField(const std::string& field_name) const
{
  // Check if it's a state field
  if (to_states_index_.count(field_name)) {
    size_t field_index = to_states_index_.at(field_name);
    return states_[field_index];
  }
  // Otherwise check if it's a parameter
  if (to_params_index_.count(field_name)) {
    size_t param_index = to_params_index_.at(field_name);
    return params_[param_index];
  }
  SLIC_ERROR("Field or parameter '" << field_name << "' not found");
  return states_[0];  // unreachable, but needed for compilation
}

FieldState FieldStore::getParameter(const std::string& param_name) const
{
  size_t param_index = to_params_index_.at(param_name);
  return params_[param_index];
}

void FieldStore::setField(const std::string& field_name, FieldState updated_field)
{
  size_t field_index = getFieldIndex(field_name);
  states_[field_index] = updated_field;
}

FieldState FieldStore::getShapeDisp() const { return shape_disp_[0]; }

const std::vector<FieldState>& FieldStore::getAllFields() const { return states_; }

std::vector<FieldState> FieldStore::getStates(const std::string& weak_form_name) const
{
  // Validate that weak form is registered
  SLIC_ERROR_ROOT_IF(weak_form_name_to_field_names_.count(weak_form_name) == 0,
                     axom::fmt::format("Weak form '{}' not found in FieldStore. Did you forget to call addResidual()?",
                                       weak_form_name));

  auto field_names = weak_form_name_to_field_names_.at(weak_form_name);
  std::vector<FieldState> fields_for_residual;
  for (auto& name : field_names) {
    // Validate that field exists
    SLIC_ERROR_ROOT_IF(
        to_states_index_.count(name) == 0 && to_params_index_.count(name) == 0,
        axom::fmt::format("Field '{}' (required by weak form '{}') not found in FieldStore", name, weak_form_name));

    // Only include state fields, not parameters
    // Parameters are passed separately to avoid duplication in block_solve
    if (to_states_index_.count(name)) {
      fields_for_residual.push_back(getField(name));
    }
  }
  return fields_for_residual;
}

std::vector<FieldState> FieldStore::getStatesFromVectors(const std::string& weak_form_name,
                                                         const std::vector<FieldState>& state_fields,
                                                         const std::vector<FieldState>& param_fields) const
{
  // Validate that weak form is registered
  SLIC_ERROR_ROOT_IF(weak_form_name_to_field_names_.count(weak_form_name) == 0,
                     axom::fmt::format("Weak form '{}' not found in FieldStore. Did you forget to call addResidual()?",
                                       weak_form_name));

  auto field_names = weak_form_name_to_field_names_.at(weak_form_name);
  std::vector<FieldState> fields_for_residual;
  for (auto& name : field_names) {
    // Check if it's a state field
    if (to_states_index_.count(name)) {
      size_t idx = to_states_index_.at(name);
      SLIC_ERROR_ROOT_IF(idx >= state_fields.size(),
                         axom::fmt::format("State field index {} out of bounds (size={}) for field '{}'", idx,
                                           state_fields.size(), name));
      fields_for_residual.push_back(state_fields[idx]);
    }
    // Otherwise check if it's a parameter
    else if (to_params_index_.count(name)) {
      size_t idx = to_params_index_.at(name);
      SLIC_ERROR_ROOT_IF(idx >= param_fields.size(),
                         axom::fmt::format("Parameter field index {} out of bounds (size={}) for field '{}'", idx,
                                           param_fields.size(), name));
      fields_for_residual.push_back(param_fields[idx]);
    } else {
      SLIC_ERROR_ROOT(axom::fmt::format("Field or parameter '{}' (required by weak form '{}') not found in FieldStore",
                                        name, weak_form_name));
    }
  }
  return fields_for_residual;
}

const std::shared_ptr<smith::Mesh>& FieldStore::getMesh() const { return mesh_; }

const std::vector<std::pair<std::shared_ptr<TimeIntegrationRule>, FieldStore::TimeIntegrationMapping>>&
FieldStore::getTimeIntegrationRules() const
{
  return time_integration_rules_;
}

size_t FieldStore::getUnknownIndex(const std::string& field_name) const { return to_unknown_index_.at(field_name); }

void FieldStore::setField(size_t index, FieldState updated_field) { states_[index] = updated_field; }

void FieldStore::addWeakFormTestField(std::string weak_form_name, std::string field_name)
{
  weak_form_to_test_field_[weak_form_name] = field_name;
}

std::string FieldStore::getWeakFormTestField(const std::string& weak_form_name) const
{
  return weak_form_to_test_field_.at(weak_form_name);
}

MultiPhysicsTimeIntegrator::MultiPhysicsTimeIntegrator(std::shared_ptr<FieldStore> field_store,
                                                       const std::vector<std::shared_ptr<WeakForm>>& weak_forms,
                                                       std::shared_ptr<smith::DifferentiableBlockSolver> solver)
    : field_store_(field_store), weak_forms_(weak_forms), solver_(solver)
{
}

std::pair<std::vector<FieldState>, std::vector<ReactionState>> MultiPhysicsTimeIntegrator::advanceState(
    const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
    const std::vector<FieldState>& params) const
{
  // Sync FieldStore with input states
  for (size_t i = 0; i < states.size(); ++i) {
    field_store_->setField(i, states[i]);
  }

  std::vector<FieldState> primary_unknowns = solve(weak_forms_, *field_store_, solver_.get(), time_info, params);

  // Compute reactions BEFORE time integration updates, using newly solved primal unknowns and old states
  // Build state vector with: new primary unknowns + old states from input
  std::vector<FieldState> states_for_reactions = states;
  for (const auto& [rule, mapping] : field_store_->getTimeIntegrationRules()) {
    size_t u_idx = field_store_->getFieldIndex(mapping.primary_name);
    size_t unknown_idx = field_store_->getUnknownIndex(mapping.primary_name);
    FieldState u_new = primary_unknowns[unknown_idx];
    states_for_reactions[u_idx] = u_new;  // Update primary unknown with newly solved value
    // Keep old states (displacement_old, temperature_old) from input 'states'
  }

  std::vector<ReactionState> reactions;
  for (const auto& wf : weak_forms_) {
    std::vector<FieldState> wf_fields = field_store_->getStatesFromVectors(wf->name(), states_for_reactions, params);
    std::string test_field_name = field_store_->getWeakFormTestField(wf->name());
    size_t test_field_idx = field_store_->getFieldIndex(test_field_name);
    FieldState test_field = states_for_reactions[test_field_idx];
    reactions.push_back(smith::evaluateWeakForm(wf, time_info, shape_disp, wf_fields, test_field));
  }

  // Now do time integration to compute corrected velocities/accelerations and update all states
  std::vector<FieldState> new_states = states;

  for (const auto& [rule, mapping] : field_store_->getTimeIntegrationRules()) {
    size_t u_idx = field_store_->getFieldIndex(mapping.primary_name);
    size_t unknown_idx = field_store_->getUnknownIndex(mapping.primary_name);
    FieldState u_new = primary_unknowns[unknown_idx];
    new_states[u_idx] = u_new;

    std::vector<FieldState> rule_inputs;
    rule_inputs.push_back(u_new);          // u_{n+1}
    rule_inputs.push_back(states[u_idx]);  // u_n

    if (!mapping.dot_name.empty()) {
      size_t v_idx = field_store_->getFieldIndex(mapping.dot_name);
      rule_inputs.push_back(states[v_idx]);
    }

    if (!mapping.ddot_name.empty()) {
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
