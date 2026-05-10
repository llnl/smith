// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"

namespace smith {

FieldStore::FieldStore(std::shared_ptr<Mesh> mesh, size_t storage_size, std::string prepend_name)
    : mesh_(mesh),
      graph_(std::make_shared<gretl::DataStore>(std::make_unique<gretl::WangCheckpointStrategy>(storage_size))),
      prepend_name_(std::move(prepend_name))
{
}

std::string FieldStore::prefix(const std::string& base) const
{
  if (prepend_name_.empty()) {
    return base;
  }
  return prepend_name_ + "_" + base;
}

std::shared_ptr<DirichletBoundaryConditions> FieldStore::addBoundaryConditions(FEFieldPtr field)
{
  return std::make_shared<DirichletBoundaryConditions>(mesh_->mfemParMesh(), field->space());
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
  // Build a local column space: each residual in the subsystem contributes one local column,
  // corresponding to its "self" diagonal unknown.  The self-unknown is preferably the residual's
  // reaction (test) field if that field appears in the unknown-arg list for this weak form;
  // otherwise fall back on the first unknown argument (handles cases like the cycle-zero
  // acceleration solve, where the reaction field is a dependent/history field).
  std::map<size_t, size_t> global_state_to_local_col;
  for (size_t res_i = 0; res_i < residual_names.size(); ++res_i) {
    const std::string& res_name = residual_names[res_i];
    size_t global_state_idx = invalid_block_index;

    std::string reaction_name;
    for (const auto& kv : weak_form_to_test_field_) {
      if (kv.first == res_name) {
        reaction_name = kv.second;
        break;
      }
    }

    // Check if the reaction field is one of the registered unknown args for this weak form.
    bool reaction_is_unknown = false;
    if (!reaction_name.empty() && weak_form_name_to_unknown_name_index_.count(res_name)) {
      for (const auto& label : weak_form_name_to_unknown_name_index_.at(res_name)) {
        if (label.field_name == reaction_name) {
          reaction_is_unknown = true;
          break;
        }
      }
    }

    if (reaction_is_unknown) {
      global_state_idx = to_states_index_.at(reaction_name);
    } else {
      const auto& arg_info = weak_form_name_to_unknown_name_index_.at(res_name);
      SLIC_ERROR_IF(arg_info.empty(),
                    "Weak form '" << res_name << "' has no unknown arguments; cannot build index map.");
      global_state_idx = to_states_index_.at(arg_info.front().field_name);
    }
    global_state_to_local_col[global_state_idx] = res_i;
  }

  std::vector<std::vector<size_t>> block_indices(residual_names.size());
  for (size_t res_i = 0; res_i < residual_names.size(); ++res_i) {
    std::vector<size_t>& res_indices = block_indices[res_i];
    res_indices = std::vector<size_t>(residual_names.size(), invalid_block_index);
    const std::string& res_name = residual_names[res_i];
    const auto& arg_info = weak_form_name_to_unknown_name_index_.at(res_name);

    for (const auto& field_name_and_arg_index : arg_info) {
      size_t global_state_index = to_states_index_.at(field_name_and_arg_index.field_name);
      auto it = global_state_to_local_col.find(global_state_index);
      if (it != global_state_to_local_col.end()) {
        res_indices[it->second] = field_name_and_arg_index.field_index_in_residual;
      }
      // else: field belongs to a different subsystem; treat as fixed input here.
    }
  }

  return block_indices;
}

std::vector<const BoundaryConditionManager*> FieldStore::getBoundaryConditionManagers(
    const std::vector<std::string>& weak_form_names) const
{
  struct BoundaryConditionRef {
    std::string primary_name;
    bool use_second_derivative;
  };
  std::map<std::string, BoundaryConditionRef> field_to_primary;
  for (const auto& [_rule, mapping] : time_integration_rules_) {
    if (!mapping.primary_name.empty()) {
      field_to_primary[mapping.primary_name] = {mapping.primary_name, false};
    }
    if (!mapping.history_name.empty()) {
      field_to_primary[mapping.history_name] = {mapping.primary_name, false};
    }
    if (!mapping.ddot_name.empty()) {
      field_to_primary[mapping.ddot_name] = {mapping.primary_name, true};
    }
  }

  std::vector<const BoundaryConditionManager*> bcs;
  for (const auto& wf_name : weak_form_names) {
    const std::string reaction_name = getWeakFormReaction(wf_name);

    // Direct DBC entry takes precedence (e.g. an independent unknown like stress with its own BC).
    auto direct = boundary_conditions_.find(reaction_name);
    if (direct != boundary_conditions_.end()) {
      bcs.push_back(&direct->second->getBoundaryConditionManager());
      continue;
    }

    // Otherwise resolve via the time-integration mapping that owns this reaction field.
    auto ref_it = field_to_primary.find(reaction_name);
    if (ref_it == field_to_primary.end()) {
      bcs.push_back(nullptr);
      continue;
    }
    auto primary_it = boundary_conditions_.find(ref_it->second.primary_name);
    if (primary_it == boundary_conditions_.end()) {
      bcs.push_back(nullptr);
      continue;
    }
    const auto& dbc = *primary_it->second;
    bcs.push_back(ref_it->second.use_second_derivative ? &dbc.getSecondDerivativeManager()
                                                       : &dbc.getBoundaryConditionManager());
  }
  return bcs;
}

bool FieldStore::hasField(const std::string& field_name) const
{
  const auto resolved_name = resolveFieldName(field_name);
  if (to_states_index_.count(resolved_name)) return true;
  if (to_params_index_.count(resolved_name)) return true;
  if (!shape_disp_.empty() && shape_disp_[0].get()->name() == resolved_name) return true;
  return false;
}

size_t FieldStore::getFieldIndex(const std::string& field_name) const
{
  const auto resolved_name = resolveFieldName(field_name);
  if (to_states_index_.count(resolved_name)) {
    return to_states_index_.at(resolved_name);
  }
  if (to_params_index_.count(resolved_name)) {
    return to_params_index_.at(resolved_name);
  }
  SLIC_ERROR("Field or parameter '" << field_name << "' not found in getFieldIndex");
  return 0;  // unreachable
}

FieldState FieldStore::getField(const std::string& field_name) const
{
  const auto resolved_name = resolveFieldName(field_name);
  // Check if it's a state field
  if (to_states_index_.count(resolved_name)) {
    size_t field_index = to_states_index_.at(resolved_name);
    return states_[field_index];
  }
  // Otherwise check if it's a parameter
  if (to_params_index_.count(resolved_name)) {
    size_t param_index = to_params_index_.at(resolved_name);
    return params_[param_index];
  }
  SLIC_ERROR("Field or parameter '" << field_name << "' not found");
  return states_[0];  // unreachable, but needed for compilation
}

FieldState FieldStore::getParameter(const std::string& param_name) const
{
  const auto resolved_name = resolveFieldName(param_name);
  size_t param_index = to_params_index_.at(resolved_name);
  return params_[param_index];
}

void FieldStore::setField(const std::string& field_name, FieldState updated_field)
{
  const auto resolved_name = resolveFieldName(field_name);
  if (to_states_index_.count(resolved_name)) {
    states_[to_states_index_.at(resolved_name)] = updated_field;
    return;
  }
  if (to_params_index_.count(resolved_name)) {
    params_[to_params_index_.at(resolved_name)] = updated_field;
    return;
  }
  if (!shape_disp_.empty() && shape_disp_[0].get()->name() == resolved_name) {
    shape_disp_[0] = updated_field;
    return;
  }
  SLIC_ERROR("Field '" << field_name << "' not found in setField");
}

std::string FieldStore::resolveFieldName(const std::string& field_name) const
{
  if (to_states_index_.count(field_name) || to_params_index_.count(field_name)) {
    return field_name;
  }
  if (!shape_disp_.empty() && shape_disp_[0].get()->name() == field_name) {
    return field_name;
  }

  const auto prefixed_name = prefix(field_name);
  if (prefixed_name != field_name) {
    if (to_states_index_.count(prefixed_name) || to_params_index_.count(prefixed_name)) {
      return prefixed_name;
    }
    if (!shape_disp_.empty() && shape_disp_[0].get()->name() == prefixed_name) {
      return prefixed_name;
    }
  }

  return field_name;
}

FieldState FieldStore::getShapeDisp() const { return shape_disp_[0]; }

const std::vector<FieldState>& FieldStore::getAllFields() const { return states_; }

std::vector<FieldState> FieldStore::getStates(const std::string& weak_form_name) const
{
  // Validate that weak form is registered
  SLIC_ERROR_ROOT_IF(weak_form_name_to_field_names_.count(weak_form_name) == 0,
                     axom::fmt::format("Weak form '{}' not found in FieldStore. Did you forget to call addReaction()?",
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
                     axom::fmt::format("Weak form '{}' not found in FieldStore. Did you forget to call addReaction()?",
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

std::shared_ptr<DirichletBoundaryConditions> FieldStore::getBoundaryConditions(const std::string& field_name) const
{
  auto it = boundary_conditions_.find(field_name);
  if (it != boundary_conditions_.end()) {
    return it->second;
  }
  return nullptr;
}

const std::shared_ptr<gretl::DataStore>& FieldStore::graph() const { return graph_; }

const std::vector<std::pair<std::shared_ptr<TimeIntegrationRule>, FieldStore::TimeIntegrationMapping>>&
FieldStore::getTimeIntegrationRules() const
{
  return time_integration_rules_;
}

void FieldStore::setField(size_t index, FieldState updated_field) { states_[index] = updated_field; }

void FieldStore::markWeakFormInternal(const std::string& weak_form_name)
{
  internal_weak_forms_.insert(weak_form_name);
}

void FieldStore::addWeakFormReaction(std::string weak_form_name, std::string field_name)
{
  for (auto& kv : weak_form_to_test_field_) {
    if (kv.first == weak_form_name) {
      kv.second = field_name;
      return;
    }
  }
  weak_form_to_test_field_.push_back({weak_form_name, field_name});
}

std::string FieldStore::getWeakFormReaction(const std::string& weak_form_name) const
{
  for (const auto& kv : weak_form_to_test_field_) {
    if (kv.first == weak_form_name) {
      return kv.second;
    }
  }
  SLIC_ERROR("Reaction field not found for weak form " << weak_form_name);
  return "";
}

const std::vector<FieldState>& FieldStore::getParameterFields() const { return params_; }

const std::vector<FieldState>& FieldStore::getStateFields() const { return states_; }

std::vector<FieldState> FieldStore::getOutputFieldStates() const
{
  std::vector<FieldState> output;
  std::set<std::string> public_static_fields;
  for (const auto& [rule, mapping] : time_integration_rules_) {
    if (mapping.history_name.empty() && mapping.dot_name.empty() && mapping.ddot_name.empty()) {
      public_static_fields.insert(mapping.primary_name);
    }
  }
  for (size_t i = 0; i < states_.size(); ++i) {
    if (!is_solve_state_[i] || public_static_fields.count(states_[i].get()->name()) > 0) {
      output.push_back(states_[i]);
    }
  }
  return output;
}

std::vector<ReactionInfo> FieldStore::getReactionInfos() const
{
  std::vector<ReactionInfo> infos;
  for (const auto& kv : weak_form_to_test_field_) {
    const std::string& weak_form_name = kv.first;
    if (internal_weak_forms_.count(weak_form_name)) {
      continue;
    }
    const std::string& field_name = kv.second;
    infos.push_back({weak_form_name, &getField(field_name).get()->space()});
  }
  return infos;
}

}  // namespace smith
