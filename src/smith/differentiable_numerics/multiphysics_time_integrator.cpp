// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"

namespace smith {

FieldStore::FieldStore(std::shared_ptr<Mesh> mesh, size_t storage_size)
    : mesh_(mesh), data_store_(std::make_shared<gretl::DataStore>(storage_size))
{
}

std::shared_ptr<DirichletBoundaryConditions> FieldStore::addBoundaryConditions(FEFieldPtr field) {
  boundary_conditions_.push_back(
      std::make_shared<DirichletBoundaryConditions>(mesh_->mfemParMesh(), field->space()));
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
  size_t field_index = to_fields_index_.at(argument_name);
  if (weak_form_name_to_field_indices_.count(weak_form_name)) {
    weak_form_name_to_field_indices_.at(weak_form_name).push_back(field_index);
  } else {
    weak_form_name_to_field_indices_[weak_form_name] = std::vector<size_t>{field_index};
  }
  SLIC_ERROR_IF(argument_index + 1 != weak_form_name_to_field_indices_.at(weak_form_name).size(),
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

size_t FieldStore::getFieldIndex(const std::string& field_name) const { return to_fields_index_.at(field_name); }

const FieldState& FieldStore::getField(const std::string& field_name) const
{
  size_t field_index = getFieldIndex(field_name);
  return fields_[field_index];
}

void FieldStore::setField(const std::string& field_name, FieldState updated_field)
{
  size_t field_index = getFieldIndex(field_name);
  fields_[field_index] = updated_field;
}

const FieldState& FieldStore::getShapeDisp() const { return shape_disp_[0]; }

const std::vector<FieldState>& FieldStore::getAllFields() const { return fields_; }

std::vector<FieldState> FieldStore::getFields(const std::string& weak_form_name) const
{
  auto unknown_field_indices = weak_form_name_to_field_indices_.at(weak_form_name);
  std::vector<FieldState> fields_for_residual;
  for (auto& i : unknown_field_indices) {
    fields_for_residual.push_back(fields_[i]);
  }
  return fields_for_residual;
}

const std::shared_ptr<smith::Mesh>& FieldStore::getMesh() const { return mesh_; }

std::vector<FieldState> solve(const std::vector<WeakForm*>& weak_forms, const FieldStore& field_store,
                              const DifferentiableBlockSolver* solver, const TimeInfo& time_info)
{
  std::vector<std::string> weak_form_names;
  for (const auto& wf : weak_forms) {
    weak_form_names.push_back(wf->name());
  }
  std::vector<std::vector<size_t>> index_map = field_store.indexMap(weak_form_names);

  std::vector<std::vector<FieldState>> inputs;
  for (size_t i = 0; i < weak_forms.size(); ++i) {
    std::string wf_name = weak_forms[i]->name();
    std::vector<FieldState> fields_for_wk = field_store.getFields(wf_name);
    inputs.push_back(fields_for_wk);
  }
  std::vector<std::vector<FieldState>> params(weak_forms.size());

  return block_solve(weak_forms, index_map, field_store.getShapeDisp(), inputs, params, time_info, solver,
                     field_store.getBoundaryConditionManagers());
}

}  // namespace smith
