// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/physics/mesh.hpp"

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace smith {

class DirichletBoundaryConditions;
class BoundaryConditionManager;

/**
 * @brief Representation of a field type with a name and an optional unknown index.
 * @tparam Space The finite element space type.
 * @tparam Time The time integration type (unused by default).
 */
template <typename Space, typename Time = void*>
struct FieldType {
  /**
   * @brief Construct a new FieldType object.
   * @param n Name of the field.
   * @param unknown_index_ Index of the unknown in the solver (default: -1).
   */
  FieldType(std::string n, int unknown_index_ = -1) : name(n), unknown_index(unknown_index_) {}
  std::string name;   ///< Name of the field.
  int unknown_index;  ///< Index of the unknown in the solver.
};

/**
 * @brief Manages storage and metadata for fields, parameters, and weak forms.
 */
struct FieldStore {
  /**
   * @brief Construct a new FieldStore object.
   * @param mesh The mesh associated with the fields.
   * @param storage_size Initial storage size for fields (default: 50).
   */
  FieldStore(std::shared_ptr<Mesh> mesh, size_t storage_size = 50);

  /**
   * @brief Enum for different types of time derivatives.
   */
  enum class TimeDerivative
  {
    VALUE,  ///< The value of the field.
    DOT,    ///< The first time derivative.
    DDOT,   ///< The second time derivative.
    DDDOT   ///< The third time derivative.
  };

  /**
   * @brief Add a shape displacement field to the store.
   * @tparam Space The finite element space type.
   * @param type The field type specification.
   */
  template <typename Space>
  void addShapeDisp(FieldType<Space> type)
  {
    shape_disp_.push_back(smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag()));
  }

  /**
   * @brief Add a parameter field to the store.
   * @tparam Space The finite element space type.
   * @param type The field type specification.
   */
  template <typename Space>
  void addParameter(FieldType<Space> type)
  {
    to_params_index_[type.name] = params_.size();
    params_.push_back(smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag()));
  }

  /**
   * @brief Add an independent field (an unknown) to the store.
   * @tparam Space The finite element space type.
   * @param type The field type specification.
   * @param time_rule The time integration rule for this field.
   * @return std::shared_ptr<DirichletBoundaryConditions> The boundary conditions for this field.
   */
  template <typename Space>
  std::shared_ptr<DirichletBoundaryConditions> addIndependent(FieldType<Space>& type,
                                                              std::shared_ptr<TimeIntegrationRule> time_rule)
  {
    type.unknown_index = static_cast<int>(num_unknowns_);
    to_states_index_[type.name] = states_.size();
    to_unknown_index_[type.name] = num_unknowns_;
    FieldState new_field = smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag());
    states_.push_back(new_field);
    auto latest_bc = addBoundaryConditions(new_field.get());
    ++num_unknowns_;
    SLIC_ERROR_IF(num_unknowns_ != boundary_conditions_.size(),
                  "Inconcistency between num unknowns and boundary condition size");

    SLIC_ERROR_IF(!time_rule, "Invalid time_rule");

    TimeIntegrationMapping mapping;
    mapping.primary_name = type.name;
    independent_name_to_rule_index_[type.name] = time_integration_rules_.size();
    time_integration_rules_.push_back({time_rule, mapping});

    return latest_bc;
  }

  /**
   * @brief Add a dependent field (e.g., history, velocity, acceleration) to the store.
   * @tparam Space The finite element space type.
   * @param independent_field The independent field this field depends on.
   * @param derivative The type of time derivative this field represents.
   * @param name_override Optional name override for the dependent field.
   * @return FieldType<Space> The added dependent field's type specification.
   */
  template <typename Space>
  auto addDependent(FieldType<Space> independent_field, TimeDerivative derivative, std::string name_override = "")
  {
    std::string suffix;
    if (derivative == TimeDerivative::VALUE) {
      suffix = "_old";
    } else if (derivative == TimeDerivative::DOT) {
      suffix = "_dot_old";
    } else if (derivative == TimeDerivative::DDOT) {
      suffix = "_ddot_old";
    } else {
      SLIC_ERROR("Unsupported TimeDerivative");
    }

    std::string name = name_override.empty() ? independent_field.name + suffix : name_override;

    if (independent_name_to_rule_index_.count(independent_field.name)) {
      size_t rule_idx = independent_name_to_rule_index_.at(independent_field.name);
      auto& mapping = time_integration_rules_[rule_idx].second;
      if (derivative == TimeDerivative::VALUE) {
        mapping.history_name = name;
      } else if (derivative == TimeDerivative::DOT) {
        mapping.dot_name = name;
      } else if (derivative == TimeDerivative::DDOT) {
        mapping.ddot_name = name;
      }
    } else {
      SLIC_WARNING("Adding dependent time integration field for independent field '"
                   << independent_field.name << "' which has no registered TimeIntegrationRule.");
    }

    to_states_index_[name] = states_.size();
    states_.push_back(smith::createFieldState<Space>(*graph_, Space{}, name, mesh_->tag()));
    return FieldType<Space>(name);
  }

  /**
   * @brief Register an argument to a weak form as an unknown.
   * @param weak_form_name Name of the weak form.
   * @param argument_name Name of the argument field.
   * @param argument_index Index of the argument in the weak form's argument list.
   */
  void addWeakFormUnknownArg(std::string weak_form_name, std::string argument_name, size_t argument_index);

  /**
   * @brief Register an argument to a weak form.
   * @param weak_form_name Name of the weak form.
   * @param argument_name Name of the argument field.
   * @param argument_index Index of the argument in the weak form's argument list.
   */
  void addWeakFormArg(std::string weak_form_name, std::string argument_name, size_t argument_index);

  /**
   * @brief Register the test field for a weak form.
   * @param weak_form_name Name of the weak form.
   * @param field_name Name of the test field.
   */
  void addWeakFormTestField(std::string weak_form_name, std::string field_name);

  /**
   * @brief Get the name of the test field for a weak form.
   * @param weak_form_name Name of the weak form.
   * @return std::string Name of the test field.
   */
  std::string getWeakFormTestField(const std::string& weak_form_name) const;

  /**
   * @brief Mapping between primary and history/derivative fields for time integration.
   */
  struct TimeIntegrationMapping {
    std::string primary_name;  ///< Primary unknown field name.
    std::string history_name;  ///< Previous time step value field name.
    std::string dot_name;      ///< First time derivative field name.
    std::string ddot_name;     ///< Second time derivative field name.
  };

  /**
   * @brief Get all registered time integration rules and their mappings.
   * @return const std::vector<std::pair<std::shared_ptr<TimeIntegrationRule>, TimeIntegrationMapping>>& List of rules
   * and mappings.
   */
  const std::vector<std::pair<std::shared_ptr<TimeIntegrationRule>, TimeIntegrationMapping>>& getTimeIntegrationRules()
      const;

  /**
   * @brief Print the internal field maps for debugging.
   */
  void printMap();

  /**
   * @brief Generate an index map for the residuals.
   * @param residual_names Names of the residuals.
   * @return std::vector<std::vector<size_t>> The index map.
   */
  std::vector<std::vector<size_t>> indexMap(const std::vector<std::string>& residual_names) const;

  /**
   * @brief Get the boundary condition managers for all independent fields.
   * @return std::vector<const BoundaryConditionManager*> List of boundary condition managers.
   */
  std::vector<const BoundaryConditionManager*> getBoundaryConditionManagers() const;

  /**
   * @brief Get the Dirichlet boundary conditions for an independent field by its unknown index.
   * @param unknown_index The unknown index of the independent field.
   * @return std::shared_ptr<DirichletBoundaryConditions> The boundary conditions.
   */
  std::shared_ptr<DirichletBoundaryConditions> getBoundaryConditions(size_t unknown_index) const;

  /**
   * @brief Get the internal index of a field by name.
   * @param field_name Name of the field.
   * @return size_t Index of the field.
   */
  size_t getFieldIndex(const std::string& field_name) const;

  /**
   * @brief Get the unknown index of a field by name.
   * @param field_name Name of the field.
   * @return size_t Unknown index of the field.
   */
  size_t getUnknownIndex(const std::string& field_name) const;

  /**
   * @brief Get a FieldState by name.
   * @param field_name Name of the field.
   * @return FieldState The field state.
   */
  FieldState getField(const std::string& field_name) const;

  /**
   * @brief Get a parameter field by name.
   * @param param_name Name of the parameter.
   * @return FieldState The parameter field state.
   */
  FieldState getParameter(const std::string& param_name) const;

  /**
   * @brief Update a field in the store by name.
   * @param field_name Name of the field.
   * @param updated_field The new field state.
   */
  void setField(const std::string& field_name, FieldState updated_field);

  /**
   * @brief Update a field in the store by index.
   * @param index Index of the field.
   * @param updated_field The new field state.
   */
  void setField(size_t index, FieldState updated_field);

  /**
   * @brief Get the shape displacement field.
   * @return FieldState The shape displacement field.
   */
  FieldState getShapeDisp() const;

  /**
   * @brief Get all fields stored in the FieldStore.
   * @return const std::vector<FieldState>& List of all fields.
   */
  const std::vector<FieldState>& getAllFields() const;

  /**
   * @brief Get the state fields associated with a weak form.
   * @param weak_form_name Name of the weak form.
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStates(const std::string& weak_form_name) const;

  /**
   * @brief Extract state fields for a weak form from provided state and parameter vectors.
   * @param weak_form_name Name of the weak form.
   * @param state_fields Vector of all state fields.
   * @param param_fields Vector of all parameter fields.
   * @return std::vector<FieldState> Subset of fields relevant to the weak form.
   */
  std::vector<FieldState> getStatesFromVectors(const std::string& weak_form_name,
                                               const std::vector<FieldState>& state_fields,
                                               const std::vector<FieldState>& param_fields) const;

  /**
   * @brief Get the associated mesh.
   * @return const std::shared_ptr<smith::Mesh>& The mesh.
   */
  const std::shared_ptr<smith::Mesh>& getMesh() const;

  /**
   * @brief Get the associated data store graph.
   * @return const std::shared_ptr<gretl::DataStore>& The graph.
   */
  const std::shared_ptr<gretl::DataStore>& graph() const;

 private:
  std::shared_ptr<Mesh> mesh_;
  std::shared_ptr<gretl::DataStore> graph_;

  std::vector<FieldState> shape_disp_;
  std::vector<FieldState> params_;
  std::vector<FieldState> states_;

  std::map<std::string, size_t> to_states_index_;
  std::map<std::string, size_t> to_params_index_;

  size_t num_unknowns_ = 0;
  std::map<std::string, size_t> to_unknown_index_;
  std::vector<std::shared_ptr<DirichletBoundaryConditions>> boundary_conditions_;

  struct FieldLabel {
    std::string field_name;
    size_t field_index_in_residual;
  };

  std::shared_ptr<DirichletBoundaryConditions> addBoundaryConditions(FEFieldPtr field);

  std::map<std::string, std::vector<FieldLabel>> weak_form_name_to_unknown_name_index_;

  std::map<std::string, std::vector<size_t>> weak_form_name_to_field_indices_;
  std::map<std::string, std::vector<std::string>> weak_form_name_to_field_names_;

  std::map<std::string, std::string> weak_form_to_test_field_;

  std::vector<std::pair<std::shared_ptr<TimeIntegrationRule>, TimeIntegrationMapping>> time_integration_rules_;
  std::map<std::string, size_t> independent_name_to_rule_index_;
};

/**
 * @brief Helper function to recursively register finite element spaces for a weak form.
 */
template <typename FirstType, typename... Types>
void createSpaces(const std::string& weak_form_name, FieldStore& field_store,
                  std::vector<const mfem::ParFiniteElementSpace*>& spaces, size_t arg_num, FirstType type,
                  Types... types)
{
  SLIC_ERROR_IF(spaces.size() != arg_num, "Error creating spaces recursively");
  spaces.push_back(&field_store.getField(type.name).get()->space());
  field_store.addWeakFormArg(weak_form_name, type.name, arg_num);
  if (type.unknown_index >= 0) {
    field_store.addWeakFormUnknownArg(weak_form_name, type.name, arg_num);
  }
  if constexpr (sizeof...(types) > 0) {
    createSpaces(weak_form_name, field_store, spaces, arg_num + 1, types...);
  }
}

/**
 * @brief Create a TimeDiscretizedWeakForm and register its fields in the FieldStore.
 */
template <int spatial_dim, typename TestSpaceType, typename... InputSpaceTypes>
auto createWeakForm(std::string name, FieldType<TestSpaceType> test_type, FieldStore& field_store,
                    FieldType<InputSpaceTypes>... field_types)
{
  field_store.addWeakFormTestField(name, test_type.name);
  const mfem::ParFiniteElementSpace& test_space = field_store.getField(test_type.name).get()->space();
  std::vector<const mfem::ParFiniteElementSpace*> input_spaces;
  createSpaces(name, field_store, input_spaces, 0, field_types...);
  return std::make_shared<TimeDiscretizedWeakForm<spatial_dim, TestSpaceType, Parameters<InputSpaceTypes...>>>(
      name, field_store.getMesh(), test_space, input_spaces);
}

}  // namespace smith
