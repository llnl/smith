// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/physics/functional_weak_form.hpp"
#include "smith/physics/mesh.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>

namespace smith {

class DirichletBoundaryConditions;
class BoundaryConditionManager;

/**
 * @brief Information about a dual field.
 */
struct ReactionInfo {
  std::string name;                                    ///< The name of the dual field.
  const mfem::ParFiniteElementSpace* space = nullptr;  ///< The finite element space of the dual field.
};

/// Names identifying a weak form's test field and solver-owned unknown.
struct UnknownAndTestFieldNames {
  std::string unknown;  ///< Field owned and solved by this residual row.
  std::string test;     ///< Field defining residual test space and reaction output.
};

/**
 * @brief Representation of a named field type.
 * @tparam Space The finite element space type.
 * @tparam Time The time integration type (unused by default).
 */
template <typename Space, typename Time = void*>
struct FieldType {
  using space_type = Space;  ///< The finite element space type.

  /// Construct a field type with the given name.
  FieldType(std::string n) : name(n) {}
  std::string name;  ///< Name of the field.
};

/**
 * @brief Manages storage and metadata for fields, parameters, and weak forms.
 */
struct FieldStore {
  /**
   * @brief Construct a new FieldStore object.
   * @param mesh The mesh associated with the fields.
   * @param storage_size Initial storage size for fields (default: 50).
   * @param prepend_name Namespace prefix applied by @c prefix(). Empty means no prefix.
   */
  FieldStore(std::shared_ptr<Mesh> mesh, size_t storage_size = 50, std::string prepend_name = "");

  /**
   * @brief Apply this store's namespace prefix to a base name.
   *
   * Returns @p base unchanged when the store was constructed with an empty prepend name,
   * otherwise returns @c prepend_name_ + "_" + base. Factories use this to namespace
   * weak form, field, and parameter names consistently without re-implementing the rule.
   */
  std::string prefix(const std::string& base) const;

  /**
   * @brief Enum for different types of time derivatives.
   */
  enum class TimeDerivative
  {
    VAL,   //< The value of the field.
    DOT,   ///< The first time derivative.
    DDOT,  ///< The second time derivative.
    DDDOT  ///< The third time derivative.
  };

  /**
   * @brief Add a shape displacement field to the store.
   * @tparam Space The finite element space type.
   * @param type The field type specification.
   */
  template <typename Space>
  void addShapeDisp(FieldType<Space>& type)
  {
    type.name = prefix(type.name);
    shape_disp_.push_back(smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag()));
  }

  /**
   * @brief Add a parameter field to the store.
   * @tparam Space The finite element space type.
   * @param type The field type specification.
   */
  template <typename Space>
  void addParameter(FieldType<Space>& type)
  {
    type.name = prefix(type.name);
    if (to_params_index_.count(type.name)) {
      // Already registered — expected when multiple systems share the same parameter.
      // Verify the space matches by checking vdim (== Space::components).
      auto& existing = params_[to_params_index_.at(type.name)];
      SLIC_ERROR_ROOT_IF(existing.get()->space().GetVDim() != Space::components,
                         axom::fmt::format("Parameter '{}' re-registered with a different space "
                                           "(existing vdim={}, new vdim={})",
                                           type.name, existing.get()->space().GetVDim(), Space::components));
      return;
    }
    to_params_index_[type.name] = params_.size();
    params_.push_back(smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag()));
  }

  /**
   * @brief Add an independent field (a solver unknown) to the store.
   *
   * Also creates a boundary-condition slot keyed by field name that callers can populate.
   *
   * @tparam Space The finite element space type.
   * @param type The field type specification.
   * @param time_rule The time integration rule governing how this unknown and its dependents
   *        are related across time steps.
   * @return std::shared_ptr<DirichletBoundaryConditions> The boundary conditions for this field.
   */
  template <typename Space>
  std::shared_ptr<DirichletBoundaryConditions> addIndependent(FieldType<Space>& type,
                                                              std::shared_ptr<TimeIntegrationRule> time_rule)
  {
    type.name = prefix(type.name);
    to_states_index_[type.name] = states_.size();
    FieldState new_field = smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag());
    states_.push_back(new_field);
    is_solve_state_.push_back(true);
    auto latest_bc = addBoundaryConditions(new_field.get());
    boundary_conditions_[type.name] = latest_bc;

    SLIC_ERROR_IF(!time_rule, "Invalid time_rule");

    TimeIntegrationMapping mapping;
    mapping.primary_name = type.name;
    independent_name_to_rule_index_[type.name] = time_integration_rules_.size();
    time_integration_rules_.push_back({time_rule, mapping});

    return latest_bc;
  }

  /**
   * @brief Add a dependent field (history value, velocity, or acceleration) to the store.
   *
   * Creates and registers a new field that carries the previous time-step value of a particular
   * time derivative of an independent field.  The relationship is recorded in the
   * @c TimeIntegrationMapping for the parent independent field so that, at evaluation time, the
   * time integration rule can reconstruct the current rate from the pair
   * (predicted_value, stored_old_value).
   *
   * Returns a descriptor for the newly registered field so callers can pass it directly to
   * @c createSpaces when assembling the weak-form argument list.
   *
   * @tparam Space The finite element space type (must match the independent field).
   * @param independent_field The @c FieldType of the independent (predicted) field.
   * @param derivative Which time-derivative level this history field stores.
   * @param name_override If non-empty, use this as the field name instead of the auto-generated one.
   * @return FieldType<Space> Type descriptor for the newly created dependent field.
   */
  template <typename Space>
  auto addDependent(FieldType<Space> independent_field, TimeDerivative derivative, std::string name_override = "")
  {
    std::string suffix;
    if (derivative == TimeDerivative::VAL) {
      suffix = "_old";
    } else if (derivative == TimeDerivative::DOT) {
      suffix = "_dot_old";
    } else if (derivative == TimeDerivative::DDOT) {
      suffix = "_ddot_old";
    } else {
      SLIC_ERROR("Unsupported TimeDerivative");
    }

    std::string name = name_override.empty() ? independent_field.name + suffix : prefix(name_override);

    if (independent_name_to_rule_index_.count(independent_field.name)) {
      size_t rule_idx = independent_name_to_rule_index_.at(independent_field.name);
      auto& mapping = time_integration_rules_[rule_idx].second;
      if (derivative == TimeDerivative::VAL) {
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
    is_solve_state_.push_back(false);
    return FieldType<Space>(name);
  }

  /**
   * @brief Register an argument to a weak form.
   * @param weak_form_name Name of the weak form.
   * @param argument_name Name of the argument field.
   * @param argument_index Index of the argument in the weak form's argument list.
   */
  void addWeakFormArg(std::string weak_form_name, std::string argument_name, size_t argument_index);

  /// Register test-space and solver-owned field names for a weak form.
  void addWeakFormFieldNames(std::string weak_form_name, UnknownAndTestFieldNames field_names);

  /**
   * @brief Mark a weak form as internal so it is excluded from getReactionInfos().
   *
   * Use this for subsystem forms (e.g. cycle-zero acceleration solve) that should not be
   * exposed as user-visible reactions in DifferentiablePhysics.
   */
  void markWeakFormInternal(const std::string& weak_form_name);

  /// Get explicit test-space and solver-owned field names for a weak form.
  const UnknownAndTestFieldNames& getWeakFormFieldNames(const std::string& weak_form_name) const;

  /**
   * @brief Register all input fields for a weak form and return their FE spaces.
   *
   * This is the primary setup method for constructing a weak form.  It:
   *   1. Registers explicit test-space and solver-owned field names.
   *   2. Registers every @c FieldType in @p types as an ordered input argument.
   *   3. Returns the ordered vector of finite element spaces.
   *
   * @param weak_form_name  Name of the weak form being constructed.
   * @param field_names Names of the test-space and solver-owned fields.
   * @param types  Ordered list of @c FieldType descriptors for every input argument.
   * @return std::vector<const mfem::ParFiniteElementSpace*> Ordered input FE spaces.
   */
  template <typename... FieldTypes>
  std::vector<const mfem::ParFiniteElementSpace*> createSpaces(const std::string& weak_form_name,
                                                               UnknownAndTestFieldNames field_names,
                                                               FieldTypes... types)
  {
    addWeakFormFieldNames(weak_form_name, field_names);
    std::vector<const mfem::ParFiniteElementSpace*> spaces;
    size_t arg_num = 0;
    bool unknown_found = false;
    auto register_field = [&](auto type) {
      spaces.push_back(&getField(type.name).get()->space());
      addWeakFormArg(weak_form_name, type.name, arg_num);
      unknown_found = unknown_found || type.name == field_names.unknown;
      ++arg_num;
    };
    (register_field(types), ...);
    SLIC_ERROR_IF(!unknown_found, "Unknown field '" << field_names.unknown << "' is not an argument of weak form '"
                                                    << weak_form_name << "'");
    return spaces;
  }

  /**
   * @brief Register input fields when test and unknown field names match.
   * @param weak_form_name Name of the weak form being constructed.
   * @param field_name Shared test-space and solver-owned field name.
   * @param types Ordered list of @c FieldType descriptors for every input argument.
   * @return std::vector<const mfem::ParFiniteElementSpace*> Ordered input FE spaces.
   */
  template <typename... FieldTypes>
  std::vector<const mfem::ParFiniteElementSpace*> createSpaces(const std::string& weak_form_name,
                                                               const std::string& field_name, FieldTypes... types)
  {
    return createSpaces(weak_form_name, {.unknown = field_name, .test = field_name}, types...);
  }

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
   * @return Slot-list matrix mapping residual rows and solved columns.
   */
  BlockArgumentMap indexMap(const std::vector<std::string>& residual_names) const;

  /**
   * @brief Get the boundary condition managers for the given weak forms, one per residual row.
   *
   * For each weak form in @p weak_form_names the solver-owned field name is looked up. The
   * returned manager is selected by consulting the registered @c TimeIntegrationMapping s:
   *   - unknown = primary or history slot -> value-level BC manager
   *   - unknown = second-derivative (ddot) slot -> second-derivative BC manager
   *   - unknown has its own DBC entry not tied to a mapping -> that DBC's value manager
   *   - otherwise -> @c nullptr (solver skips null entries)
   *
   * The second-derivative manager is rebuilt on each call, so late value-BC additions are reflected.
   *
   * @param weak_form_names Ordered list of weak form names whose BCs are needed.
   * @return std::vector<const BoundaryConditionManager*> One entry per weak form, in order.
   */
  std::vector<const BoundaryConditionManager*> getBoundaryConditionManagers(
      const std::vector<std::string>& weak_form_names) const;

  /// @brief Get ordered boundary condition managers corresponding to an ordered list of fields.
  std::vector<const BoundaryConditionManager*> getBoundaryConditionManagersForFields(
      const std::vector<std::string>& field_names) const;

  /**
   * @brief Check whether a field exists.
   *
   * Accepts either a fully-qualified field name or an unprefixed base name.
   */
  bool hasField(const std::string& field_name) const;

  /**
   * @brief Get the internal index of a field by name.
   * @param field_name Fully-qualified or unprefixed field name.
   * @return size_t Index of the field.
   */
  size_t getFieldIndex(const std::string& field_name) const;

  /**
   * @brief Get a FieldState by name.
   * @param field_name Fully-qualified or unprefixed field name.
   * @return FieldState The field state.
   */
  FieldState getField(const std::string& field_name) const;

  /**
   * @brief Get a parameter field by name.
   * @param param_name Fully-qualified or unprefixed parameter name.
   * @return FieldState The parameter field state.
   */
  FieldState getParameter(const std::string& param_name) const;

  /**
   * @brief Update a field in the store by name.
   * @param field_name Fully-qualified or unprefixed field name.
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
   * @brief Get the list of all parameter fields.
   */
  const std::vector<FieldState>& getParameterFields() const;

  /**
   * @brief Get the list of all state fields.
   */
  const std::vector<FieldState>& getStateFields() const;

  /**
   * @brief Get the list of physical, non-solve state fields suitable for output.
   */
  std::vector<FieldState> getOutputFieldStates() const;

  /**
   * @brief Get information about reaction fields.
   */
  std::vector<ReactionInfo> getReactionInfos() const;

  /**
   * @brief Get associated mesh shared by all registered fields.
   */
  const std::shared_ptr<smith::Mesh>& getMesh() const;

  /**
   * @brief Get the boundary conditions for a given field name.
   */
  std::shared_ptr<DirichletBoundaryConditions> getBoundaryConditions(const std::string& field_name) const;

  /**
   * @brief Get the associated data store graph.
   * @return const std::shared_ptr<gretl::DataStore>& The graph.
   */
  const std::shared_ptr<gretl::DataStore>& graph() const;

 private:
  std::string resolveFieldName(const std::string& field_name) const;

  std::shared_ptr<Mesh> mesh_;
  std::shared_ptr<gretl::DataStore> graph_;
  std::string prepend_name_;

  std::vector<FieldState> shape_disp_;
  std::vector<FieldState> params_;
  std::vector<FieldState> states_;
  std::vector<bool> is_solve_state_;

  std::map<std::string, size_t> to_states_index_;
  std::map<std::string, size_t> to_params_index_;

  /// Boundary conditions keyed by primary unknown field name.
  std::map<std::string, std::shared_ptr<DirichletBoundaryConditions>> boundary_conditions_;

  std::shared_ptr<DirichletBoundaryConditions> addBoundaryConditions(FEFieldPtr field);

  std::map<std::string, std::vector<std::string>> weak_form_name_to_field_names_;

  std::vector<std::pair<std::string, UnknownAndTestFieldNames>> weak_form_field_names_;
  std::set<std::string> internal_weak_forms_;  ///< weak forms excluded from getReactionInfos() (subsystem-internal)

  std::vector<std::pair<std::shared_ptr<TimeIntegrationRule>, TimeIntegrationMapping>> time_integration_rules_;
  std::map<std::string, size_t> independent_name_to_rule_index_;
};

/**
 * @brief Create a FunctionalWeakForm and register its fields in the FieldStore.
 *
 * Thin convenience wrapper: registers @p test_type as the reaction field, registers all
 * @p field_types as input arguments, and constructs the weak form in one call.
 */
template <int spatial_dim, typename TestSpaceType, typename UnknownSpaceType, typename... InputSpaceTypes>
auto createWeakForm(std::string name, FieldType<TestSpaceType> test_type, FieldType<UnknownSpaceType> unknown_type,
                    FieldStore& field_store, FieldType<InputSpaceTypes>... field_types)
{
  return std::make_shared<
      FunctionalWeakForm<spatial_dim, TestSpaceType, Parameters<UnknownSpaceType, InputSpaceTypes...>>>(
      name, field_store.getMesh(), field_store.getField(test_type.name).get()->space(),
      field_store.createSpaces(name, {.unknown = unknown_type.name, .test = test_type.name}, unknown_type,
                               field_types...));
}

/**
 * @brief Construct a `shared_ptr<FieldStore>` over a mesh. Inline call-site helper for
 * standalone-physics setup; equivalent to `std::make_shared<FieldStore>(mesh, ...)`.
 */
inline std::shared_ptr<FieldStore> fieldStore(std::shared_ptr<Mesh> mesh, std::size_t storage_size = 200,
                                              std::string prefix = "")
{
  return std::make_shared<FieldStore>(std::move(mesh), storage_size, std::move(prefix));
}

}  // namespace smith
