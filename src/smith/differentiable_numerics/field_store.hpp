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

/**
 * @brief Representation of a field type with a name and a flag indicating whether it is an
 *        active Jacobian unknown in the current weak-form context.
 *
 * @c is_unknown is intentionally a per-instance flag, not a global property.  The same field
 * may be a Jacobian variable in one weak form (e.g. displacement in the main solid solve) and
 * a fixed input in another (e.g. displacement in the stress projection).  Callers control this
 * by passing the same @c FieldType object (set by @c addIndependent) or a plain copy with the
 * default @c is_unknown = false.
 *
 * @tparam Space The finite element space type.
 * @tparam Time The time integration type (unused by default).
 */
template <typename Space, typename Time = void*>
struct FieldType {
  /**
   * @brief Construct a new FieldType object.
   * @param n Name of the field.
   * @param is_unknown_ Whether this field is a Jacobian unknown in the current context.
   */
  FieldType(std::string n, bool is_unknown_ = false) : name(n), is_unknown(is_unknown_) {}
  std::string name;         ///< Name of the field.
  bool is_unknown = false;  ///< True if this field is a Jacobian variable in the current weak-form context.
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
    to_params_index_[type.name] = params_.size();
    params_.push_back(smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag()));
  }

  /**
   * @brief Add an independent field (a solver unknown) to the store.
   *
   * Registers the field as an unknown by setting @c type.is_unknown = true, so the same
   * @c FieldType<Space> object can later be passed to @c createSpaces to mark this argument
   * as an active Jacobian variable.  Also creates a boundary-condition slot keyed by field
   * name that callers can populate after this call returns.
   *
   * @tparam Space The finite element space type.
   * @param type The field type specification; @c type.is_unknown is set to @c true on return.
   * @param time_rule The time integration rule governing how this unknown and its dependents
   *        are related across time steps.
   * @return std::shared_ptr<DirichletBoundaryConditions> The boundary conditions for this field.
   */
  template <typename Space>
  std::shared_ptr<DirichletBoundaryConditions> addIndependent(FieldType<Space>& type,
                                                              std::shared_ptr<TimeIntegrationRule> time_rule)
  {
    type.name = prefix(type.name);
    type.is_unknown = true;
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
   * **Return value:** a @c FieldType<Space> whose @c name is the name of the newly registered
   * field and @c is_unknown is @c false.  This object is intentionally returned (rather than
   * discarded) so that callers can pass it directly to @c createSpaces when assembling the
   * weak-form argument list.  To make it the Jacobian variable for a specific weak form (e.g.
   * the cycle-zero acceleration solve), copy the returned object and set @c is_unknown = true
   * before passing to @c createSpaces.
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
   * @brief Register the reaction (test) field for a weak form.
   *
   * The reaction field is the field whose test function space the weak form integrates against.
   * It determines which field's degrees of freedom the assembled residual is "returned to"
   * (i.e. the field whose force/flux vector is populated).
   *
   * @param weak_form_name Name of the weak form.
   * @param field_name Name of the reaction field.
   */
  void addWeakFormReaction(std::string weak_form_name, std::string field_name);

  /**
   * @brief Mark a weak form as internal so it is excluded from getReactionInfos().
   *
   * Use this for subsystem forms (e.g. cycle-zero acceleration solve) that should not be
   * exposed as user-visible reactions in DifferentiablePhysics.
   */
  void markWeakFormInternal(const std::string& weak_form_name);

  /**
   * @brief Get the name of the reaction (test) field for a weak form.
   * @param weak_form_name Name of the weak form.
   * @return std::string Name of the reaction field.
   */
  std::string getWeakFormReaction(const std::string& weak_form_name) const;

  /**
   * @brief Register all input fields for a weak form and return their FE spaces.
   *
   * This is the primary setup method for constructing a weak form.  It:
   *   1. Registers @p reaction_field_name as the reaction/test field via @c addWeakFormReaction.
   *   2. Iterates over every @c FieldType in @p types (in order), registering each as an input
   *      argument.  If @c type.is_unknown is @c true, the field is also registered as an active
   *      Jacobian unknown for this weak form.
   *   3. Returns the ordered vector of finite element spaces.
   *
   * A field may have @c is_unknown = true in one weak form and @c false in another (e.g.
   * displacement is a Jacobian variable in the main solid solve but a fixed input in the stress
   * projection).  Callers control this by passing the @c FieldType returned from @c addIndependent
   * (has @c is_unknown = true) or a plain copy constructed from the field name (has @c is_unknown
   * = false).  Similarly, a dependent field can be made the Jacobian variable for a specific weak
   * form (e.g. acceleration in the cycle-zero solve) by copying the returned @c FieldType and
   * setting @c is_unknown = true.
   *
   * @param weak_form_name  Name of the weak form being constructed.
   * @param reaction_field_name  Name of the test/reaction field (may differ from the first input).
   * @param types  Ordered list of @c FieldType descriptors for every input argument.
   * @return std::vector<const mfem::ParFiniteElementSpace*> Ordered input FE spaces.
   */
  template <typename... FieldTypes>
  std::vector<const mfem::ParFiniteElementSpace*> createSpaces(const std::string& weak_form_name,
                                                               const std::string& reaction_field_name,
                                                               FieldTypes... types)
  {
    addWeakFormReaction(weak_form_name, reaction_field_name);
    std::vector<const mfem::ParFiniteElementSpace*> spaces;
    size_t arg_num = 0;
    auto register_field = [&](auto type) {
      spaces.push_back(&getField(type.name).get()->space());
      addWeakFormArg(weak_form_name, type.name, arg_num);
      if (type.is_unknown) {
        addWeakFormUnknownArg(weak_form_name, type.name, arg_num);
      }
      ++arg_num;
    };
    (register_field(types), ...);
    return spaces;
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
   * @return std::vector<std::vector<size_t>> The index map.
   */
  std::vector<std::vector<size_t>> indexMap(const std::vector<std::string>& residual_names) const;

  /**
   * @brief Get the boundary condition managers for the given weak forms, one per residual row.
   *
   * For each weak form in @p weak_form_names the reaction (test) field name is looked up, and
   * the corresponding @c BoundaryConditionManager is returned.  If no BC was registered for
   * that field, @c nullptr is returned for that slot (the solver skips null entries).
   *
   * Zero-mirror BCs registered via @c shareBoundaryConditions are materialized lazily on the
   * first call to this method (so the user's @c set*BCs calls on the source BC are visible).
   *
   * @param weak_form_names Ordered list of weak form names whose BCs are needed.
   * @return std::vector<const BoundaryConditionManager*> One entry per weak form, in order.
   */
  std::vector<const BoundaryConditionManager*> getBoundaryConditionManagers(
      const std::vector<std::string>& weak_form_names);

  /**
   * @brief Register a zero-valued mirror BC for @p name, sharing the constrained DOF set of @p source_bc.
   *
   * Used for fields whose constrained DOFs must match a reference field (e.g. acceleration mirrors
   * displacement), but whose prescribed BC value is always zero.  The zero BC is materialized
   * lazily on the first call to @c getBoundaryConditionManagers so that any @c set*BCs calls the
   * caller makes on @p source_bc after this method returns are reflected in the mirror.
   *
   * @param name       Field name to associate with the zero BC.
   * @param source_bc  BC object whose constrained DOF set is mirrored (e.g. the displacement BC).
   */
  void shareBoundaryConditions(const std::string& name, std::shared_ptr<DirichletBoundaryConditions> source_bc);

  bool hasField(const std::string& field_name) const;

  /**
   * @brief Get the internal index of a field by name.
   * @param field_name Name of the field.
   * @return size_t Index of the field.
   */
  size_t getFieldIndex(const std::string& field_name) const;

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

  const std::shared_ptr<smith::Mesh>& getMesh() const;

  /**
   * @brief Get the associated data store graph.
   * @return const std::shared_ptr<gretl::DataStore>& The graph.
   */
  const std::shared_ptr<gretl::DataStore>& graph() const;

 private:
  std::shared_ptr<Mesh> mesh_;
  std::shared_ptr<gretl::DataStore> graph_;
  std::string prepend_name_;

  std::vector<FieldState> shape_disp_;
  std::vector<FieldState> params_;
  std::vector<FieldState> states_;
  std::vector<bool> is_solve_state_;

  std::map<std::string, size_t> to_states_index_;
  std::map<std::string, size_t> to_params_index_;

  /// Boundary conditions keyed by field name.  Populated by @c addIndependent (for primary
  /// unknowns).  Zero-mirror BCs are added lazily by @c getBoundaryConditionManagers when
  /// entries from @c zero_mirror_sources_ are materialized.
  std::map<std::string, std::shared_ptr<DirichletBoundaryConditions>> boundary_conditions_;

  /// Pending zero-mirror BCs: maps a field name to the source @c DirichletBoundaryConditions
  /// whose constrained DOF set it should copy (with zero prescribed values).  Entries are
  /// materialized and moved to @c boundary_conditions_ on the first call to
  /// @c getBoundaryConditionManagers.
  std::map<std::string, std::shared_ptr<DirichletBoundaryConditions>> zero_mirror_sources_;

  struct FieldLabel {
    std::string field_name;
    size_t field_index_in_residual;
  };

  std::shared_ptr<DirichletBoundaryConditions> addBoundaryConditions(FEFieldPtr field);

  std::map<std::string, std::vector<FieldLabel>> weak_form_name_to_unknown_name_index_;

  std::map<std::string, std::vector<size_t>> weak_form_name_to_field_indices_;
  std::map<std::string, std::vector<std::string>> weak_form_name_to_field_names_;

  std::vector<std::pair<std::string, std::string>> weak_form_to_test_field_;
  std::set<std::string> internal_weak_forms_;  ///< weak forms excluded from getReactionInfos() (subsystem-internal)

  std::vector<std::pair<std::shared_ptr<TimeIntegrationRule>, TimeIntegrationMapping>> time_integration_rules_;
  std::map<std::string, size_t> independent_name_to_rule_index_;
};

/**
 * @brief Create a TimeDiscretizedWeakForm and register its fields in the FieldStore.
 *
 * Thin convenience wrapper: registers @p test_type as the reaction field, registers all
 * @p field_types as input arguments, and constructs the weak form in one call.
 */
template <int spatial_dim, typename TestSpaceType, typename... InputSpaceTypes>
auto createWeakForm(std::string name, FieldType<TestSpaceType> test_type, FieldStore& field_store,
                    FieldType<InputSpaceTypes>... field_types)
{
  return std::make_shared<TimeDiscretizedWeakForm<spatial_dim, TestSpaceType, Parameters<InputSpaceTypes...>>>(
      name, field_store.getMesh(), field_store.getField(test_type.name).get()->space(),
      field_store.createSpaces(name, test_type.name, field_types...));
}

}  // namespace smith
