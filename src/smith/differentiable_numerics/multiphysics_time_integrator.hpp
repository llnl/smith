// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/physics/mesh.hpp"

namespace smith {

class DifferentiableBlockSolver;
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

/**
 * @brief Solve a set of weak forms.
 * @param weak_forms List of weak forms to solve.
 * @param field_store Field store containing the fields.
 * @param solver The solver to use.
 * @param time_info Current time information.
 * @param params Optional parameter fields.
 * @return std::vector<FieldState> The updated state fields.
 */
std::vector<FieldState> solve(const std::vector<std::shared_ptr<WeakForm>>& weak_forms, const FieldStore& field_store,
                              const DifferentiableBlockSolver* solver, const TimeInfo& time_info,
                              const std::vector<FieldState>& params = {});

/**
 * @brief Time integrator for multiphysics problems, coordinating multiple weak forms.
 */
class MultiPhysicsTimeIntegrator : public StateAdvancer {
 public:
  /**
   * @brief Construct a new MultiPhysicsTimeIntegrator object.
   * @param field_store Field store containing the fields.
   * @param weak_forms List of weak forms to coordinate.
   * @param solver The block solver to use.
   */
  MultiPhysicsTimeIntegrator(std::shared_ptr<FieldStore> field_store,
                             const std::vector<std::shared_ptr<WeakForm>>& weak_forms,
                             std::shared_ptr<smith::DifferentiableBlockSolver> solver);

  /**
   * @brief Advance the multiphysics state by one time step.
   * @param time_info Current time information.
   * @param shape_disp Shape displacement field.
   * @param states Current state fields.
   * @param params Parameter fields.
   * @return std::pair<std::vector<FieldState>, std::vector<ReactionState>> Updated states and reactions.
   */
  std::pair<std::vector<FieldState>, std::vector<ReactionState>> advanceState(
      const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
      const std::vector<FieldState>& params) const override;

 private:
  std::shared_ptr<FieldStore> field_store_;
  std::vector<std::shared_ptr<WeakForm>> weak_forms_;
  std::shared_ptr<smith::DifferentiableBlockSolver> solver_;
};

/**
 * @brief Container for a coupled thermo-mechanical system.
 * @tparam dim Spatial dimension.
 * @tparam disp_order Order of the displacement basis.
 * @tparam temp_order Order of the temperature basis.
 * @tparam parameter_space Finite element spaces for optional parameters.
 */
template <int dim, int disp_order, int temp_order, typename... parameter_space>
struct ThermoMechanicsSystem {
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<temp_order>, H1<temp_order>, parameter_space...>>;
  using ThermalWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<temp_order>,
      Parameters<H1<temp_order>, H1<temp_order>, H1<disp_order, dim>, H1<disp_order, dim>, parameter_space...>>;

  std::shared_ptr<FieldStore> field_store;                      ///< Field store managing the system's fields.
  std::shared_ptr<SolidWeakFormType> solid_weak_form;           ///< Solid mechanics weak form.
  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;       ///< Thermal weak form.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;         ///< Displacement boundary conditions.
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;  ///< Temperature boundary conditions.
  std::shared_ptr<DifferentiableBlockSolver> solver;            ///< The solver for the coupled system.
  std::shared_ptr<StateAdvancer> advancer;                      ///< The multiphysics state advancer.
  std::shared_ptr<QuasiStaticFirstOrderTimeIntegrationRule> disp_time_rule;  ///< Time integration for displacement.
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule>
      temperature_time_rule;                 ///< Time integration for temperature.
  std::vector<FieldState> parameter_fields;  ///< Optional parameter fields.

  /**
   * @brief Get the list of all state fields (current and old).
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    std::vector<FieldState> states;
    states.push_back(field_store->getField("displacement"));
    states.push_back(field_store->getField("displacement_old"));
    states.push_back(field_store->getField("temperature"));
    states.push_back(field_store->getField("temperature_old"));
    return states;
  }

  /**
   * @brief Get the list of all parameter fields.
   * @return const std::vector<FieldState>& List of parameter fields.
   */
  const std::vector<FieldState>& getParameterFields() const { return parameter_fields; }

  /**
   * @brief Set the material model for a domain, defining integrals for solid and thermal weak forms.
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto dtr = disp_time_rule;
    auto ttr = temperature_time_rule;

    if constexpr (sizeof...(parameter_space) == 0) {
      // NO parameters - simpler version for testing
      solid_weak_form->addBodyIntegral(
          domain_name, [=](auto t_info, auto /*X*/, auto disp, auto disp_old, auto temperature, auto temperature_old) {
            auto u = dtr->value(t_info, disp, disp_old);
            auto v = dtr->dot(t_info, disp, disp_old);
            auto T = ttr->value(t_info, temperature, temperature_old);
            typename MaterialType::State state;
            auto [pk, C_v, s0, q0] =
                material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T), get<DERIVATIVE>(T));
            return smith::tuple{smith::zero{}, pk};
          });

      thermal_weak_form->addBodyIntegral(
          domain_name, [=](auto t_info, auto /*X*/, auto temperature, auto temperature_old, auto disp, auto disp_old) {
            typename MaterialType::State state;
            auto u = dtr->value(t_info, disp, disp_old);
            auto v = dtr->dot(t_info, disp, disp_old);
            auto T = ttr->value(t_info, temperature, temperature_old);
            auto T_dot = ttr->dot(t_info, temperature, temperature_old);
            auto [pk, C_v, s0, q0] =
                material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T), get<DERIVATIVE>(T));
            auto dT_dt = get<VALUE>(T_dot);
            return smith::tuple{C_v * dT_dt - s0, -q0};
          });
    } else {
      // WITH parameters
      solid_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto disp, auto disp_old,
                                                        auto temperature, auto temperature_old, auto... params) {
        auto u = dtr->value(t_info, disp, disp_old);
        auto v = dtr->dot(t_info, disp, disp_old);
        auto T = ttr->value(t_info, temperature, temperature_old);
        typename MaterialType::State state;
        auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T),
                                          get<DERIVATIVE>(T), params...);
        return smith::tuple{smith::zero{}, pk};
      });

      thermal_weak_form->addBodyIntegral(
          domain_name, [=](auto t_info, auto /*X*/, auto temperature, auto temperature_old, auto disp, auto disp_old,
                           auto... params) {
            typename MaterialType::State state;
            auto u = dtr->value(t_info, disp, disp_old);
            auto v = dtr->dot(t_info, disp, disp_old);
            auto T = ttr->value(t_info, temperature, temperature_old);
            auto T_dot = ttr->dot(t_info, temperature, temperature_old);
            auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T),
                                              get<DERIVATIVE>(T), params...);
            auto dT_dt = get<VALUE>(T_dot);
            return smith::tuple{C_v * dT_dt - s0, -q0};
          });
    }
  }
};

/**
 * @brief Factory function to build a thermo-mechanical system and its state advancer.
 */
template <int dim, int disp_order, int temp_order, typename... parameter_space>
ThermoMechanicsSystem<dim, disp_order, temp_order, parameter_space...> buildThermoMechanicsStateAdvancer(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<DifferentiableBlockSolver> solver,
    FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<FieldStore>(mesh, 100);

  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  // Displacement field with quasi-static time integration
  auto disp_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
  FieldType<H1<disp_order, dim>> disp_type("displacement");
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE);

  // Temperature field with backward Euler time integration
  auto temperature_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  FieldType<H1<temp_order>> temperature_type("temperature");
  auto temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule);
  auto temperature_old_type = field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VALUE);

  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(parameter_types), ...);
  (parameter_fields.push_back(field_store->getField(parameter_types.name)), ...);

  // Solid mechanics weak form
  auto solid_weak_form = createWeakForm<dim>("solid_force", disp_type, *field_store, disp_type, disp_old_type,
                                             temperature_type, temperature_old_type, parameter_types...);

  // Thermal weak form
  auto thermal_weak_form = createWeakForm<dim>("thermal_flux", temperature_type, *field_store, temperature_type,
                                               temperature_old_type, disp_type, disp_old_type, parameter_types...);

  // Build solver and advancer
  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, thermal_weak_form};
  auto advancer = std::make_shared<MultiPhysicsTimeIntegrator>(field_store, weak_forms, solver);

  return ThermoMechanicsSystem<dim, disp_order, temp_order, parameter_space...>{
      field_store, solid_weak_form, thermal_weak_form,     disp_bc,         temperature_bc, solver,
      advancer,    disp_time_rule,  temperature_time_rule, parameter_fields};
}

}  // namespace smith
