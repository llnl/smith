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

template <typename Space, typename Time = void*>
struct FieldType {
  FieldType(std::string n, int unknown_index_ = -1) : name(n), unknown_index(unknown_index_) {}
  std::string name;
  int unknown_index;
};

struct FieldStore {
  FieldStore(std::shared_ptr<Mesh> mesh, size_t storage_size = 50);

  enum class TimeDerivative
  {
    VALUE,
    DOT,
    DDOT,
    DDDOT
  };

  template <typename Space>
  void addShapeDisp(FieldType<Space> type)
  {
    shape_disp_.push_back(smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag()));
  }

  template <typename Space>
  void addParameter(FieldType<Space> type)
  {
    to_params_index_[type.name] = params_.size();
    params_.push_back(smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag()));
  }

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

  void addWeakFormUnknownArg(std::string weak_form_name, std::string argument_name, size_t argument_index);

  void addWeakFormArg(std::string weak_form_name, std::string argument_name, size_t argument_index);

  void addWeakFormTestField(std::string weak_form_name, std::string field_name);

  std::string getWeakFormTestField(const std::string& weak_form_name) const;

  struct TimeIntegrationMapping {
    std::string primary_name;
    std::string history_name;
    std::string dot_name;
    std::string ddot_name;
  };

  const std::vector<std::pair<std::shared_ptr<TimeIntegrationRule>, TimeIntegrationMapping>>& getTimeIntegrationRules()
      const;

  void printMap();

  std::vector<std::vector<size_t>> indexMap(const std::vector<std::string>& residual_names) const;

  std::vector<const BoundaryConditionManager*> getBoundaryConditionManagers() const;

  size_t getFieldIndex(const std::string& field_name) const;

  size_t getUnknownIndex(const std::string& field_name) const;

  FieldState getField(const std::string& field_name) const;

  FieldState getParameter(const std::string& param_name) const;

  void setField(const std::string& field_name, FieldState updated_field);

  void setField(size_t index, FieldState updated_field);

  FieldState getShapeDisp() const;

  const std::vector<FieldState>& getAllFields() const;

  std::vector<FieldState> getStates(const std::string& weak_form_name) const;

  std::vector<FieldState> getStatesFromVectors(const std::string& weak_form_name,
                                                const std::vector<FieldState>& state_fields,
                                                const std::vector<FieldState>& param_fields) const;

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

std::vector<FieldState> solve(const std::vector<std::shared_ptr<WeakForm>>& weak_forms, const FieldStore& field_store,
                              const DifferentiableBlockSolver* solver, const TimeInfo& time_info,
                              const std::vector<FieldState>& params = {});

class MultiPhysicsTimeIntegrator : public StateAdvancer {
 public:
  MultiPhysicsTimeIntegrator(std::shared_ptr<FieldStore> field_store,
                             const std::vector<std::shared_ptr<WeakForm>>& weak_forms,
                             std::shared_ptr<smith::DifferentiableBlockSolver> solver);

  std::pair<std::vector<FieldState>, std::vector<ReactionState>> advanceState(
      const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
      const std::vector<FieldState>& params) const override;

 private:
  std::shared_ptr<FieldStore> field_store_;
  std::vector<std::shared_ptr<WeakForm>> weak_forms_;
  std::shared_ptr<smith::DifferentiableBlockSolver> solver_;
};

template <int dim, int disp_order, int temp_order, typename... parameter_space>
struct ThermoMechanicsSystem {
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<temp_order>, H1<temp_order>, parameter_space...>>;
  using ThermalWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<temp_order>,
      Parameters<H1<temp_order>, H1<temp_order>, H1<disp_order, dim>, H1<disp_order, dim>, parameter_space...>>;

  std::shared_ptr<FieldStore> field_store;
  std::shared_ptr<SolidWeakFormType> solid_weak_form;
  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;
  std::shared_ptr<DifferentiableBlockSolver> solver;
  std::shared_ptr<StateAdvancer> advancer;
  std::shared_ptr<QuasiStaticFirstOrderTimeIntegrationRule> disp_time_rule;
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule> temperature_time_rule;
  std::vector<FieldState> parameter_fields;

  std::vector<FieldState> getStateFields() const
  {
    std::vector<FieldState> states;
    states.push_back(field_store->getField("displacement"));
    states.push_back(field_store->getField("displacement_old"));
    states.push_back(field_store->getField("temperature"));
    states.push_back(field_store->getField("temperature_old"));
    return states;
  }

  const std::vector<FieldState>& getParameterFields() const { return parameter_fields; }

  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto dtr = disp_time_rule;
    auto ttr = temperature_time_rule;
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

    thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto temperature, auto temperature_old,
                                                        auto disp, auto disp_old, auto... params) {
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
};

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

  std::cout << "num params = " << parameter_fields.size() << std::endl;

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
