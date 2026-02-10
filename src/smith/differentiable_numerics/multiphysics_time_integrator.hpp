// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
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

  template <typename Space>
  void addShapeDisp(FieldType<Space> type)
  {
    shape_disp_.push_back(smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag()));
  }

  std::shared_ptr<DirichletBoundaryConditions> addBoundaryConditions(FEFieldPtr field);

  template <typename Space>
  std::shared_ptr<DirichletBoundaryConditions> addIndependent(FieldType<Space>& type)
  {
    type.unknown_index = static_cast<int>(num_unknowns_);
    to_fields_index_[type.name] = fields_.size();
    to_unknown_index_[type.name] = num_unknowns_;
    FieldState new_field = smith::createFieldState<Space>(*graph_, Space{}, type.name, mesh_->tag());
    fields_.push_back(new_field);
    auto latest_bc = addBoundaryConditions(new_field.get());
      ++num_unknowns_;
    SLIC_ERROR_IF(num_unknowns_ != boundary_conditions_.size(),
                  "Inconcistency between num unknowns and boundary condition size");
    return latest_bc; //boundary_conditions_.back();
  }

  template <typename Space>
  auto addDependent(FieldType<Space>, std::string name)
  {
    to_fields_index_[name] = fields_.size();
    fields_.push_back(smith::createFieldState<Space>(*graph_, Space{}, name, mesh_->tag()));
    return FieldType<Space>(name);
  }

  void addWeakFormUnknownArg(std::string weak_form_name, std::string argument_name, size_t argument_index);

  void addWeakFormArg(std::string weak_form_name, std::string argument_name, size_t argument_index);

  void printMap();

  std::vector<std::vector<size_t>> indexMap(const std::vector<std::string>& residual_names) const;

  std::vector<const BoundaryConditionManager*> getBoundaryConditionManagers() const;

  size_t getFieldIndex(const std::string& field_name) const;

  const FieldState& getField(const std::string& field_name) const;

  void setField(const std::string& field_name, FieldState updated_field);

  const FieldState& getShapeDisp() const;

  const std::vector<FieldState>& getAllFields() const;

  std::vector<FieldState> getFields(const std::string& weak_form_name) const;

  const std::shared_ptr<smith::Mesh>& getMesh() const;

 private:
  std::shared_ptr<Mesh> mesh_;
  std::shared_ptr<gretl::DataStore> graph_;

  std::vector<FieldState> shape_disp_;
  std::vector<FieldState> fields_;
  std::map<std::string, size_t> to_fields_index_;

  size_t num_unknowns_ = 0;
  std::map<std::string, size_t> to_unknown_index_;
  std::vector<std::shared_ptr<DirichletBoundaryConditions>> boundary_conditions_;

  struct FieldLabel {
    std::string field_name;
    size_t field_index_in_residual;
  };

  std::map<std::string, std::vector<FieldLabel>> weak_form_name_to_unknown_name_index_;

  std::map<std::string, std::vector<size_t>> weak_form_name_to_field_indices_;
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
  const mfem::ParFiniteElementSpace& test_space = field_store.getField(test_type.name).get()->space();
  std::vector<const mfem::ParFiniteElementSpace*> input_spaces;
  createSpaces(name, field_store, input_spaces, 0, field_types...);
  return std::make_shared<TimeDiscretizedWeakForm<spatial_dim, TestSpaceType, Parameters<InputSpaceTypes...>>>(
      name, field_store.getMesh(), test_space, input_spaces);
}

std::vector<FieldState> solve(const std::vector<std::shared_ptr<WeakForm>>& weak_forms, const FieldStore& field_store,
                              const DifferentiableBlockSolver* solver, const TimeInfo& time_info);


// MultiStateAdvancer advancer(mesh, {weak_forms}, field_store, {time_integration_rules}, solver);
// DifferentiablePhysics(advancer);
// auto states = field_store.getFields();

class MultiPhysicsTimeIntegrator : public StateAdvancer {
  public:
  MultiPhysicsTimeIntegrator(std::shared_ptr<FieldStore> field_store, const std::vector<std::shared_ptr<WeakForm>>& weak_forms, const std::vector<std::shared_ptr<TimeIntegrationRule>>& time_integrators, std::shared_ptr<smith::DifferentiableBlockSolver> solver) :
    field_store_(field_store), weak_forms_(weak_forms), time_integrators_(time_integrators), solver_(solver)
  {

  }

   std::vector<FieldState> advanceState(const TimeInfo& time_info, const FieldState& shape_disp,
                                               const std::vector<FieldState>& states,
                                               const std::vector<FieldState>& params) const override {

    
    std::vector<FieldState> disp_temp = solve(weak_forms_, *field_store_, solver_.get(), time_info);
    
    
    
  }

  private:
  
  std::shared_ptr<FieldStore> field_store_;
  std::vector<std::shared_ptr<WeakForm>> weak_forms_;
  std::vector<std::shared_ptr<TimeIntegrationRule>> time_integrators_;
  std::shared_ptr<smith::DifferentiableBlockSolver> solver_;
};

}  // namespace smith
