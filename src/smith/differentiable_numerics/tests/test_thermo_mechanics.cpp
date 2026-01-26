// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"

#include "smith/physics/state/state_manager.hpp"

#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"

namespace smith {

/// @brief Green-Saint Venant isotropic thermoelastic model
struct GreenSaintVenantThermoelasticMaterial {
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity

  using State = Empty;

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain(grad_u);
    const auto trEg = tr(Eg);

    // stress
    static constexpr auto I = Identity<dim>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - dim * alpha * (theta - theta_ref)) * I;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);

    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);

    // heat flux
    const auto q0 = -kappa * grad_theta;

    return smith::tuple{Piola, C_v, s0, q0};
  }
};

smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack,
                                          .relative_tol = 1e-8,
                                          .absolute_tol = 1e-8,
                                          .max_iterations = 200,
                                          .print_level = 0};

smith::NonlinearSolverOptions nonlinear_opts{.nonlin_solver = NonlinearSolver::NewtonLineSearch,
                                             .relative_tol = 1.9e-6,
                                             .absolute_tol = 1.0e-10,
                                             .max_iterations = 500,
                                             .max_line_search_iterations = 50,
                                             .print_level = 2};

static constexpr int dim = 3;
static constexpr int order = 1;

struct SolidMechanicsMeshFixture : public testing::Test {
  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 12;
  int num_elements_y = 2;
  int num_elements_z = 2;
  double elem_size = length / num_elements_x;

  void SetUp()
  {
    smith::StateManager::initialize(datastore_, "solid");
    auto mfem_shape = mfem::Element::QUADRILATERAL;
    mesh_ = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
        "mesh", 0, 0);
    mesh_->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
    mesh_->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  }

  static constexpr double total_simulation_time_ = 1.1;
  static constexpr size_t num_steps_ = 4;
  static constexpr double dt_ = total_simulation_time_ / num_steps_;

  axom::sidre::DataStore datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
};

template <typename Space, typename Time = void*>
struct FieldType {
  FieldType(std::string n, int unknown_index_ = -1) : name(n), unknown_index(unknown_index_) {}
  std::string name;
  int unknown_index;
};

struct FieldStore {
  FieldStore(std::shared_ptr<Mesh> mesh, size_t storage_size = 50)
      : mesh_(mesh), data_store_(std::make_shared<gretl::DataStore>(storage_size))
  {
  }

  template <typename Space>
  void addShapeDisp(FieldType<Space> type)
  {
    shape_disp_.push_back(smith::createFieldState<Space>(*data_store_, Space{}, type.name, mesh_->tag()));
  }

  template <typename Space>
  std::shared_ptr<DirichletBoundaryConditions>& addUnknown(FieldType<Space>& type)
  {
    type.unknown_index = static_cast<int>(num_unknowns_);
    to_fields_index_[type.name] = fields_.size();
    to_unknown_index_[type.name] = num_unknowns_;
    FieldState new_field = smith::createFieldState<Space>(*data_store_, Space{}, type.name, mesh_->tag());
    fields_.push_back(new_field);
    ++num_unknowns_;
    boundary_conditions_.push_back(
        std::make_shared<DirichletBoundaryConditions>(mesh_->mfemParMesh(), new_field.get()->space()));
    SLIC_ASSERT(num_unknowns_ == boundary_conditions.size());
    return boundary_conditions_.back();
  }

  template <typename Space>
  auto addDerived(FieldType<Space>, std::string name)
  {
    to_fields_index_[name] = fields_.size();
    fields_.push_back(smith::createFieldState<Space>(*data_store_, Space{}, name, mesh_->tag()));
    return FieldType<Space>(name);
  }

  void addWeakFormUnknownArg(std::string weak_form_name, std::string argument_name, size_t argument_index)
  {
    FieldLabel argument_name_and_index{.field_name = argument_name, .field_index_in_residual = argument_index};
    if (weak_form_name_to_unknown_name_index_.count(weak_form_name)) {
      weak_form_name_to_unknown_name_index_.at(weak_form_name).push_back(argument_name_and_index);
    } else {
      weak_form_name_to_unknown_name_index_[weak_form_name] = std::vector<FieldLabel>{argument_name_and_index};
    }
  }

  void addWeakFormArg(std::string weak_form_name, std::string argument_name, size_t argument_index)
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

  void printMap()
  {
    for (auto& keyval : weak_form_name_to_unknown_name_index_) {
      std::cout << "for residual: " << keyval.first << " ";
      for (auto& name_index : keyval.second) {
        std::cout << "arg " << name_index.field_name << " at " << name_index.field_index_in_residual << ", ";
      }
      std::cout << std::endl;
    }
  }

  std::vector<std::vector<size_t>> indexMap(const std::vector<std::string>& residual_names) const
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

  std::vector<const BoundaryConditionManager*> getBoundaryConditionManagers() const
  {
    std::vector<const BoundaryConditionManager*> bcs;
    for (auto& bc : boundary_conditions_) {
      bcs.push_back(&bc->getBoundaryConditionManager());
    }
    return bcs;
  }

  size_t getFieldIndex(const std::string& field_name) const { return to_fields_index_.at(field_name); }

  const FieldState& getField(size_t field_index) const { return fields_[field_index]; }

  const FieldState& getShapeDisp() const { return shape_disp_[0]; }

  std::vector<FieldState> getFields(const std::string& weak_form_name) const
  {
    auto unknown_field_indices = weak_form_name_to_field_indices_.at(weak_form_name);
    std::vector<FieldState> fields_for_residual;
    for (auto& i : unknown_field_indices) {
      fields_for_residual.push_back(fields_[i]);
    }
    return fields_for_residual;
  }

  const std::shared_ptr<smith::Mesh>& getMesh() const { return mesh_; }

 private:
  std::shared_ptr<Mesh> mesh_;
  std::shared_ptr<gretl::DataStore> data_store_;

  std::vector<FieldState> shape_disp_;
  std::vector<FieldState> fields_;
  std::map<std::string, size_t> to_fields_index_;

  size_t num_unknowns_ = 0;
  std::map<std::string, size_t> to_unknown_index_;
  std::vector<std::shared_ptr<DirichletBoundaryConditions>> boundary_conditions_;

  // MRT, do this for readability
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
  const size_t test_index = field_store.getFieldIndex(type.name);
  SLIC_ERROR_IF(spaces.size() != arg_num, "Error creating spaces recursively");
  spaces.push_back(&field_store.getField(test_index).get()->space());
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
  const size_t test_index = field_store.getFieldIndex(test_type.name);
  const mfem::ParFiniteElementSpace& test_space = field_store.getField(test_index).get()->space();
  std::vector<const mfem::ParFiniteElementSpace*> input_spaces;
  createSpaces(name, field_store, input_spaces, 0, field_types...);
  return std::make_shared<TimeDiscretizedWeakForm<spatial_dim, TestSpaceType, Parameters<InputSpaceTypes...>>>(
      name, field_store.getMesh(), test_space, input_spaces);
}

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

TEST_F(SolidMechanicsMeshFixture, A)
{
  SMITH_MARK_FUNCTION;

  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");

  FieldType<H1<order, dim>> disp_type("displacement");
  FieldType<H1<order>> temperature_type("temperature");

  std::shared_ptr<DifferentiableBlockSolver> d_nonlinear_solver =
      buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);

  FieldStore field_store(mesh_, 100);

  field_store.addShapeDisp(shape_disp_type);

  std::shared_ptr<DirichletBoundaryConditions>& disp_bc = field_store.addUnknown(disp_type);
  disp_bc->setVectorBCs<dim>(mesh_->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = 0.01 * t;
    return bc;
  });
  disp_bc->setFixedVectorBCs<dim, dim>(mesh_->domain("right"));

  auto disp_old_type = field_store.addDerived(disp_type, "displacement_old");
  auto velo_old_type = field_store.addDerived(disp_type, "velocity_old");
  auto accel_old_type = field_store.addDerived(disp_type, "acceleration_old");

  std::shared_ptr<DirichletBoundaryConditions>& temperature_bc = field_store.addUnknown(temperature_type);
  temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));
  temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("right"));

  auto temperature_old_type = field_store.addDerived(temperature_type, "temperature_old");

  auto disp_time_rule = SecondOrderTimeIntegrationRule(SecondOrderTimeIntegrationMethod::IMPLICIT_NEWMARK);
  auto temperature_time_rule = BackwardEulerFirstOrderTimeIntegrationRule();

  auto solid_weak_form = createWeakForm<dim>("solid_force", disp_type, field_store, disp_type, disp_old_type,
                                             velo_old_type, accel_old_type, temperature_type, temperature_old_type);

  solid_weak_form->addBodyIntegral(
      mesh_->entireBodyName(), [=](auto time_info, auto /*X*/, auto disp, auto disp_old, auto velo_old, auto accel_old,
                                   auto temperature, auto temperature_old) {
        auto u = disp_time_rule.value(time_info, disp, disp_old, velo_old, accel_old);
        auto a = disp_time_rule.ddot(time_info, disp, disp_old, velo_old, accel_old);
        auto theta_dot = temperature_time_rule.dot(time_info, temperature, temperature_old);
        return smith::tuple{2.0 * get<VALUE>(a), get<DERIVATIVE>(u) * get<VALUE>(theta_dot)};
      });

  auto thermal_weak_form = createWeakForm<dim>("thermal_flux", temperature_type, field_store, temperature_type,
                                               temperature_old_type, disp_type);

  thermal_weak_form->addBodyIntegral(
      mesh_->entireBodyName(), [=](auto time_info, auto /*X*/, auto temperature, auto temperature_old, auto) {
        auto theta = temperature_time_rule.value(time_info, temperature, temperature_old);
        auto theta_dot = temperature_time_rule.dot(time_info, temperature, temperature_old);
        return smith::tuple{get<VALUE>(theta_dot), get<DERIVATIVE>(theta)};
      });

  std::vector<WeakForm*> weak_forms{solid_weak_form.get(), thermal_weak_form.get()};
  std::vector<FieldState> disp_temp = solve(weak_forms, field_store, d_nonlinear_solver.get(), TimeInfo(0.0, 1.0));

  EXPECT_EQ(0, 0);

  // auto derived_types = createDerivedFieldTypes(ddot(disp_type), dot(temperature_type));
  // FieldType<L2<0>> kappa_type;
  // FieldType<L2<0>> mu_type;
  // auto params =
  // auto weak_form =
  // createWeakForm("solid", smith::DependsOnFields(disp_type, dot(disp_type), ddot(disp_type), temperature_type));
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
