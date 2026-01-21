// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/functional_weak_form.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"

namespace smith {

smith::LinearSolverOptions solid_linear_options{.linear_solver = smith::LinearSolver::CG,
                                                .preconditioner = smith::Preconditioner::HypreJacobi,
                                                .relative_tol = 1e-11,
                                                .absolute_tol = 1e-11,
                                                .max_iterations = 10000,
                                                .print_level = 0};

smith::NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                   .relative_tol = 1.0e-10,
                                                   .absolute_tol = 1.0e-10,
                                                   .max_iterations = 500,
                                                   .print_level = 0};

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
    smith::StateManager::initialize(datastore, "solid");
    auto mfem_shape = mfem::Element::QUADRILATERAL;
    mesh = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
        "mesh", 0, 0);
    mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
    mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  }

  static constexpr double total_simulation_time_ = 1.1;
  static constexpr size_t num_steps_ = 4;
  static constexpr double dt_ = total_simulation_time_ / num_steps_;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
};

template <typename Space, typename Time=void*>
struct FieldType {
  FieldType(std::string n) : name(n) {}
  std::string name;
};

enum class SecondOrderTimeDiscretization
{
  QuasiStatic,
  ImplicitNewmark
};

enum class FirstOrderTimeDiscretization
{
  QuasiStatic,
  BackwardEuler
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
  void addUnknown(FieldType<Space> type)
  {
    to_fields_[type.name] = fields_.size();
    fields_.push_back(smith::createFieldState<Space>(*data_store_, Space{}, type.name, mesh_->tag()));
  }

  template <typename Space>
  auto addDerived(FieldType<Space> type, std::string name)
  {
    to_fields_[name] = fields_.size();
    fields_.push_back(smith::createFieldState<Space>(*data_store_, Space{}, name, mesh_->tag()));
    return FieldType<Space>(name);
  }

  std::shared_ptr<Mesh> mesh_;
  std::shared_ptr<gretl::DataStore> data_store_;

  std::vector<FieldState> shape_disp_;

  std::vector<FieldState> fields_;
  std::map<std::string, size_t> to_fields_;
};

struct NewmarkDot 
{
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto dot([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                             [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                             [[maybe_unused]] const T4& accel_old) const
  {
    return (2.0 / t.dt()) * (field_new - field_old) - velo_old;
  }
};


template <typename FirstType>
void createSpaces(const FieldStore& field_store, std::vector<const mfem::ParFiniteElementSpace*>& spaces, FieldType<FirstType> type)
{
  const size_t test_index = field_store.to_fields_.at(type.name);
  spaces.push_back(&field_store.fields_[test_index].get()->space());
}


template <typename FirstType, typename... Types>
void createSpaces(const FieldStore& field_store, std::vector<const mfem::ParFiniteElementSpace*>& spaces, FirstType type, Types... types)
{
  const size_t test_index = field_store.to_fields_.at(type.name);
  spaces.insert(spaces.begin(), &field_store.fields_[test_index].get()->space());
  createSpaces(field_store, spaces, types...);
}


template <int spatial_dim, typename TestSpaceType, typename... InputSpaceTypes>
auto createWeakForm(std::string name, FieldStore& field_store, FieldType<TestSpaceType> test_type, FieldType<InputSpaceTypes>... field_types)
{
  const size_t test_index = field_store.to_fields_.at(test_type.name);
  const mfem::ParFiniteElementSpace& test_space = field_store.fields_[test_index].get()->space();
  std::vector<const mfem::ParFiniteElementSpace*> input_spaces;
  createSpaces(field_store, input_spaces, field_types...);
  // std::cout << "num spaces = " << input_spaces.size() << std::endl;
  auto weak_form = std::make_shared<TimeDiscretizedWeakForm<spatial_dim, TestSpaceType, Parameters<InputSpaceTypes...> > >(name, field_store.mesh_, test_space, input_spaces);
  return weak_form;
}


TEST_F(SolidMechanicsMeshFixture, A)
{
  SMITH_MARK_FUNCTION;

  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  FieldType<H1<order, dim>> disp_type("displacement");
  FieldType<H1<order>> temperature_type("temperature");

  FieldStore field_store(mesh, 100);

  field_store.addShapeDisp(shape_disp_type);

  field_store.addUnknown(disp_type);
  field_store.addUnknown(temperature_type);

  auto disp_old_type = field_store.addDerived(disp_type, "displacement_old");
  auto velo_old_type = field_store.addDerived(disp_type, "velocity_old");
  auto accel_old_type = field_store.addDerived(disp_type, "acceleration_old");
  auto temperature_old_type = field_store.addDerived(temperature_type, "temperature_old");

  auto disp_time_rule = SecondOrderTimeIntegrationRule(SecondOrderTimeIntegrationMethod::IMPLICIT_NEWMARK);
  auto temperature_time_rule = BackwardEulerFirstOrderTimeIntegrationRule();

  auto solid_weak_form = createWeakForm<dim>("solid", field_store, disp_type, disp_type, temperature_type, disp_old_type, velo_old_type, accel_old_type, temperature_old_type);

  solid_weak_form->addBodyIntegral(mesh->entireBodyName(), [=](auto time_info, auto /*X*/, auto disp, auto temperature, auto disp_old, auto velo_old, auto accel_old, auto temperature_old) {
    auto u = disp_time_rule.value(time_info, disp, disp_old, velo_old, accel_old);
    auto a = disp_time_rule.ddot(time_info, disp, disp_old, velo_old, accel_old);
    auto theta_dot = temperature_time_rule.dot(time_info, temperature, temperature_old);
    return smith::tuple{2.0 * get<VALUE>(a), get<DERIVATIVE>(u) * get<VALUE>(theta_dot)};
  });

  // auto thermal_weak_form = createWeakForm<dim>("thermal", field_store, UnknownArgs<UnknownFields::TEMPERATURE,UnknownFields::DISP>{}, DerivedArgs<>{}, ParameterArgs<>{});
  // thermal_weak_form->addBodyIntegral(mesh->entireBodyName(), [](auto /*t*/, auto /*X*/, auto temperature, auto u) {
  //   auto ones = 0.0*get<VALUE>(u);
  //   ones[0] = 1.0;
  //   return smith::tuple{get<VALUE>(u) + ones, get<DERIVATIVE>(u)};
  // });

  // solve({wf_1, wf_2}, field_store, solver, bcs, ...);
  //auto r = weak_form->residual(TimeInfo(0.0,0.0,0), field_store.shape_disp_[0].get().get(), getConstFieldPointers(field_store.unknowns_, field_store.derived_));
  // std::cout << "norm = " << r.Norml2() << std::endl;

  EXPECT_EQ(0,0);

  // auto derived_types = createDerivedFieldTypes(ddot(disp_type), dot(temperature_type));

  // FieldType<L2<0>> kappa_type;
  // FieldType<L2<0>> mu_type;
  // auto params =

  // auto weak_form =
  //   createWeakForm("solid", smith::DependsOnFields(disp_type, dot(disp_type), ddot(disp_type), temperature_type));
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
