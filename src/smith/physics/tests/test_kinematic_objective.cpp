// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/solid_weak_form.hpp"
#include "smith/physics/functional_objective.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/common.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/numerics/functional/tensor.hpp"
#include "smith/physics/field_types.hpp"
#include "smith/physics/scalar_objective.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

struct ConstrainedWeakFormFixture : public testing::Test {
  static constexpr int dim = 3;
  static constexpr int disp_order = 1;

  using VectorSpace = smith::H1<disp_order, dim>;
  using DensitySpace = smith::L2<disp_order - 1>;
  using SolidMaterial = smith::solid_mechanics::NeoHookeanWithFieldDensity;

  using SolidWeakFormT = smith::SolidWeakForm<disp_order, dim, smith::Parameters<DensitySpace>>;

  enum FIELD
  {
    DISP = SolidWeakFormT::DISPLACEMENT,
    VELO = SolidWeakFormT::VELOCITY,
    ACCEL = SolidWeakFormT::ACCELERATION,
    DENSITY = SolidWeakFormT::NUM_STATES
  };

  auto constructWeakForm(const std::string& physics_name)
  {
    auto solid_mechanics_weak_form =
        std::make_shared<SolidWeakFormT>(physics_name, mesh, states[DISP].space(), getSpaces(params));
    // setup material model
    SolidMaterial mat;
    mat.K = 1.0;
    mat.G = 0.5;
    solid_mechanics_weak_form->setMaterial(smith::DependsOn<0>{}, mesh->entireBodyName(), mat);

    // apply some traction boundary conditions
    std::string surface_name = "side";
    mesh->addDomainOfBoundaryElements(surface_name, smith::by_attr<dim>(1));
    solid_mechanics_weak_form->addBoundaryFlux(surface_name, [](auto /*x*/, auto n, auto /*t*/) { return 1.0 * n; });

    return solid_mechanics_weak_form;
  }

  auto constructConstraints()
  {
    std::vector<std::shared_ptr<smith::ScalarObjective>> constraint_evaluators;

    using ObjectiveT = smith::FunctionalObjective<dim, smith::Parameters<VectorSpace, DensitySpace>>;

    smith::TimeInfo time_info(0.0, 0.0);
    auto input_fields = getConstFieldPointers(states, params);
    auto objective_states = {input_fields[DISP], input_fields[DENSITY]};

    ObjectiveT::SpacesT param_space_ptrs{&input_fields[DISP]->space(), &input_fields[DENSITY]->space()};

    ObjectiveT mass_objective("mass constraining", mesh, param_space_ptrs);
    mass_objective.addBodyIntegral(smith::DependsOn<1>{}, mesh->entireBodyName(),
                                   [](double /*time*/, auto /*X*/, auto RHO) { return get<smith::VALUE>(RHO); });

    double mass = mass_objective.evaluate(time_info, shape_disp.get(), objective_states);

    smith::tensor<double, dim> initial_cg;

    for (int i = 0; i < dim; ++i) {
      auto cg_objective = std::make_shared<ObjectiveT>("translation" + std::to_string(i), mesh, param_space_ptrs);
      cg_objective->addBodyIntegral(
          smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
          [i](double
              /*time*/,
              auto X, auto U,
              auto RHO) { return (get<smith::VALUE>(X)[i] + get<smith::VALUE>(U)[i]) * get<smith::VALUE>(RHO); });
      initial_cg[i] = cg_objective->evaluate(time_info, shape_disp.get(), objective_states) / mass;
      constraint_evaluators.push_back(cg_objective);
    }

    for (int i = 0; i < dim; ++i) {
      auto center_rotation_objective =
          std::make_shared<ObjectiveT>("rotation" + std::to_string(i), mesh, param_space_ptrs);
      center_rotation_objective->addBodyIntegral(smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
                                                 [i, initial_cg](double /*time*/, auto X, auto U, auto RHO) {
                                                   auto u = get<smith::VALUE>(U);
                                                   auto x = get<smith::VALUE>(X) + u;
                                                   auto dx = x - initial_cg;
                                                   auto x_cross_u = smith::cross(dx, u);
                                                   return x_cross_u[i] * get<smith::VALUE>(RHO);
                                                 });
      constraint_evaluators.push_back(center_rotation_objective);
    }

    return constraint_evaluators;
  }

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    smith::StateManager::initialize(datastore, "solid_dynamics");

    double xlength = 0.5;
    double ylength = 0.7;
    double zlength = 0.3;
    mesh = std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian3D(6, 4, 4, element_shape, xlength, ylength, zlength),
                                         "this_mesh_name", 0, 0);

    shape_disp = std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());
    smith::FiniteElementState disp = smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    smith::FiniteElementState velo = smith::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
    smith::FiniteElementState accel = smith::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
    smith::FiniteElementState density = smith::StateManager::newState(DensitySpace{}, "density", mesh->tag());

    states = {disp, velo, accel};
    params = {density};

    std::string physics_name = "solid";
    weak_form = constructWeakForm(physics_name);

    params[0] = 1.2;  // set density before computing mass properties
    constraints = constructConstraints();

    // initialize displacement
    states[FIELD::DISP].setFromFieldFunction([](smith::tensor<double, dim> x) {
      auto u = 0.1 * x;
      return u;
    });
  }

  std::unique_ptr<smith::FiniteElementState> shape_disp;
  std::vector<smith::FiniteElementState> states;
  std::vector<smith::FiniteElementState> params;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::shared_ptr<smith::WeakForm> weak_form;
  std::vector<std::shared_ptr<smith::ScalarObjective>> constraints;
};

TEST_F(ConstrainedWeakFormFixture, CanComputeObjectivesAndTheirGradients)
{
  smith::TimeInfo time_info(0.0, 1.0);
  auto input_fields = getConstFieldPointers(states, params);

  smith::FiniteElementDual res_vector(states[DISP].space(), "residual");
  res_vector = weak_form->residual(time_info, shape_disp.get(), input_fields);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto objective_states = {input_fields[DISP], input_fields[DENSITY]};
  for (const auto& c : constraints) {
    ASSERT_NE(0.0, c->evaluate(time_info, shape_disp.get(), objective_states));
    for (size_t f_ordinal = 0; f_ordinal < objective_states.size(); ++f_ordinal) {
      ASSERT_NE(0.0, c->gradient(time_info, shape_disp.get(), objective_states, int(f_ordinal)).Norml2());
    }
    ASSERT_NE(0.0, c->mesh_coordinate_gradient(time_info, shape_disp.get(), objective_states).Norml2());
  }
}

int main(int argc, char* argv[])
{
  smith::ApplicationManager manager(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
