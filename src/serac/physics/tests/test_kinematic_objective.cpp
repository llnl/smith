// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "serac/physics/solid_residual.hpp"
#include "serac/physics/functional_objective.hpp"

#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/common.hpp"
#include "mfem.hpp"
#include "serac/physics/tests/physics_test_utils.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

struct ConstrainedResidualFixture : public testing::Test {
  static constexpr int dim = 3;
  static constexpr int disp_order = 1;

  using VectorSpace = serac::H1<disp_order, dim>;
  using DensitySpace = serac::L2<disp_order - 1>;
  using SolidMaterial = serac::solid_mechanics::NeoHookeanWithFieldDensity;

  enum FIELD
  {
    SHAPE_DISP,
    DISP,
    VELO,
    ACCEL,
    DENSITY
  };

  auto constructResidual(const std::string& physics_name)
  {
    using SolidResidualT = serac::SolidResidual<disp_order, dim, serac::Parameters<DensitySpace>>;
    auto solid_mechanics_residual = std::make_shared<SolidResidualT>(physics_name, mesh, states[SHAPE_DISP].space(),
                                                                     states[DISP].space(), getSpaces(params));
    // setup material model
    SolidMaterial mat;
    mat.K = 1.0;
    mat.G = 0.5;
    solid_mechanics_residual->setMaterial(serac::DependsOn<0>{}, mesh->entireBodyName(), mat);

    // apply some traction boundary conditions
    std::string surface_name = "side";
    mesh->addDomainOfBoundaryElements(surface_name, serac::by_attr<dim>(1));
    solid_mechanics_residual->addBoundaryIntegral(surface_name, [](auto /*x*/, auto n, auto /*t*/) { return 1.0 * n; });

    // residual is abstract Residual class to ensure usage only through BasePhysics interface
    return solid_mechanics_residual;
  }

  auto constructConstraints()
  {
    std::vector<std::shared_ptr<serac::ScalarObjective>> constraint_evaluators;

    using ObjectiveT = serac::FunctionalObjective<dim, VectorSpace, serac::Parameters<VectorSpace, DensitySpace>>;

    double time = 0.0;
    double dt = 0.0;
    auto all_states = getPointers(states, params);
    auto objective_states = {all_states[SHAPE_DISP], all_states[DISP], all_states[DENSITY]};

    ObjectiveT::SpacesT param_space_ptrs{&all_states[DISP]->space(), &all_states[DENSITY]->space()};

    ObjectiveT mass_objective("mass constraing", mesh, all_states[SHAPE_DISP]->space(), param_space_ptrs);
    mass_objective.addBodyIntegral(serac::DependsOn<1>{}, mesh->entireBodyName(),
                                   [](double /*time*/, auto /*X*/, auto RHO) { return get<serac::VALUE>(RHO); });

    double mass = mass_objective.evaluate(time, dt, objective_states);

    serac::tensor<double, dim> initial_cg;

    for (int i = 0; i < dim; ++i) {
      auto cg_objective = std::make_shared<ObjectiveT>("translation" + std::to_string(i), mesh,
                                                       all_states[SHAPE_DISP]->space(), param_space_ptrs);
      cg_objective->addBodyIntegral(
          serac::DependsOn<0, 1>{}, mesh->entireBodyName(),
          [i](double
              /*time*/,
              auto X, auto U,
              auto RHO) { return (get<serac::VALUE>(X)[i] + get<serac::VALUE>(U)[i]) * get<serac::VALUE>(RHO); });
      initial_cg[i] = cg_objective->evaluate(time, dt, objective_states) / mass;
      constraint_evaluators.push_back(cg_objective);
    }

    for (int i = 0; i < dim; ++i) {
      auto center_rotation_objective = std::make_shared<ObjectiveT>("rotation" + std::to_string(i), mesh,
                                                                    all_states[SHAPE_DISP]->space(), param_space_ptrs);
      center_rotation_objective->addBodyIntegral(serac::DependsOn<0, 1>{}, mesh->entireBodyName(),
                                                 [i, initial_cg](double /*time*/, auto X, auto U, auto RHO) {
                                                   auto u = get<serac::VALUE>(U);
                                                   auto x = get<serac::VALUE>(X) + u;
                                                   auto dx = x - initial_cg;
                                                   auto x_cross_u = serac::cross(dx, u);
                                                   return x_cross_u[i] * get<serac::VALUE>(RHO);
                                                 });
      constraint_evaluators.push_back(center_rotation_objective);
    }

    return constraint_evaluators;
  }

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    serac::StateManager::initialize(datastore, "solid_dynamics");

    double xlength = 0.5;
    double ylength = 0.7;
    double zlength = 0.3;
    mesh = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian3D(6, 4, 4, element_shape, xlength, ylength, zlength),
                                         "this_mesh_name", 0, 0);

    serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
    serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
    serac::FiniteElementState shape_disp = serac::StateManager::newState(VectorSpace{}, "shape_disp", mesh->tag());
    serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());

    states = {disp, velo, accel, shape_disp};
    params = {density};

    std::string physics_name = "solid";
    residual = constructResidual(physics_name);

    params[0] = 1.2;  // set density before computing mass properties
    constraints = constructConstraints();

    // initialize displacement
    states[FIELD::DISP].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = 0.1 * x;
      return u;
    });
  }

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  axom::sidre::DataStore datastore;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<serac::Residual> residual;
  std::vector<std::shared_ptr<serac::ScalarObjective>> constraints;
};

TEST_F(ConstrainedResidualFixture, CanComputeResidualObjectivesAndTheirGradients)
{
  double time = 0.0;
  double dt = 1.0;
  auto all_states = getPointers(states, params);

  serac::FiniteElementDual res_vector(states[0].space(), "residual");
  res_vector = residual->residual(time, dt, all_states);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto objective_states = {all_states[SHAPE_DISP], all_states[DISP], all_states[DENSITY]};
  for (const auto& c : constraints) {
    ASSERT_NE(0.0, c->evaluate(time, dt, objective_states));
    for (int i = 0; i < dim; ++i) {
      ASSERT_NE(0.0, c->gradient(time, dt, objective_states, i).Norml2());
    }
  }
}

int main(int argc, char* argv[])
{
  serac::ApplicationManager manager(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
