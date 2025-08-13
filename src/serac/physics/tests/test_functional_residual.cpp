// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/mesh_utils/mesh_utils.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"

#include "serac/physics/tests/physics_test_utils.hpp"
#include "serac/physics/functional_residual.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

struct ResidualFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = serac::H1<disp_order, dim>;
  using DensitySpace = serac::L2<disp_order - 1>;

  enum STATE
  {
    DISP,
    VELO,
    NUM_STATES
  };

  enum PAR
  {
    DENSITY
  };

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    serac::StateManager::initialize(datastore, "solid_dynamics");

    double length = 0.5;
    double width = 2.0;
    mesh = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian2D(6, 20, element_shape, true, length, width),
                                         "this_mesh_name", 0, 0);

    serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
    serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());

    shape_disp = std::make_unique<serac::FiniteElementState>(mesh->newShapeDisplacement());
    shape_disp_dual = std::make_unique<serac::FiniteElementDual>(mesh->newShapeDisplacementDual());

    states = {disp, velo};
    params = {density};

    for (auto s : states) {
      state_duals.push_back(serac::FiniteElementDual(s.space(), s.name() + "_dual"));
    }
    for (auto p : params) {
      param_duals.push_back(serac::FiniteElementDual(p.space(), p.name() + "_dual"));
    }

    state_tangents = states;
    param_tangents = params;

    std::string physics_name = "fake_physics";

    using TrialSpace = VectorSpace;

    using ResidualT =
        serac::FunctionalResidual<dim, TrialSpace, serac::Parameters<VectorSpace, VectorSpace, DensitySpace>>;

    std::vector<const mfem::ParFiniteElementSpace*> inputs{&states[STATE::DISP].space(), &states[STATE::VELO].space(),
                                                           &params[PAR::DENSITY].space()};

    auto f_residual = std::make_shared<ResidualT>(physics_name, mesh, states[STATE::DISP].space(), inputs);

    // apply some traction boundary conditions

    std::string surface_name = "side";
    mesh->addDomainOfBoundaryElements(surface_name, serac::by_attr<dim>(1));

    f_residual->addBoundaryIntegral(surface_name, [](double /*t*/, auto /*x*/, auto n) { return 1.0 * n; });
    f_residual->addBodyIntegral(serac::DependsOn<0>{}, mesh->entireBodyName(), [](double /*t*/, auto /*x*/, auto u) {
      return serac::tuple{serac::get<serac::VALUE>(u), 0.0 * serac::get<serac::DERIVATIVE>(u)};
    });

    f_residual->addBodyIntegral(mesh->entireBodyName(), [](double /*t*/, auto x) {
      return serac::tuple{0.5 * serac::get<serac::VALUE>(x), 0.0 * serac::get<serac::DERIVATIVE>(x)};
    });

    // initialize fields for testing

    for (auto& s : state_tangents) {
      pseudoRand(s);
    }
    for (auto& p : param_tangents) {
      pseudoRand(p);
    }

    state_duals[DISP] = 1.0;
    state_duals[VELO] = 2.0;
    param_duals[DENSITY] = 3.0;

    states[DISP].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = 0.1 * x;
      return u;
    });

    states[VELO].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = -0.01 * x;
      return u;
    });

    params[DENSITY] = 1.2;

    // residual is abstract Residual class to ensure usage only through BasePhysics interface
    residual = f_residual;
  }

  const double time = 0.0;
  const double dt = 1.0;

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<serac::Residual> residual;

  std::unique_ptr<serac::FiniteElementState> shape_disp;
  std::unique_ptr<serac::FiniteElementDual> shape_disp_dual;

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  std::vector<serac::FiniteElementDual> state_duals;
  std::vector<serac::FiniteElementDual> param_duals;

  std::vector<serac::FiniteElementState> state_tangents;
  std::vector<serac::FiniteElementState> param_tangents;
};

TEST_F(ResidualFixture, VjpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = getConstFieldPointers(states, params);

  serac::FiniteElementDual res_vector(states[DISP].space(), "residual");

  res_vector = residual->residual(time, dt, shape_disp.get(), input_fields);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto jacobian_weights = [&](size_t i) {
    std::vector<double> tangents(input_fields.size());
    tangents[i] = 1.0;
    return tangents;
  };

  // test vjp
  serac::FiniteElementState v(res_vector.space(), "v");
  pseudoRand(v);
  auto field_vjps = getFieldPointers(state_duals, param_duals);

  std::vector<serac::FiniteElementDual> field_vjps_slow;
  for (auto& vjp : field_vjps) {
    field_vjps_slow.push_back(*vjp);
  }

  for (size_t i = 0; i < input_fields.size(); ++i) {
    serac::FiniteElementDual& vjp = field_vjps_slow[i];
    auto J = residual->jacobian(time, dt, shape_disp.get(), input_fields, jacobian_weights(i));
    J->AddMultTranspose(v, vjp);
  }
  residual->vjp(time, dt, shape_disp.get(), input_fields, {}, getConstFieldPointers(v), shape_disp_dual.get(),
                field_vjps, {});

  for (size_t i = 0; i < input_fields.size(); ++i) {
    EXPECT_NEAR(field_vjps_slow[i].Norml2(), field_vjps[i]->Norml2(), 1e-12)
        << " " << field_vjps_slow[i].name() << std::endl;
  }
}

TEST_F(ResidualFixture, JvpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = getConstFieldPointers(states, params);

  serac::FiniteElementDual res_vector(states[DISP].space(), "residual");
  res_vector = residual->residual(time, dt, shape_disp.get(), input_fields);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto jacobianWeights = [&](size_t i) {
    std::vector<double> tangents(input_fields.size());
    tangents[i] = 1.0;
    return tangents;
  };

  auto selectStates = [&](size_t i) {
    auto field_tangents = getConstFieldPointers(state_tangents, param_tangents);
    for (size_t j = 0; j < field_tangents.size(); ++j) {
      if (i != j) {
        field_tangents[j] = nullptr;
      }
    }
    return field_tangents;
  };

  serac::FiniteElementDual jvp_slow(states[DISP].space(), "jvp_slow");
  serac::FiniteElementDual jvp(states[DISP].space(), "jvp");
  jvp = 4.0;  // set to some value to test jvp resets these values
  auto jvps = getFieldPointers(jvp);

  auto field_tangents = getConstFieldPointers(state_tangents, param_tangents);

  for (size_t i = 0; i < input_fields.size(); ++i) {
    auto J = residual->jacobian(time, dt, shape_disp.get(), input_fields, jacobianWeights(i));
    J->Mult(*field_tangents[i], jvp_slow);
    residual->jvp(time, dt, shape_disp.get(), input_fields, {}, nullptr, selectStates(i), {}, jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }

  // test jacobians in weighted combinations
  {
    field_tangents[NUM_STATES] = nullptr;

    double velo_factor = 0.2;
    std::vector<double> jacobian_weights = {1.0, velo_factor, 0.0};

    auto J = residual->jacobian(time, dt, shape_disp.get(), input_fields, jacobian_weights);
    J->Mult(*field_tangents[DISP], jvp_slow);

    state_tangents[VELO] *= velo_factor;
    residual->jvp(time, dt, shape_disp.get(), input_fields, {}, nullptr, field_tangents, {}, jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
