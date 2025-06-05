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

#include "serac/physics/functional_residual.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

void pseudoRand(serac::FiniteElementVector& dual)
{
  int sz = dual.Size();
  for (int i = 0; i < sz; ++i) {
    dual(i) = -1.2 + 2.02 * (double(i) / sz);
  }
}

struct ResidualFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = serac::H1<disp_order, dim>;
  using DensitySpace = serac::L2<disp_order - 1>;

  enum STATE
  {
    SHAPE_DISP,
    DISP,
    VELO
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
    serac::FiniteElementState shape_disp =
        serac::StateManager::newState(VectorSpace{}, "shape_displacement", mesh->tag());
    serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());

    states = {shape_disp, disp, velo};
    params = {density};

    for (auto s : states) {
      dual_states.push_back(serac::FiniteElementDual(s.space(), s.name() + "_dual"));
    }
    for (auto p : params) {
      dual_params.push_back(serac::FiniteElementDual(p.space(), p.name() + "_dual"));
    }

    tangent_states = states;
    tangent_params = params;

    std::string physics_name = "fake_physics";

    using TrialSpace = VectorSpace;
    using ShapeSpace = VectorSpace;

    using ResidualT = serac::FunctionalResidual<dim, ShapeSpace, TrialSpace,
                                                serac::Parameters<VectorSpace, VectorSpace, DensitySpace>>;

    std::vector<const mfem::ParFiniteElementSpace*> inputs{&states[STATE::DISP].space(), &states[STATE::VELO].space(),
                                                           &params[PAR::DENSITY].space()};

    auto f_residual = std::make_shared<ResidualT>(physics_name, mesh, states[STATE::SHAPE_DISP].space(),
                                                  states[STATE::DISP].space(), inputs);

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

    for (auto& s : tangent_states) {
      pseudoRand(s);
    }
    for (auto& p : tangent_params) {
      pseudoRand(p);
    }

    dual_states[SHAPE_DISP] = 4.0;
    dual_states[DISP] = 1.0;
    dual_states[VELO] = 2.0;
    dual_params[DENSITY] = 3.0;

    states[DISP].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = 0.1 * x;
      return u;
    });

    states[VELO].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = -0.01 * x;
      return u;
    });

    states[SHAPE_DISP] = 0.0;
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

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  std::vector<serac::FiniteElementDual> dual_states;
  std::vector<serac::FiniteElementDual> dual_params;

  std::vector<serac::FiniteElementState> tangent_states;
  std::vector<serac::FiniteElementState> tangent_params;
};

TEST_F(ResidualFixture, VjpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = constResidualPointers(states, params);

  serac::FiniteElementDual res_vector(states[DISP].space(), "residual");

  res_vector = residual->residual(time, dt, input_fields);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto jacobian_weights = [&](size_t i) {
    std::vector<double> tangents(input_fields.size());
    tangents[i] = 1.0;
    return tangents;
  };

  // test vjp
  serac::FiniteElementState v(res_vector.space(), "v");
  pseudoRand(v);
  auto all_jvps = residualPointers(dual_states, dual_params);

  std::vector<serac::FiniteElementDual> all_Jvps;
  for (auto& jvp : all_jvps) {
    all_Jvps.push_back(*jvp);
  }

  for (size_t i = 0; i < input_fields.size(); ++i) {
    serac::FiniteElementDual& vjp = all_Jvps[i];
    auto J = residual->jacobian(time, dt, input_fields, jacobian_weights(i));
    J->AddMultTranspose(v, vjp);
  }
  residual->vjp(time, dt, input_fields, constResidualPointers(v), all_jvps);

  for (size_t i = 0; i < input_fields.size(); ++i) {
    EXPECT_NEAR(all_Jvps[i].Norml2(), all_jvps[i]->Norml2(), 1e-12) << " " << all_Jvps[i].name() << std::endl;
  }
}

TEST_F(ResidualFixture, JvpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = constResidualPointers(states, params);

  serac::FiniteElementDual res_vector(states[0].space(), "residual");
  res_vector = residual->residual(time, dt, input_fields);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto jacobianWeights = [&](size_t i) {
    std::vector<double> tangents(input_fields.size());
    tangents[i] = 1.0;
    return tangents;
  };

  auto selectStates = [&](size_t i) {
    auto pts = constResidualPointers(tangent_states, tangent_params);
    for (size_t j = 0; j < pts.size(); ++j) {
      if (i != j) {
        pts[j] = nullptr;
      }
    }
    return pts;
  };

  serac::FiniteElementDual jvp_slow(states[0].space(), "jvp_slow");
  serac::FiniteElementDual jvp(states[0].space(), "jvp");
  jvp = 4.0;  // set to some value to test jvp resets these values
  auto jvps = residualPointers(jvp);

  auto all_tangent_fields = constResidualPointers(tangent_states, tangent_params);

  for (size_t i = 0; i < input_fields.size(); ++i) {
    auto J = residual->jacobian(time, dt, input_fields, jacobianWeights(i));
    J->Mult(*all_tangent_fields[i], jvp_slow);
    residual->jvp(time, dt, input_fields, selectStates(i), jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }

  // test jacobians in weighted combinations
  {
    all_tangent_fields[SHAPE_DISP] = nullptr;
    all_tangent_fields[3] = nullptr;

    double velo_factor = 0.2;
    std::vector<double> jacobian_weights = {0.0, 1.0, velo_factor, 0.0};

    auto J = residual->jacobian(time, dt, input_fields, jacobian_weights);
    J->Mult(*all_tangent_fields[DISP], jvp_slow);

    tangent_states[VELO] = tangent_states[0];
    tangent_states[VELO] *= velo_factor;
    residual->jvp(time, dt, input_fields, all_tangent_fields, jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
