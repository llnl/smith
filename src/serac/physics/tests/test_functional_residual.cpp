// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"

#include "serac/physics/functional_residual.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

template <typename T>
auto getPointers(std::vector<T>& states, std::vector<T>& params)
{
  assert(params.size() > 0);
  std::vector<T*> pointers{&params[0]};
  for (auto& t : states) {
    pointers.push_back(&t);
  }
  for (size_t n = 1; n < params.size(); ++n) {
    pointers.push_back(&params[n]);
  }
  return pointers;
}

template <typename T>
auto getPointers(T& v)
{
  return std::vector<T*>{&v};
}

void pseudoRand(serac::FiniteElementState& dual)
{
  int sz = dual.Size();
  for (int i = 0; i < sz; ++i) {
    dual(i) = -1.2 + 2.02 * (double(i) / sz);
  }
}

void pseudoRand(serac::FiniteElementDual& dual)
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
    DISP,
    VELO
  };

  enum PAR
  {
    SHAPE,
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

    states = {disp, velo};
    params = {shape_disp, density};

    dual_states = states;
    dual_params = params;

    v_rhs_states = states;
    v_rhs_params = params;

    std::string physics_name = "fake_physics";

    using TrialSpace = VectorSpace;
    using ShapeSpace = VectorSpace;

    using ResidualT =
        serac::FunctionalResidual<ShapeSpace, TrialSpace, serac::Parameters<VectorSpace, VectorSpace, DensitySpace>>;

    std::vector<const mfem::ParFiniteElementSpace*> inputs{&states[STATE::DISP].space(), &states[STATE::DISP].space(),
                                                           &params[PAR::DENSITY].space()};

    auto f_residual = std::make_shared<ResidualT>(physics_name, mesh, params[PAR::SHAPE].space(),
                                                  states[STATE::DISP].space(), inputs);

    // apply some traction boundary conditions

    std::string surface_name = "side";
    mesh->addDomainOfBoundaryElements(surface_name, serac::by_attr<dim>(1));

    f_residual->addBoundaryIntegral(surface_name,
                                   [](double /*t*/, auto /*x*/, auto n) { return 1.0 * n; });
    f_residual->addBodyIntegral(serac::DependsOn<0>{}, mesh->entireDomainName(), [](double /*t*/, auto /*x*/, auto u) {
      return serac::tuple{serac::get<serac::VALUE>(u), 0.0 * serac::get<serac::DERIVATIVE>(u)};
    });

    f_residual->addBodyIntegral(mesh->entireDomainName(), [](double /*t*/, auto x) {
      return serac::tuple{0.5 * serac::get<serac::VALUE>(x), 0.0 * serac::get<serac::DERIVATIVE>(x)};
    });

    // initialize fields for testing

    for (auto& s : v_rhs_states) {
      pseudoRand(s);
    }
    for (auto& p : v_rhs_params) {
      pseudoRand(p);
    }

    dual_states[0] = 1.0;
    dual_states[1] = 2.0;
    dual_params[0] = 1.0;
    dual_params[1] = 2.0;

    states[0].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = 0.1 * x;
      return u;
    });

    states[1].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = -0.01 * x;
      return u;
    });

    params[0] = 0.0;
    params[1] = 1.2;

    // residual is abstract Residual class to ensure usage only through BasePhysics interface
    residual = f_residual;
  }

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<serac::Residual> residual;

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  std::vector<serac::FiniteElementState> dual_states;
  std::vector<serac::FiniteElementState> dual_params;

  std::vector<serac::FiniteElementState> v_rhs_states;
  std::vector<serac::FiniteElementState> v_rhs_params;
};

TEST_F(ResidualFixture, VjpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  double time = 0.0;
  auto all_states = getPointers(states, params);

  serac::FiniteElementDual res_vector(states[0].space(), "residual");

  res_vector = residual->residual(time, all_states);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto jacobian_weights = [&](size_t i) {
    std::vector<double> tangents(all_states.size());
    tangents[i] = 1.0;
    return tangents;
  };

  // test vjp
  serac::FiniteElementDual v(res_vector.space(), "v");
  pseudoRand(v);
  auto all_jvps = getPointers(dual_states, dual_params);

  std::vector<serac::FiniteElementState> all_Jvps;
  for (auto& jvp : all_jvps) {
    all_Jvps.push_back(*jvp);
  }

  for (size_t i = 0; i < all_states.size(); ++i) {
    serac::FiniteElementState& vjp = all_Jvps[i];
    auto J = residual->jacobian(time, all_states, jacobian_weights(i));
    J->AddMultTranspose(v, vjp);
  }
  residual->vjp(time, all_states, getPointers(v), all_jvps);

  for (size_t i = 0; i < all_states.size(); ++i) {
    EXPECT_NEAR(all_Jvps[i].Norml2(), all_jvps[i]->Norml2(), 1e-12);
  }
}

TEST_F(ResidualFixture, JvpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  double time = 0.0;
  auto all_states = getPointers(states, params);

  serac::FiniteElementDual res_vector(states[0].space(), "residual");
  res_vector = residual->residual(time, all_states);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto jacobianWeights = [&](size_t i) {
    std::vector<double> tangents(all_states.size());
    tangents[i] = 1.0;
    return tangents;
  };

  auto selectStates = [&](size_t i) {
    auto pts = getPointers(v_rhs_states, v_rhs_params);
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
  std::vector<serac::FiniteElementDual*> jvps = getPointers(jvp);

  auto all_v_rhs_states = getPointers(v_rhs_states, v_rhs_params);

  for (size_t i = 0; i < all_states.size(); ++i) {
    auto J = residual->jacobian(time, all_states, jacobianWeights(i));
    J->Mult(*all_v_rhs_states[i], jvp_slow);
    residual->jvp(time, all_states, selectStates(i), jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }

  // test jacobians in weighted combinations
  {
    all_v_rhs_states[1] = nullptr;
    all_v_rhs_states[3] = nullptr;
    all_v_rhs_states[4] = nullptr;

    double acceleration_factor = 0.2;
    std::vector<double> jacobian_weights = {1.0, 0.0, acceleration_factor, 0.0, 0.0};

    auto J = residual->jacobian(time, all_states, jacobian_weights);
    J->Mult(*all_v_rhs_states[0], jvp_slow);

    *all_v_rhs_states[2] = *all_v_rhs_states[0];
    *all_v_rhs_states[2] *= acceleration_factor;
    residual->jvp(time, all_states, all_v_rhs_states, jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);

  return result;
}
