// Copyright Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/mesh_utils/mesh_utils.hpp"
#include "serac/physics/materials/parameterized_thermal_material.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/heat_transfer_residual.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

template <typename T>
auto getPointers(std::vector<T>& values)
{
  std::vector<T*> pointers;
  for (auto& t : values) {
    pointers.push_back(&t);
  }
  return pointers;
}

template <typename T>
auto getPointers(std::vector<T>& states, std::vector<T>& params)
{
  assert(params.size() > 0);
  std::vector<T*> pointers;
  for (auto& t : states) {
    pointers.push_back(&t);
  }
  for (auto& t : params) {
    pointers.push_back(&t);
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
  static constexpr int order = 1;

  using VectorSpace = serac::H1<order, dim>;
  using ScalarSpace = serac::H1<order>;
  using ParamSpace = serac::L2<order - 1>;

  using ThermalMaterial = serac::heat_transfer::ParameterizedLinearIsotropicConductor;

  enum StateOrder
  {
    SHAPE,
    TEMP,
    TRATE
  };

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    serac::StateManager::initialize(datastore, "heat_diffusion");

    mesh = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian2D(10, 10, element_shape, true, 1.0, 1.0),
                                         "this_mesh_name", 0, 0);

    serac::FiniteElementState temperature = serac::StateManager::newState(ScalarSpace{}, "temperature", mesh->tag());
    serac::FiniteElementState temp_rate = serac::StateManager::newState(ScalarSpace{}, "temperature_rate", mesh->tag());
    serac::FiniteElementState shape_disp =
        serac::StateManager::newState(VectorSpace{}, "shape_displacement", mesh->tag());
    serac::FiniteElementState conductivity_offset =
        serac::StateManager::newState(ParamSpace{}, "conductivity_offset", mesh->tag());

    states = {shape_disp, temperature, temp_rate};
    params = {conductivity_offset};

    for (auto s : states) {
      dual_states.push_back(serac::FiniteElementDual(s.space(), s.name() + "_dual"));
    }
    for (auto p : params) {
      dual_params.push_back(serac::FiniteElementDual(p.space(), p.name() + "_dual"));
    }

    v_rhs_states = states;
    v_rhs_params = params;

    std::string physics_name = "heat";

    using HeatResidualT = serac::HeatTransferResidual<order, dim, serac::Parameters<ParamSpace>>;
    auto heat_transfer_residual = std::make_shared<HeatResidualT>(physics_name, mesh, states[SHAPE].space(),
                                                                  states[TEMP].space(), getSpaces(params));

    ThermalMaterial mat(1.0, 1.0, 1.0);
    heat_transfer_residual->setMaterial(serac::DependsOn<0>{}, mesh->entireBodyName(), mat);

    std::string source_name = "source";
    mesh->addDomainOfBoundaryElements(source_name, serac::by_attr<dim>({1, 2}));

    heat_transfer_residual->addBoundaryIntegral(source_name,
                                                [](auto /* t */, auto /* x */, auto /* n */) { return -1.0; });

    for (auto& s : v_rhs_states) {
      pseudoRand(s);
    }
    for (auto& p : v_rhs_params) {
      pseudoRand(p);
    }

    dual_states[0] = 1.0;  // used to test that vjp acts via +=, add initial value to shape displacement dual

    states[TEMP].setFromFieldFunction(
        [](serac::tensor<double, dim> x) { return 0.1 * std::pow(std::pow(x[0], 2.0) + std::pow(x[1], 2.0), 0.5); });
    states[TRATE].setFromFieldFunction(
        [](serac::tensor<double, dim> x) { return 0.01 * std::pow(std::pow(x[0], 2.0) + std::pow(x[1], 2.0), 0.5); });
    states[SHAPE] = 0.0;
    params[0] = 0.5;

    residual = heat_transfer_residual;
  }

  const double time = 0.0;
  const double dt = 1.0;

  axom::sidre::DataStore datastore;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<serac::Residual> residual;

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  std::vector<serac::FiniteElementDual> dual_states;
  std::vector<serac::FiniteElementDual> dual_params;

  std::vector<serac::FiniteElementState> v_rhs_states;
  std::vector<serac::FiniteElementState> v_rhs_params;
};

TEST_F(ResidualFixture, VjpConsistency)
{
  auto all_states = getPointers(states, params);

  serac::FiniteElementDual res_vector(states[TEMP].space(), "residual");
  res_vector = residual->residual(time, dt, all_states);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto jacobian_weights = [&](size_t i) {
    std::vector<double> tangents(all_states.size());
    tangents[i] = 1.0;
    return tangents;
  };

  // test vjp
  serac::FiniteElementState v(res_vector.space(), "v");
  pseudoRand(v);
  auto all_vjps = getPointers(dual_states, dual_params);

  residual->vjp(time, dt, all_states, getPointers(v), all_vjps);

  for (size_t i = 0; i < all_states.size(); ++i) {
    serac::FiniteElementState vjp = *all_states[i];
    vjp = 0.0;
    auto J = residual->jacobian(time, dt, all_states, jacobian_weights(i));
    J->MultTranspose(v, vjp);
    if (i == 0) vjp += 1.0;  // make sure vjp uses +=
    EXPECT_NEAR(vjp.Norml2(), all_vjps[i]->Norml2(), 1e-12);
  }
}

TEST_F(ResidualFixture, JvpConsistency)
{
  auto all_states = getPointers(states, params);

  serac::FiniteElementDual res_vector(states[TEMP].space(), "residual");
  res_vector = residual->residual(time, dt, all_states);
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

  serac::FiniteElementDual jvp_slow(states[TEMP].space(), "jvp_slow");
  serac::FiniteElementDual jvp(states[TEMP].space(), "jvp");
  jvp = 4.0;
  std::vector<serac::FiniteElementDual*> jvps = getPointers(jvp);

  auto all_v_rhs_states = getPointers(v_rhs_states, v_rhs_params);

  for (size_t i = 0; i < all_states.size(); ++i) {
    auto J = residual->jacobian(time, dt, all_states, jacobianWeights(i));
    J->Mult(*all_v_rhs_states[i], jvp_slow);
    residual->jvp(time, dt, all_states, selectStates(i), jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
