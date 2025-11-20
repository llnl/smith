// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstddef>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/materials/parameterized_thermal_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/tests/physics_test_utils.hpp"
#include "smith/physics/functional_weak_form.hpp"
#include "smith/physics/heat_transfer_weak_form.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/field_types.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

struct WeakFormFixture : public testing::Test {
  WeakFormFixture() : time_info(time, dt) {}

  static constexpr int dim = 2;
  static constexpr int order = 1;

  using ScalarSpace = smith::H1<order>;
  using ParamSpace = smith::L2<order - 1>;

  using ThermalMaterial = smith::heat_transfer::ParameterizedLinearIsotropicConductor;

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    smith::StateManager::initialize(datastore, "heat_diffusion");

    mesh = std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian2D(10, 10, element_shape, true, 1.0, 1.0),
                                         "this_mesh_name", 0, 0);

    smith::FiniteElementState temperature = smith::StateManager::newState(ScalarSpace{}, "temperature", mesh->tag());
    smith::FiniteElementState temperature_rate =
        smith::StateManager::newState(ScalarSpace{}, "temperature_rate", mesh->tag());
    smith::FiniteElementState conductivity_offset =
        smith::StateManager::newState(ParamSpace{}, "conductivity_offset", mesh->tag());

    shape_disp = std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());
    shape_disp_dual = std::make_unique<smith::FiniteElementDual>(mesh->newShapeDisplacementDual());

    states = {temperature, temperature_rate};
    params = {conductivity_offset};

    for (auto s : states) {
      state_duals.push_back(smith::FiniteElementDual(s.space(), s.name() + "_dual"));
    }
    for (auto p : params) {
      param_duals.push_back(smith::FiniteElementDual(p.space(), p.name() + "_dual"));
    }

    state_tangents = states;
    param_tangents = params;

    std::string physics_name = "heat";

    auto heat_transfer_weak_form = std::make_shared<HeatWeakFormT>(
        physics_name, mesh, states[HeatWeakFormT::TEMPERATURE].space(), getSpaces(params));

    ThermalMaterial mat(1.0, 1.0, 1.0);
    heat_transfer_weak_form->setMaterial(smith::DependsOn<0>{}, mesh->entireBodyName(), mat);

    std::string source_name = "source";
    mesh->addDomainOfBoundaryElements(source_name, smith::by_attr<dim>({1, 2}));

    heat_transfer_weak_form->addBoundaryFlux(source_name, [](auto /* t */, auto /* x */, auto /* n */) { return 1.0; });

    for (auto& s : state_tangents) {
      pseudoRand(s);
    }
    for (auto& p : param_tangents) {
      pseudoRand(p);
    }

    // used to test that vjp acts via +=, add initial value to shape displacement dual
    state_duals[HeatWeakFormT::TEMPERATURE] = 1.0;

    states[HeatWeakFormT::TEMPERATURE].setFromFieldFunction(
        [](smith::tensor<double, dim> x) { return 0.1 * std::pow(std::pow(x[0], 2.0) + std::pow(x[1], 2.0), 0.5); });
    states[HeatWeakFormT::TEMPERATURE_RATE].setFromFieldFunction([](smith::tensor<double, dim> x) {
      return 0.01 * std::pow(std::pow(x[0], 2.0) + std::pow(0.5 * x[1], 2.0), 0.5);
    });
    params[0] = 0.5;

    weak_form = heat_transfer_weak_form;
  }

  using HeatWeakFormT = smith::HeatTransferWeakForm<order, dim, smith::Parameters<ParamSpace>>;

  const double time = 0.0;
  const double dt = 1.0;
  smith::TimeInfo time_info;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::shared_ptr<smith::WeakForm> weak_form;

  std::unique_ptr<smith::FiniteElementState> shape_disp;
  std::unique_ptr<smith::FiniteElementDual> shape_disp_dual;

  std::vector<smith::FiniteElementState> states;
  std::vector<smith::FiniteElementState> params;

  std::vector<smith::FiniteElementDual> state_duals;
  std::vector<smith::FiniteElementDual> param_duals;

  std::vector<smith::FiniteElementState> state_tangents;
  std::vector<smith::FiniteElementState> param_tangents;
};

TEST_F(WeakFormFixture, VjpConsistency)
{
  auto input_fields = getConstFieldPointers(states, params);

  smith::FiniteElementDual res_vector(states[HeatWeakFormT::TEMPERATURE].space(), "residual");
  res_vector = weak_form->residual(time_info, shape_disp.get(), input_fields);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto jacobian_weights = [&](size_t i) {
    std::vector<double> tangents(input_fields.size());
    tangents[i] = 1.0;
    return tangents;
  };

  // test vjp
  smith::FiniteElementState v(res_vector.space(), "v");
  pseudoRand(v);
  auto field_vjps = getFieldPointers(state_duals, param_duals);

  weak_form->vjp(time_info, shape_disp.get(), input_fields, {}, &v, shape_disp_dual.get(), field_vjps, {});

  for (size_t i = 0; i < input_fields.size(); ++i) {
    smith::FiniteElementState vjp = *input_fields[i];
    vjp = 0.0;
    auto J = weak_form->jacobian(time_info, shape_disp.get(), input_fields, jacobian_weights(i));
    J->MultTranspose(v, vjp);
    if (i == HeatWeakFormT::TEMPERATURE) vjp += 1.0;  // make sure vjp uses +=
    EXPECT_NEAR(vjp.Norml2(), field_vjps[i]->Norml2(), 1e-12);
  }
}

TEST_F(WeakFormFixture, JvpConsistency)
{
  auto input_fields = getConstFieldPointers(states, params);

  smith::FiniteElementDual res_vector(states[HeatWeakFormT::TEMPERATURE].space(), "residual");
  res_vector = weak_form->residual(time_info, shape_disp.get(), input_fields);
  ASSERT_NE(0.0, res_vector.Norml2());

  auto jacobian_weights = [&](size_t i) {
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

  smith::FiniteElementDual jvp_slow(states[HeatWeakFormT::TEMPERATURE].space(), "jvp_slow");
  smith::FiniteElementDual jvp(states[HeatWeakFormT::TEMPERATURE].space(), "jvp");
  jvp = 4.0;

  auto field_tangents = getFieldPointers(state_tangents, param_tangents);

  for (size_t i = 0; i < input_fields.size(); ++i) {
    auto J = weak_form->jacobian(time_info, shape_disp.get(), input_fields, jacobian_weights(i));
    J->Mult(*field_tangents[i], jvp_slow);
    weak_form->jvp(time_info, shape_disp.get(), input_fields, {}, nullptr, selectStates(i), {}, &jvp);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
