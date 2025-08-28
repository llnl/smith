// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
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
#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/tests/physics_test_utils.hpp"
#include "serac/physics/solid_weak_form.hpp"
#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/numerics/functional/finite_element.hpp"  // for H1
#include "serac/numerics/functional/tensor.hpp"
#include "serac/physics/common.hpp"
#include "serac/physics/field_types.hpp"
#include "serac/physics/state/finite_element_dual.hpp"
#include "serac/physics/state/finite_element_state.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

namespace serac {

struct NeoHookeanWithFieldWithRateForTesting {
  using State = Empty;  ///< this material has no internal variables

  template <typename T1, typename T2, int dim>
  SERAC_HOST_DEVICE auto pkStress(double /*dt*/, State& /* state */, const tensor<T1, dim, dim>& du_dX,
                                  const tensor<T2, dim, dim>& /*dv_dX*/) const
  {
    using std::log1p;
    constexpr auto I = Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return dot(TK, inv(transpose(F)));
  }

  SERAC_HOST_DEVICE auto density() const { return Rho; }

  double K;    ///< bulk modulus
  double G;    ///< shear modulus
  double Rho;  ///< density
};

}  // namespace serac

struct WeakFormFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = serac::H1<disp_order, dim>;
  using DensitySpace = serac::L2<disp_order - 1>;

  using SolidMaterial = serac::solid_mechanics::NeoHookeanWithFieldDensity;
  using SolidRateMaterial = serac::NeoHookeanWithFieldWithRateForTesting;

  enum PARAMS
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
    serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
    serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());

    shape_disp = std::make_unique<serac::FiniteElementState>(mesh->newShapeDisplacement());
    shape_disp_dual = std::make_unique<serac::FiniteElementDual>(mesh->newShapeDisplacementDual());

    states = {disp, velo, accel};
    params = {density};

    for (auto s : states) {
      state_duals.push_back(serac::FiniteElementDual(s.space(), s.name() + "_dual"));
    }
    for (auto p : params) {
      param_duals.push_back(serac::FiniteElementDual(p.space(), p.name() + "_dual"));
    }

    state_tangents = states;
    param_tangents = params;

    std::string physics_name = "solid";

    auto solid_mechanics_weak_form = std::make_shared<SolidWeakFormT>(
        physics_name, mesh, states[SolidWeakFormT::DISPLACEMENT].space(), getSpaces(params));
    SolidMaterial mat;
    mat.K = 1.0;
    mat.G = 0.5;
    SolidRateMaterial rate_mat;
    rate_mat.K = 1.0;
    rate_mat.G = 0.5;
    rate_mat.Rho = 1.5;

    solid_mechanics_weak_form->setMaterial(serac::DependsOn<0>{}, mesh->entireBodyName(), mat);
    solid_mechanics_weak_form->setRateMaterial(serac::DependsOn<>{}, mesh->entireBodyName(), rate_mat);

    // apply traction boundary conditions
    std::string surface_name = "side";
    mesh->addDomainOfBoundaryElements(surface_name, serac::by_attr<dim>(1));
    solid_mechanics_weak_form->addBoundaryFlux(surface_name, [](auto /*t*/, auto /*x*/, auto n) { return 1.0 * n; });
    solid_mechanics_weak_form->addPressure(surface_name, [](auto /*t*/, auto /*x*/) { return 0.6; });

    // initialize fields for testing
    for (auto& s : state_tangents) {
      pseudoRand(s);
    }
    for (auto& p : param_tangents) {
      pseudoRand(p);
    }

    state_duals[0] = 1.0;  // used to test that vjp acts via +=, add initial value to shape displacement dual

    states[SolidWeakFormT::DISPLACEMENT].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = 0.1 * x;
      return u;
    });
    states[SolidWeakFormT::VELOCITY].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = -0.1 * x;
      return u;
    });
    states[SolidWeakFormT::ACCELERATION].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = -0.01 * x;
      return u;
    });
    params[0] = 1.2;

    // weak_form is abstract WeakForm class to ensure usage only through WeakForm interface
    weak_form = solid_mechanics_weak_form;
  }

  using SolidWeakFormT = serac::SolidWeakForm<disp_order, dim, serac::Parameters<DensitySpace>>;

  const double time = 0.0;
  const double dt = 1.0;

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<serac::WeakForm> weak_form;

  std::unique_ptr<serac::FiniteElementState> shape_disp;
  std::unique_ptr<serac::FiniteElementDual> shape_disp_dual;

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  std::vector<serac::FiniteElementDual> state_duals;
  std::vector<serac::FiniteElementDual> param_duals;

  std::vector<serac::FiniteElementState> state_tangents;
  std::vector<serac::FiniteElementState> param_tangents;
};

TEST_F(WeakFormFixture, VjpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = getConstFieldPointers(states, params);

  serac::FiniteElementDual res_vector(states[SolidWeakFormT::DISPLACEMENT].space(), "residual");
  res_vector = weak_form->residual(time, dt, shape_disp.get(), input_fields);
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

  weak_form->vjp(time, dt, shape_disp.get(), input_fields, {}, getConstFieldPointers(v), shape_disp_dual.get(),
                 field_vjps, {});

  for (size_t i = 0; i < input_fields.size(); ++i) {
    serac::FiniteElementState vjp_slow = *input_fields[i];
    vjp_slow = 0.0;
    auto J = weak_form->jacobian(time, dt, shape_disp.get(), input_fields, jacobian_weights(i));
    J->MultTranspose(v, vjp_slow);
    if (i == 0) vjp_slow += 1.0;  // make sure vjp uses +=
    EXPECT_NEAR(vjp_slow.Norml2(), field_vjps[i]->Norml2(), 1e-12);
  }
}

TEST_F(WeakFormFixture, JvpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = getConstFieldPointers(states, params);

  serac::FiniteElementDual res_vector(states[SolidWeakFormT::DISPLACEMENT].space(), "residual");
  res_vector = weak_form->residual(time, dt, shape_disp.get(), input_fields);
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

  serac::FiniteElementDual jvp_slow(states[SolidWeakFormT::DISPLACEMENT].space(), "jvp_slow");
  serac::FiniteElementDual jvp(states[SolidWeakFormT::DISPLACEMENT].space(), "jvp");
  jvp = 4.0;  // set to some value to test jvp resets these values
  auto jvps = getFieldPointers(jvp);

  auto field_tangents = getConstFieldPointers(state_tangents, param_tangents);

  for (size_t i = 0; i < input_fields.size(); ++i) {
    auto J = weak_form->jacobian(time, dt, shape_disp.get(), input_fields, jacobianWeights(i));
    J->Mult(*field_tangents[i], jvp_slow);
    weak_form->jvp(time, dt, shape_disp.get(), input_fields, {}, nullptr, selectStates(i), {}, jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }

  // test jacobians in weighted combinations
  {
    field_tangents[SolidWeakFormT::VELOCITY] = nullptr;
    field_tangents[size_t(SolidWeakFormT::NUM_STATES) + size_t(DENSITY)] = nullptr;

    double acceleration_factor = 0.2;
    std::vector<double> jacobian_weights = {1.0, 0.0, acceleration_factor, 0.0};

    auto J = weak_form->jacobian(time, dt, shape_disp.get(), input_fields, jacobian_weights);
    J->Mult(*field_tangents[SolidWeakFormT::DISPLACEMENT], jvp_slow);

    state_tangents[SolidWeakFormT::ACCELERATION] *= acceleration_factor;

    weak_form->jvp(time, dt, shape_disp.get(), input_fields, {}, nullptr, field_tangents, {}, jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
