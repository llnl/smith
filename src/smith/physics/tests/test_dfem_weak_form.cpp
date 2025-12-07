// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstddef>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/tests/physics_test_utils.hpp"
#include "smith/physics/dfem_weak_form.hpp"
#include "smith/physics/functional_weak_form.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/field_types.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

struct WeakFormsFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = smith::H1<disp_order, dim>;
  using DensitySpace = smith::L2<disp_order - 1>;

  static constexpr int NUM_FUNCTIONAL_STATES = 2;
  static constexpr int NUM_DFEM_STATES = 3;

  enum STATE
  {
    DISP,
    VELO,
    COORD
  };

  enum PAR
  {
    DENSITY
  };

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    smith::StateManager::initialize(datastore, "solid_dynamics");

    double length = 0.5;
    double width = 2.0;
    mesh = std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian2D(6, 20, element_shape, true, length, width),
                                         "this_mesh_name", 0, 0);

    smith::FiniteElementState disp = smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    smith::FiniteElementState velo = smith::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
    smith::FiniteElementState density = smith::StateManager::newState(DensitySpace{}, "density", mesh->tag());

    shape_disp = std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());
    shape_disp_dual = std::make_unique<smith::FiniteElementDual>(mesh->newShapeDisplacementDual());

    states = {disp, velo};
    params = {density};

    for (auto s : states) {
      state_duals.push_back(smith::FiniteElementDual(s.space(), s.name() + "_dual"));
    }
    for (auto p : params) {
      param_duals.push_back(smith::FiniteElementDual(p.space(), p.name() + "_dual"));
    }

    state_tangents = states;
    param_tangents = params;

    std::string physics_name = "fake_physics";

    using TrialSpace = VectorSpace;
    using FunctionalWeakFormT =
        smith::FunctionalWeakForm<dim, TrialSpace, smith::Parameters<VectorSpace, VectorSpace, DensitySpace>>;
    std::vector<const mfem::ParFiniteElementSpace*> f_inputs{&states[DISP].space(), &states[VELO].space(),
                                                             &params[DENSITY].space()};
    auto f_weak_form = std::make_shared<FunctionalWeakFormT>(physics_name, mesh, states[DISP].space(), f_inputs);

    std::vector<const mfem::ParFiniteElementSpace*> d_inputs{
        &states[DISP].space(), &states[VELO].space(),
        static_cast<const mfem::ParFiniteElementSpace*>(mesh->mfemParMesh().GetNodalFESpace())};
    auto d_weak_form = std::make_shared<smith::DfemWeakForm>(physics_name, mesh, states[DISP].space(), d_inputs);
    mfem::Array<int> solid_attrib({1});
    constexpr int ir_order = 2;
    const mfem::IntegrationRule& displacement_ir = mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order);
    // none of the q functions are dependent on the parameter field
    constexpr auto deriv_indices = std::make_index_sequence<NUM_DFEM_STATES>();

    // body force 1: displacement
    f_weak_form->addBodySource(smith::DependsOn<0>{}, mesh->entireBodyName(),
                               [](double /*t*/, auto /*x*/, auto u) { return u; });
    mfem::future::tuple<mfem::future::Value<DISP>, mfem::future::Value<VELO>, mfem::future::Gradient<COORD>,
                        mfem::future::Weight>
        qfn1_inputs{};
    mfem::future::tuple<mfem::future::Value<NUM_DFEM_STATES>> qfn1_outputs{};
    d_weak_form->addBodyIntegral(
        solid_attrib,
        [=] SMITH_HOST_DEVICE(const mfem::future::tensor<mfem::real_t, dim>& u,
                              const mfem::future::tensor<mfem::real_t, dim>&,
                              const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi, mfem::real_t weight) {
          auto J = mfem::future::det(dX_dxi) * weight;
          return mfem::future::tuple{-u * J};
        },
        qfn1_inputs, qfn1_outputs, displacement_ir, deriv_indices);

    // body force 2: position
    f_weak_form->addBodySource(mesh->entireBodyName(), [](double /*t*/, auto x) { return 0.5 * x; });
    mfem::future::tuple<mfem::future::Value<DISP>, mfem::future::Value<VELO>, mfem::future::Value<COORD>,
                        mfem::future::Gradient<COORD>, mfem::future::Weight>
        qfn2_inputs{};
    mfem::future::tuple<mfem::future::Value<NUM_DFEM_STATES>> qfn2_outputs{};
    d_weak_form->addBodyIntegral(
        solid_attrib,
        [=] SMITH_HOST_DEVICE(const mfem::future::tensor<mfem::real_t, dim>&,
                              const mfem::future::tensor<mfem::real_t, dim>&,
                              const mfem::future::tensor<mfem::real_t, dim>& X,
                              const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi, mfem::real_t weight) {
          auto J = mfem::future::det(dX_dxi) * weight;
          return mfem::future::tuple{-0.5 * X * J};
        },
        qfn2_inputs, qfn2_outputs, displacement_ir, deriv_indices);

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

    states[DISP].setFromFieldFunction([](smith::tensor<double, dim> x) {
      auto u = 0.1 * x;
      return u;
    });

    states[VELO].setFromFieldFunction([](smith::tensor<double, dim> x) {
      auto u = -0.01 * x;
      return u;
    });

    params[DENSITY] = 1.2;

    // functional_weak_form and dfem_weak_form are abstract WeakForm classes to ensure usage only through WeakForm
    // interface
    functional_weak_form = f_weak_form;
    dfem_weak_form = d_weak_form;
  }

  const double time = 0.0;
  const double dt = 1.0;

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::shared_ptr<smith::WeakForm> functional_weak_form;
  std::shared_ptr<smith::WeakForm> dfem_weak_form;

  std::unique_ptr<smith::FiniteElementState> shape_disp;
  std::unique_ptr<smith::FiniteElementDual> shape_disp_dual;

  std::vector<smith::FiniteElementState> states;
  std::vector<smith::FiniteElementState> params;

  std::vector<smith::FiniteElementDual> state_duals;
  std::vector<smith::FiniteElementDual> param_duals;

  std::vector<smith::FiniteElementState> state_tangents;
  std::vector<smith::FiniteElementState> param_tangents;
};

TEST_F(WeakFormsFixture, VjpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto f_input_fields = getConstFieldPointers(states, params);
  smith::FiniteElementState coords(
      *static_cast<const mfem::ParFiniteElementSpace*>(mesh->mfemParMesh().GetNodalFESpace()), "coords");
  coords.setFromGridFunction(*static_cast<const mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes()));
  std::vector<smith::ConstFieldPtr> d_input_fields = {&states[DISP], &states[VELO], &coords};

  smith::FiniteElementDual f_res_vector(states[DISP].space(), "functional_residual");
  f_res_vector = functional_weak_form->residual(time, dt, shape_disp.get(), f_input_fields);
  smith::FiniteElementDual d_res_vector(states[DISP].space(), "dfem_residual");
  d_res_vector = dfem_weak_form->residual(time, dt, shape_disp.get(), d_input_fields);
  EXPECT_NEAR(f_res_vector.Norml2(), d_res_vector.Norml2(), 1.e-12);

  // test vjp
  smith::FiniteElementState v(f_res_vector.space(), "v");
  pseudoRand(v);
  auto f_field_vjps = getFieldPointers(state_duals, param_duals);

  smith::FiniteElementDual coords_dual(coords.space(), coords.name() + "_dual");
  coords_dual = 0.0;
  std::vector<smith::FiniteElementDual> field_vjps_copy = {state_duals[DISP], state_duals[VELO],
                                                           std::move(coords_dual)};
  auto d_field_vjps = getFieldPointers(field_vjps_copy);

  dfem_weak_form->vjp(time, dt, shape_disp.get(), d_input_fields, {}, getConstFieldPointers(v), shape_disp_dual.get(),
                      d_field_vjps, {});
  functional_weak_form->vjp(time, dt, shape_disp.get(), f_input_fields, {}, getConstFieldPointers(v),
                            shape_disp_dual.get(), f_field_vjps, {});

  for (size_t i = 0; i < NUM_FUNCTIONAL_STATES; ++i) {
    EXPECT_NEAR(f_field_vjps[i]->Norml2(), d_field_vjps[i]->Norml2(), 1e-12)
        << " " << d_field_vjps[i]->name() << std::endl;
  }
}

TEST_F(WeakFormsFixture, JvpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = getConstFieldPointers(states, params);

  smith::FiniteElementDual res_vector(states[DISP].space(), "residual");
  res_vector = functional_weak_form->residual(time, dt, shape_disp.get(), input_fields);
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

  smith::FiniteElementDual jvp_slow(states[DISP].space(), "jvp_slow");
  smith::FiniteElementDual jvp(states[DISP].space(), "jvp");
  jvp = 4.0;  // set to some value to test jvp resets these values
  auto jvps = getFieldPointers(jvp);

  auto field_tangents = getConstFieldPointers(state_tangents, param_tangents);

  for (size_t i = 0; i < input_fields.size(); ++i) {
    auto J = functional_weak_form->jacobian(time, dt, shape_disp.get(), input_fields, jacobianWeights(i));
    J->Mult(*field_tangents[i], jvp_slow);
    functional_weak_form->jvp(time, dt, shape_disp.get(), input_fields, {}, nullptr, selectStates(i), {}, jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }

  // test jacobians in weighted combinations
  {
    field_tangents[NUM_FUNCTIONAL_STATES] = nullptr;

    double velo_factor = 0.2;
    std::vector<double> jacobian_weights = {1.0, velo_factor, 0.0};

    auto J = functional_weak_form->jacobian(time, dt, shape_disp.get(), input_fields, jacobian_weights);
    J->Mult(*field_tangents[DISP], jvp_slow);

    state_tangents[VELO] *= velo_factor;
    functional_weak_form->jvp(time, dt, shape_disp.get(), input_fields, {}, nullptr, field_tangents, {}, jvps);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
