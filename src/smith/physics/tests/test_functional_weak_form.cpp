// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstddef>
#include <memory>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/tests/physics_test_utils.hpp"
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

struct WeakFormFixture : public testing::Test {
  WeakFormFixture() : time_info(time, dt) {}

  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = smith::H1<disp_order, dim>;
  using DensitySpace = smith::L2<disp_order - 1>;

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

    using WeakFormT =
        smith::FunctionalWeakForm<dim, TrialSpace, smith::Parameters<VectorSpace, VectorSpace, DensitySpace>>;

    std::vector<const mfem::ParFiniteElementSpace*> inputs{&states[STATE::DISP].space(), &states[STATE::VELO].space(),
                                                           &params[PAR::DENSITY].space()};

    auto f_weak_form = std::make_shared<WeakFormT>(physics_name, mesh, states[STATE::DISP].space(), inputs);

    // apply some traction boundary conditions

    std::string surface_name = "side";
    mesh->addDomainOfBoundaryElements(surface_name, smith::by_attr<dim>(1));

    f_weak_form->addBoundaryFlux(surface_name,
                                 [](auto /*t_info*/, auto /*x*/, auto n, auto... /*args*/) { return 1.0 * n; });
    f_weak_form->addBodySource(mesh->entireBodyName(),
                               [](auto /*t_info*/, auto /*x*/, auto u, auto... /*args*/) { return u; });
    f_weak_form->addBodySource(mesh->entireBodyName(),
                               [](auto /*t_info*/, auto x, auto... /*args*/) { return 0.5 * x; });

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

    // weak_form is abstract WeakForm class to ensure usage only through WeakForm interface
    weak_form = f_weak_form;
  }

  const double time = 0.0;
  const double dt = 1.0;
  smith::TimeInfo time_info;

  std::string velo_name = "solid_velocity";

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
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = getConstFieldPointers(states, params);

  smith::FiniteElementDual res_vector(states[DISP].space(), "residual");
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

  std::vector<smith::FiniteElementDual> field_vjps_slow;
  for (auto& vjp : field_vjps) {
    field_vjps_slow.push_back(*vjp);
  }

  for (size_t i = 0; i < input_fields.size(); ++i) {
    smith::FiniteElementDual& vjp = field_vjps_slow[i];
    auto J = weak_form->jacobian(time_info, shape_disp.get(), input_fields, jacobian_weights(i));
    J->AddMultTranspose(v, vjp);
  }
  weak_form->vjp(time_info, shape_disp.get(), input_fields, {}, &v, shape_disp_dual.get(), field_vjps, {});

  for (size_t i = 0; i < input_fields.size(); ++i) {
    EXPECT_NEAR(field_vjps_slow[i].Norml2(), field_vjps[i]->Norml2(), 1e-12)
        << " " << field_vjps_slow[i].name() << std::endl;
  }
}

TEST_F(WeakFormFixture, JvpConsistency)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = getConstFieldPointers(states, params);

  smith::FiniteElementDual res_vector(states[DISP].space(), "residual");
  res_vector = weak_form->residual(time_info, shape_disp.get(), input_fields);
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

  auto field_tangents = getConstFieldPointers(state_tangents, param_tangents);

  for (size_t i = 0; i < input_fields.size(); ++i) {
    auto J = weak_form->jacobian(time_info, shape_disp.get(), input_fields, jacobianWeights(i));
    J->Mult(*field_tangents[i], jvp_slow);
    weak_form->jvp(time_info, shape_disp.get(), input_fields, {}, nullptr, selectStates(i), {}, &jvp);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }

  // test jacobians in weighted combinations
  {
    field_tangents[NUM_STATES] = nullptr;

    double velo_factor = 0.2;
    std::vector<double> jacobian_weights = {1.0, velo_factor, 0.0};

    auto J = weak_form->jacobian(time_info, shape_disp.get(), input_fields, jacobian_weights);
    J->Mult(*field_tangents[DISP], jvp_slow);

    state_tangents[VELO] *= velo_factor;
    weak_form->jvp(time_info, shape_disp.get(), input_fields, {}, nullptr, field_tangents, {}, &jvp);
    EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
  }
}

TEST_F(WeakFormFixture, PreAssemblyCallbackRefreshesAuxiliaryFieldForEveryEvaluationPath)
{
  // Use the deliberately simple model R(u; p) = p u with the stored auxiliary
  // parameter p refreshed as G(u) = u(0). Because p is stored independently,
  // changing u alone cannot keep it current. Resetting p before each public
  // evaluation detects stale values or a missing callback on that path. This
  // test checks Smith's refresh and frozen-auxiliary contract, not exact total
  // differentiation through G or any particular auxiliary-field algorithm.
  using WeakFormT =
      smith::FunctionalWeakForm<dim, VectorSpace, smith::Parameters<VectorSpace, VectorSpace, DensitySpace>>;

  std::vector<const mfem::ParFiniteElementSpace*> inputs{&states[DISP].space(), &states[VELO].space(),
                                                         &params[DENSITY].space()};
  auto callback_form = std::make_shared<WeakFormT>("callback_refresh", mesh, states[DISP].space(), inputs);
  callback_form->addBodySource(smith::DependsOn<DISP, NUM_STATES>{}, mesh->entireBodyName(),
                               [](const smith::TimeInfo& /*t_info*/, auto /*x*/, auto displacement, auto density) {
                                 return density * displacement;
                               });

  int callback_count = 0;
  auto refresh_auxiliary = [&](const std::vector<smith::ConstFieldPtr>& current_fields) {
    ++callback_count;
    params[DENSITY] = (*current_fields.at(DISP))(0);
  };
  callback_form->setPreAssemblyCallback(refresh_auxiliary);

  states[DISP] = 2.0;
  states[VELO] = 0.0;
  auto input_fields = getConstFieldPointers(states, params);
  auto reset_auxiliary = [&]() { params[DENSITY] = 0.0; };

  reset_auxiliary();
  auto residual = callback_form->residual(time_info, shape_disp.get(), input_fields);
  EXPECT_DOUBLE_EQ(params[DENSITY](0), 2.0);
  EXPECT_GT(residual.Norml2(), 0.0);

  states[DISP] = 3.0;
  reset_auxiliary();
  auto updated_residual = callback_form->residual(time_info, shape_disp.get(), input_fields);
  EXPECT_DOUBLE_EQ(params[DENSITY](0), 3.0);
  // Both factors change from 2 to 3, so R must scale by (3 / 2)^2. This proves
  // the refreshed parameter is consumed by assembly, not merely assigned.
  mfem::Vector expected_updated_residual(residual);
  expected_updated_residual *= 9.0 / 4.0;
  updated_residual -= expected_updated_residual;
  EXPECT_LE(updated_residual.Norml2(), 1.0e-12);

  reset_auxiliary();
  std::vector<double> jacobian_weights(input_fields.size(), 0.0);
  jacobian_weights[DISP] = 1.0;
  auto J = callback_form->jacobian(time_info, shape_disp.get(), input_fields, jacobian_weights);
  state_tangents[DISP] = 1.0;
  smith::FiniteElementDual jacobian_action(states[DISP].space(), "callback_jacobian_action");
  J->Mult(state_tangents[DISP], jacobian_action);
  EXPECT_DOUBLE_EQ(params[DENSITY](0), 3.0);
  EXPECT_GT(jacobian_action.Norml2(), 0.0);

  // The callback is outside automatic differentiation. Holding the refreshed
  // p = 3 fixed, the assembled partial derivative dR/du = p must match this
  // central difference. This is intentionally the frozen-parameter derivative.
  constexpr double perturbation = 1.0e-6;
  callback_form->setPreAssemblyCallback({});
  params[DENSITY] = 3.0;
  states[DISP] = 3.0 + perturbation;
  auto frozen_plus = callback_form->residual(time_info, shape_disp.get(), input_fields);
  params[DENSITY] = 3.0;
  states[DISP] = 3.0 - perturbation;
  auto frozen_minus = callback_form->residual(time_info, shape_disp.get(), input_fields);
  states[DISP] = 3.0;
  frozen_plus -= frozen_minus;
  frozen_plus /= 2.0 * perturbation;
  frozen_plus -= jacobian_action;
  EXPECT_LE(frozen_plus.Norml2(), 1.0e-8 * jacobian_action.Norml2() + 1.0e-12);

  // With refresh enabled, p(u) = u(0), so the total derivative along the
  // constant direction is d(u^2)/du = 2u. This analytic comparison distinguishes
  // that composite derivative from the frozen-p Jacobian Smith assembles.
  callback_form->setPreAssemblyCallback(refresh_auxiliary);
  reset_auxiliary();
  states[DISP] = 3.0 + perturbation;
  auto refreshed_plus = callback_form->residual(time_info, shape_disp.get(), input_fields);
  reset_auxiliary();
  states[DISP] = 3.0 - perturbation;
  auto refreshed_minus = callback_form->residual(time_info, shape_disp.get(), input_fields);
  states[DISP] = 3.0;
  refreshed_plus -= refreshed_minus;
  refreshed_plus /= 2.0 * perturbation;
  mfem::Vector twice_frozen_jacobian(jacobian_action);
  twice_frozen_jacobian *= 2.0;
  refreshed_plus -= twice_frozen_jacobian;
  EXPECT_LE(refreshed_plus.Norml2(), 1.0e-8 * twice_frozen_jacobian.Norml2() + 1.0e-12);

  reset_auxiliary();
  state_tangents[VELO] = 0.0;
  param_tangents[DENSITY] = 0.0;
  // JVP must reproduce the full Jacobian action, not only its norm.
  smith::FiniteElementDual jvp(states[DISP].space(), "callback_jvp");
  callback_form->jvp(time_info, shape_disp.get(), input_fields, {}, nullptr,
                     getConstFieldPointers(state_tangents, param_tangents), {}, &jvp);
  EXPECT_DOUBLE_EQ(params[DENSITY](0), 3.0);
  jvp -= jacobian_action;
  EXPECT_LE(jvp.Norml2(), 1.0e-12);

  reset_auxiliary();
  smith::FiniteElementState v(states[DISP].space(), "callback_v");
  v = 1.0;
  for (auto& sensitivity : state_duals) sensitivity = 0.0;
  for (auto& sensitivity : param_duals) sensitivity = 0.0;
  auto sensitivities = getFieldPointers(state_duals, param_duals);
  callback_form->vjp(time_info, shape_disp.get(), input_fields, {}, &v, shape_disp_dual.get(), sensitivities, {});
  EXPECT_DOUBLE_EQ(params[DENSITY](0), 3.0);
  // Likewise, VJP must reproduce the vector J^T v for the same frozen p.
  smith::FiniteElementDual expected_vjp(states[DISP].space(), "callback_expected_vjp");
  expected_vjp = 0.0;
  J->AddMultTranspose(v, expected_vjp);
  state_duals[DISP] -= expected_vjp;
  EXPECT_LE(state_duals[DISP].Norml2(), 1.0e-12);

  EXPECT_EQ(callback_count, 7);
}

TEST_F(WeakFormFixture, ForwardsOriginalTimeInfoToIntegrands)
{
  using TrialSpace = VectorSpace;
  using WeakFormT =
      smith::FunctionalWeakForm<dim, TrialSpace, smith::Parameters<VectorSpace, VectorSpace, DensitySpace>>;

  std::vector<const mfem::ParFiniteElementSpace*> inputs{&states[STATE::DISP].space(), &states[STATE::VELO].space(),
                                                         &params[PAR::DENSITY].space()};
  auto f_weak_form = std::make_shared<WeakFormT>("time_info_forwarding", mesh, states[STATE::DISP].space(), inputs);

  double observed_time = std::numeric_limits<double>::quiet_NaN();
  double observed_dt = std::numeric_limits<double>::quiet_NaN();
  size_t observed_cycle = 0;
  bool observed_cycle_zero = false;

  f_weak_form->addBodySource(mesh->entireBodyName(),
                             [&observed_time, &observed_dt, &observed_cycle, &observed_cycle_zero](
                                 const smith::TimeInfo& t_info, auto x, auto... /*args*/) {
                               observed_time = t_info.time();
                               observed_dt = t_info.dt();
                               observed_cycle = t_info.cycle();
                               observed_cycle_zero = t_info.isCycleZeroEvaluation();
                               return 0.0 * x;
                             });

  smith::TimeInfo step_time(2.0, 0.25, 7, smith::TimeInfo::EvaluationMode::CycleZero);
  auto input_fields = getConstFieldPointers(states, params);
  f_weak_form->residual(step_time, shape_disp.get(), input_fields);

  EXPECT_DOUBLE_EQ(observed_time, step_time.time());
  EXPECT_DOUBLE_EQ(observed_dt, step_time.dt());
  EXPECT_EQ(observed_cycle, step_time.cycle());
  EXPECT_TRUE(observed_cycle_zero);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
