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

TEST_F(WeakFormFixture, ShapeJvpVjpConsistency)
{
  const auto fields = getConstFieldPointers(states, params);
  const auto comm = states[DISP].space().GetComm();

  // Shape direction ds, a smooth dilation.
  smith::FiniteElementState ds(shape_disp->space(), "shape_direction");
  ds.setFromFieldFunction([](smith::tensor<double, dim> x) { return 0.1 * x; });

  // Test vector w in the residual's test space.
  smith::FiniteElementState w(states[DISP].space(), "test_weight");
  w = 1.0;

  smith::FiniteElementDual j_ds(states[DISP].space(), "shape_jvp");
  smith::FiniteElementDual jt_w(shape_disp->space(), "shape_vjp");
  j_ds = 0.0;
  jt_w = 0.0;  // VJP accumulates into this output.

  // Zero directions for all non-shape fields.
  std::vector<smith::ConstFieldPtr> zero_directions(fields.size(), nullptr);

  // Compute j_ds = R_s ds.
  weak_form->jvp(time_info, shape_disp.get(), fields, {}, &ds, zero_directions, {}, &j_ds);

  // Request only the shape VJP.
  std::vector<smith::DualFieldPtr> unused_field_outputs(fields.size(), nullptr);

  // Compute jt_w = R_s^T w.
  weak_form->vjp(time_info, shape_disp.get(), fields, {}, &w, &jt_w, unused_field_outputs, {});

  // Global inner products across MPI ranks.
  const double lhs = mfem::InnerProduct(comm, ds, jt_w);
  const double rhs = mfem::InnerProduct(comm, w, j_ds);

  const double tolerance = 1.0e-12 + 1.0e-10 * std::max(std::abs(lhs), std::abs(rhs));

  EXPECT_NEAR(lhs, rhs, tolerance) << "Shape transpose consistency failed:\n"
                                   << "ds^T (R_s^T w) = " << lhs << "\n"
                                   << "w^T (R_s ds)  = " << rhs;
}

TEST_F(WeakFormFixture, ShapeFDDerivativeCheckOfVJPandJVP)
{
  const auto fields = getConstFieldPointers(states, params);

  // Shape direction: a smooth dilation.
  smith::FiniteElementState ds(shape_disp->space(), "shape_direction");
  ds.setFromFieldFunction([](smith::tensor<double, dim> x) { return 0.1 * x; });

  // Perturb only the shape field.
  constexpr double eps = 1.0e-6;
  smith::FiniteElementState s_plus(shape_disp->space(), "shape_plus");
  smith::FiniteElementState s_minus(shape_disp->space(), "shape_minus");

  s_plus.Set(1.0, *shape_disp);
  s_minus.Set(1.0, *shape_disp);
  s_plus.Add(eps, ds);
  s_minus.Add(-eps, ds);

  mfem::Vector fd = weak_form->residual(time_info, &s_plus, fields);
  mfem::Vector r_minus = weak_form->residual(time_info, &s_minus, fields);
  fd -= r_minus;
  fd /= (2.0 * eps);

  auto global_norm = [](const mfem::Vector& x) { return std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, x, x)); };

  const double fd_norm = global_norm(fd);
  ASSERT_GT(fd_norm, 1.0e-10) << "This direction must produce a nonzero shape derivative.";

  // JVP: all other field directions are zero.
  std::vector<smith::ConstFieldPtr> zero_directions(fields.size(), nullptr);
  smith::FiniteElementDual shape_jvp(states[DISP].space(), "shape_jvp");
  shape_jvp = 0.0;

  weak_form->jvp(time_info, shape_disp.get(), fields, {}, &ds, zero_directions, {}, &shape_jvp);

  mfem::Vector error(shape_jvp);
  error -= fd;
  EXPECT_LE(global_norm(error), 1.0e-8 + 1.0e-6 * fd_norm)
      << "Shape JVP disagrees with the residual finite difference.";

  // VJP: check ds^T (R_s^T w) against w^T fd.
  smith::FiniteElementState w(states[DISP].space(), "shape_test_weight");
  w = 1.0;

  // VJP accumulates, so initialize its output.
  *shape_disp_dual = 0.0;
  std::vector<smith::DualFieldPtr> unused_outputs(fields.size(), nullptr);

  weak_form->vjp(time_info, shape_disp.get(), fields, {}, &w, shape_disp_dual.get(), unused_outputs, {});

  const double reference = mfem::InnerProduct(MPI_COMM_WORLD, w, fd);
  const double from_vjp = mfem::InnerProduct(MPI_COMM_WORLD, ds, *shape_disp_dual);

  EXPECT_NEAR(from_vjp, reference, 1.0e-8 + 1.0e-6 * std::abs(reference))
      << "Shape VJP disagrees with the residual finite difference.";
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
