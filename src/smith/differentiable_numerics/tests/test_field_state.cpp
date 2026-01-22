// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/mesh.hpp"

#include "gretl/data_store.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"

// This tests the interface between the new smith::WeakForm with gretl and its conformity to the existing base_physics
// interface

const std::string MESHTAG = "mesh";

struct MeshFixture : public ::testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;
  using VectorSpace = smith::H1<disp_order, dim>;

  void SetUp()
  {
    smith::StateManager::initialize(datastore_, "generic");

    // create mesh
    auto mfem_shape = mfem::Element::QUADRILATERAL;
    double length = 1.0;
    double width = 1.0;
    mesh_ = std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian2D(2, 2, mfem_shape, true, length, width), MESHTAG,
                                          0, 0);
    checkpointer_ = std::make_shared<gretl::DataStore>(5);

    std::string physics_name = "generic";
    auto disp = createFieldState(*checkpointer_, VectorSpace{}, physics_name + "_displacement", mesh_->tag());
    auto velo = createFieldState(*checkpointer_, VectorSpace{}, physics_name + "_velocity", mesh_->tag());
    auto accel = createFieldState(*checkpointer_, VectorSpace{}, physics_name + "_acceleration", mesh_->tag());
    dt_ = std::make_unique<gretl::State<double, double>>(checkpointer_->create_state<double, double>(0.9));
    h_ = std::make_unique<gretl::State<double, double>>(checkpointer_->create_state<double, double>(0.7));

    disp.get()->setFromFieldFunction([](smith::tensor<double, dim> x) {
      auto v = x;
      v[0] = 4.0 * x[0];
      v[1] = -0.1 * x[1];
      return v;
    });

    velo.get()->setFromFieldFunction([](smith::tensor<double, dim> x) {
      auto v = x;
      v[0] = 3.0 * x[0] + 1.0 * x[1];
      v[1] = -0.2 * x[1];
      return v;
    });

    accel.get()->setFromFieldFunction([](smith::tensor<double, dim> x) {
      auto v = x;
      v[0] = -1.0 * x[0] + 1.0 * x[1];
      v[1] = 0.1 * x[1] + 0.25 * x[1];
      return v;
    });

    states_ = {disp, velo, accel};
  }

  axom::sidre::DataStore datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
  std::shared_ptr<gretl::DataStore> checkpointer_;

  std::vector<smith::FieldState> states_;
  std::unique_ptr<gretl::State<double, double>> dt_;
  std::unique_ptr<gretl::State<double, double>> h_;
};

TEST_F(MeshFixture, FieldStateWithDifferentiable_axpby)
{
  smith::FieldState disp = states_[0];
  smith::FieldState velo = states_[1];
  smith::FieldState accel = states_[2];
  gretl::State<double> dt = *dt_;
  double dt_f = dt.get();  // fixed dt for correctness checks

  auto u = axpby(dt, disp, dt, velo);
  auto u_exact = axpby(dt_f, disp, dt_f, velo);

  auto uu_exact = smith::innerProduct(u_exact, u_exact);
  auto uu = smith::innerProduct(u, u);
  gretl::set_as_objective(uu);

  EXPECT_EQ(uu.get(), uu_exact.get());

  EXPECT_GT(smith::checkGradWrt(uu, disp, 1e-5, 4, true), 0.95);
  EXPECT_GT(smith::checkGradWrt(uu, velo, 1e-5, 4, true), 0.95);
  EXPECT_GT(smith::checkGradWrt(uu, dt, 1e-7, 4, true), 0.95);
}

TEST_F(MeshFixture, FieldStateDifferentiablyWeightedSum_WithOperators)
{
  smith::FieldState disp = states_[0];
  smith::FieldState velo = states_[1];
  smith::FieldState accel = states_[2];
  gretl::State<double> dt = *dt_;
  gretl::State<double> h = *h_;
  double initial_dt = dt.get();
  double initial_h = h.get();

  auto u = dt * velo;
  auto v = accel + velo;
  auto w = disp - velo;
  w = -w;
  v = v - w;
  u = dt * disp + h * u;
  u = 0.65 * disp + dt * accel + h * u - v - u;
  u = 0.65 * disp + accel + h * u;
  u = u - disp;
  u = 2.0 * (velo - u);

  gretl::State<double> dth = dt * h;
  u = dth * u;

  auto u_exact = smith::axpby(initial_dt, velo, 0.0, velo);
  auto v_exact = smith::axpby(1.0, accel, 1.0, disp);
  u_exact = axpby(initial_dt, disp, initial_h, u_exact);
  u_exact = axpby(1.0, axpby(1.0, axpby(0.65, disp, initial_dt, accel), initial_h, u_exact), -1.0,
                  axpby(1.0, v_exact, 1.0, u_exact));
  u_exact = axpby(1.0, axpby(0.65, disp, 1.0, accel), initial_h, u_exact);
  u_exact = axpby(1.0, u_exact, -1.0, disp);
  u_exact = axpby(2.0, velo, -2.0, u_exact);
  u_exact = axpby(initial_dt * initial_h, u_exact, 0.0, u_exact);

  auto uu_exact = smith::innerProduct(u_exact, u_exact);
  auto uu = smith::innerProduct(u, u);

  gretl::set_as_objective(uu);

  ASSERT_DOUBLE_EQ(uu.get(), uu_exact.get());

  checkpointer_->back_prop();

  EXPECT_GT(smith::checkGradWrt(uu, disp, 1e-5, 4, true), 0.95);
  EXPECT_GT(smith::checkGradWrt(uu, velo, 1e-5, 4, true), 0.95);
  EXPECT_GT(smith::checkGradWrt(uu, accel, 1e-5, 4, true), 0.95);
  EXPECT_GT(smith::checkGradWrt(uu, dt, 1e-5, 4, true), 0.95);
  EXPECT_GT(smith::checkGradWrt(uu, h, 1e-5, 4, true), 0.95);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
