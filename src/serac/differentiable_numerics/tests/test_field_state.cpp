#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/mesh.hpp"

#include "serac/gretl/data_store.hpp"
#include "serac/differentiable_numerics/field_state.hpp"
#include "serac/differentiable_numerics/differentiable_utils.hpp"

// This tests the interface between the new serac::WeakForm with gretl and its conformity to the existing base_physics
// interface

const std::string MESHTAG = "mesh";

struct MeshFixture : public ::testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;
  using VectorSpace = serac::H1<disp_order, dim>;

  void SetUp()
  {
    serac::StateManager::initialize(datastore_, "generic");

    // create mesh
    auto mfem_shape = mfem::Element::QUADRILATERAL;
    double length = 1.0;
    double width = 1.0;
    mesh_ = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian2D(2, 2, mfem_shape, true, length, width), MESHTAG,
                                          0, 0);
    checkpointer_ = std::make_shared<gretl::DataStore>(5);

    std::string physics_name = "generic";
    auto disp = create_field_state(*checkpointer_, VectorSpace{}, physics_name + "_displacement", mesh_->tag());
    auto velo = create_field_state(*checkpointer_, VectorSpace{}, physics_name + "_velocity", mesh_->tag());
    auto accel = create_field_state(*checkpointer_, VectorSpace{}, physics_name + "_acceleration", mesh_->tag());
    dt_ = std::make_unique<gretl::State<double, double>>(checkpointer_->create_state<double, double>(0.9));
    h_ = std::make_unique<gretl::State<double, double>>(checkpointer_->create_state<double, double>(0.7));

    disp.get()->setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto v = x;
      v[0] = 4.0 * x[0];
      v[1] = -0.1 * x[1];
      return v;
    });

    velo.get()->setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto v = x;
      v[0] = 3.0 * x[0] + 1.0 * x[1];
      v[1] = -0.2 * x[1];
      return v;
    });

    accel.get()->setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto v = x;
      v[0] = -1.0 * x[0] + 1.0 * x[1];
      v[1] = 0.1 * x[1] + 0.25 * x[1];
      return v;
    });

    states_ = {disp, velo, accel};
  }

  axom::sidre::DataStore datastore_;
  std::shared_ptr<serac::Mesh> mesh_;
  std::shared_ptr<gretl::DataStore> checkpointer_;

  std::vector<serac::FieldState> states_;
  std::unique_ptr<gretl::State<double, double>> dt_;
  std::unique_ptr<gretl::State<double, double>> h_;
};

TEST_F(MeshFixture, FieldStateWithDifferentiable_axpby)
{
  serac::FieldState disp = states_[0];
  serac::FieldState velo = states_[1];
  serac::FieldState accel = states_[2];
  gretl::State<double> dt = *dt_;
  double dt_f = dt.get();  // fixed dt for correctness checks

  auto u = axpby(dt, disp, dt, velo);
  auto u_exact = axpby(dt_f, disp, dt_f, velo);

  auto uu_exact = serac::inner_product(u_exact, u_exact);
  auto uu = serac::inner_product(u, u);
  gretl::set_as_objective(uu);

  EXPECT_EQ(uu.get(), uu_exact.get());

  checkpointer_->back_prop();

  EXPECT_GT(serac::check_grad_wrt(uu, disp, *checkpointer_, 1e-5, 4, true), 0.95);
  EXPECT_GT(serac::check_grad_wrt(uu, velo, *checkpointer_, 1e-5, 4, true), 0.95);
  EXPECT_GT(serac::check_grad_wrt(uu, dt, *checkpointer_, 1e-7, 4, true), 0.95);
}

TEST_F(MeshFixture, FieldStateDifferentiablyWeightedSum_WithOperators)
{
  serac::FieldState disp = states_[0];
  serac::FieldState velo = states_[1];
  serac::FieldState accel = states_[2];
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

  auto u_exact = serac::axpby(initial_dt, velo, 0.0, velo);
  auto v_exact = serac::axpby(1.0, accel, 1.0, disp);
  u_exact = axpby(initial_dt, disp, initial_h, u_exact);
  u_exact = axpby(1.0, axpby(1.0, axpby(0.65, disp, initial_dt, accel), initial_h, u_exact), -1.0,
                  axpby(1.0, v_exact, 1.0, u_exact));
  u_exact = axpby(1.0, axpby(0.65, disp, 1.0, accel), initial_h, u_exact);
  u_exact = axpby(1.0, u_exact, -1.0, disp);
  u_exact = axpby(2.0, velo, -2.0, u_exact);
  u_exact = axpby(initial_dt * initial_h, u_exact, 0.0, u_exact);

  auto uu_exact = serac::inner_product(u_exact, u_exact);
  auto uu = serac::inner_product(u, u);

  gretl::set_as_objective(uu);

  ASSERT_DOUBLE_EQ(uu.get(), uu_exact.get());

  checkpointer_->back_prop();

  EXPECT_GT(serac::check_grad_wrt(uu, disp, *checkpointer_, 1e-5, 4, true), 0.95);
  EXPECT_GT(serac::check_grad_wrt(uu, velo, *checkpointer_, 1e-5, 4, true), 0.95);
  EXPECT_GT(serac::check_grad_wrt(uu, accel, *checkpointer_, 1e-5, 4, true), 0.95);
  EXPECT_GT(serac::check_grad_wrt(uu, dt, *checkpointer_, 1e-5, 4, true), 0.95);
  EXPECT_GT(serac::check_grad_wrt(uu, h, *checkpointer_, 1e-5, 4, true), 0.95);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}