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
    serac::StateManager::initialize(datastore, "generic");

    // create mesh
    auto mfem_shape = mfem::Element::QUADRILATERAL;
    double length = 1.0;
    double width = 1.0;
    mesh = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian2D(2, 2, mfem_shape, true, length, width), MESHTAG, 0,
                                         0);
    checkpointer = std::make_shared<gretl::DataStore>(5);

    std::string physics_name = "generic";
    auto disp = create_field_state(*checkpointer, VectorSpace{}, physics_name + "_displacement", mesh->tag());
    auto velo = create_field_state(*checkpointer, VectorSpace{}, physics_name + "_velocity", mesh->tag());
    auto accel = create_field_state(*checkpointer, VectorSpace{}, physics_name + "_acceleration", mesh->tag());
    dt = std::make_unique<gretl::State<double, double>>(checkpointer->create_state<double, double>(1e-4));

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

    states = {disp, velo, accel};
  }

  axom::sidre::DataStore datastore;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<gretl::DataStore> checkpointer;

  std::vector<serac::FieldState> states;
  std::unique_ptr<gretl::State<double, double>> dt;
};

TEST_F(MeshFixture, FieldStateUpdates)
{
  using gretl::print;
  serac::FieldState disp = states[0];
  serac::FieldState velo = states[1];
  serac::FieldState accel = states[2];

  auto u = axpby(*dt, disp, *dt, velo);
  auto u_exact = axpby(dt->get(), disp, dt->get(), velo);
  auto uu_exact = serac::inner_product(u_exact, u_exact);
  auto uu = serac::inner_product(u, u);
  EXPECT_EQ(uu.get(), uu_exact.get());

  gretl::set_as_objective(uu);
  print("f");
  checkpointer->back_prop();
  print("g");
  EXPECT_GT(serac::check_grad_wrt(uu, disp, *checkpointer, 1e-5, 4, true), 0.95);
  EXPECT_GT(serac::check_grad_wrt(uu, velo, *checkpointer, 1e-5, 4, true), 0.95);
  // EXPECT_GT(serac::check_grad_wrt(uu, *dt, *checkpointer, 1e-5, 4, true), 0.95);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}