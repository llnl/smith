#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/mesh.hpp"

#include "serac/gretl/data_store.hpp"
#include "serac/differentiable_numerics/field_state.hpp"

// This tests the interface between the new serac::WeakForm with gretl and its conformity to the existing base_physics
// interface

const std::string MESHTAG = "mesh";

struct MeshFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;
  using VectorSpace = serac::H1<disp_order, dim>;

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    serac::StateManager::initialize(datastore, "generic");

    // create mesh
    auto mfem_shape = mfem::Element::QUADRILATERAL;
    double length = 1.0;
    double width = 1.0;
    mesh = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian2D(2, 2, mfem_shape, true, length, width), MESHTAG, 0,
                                         0);
    checkpointer = std::make_shared<gretl::DataStore>(20);

    std::string physics_name = "generic";
    auto disp = create_field_state(*checkpointer, VectorSpace{}, physics_name + "_displacement", mesh->tag());
    auto velo = create_field_state(*checkpointer, VectorSpace{}, physics_name + "_velocity", mesh->tag());
    auto accel = create_field_state(*checkpointer, VectorSpace{}, physics_name + "_acceleration", mesh->tag());
    auto dt = checkpointer->create_state<double, double>(1e-4);

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
};

TEST_F(MeshFixture, FieldStateUpdates)
{
  FieldState disp = states[0];
  FieldState velo = states[1];
  FieldState accel = states[2];
  auto u = disp + dt * velo;
  auto uu = inner
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
