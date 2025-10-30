// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_finite_element_state.cpp
 */

#include "gtest/gtest.h"
#include <memory>
#include <string>

#include "smith/physics/state/finite_element_state.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/smith_config.hpp"

namespace smith {

class TestFiniteElementState : public testing::Test {
 protected:
  void SetUp() override
  {
    int serial_refinement = 0;
    int parallel_refinement = 0;

    std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";
    mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
    ASSERT_EQ(spatial_dim, mesh->SpaceDimension())
        << "Test configured incorrectly. The variable spatial_dim must match the spatial dimension of the mesh.";
  }

  static constexpr int spatial_dim{3};
  std::unique_ptr<mfem::ParMesh> mesh;
};

TEST_F(TestFiniteElementState, SetScalarStateFromFieldFunction)
{
  // make a scalar-valued state
  constexpr int p = 1;
  FiniteElementState scalar_state(*mesh, H1<p>{}, "scalar_field");

  // Set state with field function.
  // Check that lambda captures work with this.
  double c = 2.0;
  auto scalar_field = [c](const tensor<double, spatial_dim>& X) -> double { return c * X[0]; };
  scalar_state.setFromFieldFunction(scalar_field);

  // Get the nodal positions corresponding to state dofs in a grid function
  auto [coords_fe_space, coords_fe_coll] = smith::generateParFiniteElementSpace<H1<p, spatial_dim>>(mesh.get());
  mfem::ParGridFunction nodal_coords_gf(coords_fe_space.get());
  mesh->GetNodes(nodal_coords_gf);

  for (int node = 0; node < scalar_state.space().GetNDofs(); node++) {
    // Fill a tensor with the coordinates of the node
    tensor<double, spatial_dim> Xn;
    for (int i = 0; i < spatial_dim; i++) {
      int dof_index = nodal_coords_gf.FESpace()->DofToVDof(node, i);
      Xn[i] = nodal_coords_gf(dof_index);
    }

    // check that value set in the state for this node matches the field function
    EXPECT_DOUBLE_EQ(scalar_field(Xn), scalar_state(node));
  }
}

TEST_F(TestFiniteElementState, SetVectorStateFromFieldFunction)
{
  constexpr int p = 2;

  // Choose vector dimension for state field that is different from spatial dimension
  // to test the field indexing more thoroughly.
  constexpr int vdim = 2;
  FiniteElementState state(*mesh, H1<p, vdim>{}, "vector_field");

  // set the field with an arbitrarily chosen field function
  auto vector_field = [](tensor<double, spatial_dim> X) {
    return tensor<double, vdim>{norm(X), 1.0 / (1.0 + norm(X))};
  };
  state.setFromFieldFunction(vector_field);

  // Get the nodal positions for the state in a grid function
  auto [coords_fe_space, coords_fe_coll] = smith::generateParFiniteElementSpace<H1<p, spatial_dim>>(mesh.get());
  mfem::ParGridFunction nodal_coords_gf(coords_fe_space.get());
  mesh->GetNodes(nodal_coords_gf);

  // we need the state values and the nodal coordinates in the same kind of container,
  // so we will get the grid function view of the state
  mfem::ParGridFunction& state_gf = state.gridFunction();

  for (int node = 0; node < state_gf.FESpace()->GetNDofs(); node++) {
    // Fill a tensor with the coordinates of the node
    tensor<double, spatial_dim> Xn;
    for (int i = 0; i < spatial_dim; i++) {
      int dof_index = nodal_coords_gf.FESpace()->DofToVDof(node, i);
      Xn[i] = nodal_coords_gf(dof_index);
    }

    // apply the field function to the node coords
    auto v = vector_field(Xn);

    // check that value set in the state matches the field function
    for (int j = 0; j < vdim; j++) {
      int dof_index = state_gf.FESpace()->DofToVDof(node, j);
      EXPECT_DOUBLE_EQ(v[j], state_gf(dof_index));
    }
  }
}

TEST_F(TestFiniteElementState, DISABLED_ErrorsIfFieldFunctionDimensionMismatchedToState)
{
  constexpr int p = 2;

  // Choose vector dimension for state field that is different from spatial dimension
  constexpr int vdim = 2;
  FiniteElementState state(*mesh, H1<p, vdim>{}, "vector_field");

  // Set the field with a field function with the wrong vector dimension.
  // Should return tensor of size vdim!
  auto vector_field = [](tensor<double, spatial_dim> X) { return X; };

  EXPECT_DEATH(state.setFromFieldFunction(vector_field),
               "Cannot copy tensor into an MFEM Vector with incompatible size.");
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
