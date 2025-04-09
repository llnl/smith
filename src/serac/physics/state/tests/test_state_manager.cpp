// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_state_manager.cpp
 */

#include "serac/physics/state/state_manager.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac {

TEST(StateManager, StoresHighOrderMeshes)
{
  // This test ensures that when high order meshes are given to
  // the state manager, it indeed stores the high order mesh, and
  // does not cast it down to first order.
  //
  // This test will break if you change the mesh file.
  // It relies on knowledge of the specific mesh
  // in "single_curved_quad.g".

  // The mesh has a single element with one curved edge.
  // It looks something like this:
  //
  //     curved edge on top
  //       __--O--__
  //    O--         --O
  //    |             |
  //    |             |    straight edges on sides and bottom
  //    O             O
  //    |             |
  //    |             |
  //    O------O------O

  constexpr int dim = 2;
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "curved_element_output_test");

  const std::string filename = SERAC_REPO_DIR "/data/meshes/single_curved_quad.g";
  int serial_refinement = 0;
  int parallel_refinement = 0;
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), "mesh");

  ASSERT_EQ(dim, pmesh.SpaceDimension());

  // Make sure that the stored mesh maintained second order character
  EXPECT_EQ(pmesh.GetNodalFESpace()->GetMaxElementOrder(), 2);
  EXPECT_EQ(pmesh.GetNodalFESpace()->GetNDofs(), 9);

  // make sure that the curved boundary hasn't been replaced
  // with a straight edge

  const mfem::GridFunction* nodes = pmesh.GetNodes();

  // Get dofs on curved edge
  const int curved_boundary_element = 2;  // edge elem id of the curved edge
  mfem::Array<int> dofs;
  pmesh.GetNodalFESpace()->GetBdrElementDofs(curved_boundary_element, dofs);
  constexpr int num_nodes_on_edge = dim + 1;
  ASSERT_EQ(dofs.Size(), num_nodes_on_edge);

  // Get coordinates of curved edge nodes
  mfem::Array<tensor<double, dim>> edge_coords(num_nodes_on_edge);
  for (int k = 0; k < dofs.Size(); k++) {
    int d = dofs[k];
    for (int i = 0; i < dim; i++) {
      edge_coords[k][i] = (*nodes)(pmesh.GetNodalFESpace()->DofToVDof(d, i));
    }
  }

  // Make sure edge nodes are not colinear
  auto v1 = edge_coords[0] - edge_coords[1];
  auto v2 = edge_coords[0] - edge_coords[2];
  double area = std::abs(v1[0] * v2[1] - v1[1] * v2[0]);
  EXPECT_GT(area, 1e-6);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);
}