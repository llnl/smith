// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <iostream>
#include <string>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace smith {

TEST(Mesh, TestMeshFromParMesh)
{
  int serial_refinement = 2;
  int parallel_refinement = 2;

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "datastore");

  auto parMesh =
      mesh::refineAndDistribute(buildRectangleMesh(2, 3, 2, 3), serial_refinement, parallel_refinement, MPI_COMM_WORLD);
  auto mesh = std::make_shared<smith::Mesh>(std::move(*parMesh), "mesh_tag");

  EXPECT_EQ(mesh->entireBody().total_elements(), 1536);
  EXPECT_EQ(mesh->entireBoundary().total_elements(), 160);
  EXPECT_EQ(mesh->internalBoundary().total_elements(), 2992);
}

TEST(Mesh, TestMeshFromSerialMesh)
{
  constexpr int dim = 3;
  constexpr double L = 8.0;
  constexpr double W = 1.0;
  constexpr double H = 1.0;

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "datastore");

  auto mesh = std::make_shared<smith::Mesh>(buildCuboidMesh(8, 1, 1, L, W, H), "mesh_tag");

  EXPECT_EQ(mesh->entireBody().total_elements(), 8);
  EXPECT_EQ(mesh->entireBoundary().total_elements(), 34);
  EXPECT_EQ(mesh->internalBoundary().total_elements(), 7);

  auto is_in_left = [](std::vector<tensor<double, dim>> coords, int /* attribute */) {
    return average(coords)[0] < 0.5 * L;
  };
  mesh->addDomainOfBodyElements("left", is_in_left);
  EXPECT_EQ(mesh->domain("left").total_elements(), 4);
}

TEST(Mesh, TestMeshFromFile)
{
  constexpr int dim = 3;

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "datastore");

  std::string filename = SMITH_REPO_DIR "/data/meshes/ball.mesh";
  auto mesh = std::make_shared<smith::Mesh>(filename, "mesh_tag", 0, 0, MPI_COMM_WORLD);
  EXPECT_EQ(mesh->tag(), "mesh_tag");
  EXPECT_EQ(mesh->getComm(), mesh->mfemParMesh().GetComm());

  EXPECT_EQ(mesh->entireBody().total_elements(), 56);
  EXPECT_EQ(mesh->entireBoundary().total_elements(), 24);
  EXPECT_EQ(mesh->internalBoundary().total_elements(), 156);

  mesh->addDomainOfBoundaryElements("domain1", smith::by_attr<dim>(1));
  EXPECT_EQ(mesh->domain("domain1").total_elements(), 24);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
