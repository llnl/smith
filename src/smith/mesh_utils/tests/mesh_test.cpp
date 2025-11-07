// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>

#include "mpi.h"
#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

namespace smith {

TEST(Mesh, LoadExodus)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string mesh_file = std::string(SMITH_REPO_DIR) + "/data/meshes/bortel_echem.e";

  auto pmesh = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 1, 1);
  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
