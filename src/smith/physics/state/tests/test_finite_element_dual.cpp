// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_finite_element_staet.cpp
 */

#include <memory>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/smith_config.hpp"

namespace smith {

TEST(FiniteELementDual, MoveAssignment)
{
  /* Check that move assignment operator does deep copy of values */
  int serial_refinement = 0;
  int parallel_refinement = 0;
  constexpr int spatial_dim{3};

  std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";
  std::unique_ptr<mfem::ParMesh> mesh =
      mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  ASSERT_EQ(spatial_dim, mesh->SpaceDimension())
      << "Test configured incorrectly. The variable spatial_dim must match the spatial dimension of the mesh.";

  FiniteElementDual dual0(*mesh, H1<1>{}, "dual0");
  dual0 = 1.0;

  FiniteElementDual dual1 = std::move(dual0);

  for (const mfem::real_t* d = dual1.begin(); d != dual1.end(); ++d) {
    EXPECT_EQ(*d, 1.0);
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
