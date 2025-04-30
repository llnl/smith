// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_finite_element_staet.cpp
 */

 #include "serac/physics/state/finite_element_dual.hpp"

 #include "axom/slic/core/SimpleLogger.hpp"
 #include <gtest/gtest.h>
 #include "mfem.hpp"
 
 #include "serac/mesh/mesh_utils.hpp"
 #include "serac/numerics/functional/tensor.hpp"
 #include "serac/infrastructure/application_manager.hpp"
 
 namespace serac {

TEST(FiniteELementDual, MoveAssignment)
{
    int serial_refinement = 0;
    int parallel_refinement = 0;
    constexpr int spatial_dim{3};

    std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
    std::unique_ptr<mfem::ParMesh> mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
    ASSERT_EQ(spatial_dim, mesh->SpaceDimension())
        << "Test configured incorrectly. The variable spatial_dim must match the spatial dimension of the mesh.";

    FiniteElementDual dual0(*mesh, H1<1>{}, "dual0");
    dual0 = 1.0;

    FiniteElementDual dual1(*mesh, H1<1>{}, "dual1");
    dual1 = std::move(dual0);
    EXPECT_EQ(dual1(0), 1.0);
    dual1.Print();
}

 } // namespace serac

 int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}