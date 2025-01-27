// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <exception>
#include <iostream>
#include <stdlib.h>

#include "axom/core.hpp"
#include "axom/slic/core/SimpleLogger.hpp"

#include <gtest/gtest.h>

#include "serac/mesh/mesh_utils.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/input.hpp"

class SlicErrorException : public std::exception {};

// Copied from src/serac/infrastructure/tests/serac_input.cpp
class MeshTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
    axom::slic::setAbortOnError(true);
    axom::slic::setAbortOnWarning(false);
  }

  void SetUp() override
  {
    // Set up mesh
    auto reader = std::make_unique<axom::inlet::LuaReader>();
    inlet_.emplace(std::move(reader), datastore_.getRoot());
    reader_ = &(inlet_->reader());
  }

  // Initialization cannot occur during construction, has to be initialized later
  axom::inlet::Reader* reader_ = nullptr;
  std::optional<axom::inlet::Inlet> inlet_;

  // Where all of the .mesh files are located
  std::string base_mesh_dir_ = std::string(SERAC_REPO_DIR) + "/data/meshes/";

 private:
  axom::sidre::DataStore datastore_;
};

namespace serac {

TEST_F(MeshTest, LuaInputMainMeshFromFile)
{
  MPI_Barrier(MPI_COMM_WORLD);

  reader_->parseString(std::string("main_mesh_from_file = { type = \"file\",") +
                       "mesh = \"beam-hex.mesh\", ser_ref_levels = 1, par_ref_levels = 0, }");
  auto& mesh_table = inlet_->addStruct("main_mesh_from_file");
  mesh::InputOptions::defineInputFileSchema(mesh_table);

  // Build and test mesh
  auto mesh_options = mesh_table.get<mesh::InputOptions>();
  const auto file_options = std::get_if<mesh::FileInputOptions>(&mesh_options.extra_options);
  ASSERT_NE(file_options, nullptr);

  // Check mesh path
  std::string mesh_path = base_mesh_dir_ + file_options->relative_mesh_file_name;
  EXPECT_EQ(axom::utilities::filesystem::pathExists(mesh_path), true);
  file_options->absolute_mesh_file_name = mesh_path;

  auto mesh = mesh::buildParallelMesh(mesh_options);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MeshTest, LuaInputMainMeshCuboid)
{
  MPI_Barrier(MPI_COMM_WORLD);
  reader_->parseString(std::string("main_mesh_cuboid = { type = \"box\",") +
                       "elements = {x = 3, y = 3, z = 3}, size = {x = 1, y = 2, z = 3}," +
                       "ser_ref_levels = 0, par_ref_levels = 0, }");
  auto& mesh_table = inlet_->addStruct("main_mesh_cuboid");
  mesh::InputOptions::defineInputFileSchema(mesh_table);

  // Build and test mesh
  auto mesh_options = mesh_table.get<serac::mesh::InputOptions>();
  const auto cuboid_options = std::get_if<serac::mesh::BoxInputOptions>(&mesh_options.extra_options);
  ASSERT_NE(cuboid_options, nullptr);
  EXPECT_EQ(cuboid_options->elements.size(), 3);
  auto mesh = serac::mesh::buildParallelMesh(mesh_options);
  EXPECT_EQ(mesh->GetNE(), cuboid_options->elements[0] * cuboid_options->elements[1] * cuboid_options->elements[2]);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MeshTest, LuaInputMainMeshRect)
{
  MPI_Barrier(MPI_COMM_WORLD);
  reader_->parseString(std::string("main_mesh_rect = { type = \"box\",") +
                       "elements = {x = 3, y = 3}, ser_ref_levels = 0, par_ref_levels = 0, }");
  auto& mesh_table = inlet_->addStruct("main_mesh_rect");
  mesh::InputOptions::defineInputFileSchema(mesh_table);

  // Build and test mesh
  auto mesh_options = mesh_table.get<serac::mesh::InputOptions>();
  const auto rect_options = std::get_if<serac::mesh::BoxInputOptions>(&mesh_options.extra_options);
  ASSERT_NE(rect_options, nullptr);
  EXPECT_EQ(rect_options->elements.size(), 2);
  auto mesh = serac::mesh::buildParallelMesh(mesh_options);
  EXPECT_EQ(mesh->GetNE(), rect_options->elements[0] * rect_options->elements[1]);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MeshTest, LuaInputMainMeshFail)
{
  MPI_Barrier(MPI_COMM_WORLD);
  reader_->parseString(std::string("main_mesh_fail = { type = \"invalid\",") +
                       "mesh = \"beam-hex.mesh\", ser_ref_levels = 1, par_ref_levels = 0, }");

  // Check that we fail on an invalid mesh description
  auto& mesh_table = inlet_->addStruct("main_mesh_fail", "An invalid mesh description");
  mesh::InputOptions::defineInputFileSchema(mesh_table);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(MeshGen, SuccessfulCreation)
{
  // the disk and ball meshes don't exactly hit the number
  // of elements specified, they refine to get as close as possible
  ASSERT_EQ(buildDiskMesh(1000).GetNE(), 1024);
  ASSERT_EQ(buildBallMesh(6000).GetNE(), 4096);
  ASSERT_EQ(buildRectangleMesh(20, 20, 1., 2.).GetNE(), 400);
  ASSERT_EQ(buildCuboidMesh(20, 20, 20, 1., 2., 3.).GetNE(), 8000);
  ASSERT_EQ(buildCylinderMesh(2, 2, 2.0, 5.0).GetNE(), 384);
  ASSERT_EQ(buildHollowCylinderMesh(2, 2, 2.0, 3.0, 5.0).GetNE(), 256);
  ASSERT_EQ(buildHollowCylinderMesh(2, 2, 2.0, 3.0, 5.0, 2. * M_PI, 8).GetNE(), 256);
  ASSERT_EQ(buildHollowCylinderMesh(2, 2, 2.0, 3.0, 5.0, 2. * M_PI, 6).GetNE(), 192);
  ASSERT_EQ(buildHollowCylinderMesh(2, 1, 2.0, 3.0, 5.0, M_PI / 3., 1).GetNE(), 16);
  ASSERT_EQ(buildHollowCylinderMesh(2, 1, 2.0, 3.0, 5.0, 2. * M_PI, 7).GetNE(), 112);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
  axom::slic::setAbortOnError(true);
  axom::slic::setAbortOnWarning(false);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
