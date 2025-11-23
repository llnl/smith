// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "smith/numerics/functional/geometric_factors.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

using namespace smith;

std::string mesh_dir = SMITH_REPO_DIR "/data/meshes/";

mfem::Mesh import_mesh(std::string meshfile)
{
  mfem::named_ifgzstream imesh(mesh_dir + meshfile);

  if (!imesh) {
    smith::logger::flush();
    std::string err_msg = axom::fmt::format("Can not open mesh file: '{0}'", mesh_dir + meshfile);
    SLIC_ERROR_ROOT(err_msg);
  }

  mfem::Mesh mesh(imesh, 1, 1, true);
  mesh.EnsureNodes();
  return mesh;
}

TEST(geometric_factors, with_2D_domains)
{
  auto bmesh = import_mesh("patch2D_tris_and_quads.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  // `d` will consist of one tri and one quad
  Domain d = Domain::ofElements(
      *mesh, std::function([](std::vector<vec2> vertices, int /* attr */) { return average(vertices)[0] < 0.45; }));

  int q = 2;

  {
    GeometricFactors gf(d, q, mfem::Geometry::TRIANGLE);
    int components_per_J = 4;
    int qpts_per_elem = (q * (q + 1)) / 2;
    int num_elems = 1;
    EXPECT_EQ(gf.J.Size(), components_per_J * qpts_per_elem * num_elems);
  }

  std::cout << std::endl;

  {
    GeometricFactors gf(d, q, mfem::Geometry::SQUARE);
    int components_per_J = 4;
    int qpts_per_elem = q * q;
    int num_elems = 1;
    EXPECT_EQ(gf.J.Size(), components_per_J * qpts_per_elem * num_elems);
  }
}

TEST(geometric_factors, with_3D_domains)
{
  auto bmesh = import_mesh("patch3D_tets_and_hexes.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  // `d` will consist of 6 tets and 1 hex
  Domain d = Domain::ofElements(*mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
    return average(vertices)[1] < 0.75;  // y coordinate of face center
  }));

  int q = 2;

  {
    GeometricFactors gf(d, q, mfem::Geometry::TETRAHEDRON);
    int components_per_J = 9;
    int qpts_per_elem = (q * (q + 1) * (q + 2)) / 6;
    int num_elems = 6;
    EXPECT_EQ(gf.J.Size(), components_per_J * qpts_per_elem * num_elems);
  }

  {
    GeometricFactors gf(d, q, mfem::Geometry::CUBE);
    int components_per_J = 9;
    int qpts_per_elem = q * q * q;
    int num_elems = 1;
    EXPECT_EQ(gf.J.Size(), components_per_J * qpts_per_elem * num_elems);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
