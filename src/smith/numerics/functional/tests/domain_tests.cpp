// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "smith/numerics/functional/domain.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/functional/tensor.hpp"
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

TEST(domain, of_edges)
{
  {
    auto bmesh = import_mesh("onehex.mesh");
    auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));
    Domain d0 = Domain::ofEdges(*mesh, std::function([](std::vector<vec3> x) {
      return (0.5 * (x[0][0] + x[1][0])) < 0.25;  // x coordinate of edge midpoint
    }));
    EXPECT_EQ(d0.edge_ids_.size(), 4);
    EXPECT_EQ(d0.dim_, 1);

    Domain d1 = Domain::ofEdges(*mesh, std::function([](std::vector<vec3> x) {
      return (0.5 * (x[0][1] + x[1][1])) < 0.25;  // y coordinate of edge midpoint
    }));
    EXPECT_EQ(d1.edge_ids_.size(), 4);
    EXPECT_EQ(d1.dim_, 1);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.edge_ids_.size(), 7);
    EXPECT_EQ(d2.dim_, 1);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.edge_ids_.size(), 1);
    EXPECT_EQ(d3.dim_, 1);

    // note: by_attr doesn't apply to edge sets in 3D, since
    //       mfem doesn't have the notion of edge attributes
    // Domain d4 = Domain::ofEdges(mesh, by_attr<dim>(3));
  }

  {
    auto bmesh = import_mesh("onetet.mesh");
    auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));
    Domain d0 = Domain::ofEdges(*mesh, std::function([](std::vector<vec3> x) {
      return (0.5 * (x[0][0] + x[1][0])) < 0.25;  // x coordinate of edge midpoint
    }));
    EXPECT_EQ(d0.edge_ids_.size(), 3);
    EXPECT_EQ(d0.dim_, 1);

    Domain d1 = Domain::ofEdges(*mesh, std::function([](std::vector<vec3> x) {
      return (0.5 * (x[0][1] + x[1][1])) < 0.25;  // y coordinate of edge midpoint
    }));
    EXPECT_EQ(d1.edge_ids_.size(), 3);
    EXPECT_EQ(d1.dim_, 1);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.edge_ids_.size(), 5);
    EXPECT_EQ(d2.dim_, 1);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.edge_ids_.size(), 1);
    EXPECT_EQ(d3.dim_, 1);

    // note: by_attr doesn't apply to edge sets in 3D, since
    //       mfem doesn't have the notion of edge attributes
    // Domain d4 = Domain::ofEdges(mesh, by_attr<dim>(3));
  }

  {
    constexpr int dim = 2;
    auto bmesh = import_mesh("beam-quad.mesh");
    bmesh.FinalizeQuadMesh(true);
    auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));
    Domain d0 = Domain::ofEdges(*mesh, std::function([](std::vector<vec2> x, int /* bdr_attr */) {
      return (0.5 * (x[0][0] + x[1][0])) < 0.25;  // x coordinate of edge midpoint
    }));
    EXPECT_EQ(d0.edge_ids_.size(), 1);
    EXPECT_EQ(d0.dim_, 1);

    Domain d1 = Domain::ofEdges(*mesh, std::function([](std::vector<vec2> x, int /* bdr_attr */) {
      return (0.5 * (x[0][1] + x[1][1])) < 0.25;  // y coordinate of edge midpoint
    }));
    EXPECT_EQ(d1.edge_ids_.size(), 8);
    EXPECT_EQ(d1.dim_, 1);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.edge_ids_.size(), 9);
    EXPECT_EQ(d2.dim_, 1);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.edge_ids_.size(), 0);
    EXPECT_EQ(d3.dim_, 1);

    // check that by_attr compiles
    Domain d4 = Domain::ofEdges(*mesh, by_attr<dim>(3));
    EXPECT_EQ(d4.mfem_edge_ids_.size(), 16);

    Domain d5 = Domain::ofBoundaryElements(*mesh, [](std::vector<vec2>, int) { return true; });
    EXPECT_EQ(d5.edge_ids_.size(), 18);  // 1x8 row of quads has 18 boundary edges
  }
}

TEST(domain, of_faces)
{
  {
    constexpr int dim = 3;
    auto bmesh = import_mesh("onehex.mesh");
    auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));
    Domain d0 = Domain::ofFaces(*mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
      return average(vertices)[0] < 0.25;  // x coordinate of face center
    }));
    EXPECT_EQ(d0.quad_ids_.size(), 1);
    EXPECT_EQ(d0.dim_, 2);

    Domain d1 = Domain::ofFaces(*mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
      return average(vertices)[1] < 0.25;  // y coordinate of face center
    }));
    EXPECT_EQ(d1.quad_ids_.size(), 1);
    EXPECT_EQ(d1.dim_, 2);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.quad_ids_.size(), 2);
    EXPECT_EQ(d2.dim_, 2);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.quad_ids_.size(), 0);
    EXPECT_EQ(d3.dim_, 2);

    // check that by_attr compiles
    Domain d4 = Domain::ofFaces(*mesh, by_attr<dim>(3));

    Domain d5 = Domain::ofBoundaryElements(*mesh, [](std::vector<vec3>, int) { return true; });
    EXPECT_EQ(d5.quad_ids_.size(), 6);
  }

  {
    constexpr int dim = 3;
    auto bmesh = import_mesh("onetet.mesh");
    auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));
    Domain d0 = Domain::ofFaces(*mesh, std::function([](std::vector<vec3> vertices, int /* bdr_attr */) {
      // accept face if it contains a vertex whose x coordinate is less than 0.1
      for (auto v : vertices) {
        if (v[0] < 0.1) return true;
      }
      return false;
    }));
    EXPECT_EQ(d0.tri_ids_.size(), 4);
    EXPECT_EQ(d0.dim_, 2);

    Domain d1 = Domain::ofFaces(
        *mesh, std::function([](std::vector<vec3> x, int /* bdr_attr */) { return average(x)[1] < 0.1; }));
    EXPECT_EQ(d1.tri_ids_.size(), 1);
    EXPECT_EQ(d1.dim_, 2);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.tri_ids_.size(), 4);
    EXPECT_EQ(d2.dim_, 2);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.tri_ids_.size(), 1);
    EXPECT_EQ(d3.dim_, 2);

    // check that by_attr compiles
    Domain d4 = Domain::ofFaces(*mesh, by_attr<dim>(3));

    Domain d5 = Domain::ofBoundaryElements(*mesh, [](std::vector<vec3>, int) { return true; });
    EXPECT_EQ(d5.tri_ids_.size(), 4);
  }

  {
    constexpr int dim = 2;
    auto bmesh = import_mesh("beam-quad.mesh");
    auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));
    Domain d0 = Domain::ofFaces(*mesh, std::function([](std::vector<vec2> vertices, int /* attr */) {
      return average(vertices)[0] < 2.25;  // x coordinate of face center
    }));
    EXPECT_EQ(d0.quad_ids_.size(), 2);
    EXPECT_EQ(d0.dim_, 2);

    Domain d1 = Domain::ofFaces(*mesh, std::function([](std::vector<vec2> vertices, int /* attr */) {
      return average(vertices)[1] < 0.55;  // y coordinate of face center
    }));
    EXPECT_EQ(d1.quad_ids_.size(), 8);
    EXPECT_EQ(d1.dim_, 2);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.quad_ids_.size(), 8);
    EXPECT_EQ(d2.dim_, 2);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.quad_ids_.size(), 2);
    EXPECT_EQ(d3.dim_, 2);

    // check that by_attr compiles
    Domain d4 = Domain::ofFaces(*mesh, by_attr<dim>(3));
  }
}

TEST(domain, of_elements)
{
  {
    constexpr int dim = 3;
    auto bmesh = import_mesh("patch3D_tets_and_hexes.mesh");
    auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));
    Domain d0 = Domain::ofElements(*mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
      return average(vertices)[0] < 0.7;  // x coordinate of face center
    }));

    EXPECT_EQ(d0.tet_ids_.size(), 0);
    EXPECT_EQ(d0.hex_ids_.size(), 1);
    EXPECT_EQ(d0.dim_, 3);

    Domain d1 = Domain::ofElements(*mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
      return average(vertices)[1] < 0.75;  // y coordinate of face center
    }));
    EXPECT_EQ(d1.tet_ids_.size(), 6);
    EXPECT_EQ(d1.hex_ids_.size(), 1);
    EXPECT_EQ(d1.dim_, 3);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.tet_ids_.size(), 6);
    EXPECT_EQ(d2.hex_ids_.size(), 2);
    EXPECT_EQ(d2.dim_, 3);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.tet_ids_.size(), 0);
    EXPECT_EQ(d3.hex_ids_.size(), 0);
    EXPECT_EQ(d3.dim_, 3);

    // check that by_attr works
    Domain d4 = Domain::ofElements(*mesh, by_attr<dim>(3));
  }

  {
    constexpr int dim = 2;
    auto bmesh = import_mesh("patch2D_tris_and_quads.mesh");
    auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));
    Domain d0 = Domain::ofElements(
        *mesh, std::function([](std::vector<vec2> vertices, int /* attr */) { return average(vertices)[0] < 0.45; }));
    EXPECT_EQ(d0.tri_ids_.size(), 1);
    EXPECT_EQ(d0.quad_ids_.size(), 1);
    EXPECT_EQ(d0.dim_, 2);

    Domain d1 = Domain::ofElements(
        *mesh, std::function([](std::vector<vec2> vertices, int /* attr */) { return average(vertices)[1] < 0.45; }));
    EXPECT_EQ(d1.tri_ids_.size(), 1);
    EXPECT_EQ(d1.quad_ids_.size(), 1);
    EXPECT_EQ(d1.dim_, 2);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.tri_ids_.size(), 2);
    EXPECT_EQ(d2.quad_ids_.size(), 2);
    EXPECT_EQ(d2.dim_, 2);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.tri_ids_.size(), 0);
    EXPECT_EQ(d3.quad_ids_.size(), 0);
    EXPECT_EQ(d3.dim_, 2);

    // check that by_attr compiles
    Domain d4 = Domain::ofElements(*mesh, by_attr<dim>(3));
  }
}

TEST(domain, entireDomain2d)
{
  constexpr int dim = 2;
  constexpr int p = 1;
  auto bmesh = import_mesh("patch2D_tris_and_quads.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  Domain d0 = EntireDomain(*mesh);

  EXPECT_EQ(d0.dim_, 2);
  EXPECT_EQ(d0.tri_ids_.size(), 2);
  EXPECT_EQ(d0.quad_ids_.size(), 4);

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::ParFiniteElementSpace(mesh.get(), &fec);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);
  EXPECT_EQ(dof_indices.Size(), 8);
}

TEST(domain, entireDomain3d)
{
  constexpr int dim = 3;
  constexpr int p = 1;
  auto bmesh = import_mesh("patch3D_tets_and_hexes.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  Domain d0 = EntireDomain(*mesh);

  EXPECT_EQ(d0.dim_, 3);
  EXPECT_EQ(d0.tet_ids_.size(), 12);
  EXPECT_EQ(d0.hex_ids_.size(), 7);

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::ParFiniteElementSpace(mesh.get(), &fec);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);
  EXPECT_EQ(dof_indices.Size(), 25);
}

TEST(domain, of2dElementsFindsDofs)
{
  constexpr int dim = 2;
  constexpr int p = 2;
  auto bmesh = import_mesh("patch2D_tris_and_quads.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::ParFiniteElementSpace(mesh.get(), &fec);

  auto find_element_0 = [](std::vector<vec2> vertices, int /* attr */) {
    auto centroid = average(vertices);
    return (centroid[0] < 0.5) && (centroid[1] < 0.25);
  };

  Domain d0 = Domain::ofElements(*mesh, find_element_0);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 9);

  ///////////////////////////////////////

  auto find_element_4 = [](std::vector<vec2> vertices, int) {
    auto centroid = average(vertices);
    tensor<double, 2> target{{0.533, 0.424}};
    return norm(centroid - target) < 1e-2;
  };
  Domain d1 = Domain::ofElements(*mesh, find_element_4);

  Domain elements_0_and_4 = d0 | d1;

  dof_indices = elements_0_and_4.dof_list(&fes);
  EXPECT_EQ(dof_indices.Size(), 12);

  ///////////////////////////////////////

  Domain d2 = EntireDomain(*mesh) - elements_0_and_4;

  dof_indices = d2.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 22);
}

TEST(domain, of3dElementsFindsDofs)
{
  constexpr int dim = 3;
  constexpr int p = 2;
  auto bmesh = import_mesh("patch3D_tets_and_hexes.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::ParFiniteElementSpace(mesh.get(), &fec);

  auto find_element_0 = [](std::vector<vec3> vertices, int /* attr */) {
    auto centroid = average(vertices);
    vec3 target{{3.275, 0.7, 1.225}};
    return norm(centroid - target) < 1e-2;
  };

  Domain d0 = Domain::ofElements(*mesh, find_element_0);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);

  // element 0 is a P2 tetrahedron, so it should have 10 dofs
  EXPECT_EQ(dof_indices.Size(), 10);

  ///////////////////////////////////////

  auto find_element_1 = [](std::vector<vec3> vertices, int) {
    auto centroid = average(vertices);
    vec3 target{{3.275, 1.2, 0.725}};
    return norm(centroid - target) < 1e-2;
  };
  Domain d1 = Domain::ofElements(*mesh, find_element_1);

  Domain elements_0_and_1 = d0 | d1;

  dof_indices = elements_0_and_1.dof_list(&fes);

  // Elements 0 and 1 are P2 tets that share one face -> 14 dofs
  EXPECT_EQ(dof_indices.Size(), 14);

  /////////////////////////////////////////

  Domain d2 = EntireDomain(*mesh) - elements_0_and_1;

  dof_indices = d2.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 113);
}

TEST(domain, of2dBoundaryElementsFindsDofs)
{
  constexpr int dim = 2;
  constexpr int p = 2;
  auto bmesh = import_mesh("patch2D_tris_and_quads.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  auto find_right_boundary = [](std::vector<vec2> vertices, int /* attr */) {
    return std::all_of(vertices.begin(), vertices.end(), [](vec2 X) { return X[0] > 1.0 - 1e-2; });
  };

  Domain d0 = Domain::ofBoundaryElements(*mesh, find_right_boundary);
  EXPECT_EQ(d0.edge_ids_.size(), 1);

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::ParFiniteElementSpace(mesh.get(), &fec);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 3);

  auto find_top_boundary = [](std::vector<vec2> vertices, int /* attr */) {
    return std::all_of(vertices.begin(), vertices.end(), [](vec2 X) { return X[1] > 1.0 - 1e-2; });
  };

  Domain d1 = Domain::ofBoundaryElements(*mesh, find_top_boundary);
  EXPECT_EQ(d1.edge_ids_.size(), 1);

  Domain d2 = d0 | d1;

  dof_indices = d2.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 5);
}

TEST(domain, of3dBoundaryElementsFindsDofs)
{
  constexpr int dim = 3;
  constexpr int p = 2;
  auto bmesh = import_mesh("patch3D_tets.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  auto find_xmax_boundary = [](std::vector<vec3> vertices, int /* attr */) {
    return std::all_of(vertices.begin(), vertices.end(), [](vec3 X) { return X[0] > 1.0 - 1e-2; });
  };

  Domain d0 = Domain::ofBoundaryElements(*mesh, find_xmax_boundary);
  EXPECT_EQ(d0.tri_ids_.size(), 2);

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::ParFiniteElementSpace(mesh.get(), &fec);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 9);

  auto find_ymax_boundary = [](std::vector<vec3> vertices, int /* attr */) {
    return std::all_of(vertices.begin(), vertices.end(), [](vec3 X) { return X[1] > 1.0 - 1e-2; });
  };

  Domain d1 = Domain::ofBoundaryElements(*mesh, find_ymax_boundary);
  EXPECT_EQ(d1.tri_ids_.size(), 2);

  Domain d2 = d0 | d1;

  dof_indices = d2.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 15);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
