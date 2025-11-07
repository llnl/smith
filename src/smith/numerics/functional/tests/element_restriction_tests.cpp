// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <ostream>
#include <vector>

#include "gtest/gtest.h"

#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/element_restriction.hpp"
#include "smith/infrastructure/memory.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

using namespace smith;

std::ostream& operator<<(std::ostream& out, axom::Array<DoF, 2, smith::detail::host_memory_space> arr)
{
  for (int i = 0; i < arr.shape()[0]; i++) {
    for (int j = 0; j < arr.shape()[1]; j++) {
      out << arr[i][j].index() << " ";
    }
    out << std::endl;
  }
  return out;
}

TEST(patch_test_meshes, triangle_domains)
{
  int p = 2;
  int dim = 2;
  mfem::Mesh bmesh(SMITH_REPO_DIR "/data/meshes/patch2D_tris.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  auto H1_fec = std::make_unique<mfem::H1_FECollection>(p, dim);
  auto H1_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), H1_fec.get());

  auto Hcurl_fec = std::make_unique<mfem::ND_FECollection>(p, dim);
  auto Hcurl_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), Hcurl_fec.get());

  auto L2_fec = std::make_unique<mfem::L2_FECollection>(p, dim, mfem::BasisType::GaussLobatto);
  auto L2_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), L2_fec.get());

  Domain whole = EntireDomain(*mesh);
  EXPECT_EQ(whole.mfem_edge_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_tri_ids_.size(), 4);
  EXPECT_EQ(whole.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), whole);
    EXPECT_EQ(H1_BER.ESize(), 4 * 6);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), whole);
    EXPECT_EQ(Hcurl_BER.ESize(), 4 * 8);

    BlockElementRestriction L2_BER(L2_fes.get(), whole);
    EXPECT_EQ(L2_BER.ESize(), 4 * 6);
  }

  Domain boundary = EntireBoundary(*mesh);
  EXPECT_EQ(boundary.mfem_edge_ids_.size(), 4);
  EXPECT_EQ(boundary.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), boundary);
    EXPECT_EQ(H1_BER.ESize(), 4 * 3);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), boundary);
    EXPECT_EQ(Hcurl_BER.ESize(), 4 * 2);

    BlockElementRestriction L2_BER(L2_fes.get(), boundary);
    EXPECT_EQ(L2_BER.ESize(), 4 * 3);
  }

  Domain interior = InteriorFaces(*mesh);
  EXPECT_EQ(interior.mfem_edge_ids_.size(), 4);
  EXPECT_EQ(interior.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), interior);
    EXPECT_EQ(H1_BER.ESize(), 4 * 3);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), interior);
    EXPECT_EQ(Hcurl_BER.ESize(), 4 * 2);

    BlockElementRestriction L2_BER(L2_fes.get(), interior);
    EXPECT_EQ(L2_BER.ESize(), 4 * (3 * 2));
  }
}

TEST(patch_test_meshes, quadrilateral_domains)
{
  int p = 2;
  int dim = 2;
  mfem::Mesh bmesh(SMITH_REPO_DIR "/data/meshes/patch2D_quads.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  auto H1_fec = std::make_unique<mfem::H1_FECollection>(p, dim);
  auto Hcurl_fec = std::make_unique<mfem::ND_FECollection>(p, dim);
  auto L2_fec = std::make_unique<mfem::L2_FECollection>(p, dim, mfem::BasisType::GaussLobatto);

  auto H1_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), H1_fec.get());
  auto Hcurl_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), Hcurl_fec.get());
  auto L2_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), L2_fec.get());

  Domain whole = EntireDomain(*mesh);
  EXPECT_EQ(whole.mfem_edge_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_quad_ids_.size(), 5);
  EXPECT_EQ(whole.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), whole);
    EXPECT_EQ(H1_BER.ESize(), 5 * 9);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), whole);
    EXPECT_EQ(Hcurl_BER.ESize(), 5 * 12);

    BlockElementRestriction L2_BER(L2_fes.get(), whole);
    EXPECT_EQ(L2_BER.ESize(), 5 * 9);
  }

  Domain boundary = EntireBoundary(*mesh);
  EXPECT_EQ(boundary.mfem_edge_ids_.size(), 4);
  EXPECT_EQ(boundary.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), boundary);
    EXPECT_EQ(H1_BER.ESize(), 4 * 3);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), boundary);
    EXPECT_EQ(Hcurl_BER.ESize(), 4 * 2);

    BlockElementRestriction L2_BER(L2_fes.get(), boundary);
    EXPECT_EQ(L2_BER.ESize(), 4 * 3);
  }

  Domain interior = InteriorFaces(*mesh);
  EXPECT_EQ(interior.mfem_edge_ids_.size(), 8);
  EXPECT_EQ(interior.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), interior);
    EXPECT_EQ(H1_BER.ESize(), 8 * 3);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), interior);
    EXPECT_EQ(Hcurl_BER.ESize(), 8 * 2);

    BlockElementRestriction L2_BER(L2_fes.get(), interior);
    EXPECT_EQ(L2_BER.ESize(), 8 * (3 * 2));
  }
}

TEST(patch_test_meshes, triangle_and_quadrilateral_domains)
{
  int p = 2;
  int dim = 2;
  mfem::Mesh bmesh(SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  auto H1_fec = std::make_unique<mfem::H1_FECollection>(p, dim);
  auto Hcurl_fec = std::make_unique<mfem::ND_FECollection>(p, dim);
  auto L2_fec = std::make_unique<mfem::L2_FECollection>(p, dim, mfem::BasisType::GaussLobatto);

  auto H1_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), H1_fec.get());
  auto Hcurl_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), Hcurl_fec.get());
  auto L2_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), L2_fec.get());

  Domain whole = EntireDomain(*mesh);
  EXPECT_EQ(whole.mfem_edge_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_tri_ids_.size(), 2);
  EXPECT_EQ(whole.mfem_quad_ids_.size(), 4);
  EXPECT_EQ(whole.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), whole);
    EXPECT_EQ(H1_BER.ESize(), 2 * 6 + 4 * 9);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), whole);
    EXPECT_EQ(Hcurl_BER.ESize(), 2 * 8 + 4 * 12);

    BlockElementRestriction L2_BER(L2_fes.get(), whole);
    EXPECT_EQ(L2_BER.ESize(), 2 * 6 + 4 * 9);
  }

  Domain boundary = EntireBoundary(*mesh);
  EXPECT_EQ(boundary.mfem_edge_ids_.size(), 4);
  EXPECT_EQ(boundary.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), boundary);
    EXPECT_EQ(H1_BER.ESize(), 4 * 3);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), boundary);
    EXPECT_EQ(Hcurl_BER.ESize(), 4 * 2);

    BlockElementRestriction L2_BER(L2_fes.get(), boundary);
    EXPECT_EQ(L2_BER.ESize(), 4 * 3);
  }

  Domain interior = InteriorFaces(*mesh);
  EXPECT_EQ(interior.mfem_edge_ids_.size(), 9);
  EXPECT_EQ(interior.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), interior);
    EXPECT_EQ(H1_BER.ESize(), 9 * 3);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), interior);
    EXPECT_EQ(Hcurl_BER.ESize(), 9 * 2);

    BlockElementRestriction L2_BER(L2_fes.get(), interior);
    EXPECT_EQ(L2_BER.ESize(), 9 * (3 * 2));
  }
}

TEST(patch_test_meshes, tetrahedron_domains)
{
  int p = 2;
  int dim = 3;
  mfem::Mesh bmesh(SMITH_REPO_DIR "/data/meshes/patch3D_tets.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  auto H1_fec = std::make_unique<mfem::H1_FECollection>(p, dim);
  auto Hcurl_fec = std::make_unique<mfem::ND_FECollection>(p, dim);
  auto L2_fec = std::make_unique<mfem::L2_FECollection>(p, dim, mfem::BasisType::GaussLobatto);

  auto H1_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), H1_fec.get());
  auto Hcurl_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), Hcurl_fec.get());
  auto L2_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), L2_fec.get());

  Domain whole = EntireDomain(*mesh);
  EXPECT_EQ(whole.mfem_edge_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_tet_ids_.size(), 12);
  EXPECT_EQ(whole.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), whole);
    EXPECT_EQ(H1_BER.ESize(), 12 * 10);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), whole);
    EXPECT_EQ(Hcurl_BER.ESize(), 12 * 20);

    BlockElementRestriction L2_BER(L2_fes.get(), whole);
    EXPECT_EQ(L2_BER.ESize(), 12 * 10);
  }

  Domain boundary = EntireBoundary(*mesh);
  EXPECT_EQ(boundary.mfem_edge_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_tri_ids_.size(), 12);
  EXPECT_EQ(boundary.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), boundary);
    EXPECT_EQ(H1_BER.ESize(), 12 * 6);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), boundary);
    EXPECT_EQ(Hcurl_BER.ESize(), 12 * 8);

    BlockElementRestriction L2_BER(L2_fes.get(), boundary);
    EXPECT_EQ(L2_BER.ESize(), 12 * 6);
  }

  Domain interior = InteriorFaces(*mesh);
  EXPECT_EQ(interior.mfem_edge_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_tri_ids_.size(), 18);
  EXPECT_EQ(interior.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), interior);
    EXPECT_EQ(H1_BER.ESize(), 18 * 6);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), interior);
    EXPECT_EQ(Hcurl_BER.ESize(), 18 * 8);

    BlockElementRestriction L2_BER(L2_fes.get(), interior);
    EXPECT_EQ(L2_BER.ESize(), 18 * (6 * 2));
  }
}

TEST(patch_test_meshes, hexahedron_domains)
{
  int p = 2;
  int dim = 3;
  mfem::Mesh bmesh(SMITH_REPO_DIR "/data/meshes/patch3D_hexes.mesh");
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  auto H1_fec = std::make_unique<mfem::H1_FECollection>(p, dim);
  auto Hcurl_fec = std::make_unique<mfem::ND_FECollection>(p, dim);
  auto L2_fec = std::make_unique<mfem::L2_FECollection>(p, dim, mfem::BasisType::GaussLobatto);

  auto H1_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), H1_fec.get());
  auto Hcurl_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), Hcurl_fec.get());
  auto L2_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), L2_fec.get());

  Domain whole = EntireDomain(*mesh);
  EXPECT_EQ(whole.mfem_edge_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_quad_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(whole.mfem_hex_ids_.size(), 7);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), whole);
    EXPECT_EQ(H1_BER.ESize(), 7 * 27);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), whole);
    EXPECT_EQ(Hcurl_BER.ESize(), 7 * 54);

    BlockElementRestriction L2_BER(L2_fes.get(), whole);
    EXPECT_EQ(L2_BER.ESize(), 7 * 27);
  }

  Domain boundary = EntireBoundary(*mesh);
  EXPECT_EQ(boundary.mfem_edge_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_quad_ids_.size(), 6);
  EXPECT_EQ(boundary.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(boundary.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), boundary);
    EXPECT_EQ(H1_BER.ESize(), 6 * 9);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), boundary);
    EXPECT_EQ(Hcurl_BER.ESize(), 6 * 12);

    BlockElementRestriction L2_BER(L2_fes.get(), boundary);
    EXPECT_EQ(L2_BER.ESize(), 6 * 9);
  }

  Domain interior = InteriorFaces(*mesh);
  EXPECT_EQ(interior.mfem_edge_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_tri_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_quad_ids_.size(), 18);
  EXPECT_EQ(interior.mfem_tet_ids_.size(), 0);
  EXPECT_EQ(interior.mfem_hex_ids_.size(), 0);

  {
    BlockElementRestriction H1_BER(H1_fes.get(), interior);
    EXPECT_EQ(H1_BER.ESize(), 18 * 9);

    BlockElementRestriction Hcurl_BER(Hcurl_fes.get(), interior);
    EXPECT_EQ(Hcurl_BER.ESize(), 18 * 12);

    BlockElementRestriction L2_BER(L2_fes.get(), interior);
    EXPECT_EQ(L2_BER.ESize(), 18 * (9 * 2));
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
