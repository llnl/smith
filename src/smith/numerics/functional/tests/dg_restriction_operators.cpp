// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/element_restriction.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"

using namespace smith;

std::string mesh_dir = SMITH_REPO_DIR "/data/meshes/";

constexpr mfem::Geometry::Type face_type(mfem::Geometry::Type geom)
{
  if (geom == mfem::Geometry::TRIANGLE) return mfem::Geometry::SEGMENT;
  if (geom == mfem::Geometry::SQUARE) return mfem::Geometry::SEGMENT;
  if (geom == mfem::Geometry::TETRAHEDRON) return mfem::Geometry::TRIANGLE;
  if (geom == mfem::Geometry::CUBE) return mfem::Geometry::SQUARE;
  return mfem::Geometry::INVALID;
}

int possible_permutations(mfem::Geometry::Type geom)
{
  if (geom == mfem::Geometry::TRIANGLE) return 3;
  if (geom == mfem::Geometry::SQUARE) return 4;
  if (geom == mfem::Geometry::TETRAHEDRON) return 12;
  if (geom == mfem::Geometry::CUBE) return 24;
  return -1;
}

template <uint32_t n>
std::array<int, n> apply_permutation(const int (&arr)[n], const int (&p)[n])
{
  std::array<int, n> permuted_arr{};
  for (uint32_t i = 0; i < n; i++) {
    permuted_arr[i] = arr[p[i]];
  }
  return permuted_arr;
}

mfem::Mesh generate_permuted_mesh(mfem::Geometry::Type geom, int i)
{
  if (geom == mfem::Geometry::TRIANGLE) {
    constexpr int dim = 2;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 4;
    constexpr int num_permutations = 3;
    int positive_permutations[num_permutations][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};

    /*
        y
        ^
        |
        3----------2
        |'.        |
        |  '.      |
        |    '.    |
        |      '.  |
        |        '.|
        0----------1--> x
    */
    int elements[num_elements][3] = {{0, 1, 3}, {1, 2, 3}};
    double vertices[num_vertices][dim] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) {
      output.AddVertex(vertex);
    }

    // the first element is always fixed
    output.AddTri(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddTri(permuted_element.data());

    output.FinalizeMesh();

    return output;
  }

  if (geom == mfem::Geometry::SQUARE) {
    constexpr int dim = 2;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 6;
    constexpr int num_permutations = 4;
    int positive_permutations[num_permutations][4] = {{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}};

    /*
        y
        ^
        |
        3----------4----------5
        |          |          |
        |          |          |
        |          |          |
        |          |          |
        0----------1----------2--> x
    */
    int elements[num_elements][4] = {{0, 1, 4, 3}, {1, 2, 5, 4}};
    double vertices[num_vertices][dim] = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, 1.0}};

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) {
      output.AddVertex(vertex);
    }

    // the first element is always fixed
    output.AddQuad(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddQuad(permuted_element.data());

    output.FinalizeMesh();

    return output;
  }

  if (geom == mfem::Geometry::TETRAHEDRON) {
    constexpr int dim = 3;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 5;
    constexpr int num_permutations = 12;
    int positive_permutations[num_permutations][4] = {{0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {1, 0, 3, 2},
                                                      {1, 2, 0, 3}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 1, 3, 0},
                                                      {2, 3, 0, 1}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 2, 1, 0}};

    /*

                    .4.
          y      .*'/  '*.
           \  .*'  /      '*.
            2--.../          '*.
            |\   / '---...      '*.          x
            | \ /         '''---...'*.   .*'
            |  /                   :::>1
      z     | / \         ...---'''.*'
        '*. |/   ...---'''      .*'
            3--'''\          .*'
              '*.  \      .*'
                 '*.\  .*'
                    '0'

    */
    int elements[num_elements][4] = {{0, 1, 2, 3}, {1, 2, 3, 4}};
    double vertices[num_vertices][dim] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) {
      output.AddVertex(vertex);
    }

    // the first element is always fixed
    output.AddTet(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddTet(permuted_element.data());

    output.FinalizeMesh();

    return output;
  }

  if (geom == mfem::Geometry::CUBE) {
    constexpr int dim = 3;
    constexpr int num_elements = 2;
    constexpr int num_vertices = 12;
    constexpr int num_permutations = 24;
    int positive_permutations[num_permutations][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7}, {0, 3, 7, 4, 1, 2, 6, 5}, {0, 4, 5, 1, 3, 7, 6, 2}, {1, 0, 4, 5, 2, 3, 7, 6},
        {1, 2, 3, 0, 5, 6, 7, 4}, {1, 5, 6, 2, 0, 4, 7, 3}, {2, 1, 5, 6, 3, 0, 4, 7}, {2, 3, 0, 1, 6, 7, 4, 5},
        {2, 6, 7, 3, 1, 5, 4, 0}, {3, 0, 1, 2, 7, 4, 5, 6}, {3, 2, 6, 7, 0, 1, 5, 4}, {3, 7, 4, 0, 2, 6, 5, 1},
        {4, 0, 3, 7, 5, 1, 2, 6}, {4, 5, 1, 0, 7, 6, 2, 3}, {4, 7, 6, 5, 0, 3, 2, 1}, {5, 1, 0, 4, 6, 2, 3, 7},
        {5, 4, 7, 6, 1, 0, 3, 2}, {5, 6, 2, 1, 4, 7, 3, 0}, {6, 2, 1, 5, 7, 3, 0, 4}, {6, 5, 4, 7, 2, 1, 0, 3},
        {6, 7, 3, 2, 5, 4, 0, 1}, {7, 3, 2, 6, 4, 0, 1, 5}, {7, 4, 0, 3, 6, 5, 1, 2}, {7, 6, 5, 4, 3, 2, 1, 0}};

    /*
        z
        ^
        |
        8----------11
        |\         |\
        | \        | \
        |  \       |  \
        |   9------+---10
        |   |      |   |
        4---+------7   |
        |\  |      |\  |
        | \ |      | \ |
        |  \|      |  \|
        |   5------+---6
        |   |      |   |
        0---+------3---|--> y
         \  |       \  |
          \ |        \ |
           \|         \|
            1----------2
             \
              v
               x
    */
    double vertices[num_vertices][dim] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                          {1, 1, 1}, {0, 1, 1}, {0, 0, 2}, {1, 0, 2}, {1, 1, 2}, {0, 1, 2}};

    int elements[num_elements][8] = {{0, 1, 2, 3, 4, 5, 6, 7}, {4, 5, 6, 7, 8, 9, 10, 11}};

    mfem::Mesh output(dim, num_vertices, num_elements);

    for (auto vertex : vertices) {
      output.AddVertex(vertex);
    }

    // the first element is always fixed
    output.AddHex(elements[0]);

    // but the second element is permuted to the specified orientation
    auto permuted_element = apply_permutation(elements[1], positive_permutations[i]);
    output.AddHex(permuted_element.data());

    output.FinalizeMesh();

    return output;
  }

  return {};
}

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

template <int dim>
double scalar_func_mfem(const mfem::Vector& x, double /*t*/)
{
  if constexpr (dim == 2) {
    return x[0] + 10 * x[1];
  } else {
    return x[0] + 10 * x[1] + 100 * x[2];
  }
}

template <int dim>
double scalar_func(const tensor<double, dim> x)
{
  if constexpr (dim == 2) {
    return x[0] + 10 * x[1];
  } else {
    return x[0] + 10 * x[1] + 100 * x[2];
  }
}

template <int dim>
void vector_func_mfem(const mfem::Vector& x, double /*t*/, mfem::Vector& output)
{
  if constexpr (dim == 2) {
    output[0] = +x[1];
    output[1] = -x[0];
  } else {
    output[0] = +x[1];
    output[1] = -x[0] + x[2];
    output[2] = -x[1];
  }
}

template <mfem::Geometry::Type geom, int polynomial_order>
void parametrized_test(int permutation)
{
  constexpr mfem::Geometry::Type face_geom = face_type(geom);
  constexpr int dim = dimension_of(geom);

  mfem::Mesh bmesh = generate_permuted_mesh(geom, permutation);
  auto mesh = smith::mesh::refineAndDistribute(std::move(bmesh));

  Domain interior_faces = InteriorFaces(*mesh);

  // each one of these meshes should have two elements
  // and a single "face" that separates them
  EXPECT_EQ(mesh->GetNE(), 2);

  std::vector<int> face_ids;

  if (face_geom == mfem::Geometry::SEGMENT) {
    EXPECT_EQ(interior_faces.edge_ids_.size(), 1);
    EXPECT_EQ(interior_faces.mfem_edge_ids_.size(), 1);
    face_ids = interior_faces.mfem_edge_ids_;
  }

  if (face_geom == mfem::Geometry::TRIANGLE) {
    EXPECT_EQ(interior_faces.tri_ids_.size(), 1);
    EXPECT_EQ(interior_faces.mfem_tri_ids_.size(), 1);
    face_ids = interior_faces.mfem_tri_ids_;
  }

  if (face_geom == mfem::Geometry::SQUARE) {
    EXPECT_EQ(interior_faces.quad_ids_.size(), 1);
    EXPECT_EQ(interior_faces.mfem_quad_ids_.size(), 1);
    face_ids = interior_faces.mfem_quad_ids_;
  }

  auto H1_fec = std::make_unique<mfem::H1_FECollection>(polynomial_order, dim);
  auto Hcurl_fec = std::make_unique<mfem::ND_FECollection>(polynomial_order, dim);
  auto L2_fec = std::make_unique<mfem::L2_FECollection>(polynomial_order, dim, mfem::BasisType::GaussLobatto);

  auto H1_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), H1_fec.get());
  auto Hcurl_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), Hcurl_fec.get());
  auto L2_fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh.get(), L2_fec.get());

  mfem::ParGridFunction H1_gf(H1_fes.get());
  mfem::ParGridFunction Hcurl_gf(Hcurl_fes.get());
  mfem::ParGridFunction L2_gf(L2_fes.get());

  mfem::FunctionCoefficient sfunc(scalar_func_mfem<dim>);
  mfem::VectorFunctionCoefficient vfunc(dim, vector_func_mfem<dim>);

  H1_gf.ProjectCoefficient(sfunc);
  Hcurl_gf.ProjectCoefficient(vfunc);
  L2_gf.ProjectCoefficient(sfunc);

  auto H1_dofs = GetFaceDofs(H1_fes.get(), face_geom, FaceType::INTERIOR);
  auto Hcurl_dofs = GetFaceDofs(Hcurl_fes.get(), face_geom, FaceType::INTERIOR);

  auto L2_dofs = GetFaceDofs(L2_fes.get(), face_geom, face_ids);

  // verify that the dofs for the L2 faces are aligned properly
  int dofs_per_side = L2_dofs.shape()[1] / 2;
  for (int i = 0; i < dofs_per_side; i++) {
    int id1 = int(L2_dofs(0, i).index());
    int id2 = int(L2_dofs(0, i + dofs_per_side).index());
    // std::cout << id1 << " " << id2 << " " << L2_gf[id1] << " " << L2_gf[id2] << std::endl;
    EXPECT_NEAR(L2_gf[id1], L2_gf[id2], 5.0e-14);
  }

  ////////////////////////////////////////////////////////////////////////////////

  using space = L2<polynomial_order>;

  Functional<double(space)> r({L2_fes.get()});

  r.AddInteriorFaceIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [](double /*t*/, auto X, auto rho) {
        // std::cout << get<0>(X) << std::endl;
        double expected = scalar_func(get<0>(X));
        auto [rho1, rho2] = rho;
        EXPECT_NEAR(expected, get_value(rho1), 5.0e-14);
        EXPECT_NEAR(expected, get_value(rho2), 5.0e-14);
        auto difference = rho1 - rho2;
        return difference * difference;
      },
      interior_faces);

  mfem::Vector U(L2_fes->TrueVSize());
  mfem::ParGridFunction U_pgf(L2_fes.get());
  U_pgf.ProjectCoefficient(sfunc);
  U_pgf.GetTrueDofs(U);

  double t = 0.0;
  double output = r(t, U);

  EXPECT_NEAR(output, 0.0, 5.0e-14);

  // TODO: check that the actual values match their respective functions
  //       evaluated directly at the nodes (for H1, Hcurl, and L2 gfs)
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TEST(DomainInterior, TriMesh10) { parametrized_test<mfem::Geometry::TRIANGLE, 1>(0); }
TEST(DomainInterior, TriMesh11) { parametrized_test<mfem::Geometry::TRIANGLE, 1>(1); }
TEST(DomainInterior, TriMesh12) { parametrized_test<mfem::Geometry::TRIANGLE, 1>(2); }

TEST(DomainInterior, TriMesh20) { parametrized_test<mfem::Geometry::TRIANGLE, 2>(0); }
TEST(DomainInterior, TriMesh21) { parametrized_test<mfem::Geometry::TRIANGLE, 2>(1); }
TEST(DomainInterior, TriMesh22) { parametrized_test<mfem::Geometry::TRIANGLE, 2>(2); }

TEST(DomainInterior, TriMesh30) { parametrized_test<mfem::Geometry::TRIANGLE, 3>(0); }
TEST(DomainInterior, TriMesh31) { parametrized_test<mfem::Geometry::TRIANGLE, 3>(1); }
TEST(DomainInterior, TriMesh32) { parametrized_test<mfem::Geometry::TRIANGLE, 3>(2); }

////////////////////////////////////////////////////////////////////////////////

TEST(DomainInterior, QuadMesh10) { parametrized_test<mfem::Geometry::SQUARE, 1>(0); }
TEST(DomainInterior, QuadMesh11) { parametrized_test<mfem::Geometry::SQUARE, 1>(1); }
TEST(DomainInterior, QuadMesh12) { parametrized_test<mfem::Geometry::SQUARE, 1>(2); }
TEST(DomainInterior, QuadMesh13) { parametrized_test<mfem::Geometry::SQUARE, 1>(3); }

TEST(DomainInterior, QuadMesh20) { parametrized_test<mfem::Geometry::SQUARE, 2>(0); }
TEST(DomainInterior, QuadMesh21) { parametrized_test<mfem::Geometry::SQUARE, 2>(1); }
TEST(DomainInterior, QuadMesh22) { parametrized_test<mfem::Geometry::SQUARE, 2>(2); }
TEST(DomainInterior, QuadMesh23) { parametrized_test<mfem::Geometry::SQUARE, 2>(3); }

TEST(DomainInterior, QuadMesh30) { parametrized_test<mfem::Geometry::SQUARE, 3>(0); }
TEST(DomainInterior, QuadMesh31) { parametrized_test<mfem::Geometry::SQUARE, 3>(1); }
TEST(DomainInterior, QuadMesh32) { parametrized_test<mfem::Geometry::SQUARE, 3>(2); }
TEST(DomainInterior, QuadMesh33) { parametrized_test<mfem::Geometry::SQUARE, 3>(3); }

////////////////////////////////////////////////////////////////////////////////

TEST(DomainInterior, TetMesh100) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(0); }
TEST(DomainInterior, TetMesh101) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(1); }
TEST(DomainInterior, TetMesh102) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(2); }
TEST(DomainInterior, TetMesh103) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(3); }
TEST(DomainInterior, TetMesh104) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(4); }
TEST(DomainInterior, TetMesh105) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(5); }
TEST(DomainInterior, TetMesh106) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(6); }
TEST(DomainInterior, TetMesh107) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(7); }
TEST(DomainInterior, TetMesh108) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(8); }
TEST(DomainInterior, TetMesh109) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(9); }
TEST(DomainInterior, TetMesh110) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(10); }
TEST(DomainInterior, TetMesh111) { parametrized_test<mfem::Geometry::TETRAHEDRON, 1>(11); }

TEST(DomainInterior, TetMesh200) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(0); }
TEST(DomainInterior, TetMesh201) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(1); }
TEST(DomainInterior, TetMesh202) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(2); }
TEST(DomainInterior, TetMesh203) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(3); }
TEST(DomainInterior, TetMesh204) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(4); }
TEST(DomainInterior, TetMesh205) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(5); }
TEST(DomainInterior, TetMesh206) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(6); }
TEST(DomainInterior, TetMesh207) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(7); }
TEST(DomainInterior, TetMesh208) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(8); }
TEST(DomainInterior, TetMesh209) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(9); }
TEST(DomainInterior, TetMesh210) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(10); }
TEST(DomainInterior, TetMesh211) { parametrized_test<mfem::Geometry::TETRAHEDRON, 2>(11); }

TEST(DomainInterior, TetMesh300) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(0); }
TEST(DomainInterior, TetMesh301) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(1); }
TEST(DomainInterior, TetMesh302) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(2); }
TEST(DomainInterior, TetMesh303) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(3); }
TEST(DomainInterior, TetMesh304) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(4); }
TEST(DomainInterior, TetMesh305) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(5); }
TEST(DomainInterior, TetMesh306) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(6); }
TEST(DomainInterior, TetMesh307) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(7); }
TEST(DomainInterior, TetMesh308) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(8); }
TEST(DomainInterior, TetMesh309) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(9); }
TEST(DomainInterior, TetMesh310) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(10); }
TEST(DomainInterior, TetMesh311) { parametrized_test<mfem::Geometry::TETRAHEDRON, 3>(11); }

////////////////////////////////////////////////////////////////////////////////

TEST(DomainInterior, HexMesh100) { parametrized_test<mfem::Geometry::CUBE, 1>(0); }
TEST(DomainInterior, HexMesh101) { parametrized_test<mfem::Geometry::CUBE, 1>(1); }
TEST(DomainInterior, HexMesh102) { parametrized_test<mfem::Geometry::CUBE, 1>(2); }
TEST(DomainInterior, HexMesh103) { parametrized_test<mfem::Geometry::CUBE, 1>(3); }
TEST(DomainInterior, HexMesh104) { parametrized_test<mfem::Geometry::CUBE, 1>(4); }
TEST(DomainInterior, HexMesh105) { parametrized_test<mfem::Geometry::CUBE, 1>(5); }
TEST(DomainInterior, HexMesh106) { parametrized_test<mfem::Geometry::CUBE, 1>(6); }
TEST(DomainInterior, HexMesh107) { parametrized_test<mfem::Geometry::CUBE, 1>(7); }
TEST(DomainInterior, HexMesh108) { parametrized_test<mfem::Geometry::CUBE, 1>(8); }
TEST(DomainInterior, HexMesh109) { parametrized_test<mfem::Geometry::CUBE, 1>(9); }
TEST(DomainInterior, HexMesh110) { parametrized_test<mfem::Geometry::CUBE, 1>(10); }
TEST(DomainInterior, HexMesh111) { parametrized_test<mfem::Geometry::CUBE, 1>(11); }
TEST(DomainInterior, HexMesh112) { parametrized_test<mfem::Geometry::CUBE, 1>(12); }
TEST(DomainInterior, HexMesh113) { parametrized_test<mfem::Geometry::CUBE, 1>(13); }
TEST(DomainInterior, HexMesh114) { parametrized_test<mfem::Geometry::CUBE, 1>(14); }
TEST(DomainInterior, HexMesh115) { parametrized_test<mfem::Geometry::CUBE, 1>(15); }
TEST(DomainInterior, HexMesh116) { parametrized_test<mfem::Geometry::CUBE, 1>(16); }
TEST(DomainInterior, HexMesh117) { parametrized_test<mfem::Geometry::CUBE, 1>(17); }
TEST(DomainInterior, HexMesh118) { parametrized_test<mfem::Geometry::CUBE, 1>(18); }
TEST(DomainInterior, HexMesh119) { parametrized_test<mfem::Geometry::CUBE, 1>(19); }
TEST(DomainInterior, HexMesh120) { parametrized_test<mfem::Geometry::CUBE, 1>(20); }
TEST(DomainInterior, HexMesh121) { parametrized_test<mfem::Geometry::CUBE, 1>(21); }
TEST(DomainInterior, HexMesh122) { parametrized_test<mfem::Geometry::CUBE, 1>(22); }
TEST(DomainInterior, HexMesh123) { parametrized_test<mfem::Geometry::CUBE, 1>(23); }

TEST(DomainInterior, HexMesh200) { parametrized_test<mfem::Geometry::CUBE, 2>(0); }
TEST(DomainInterior, HexMesh201) { parametrized_test<mfem::Geometry::CUBE, 2>(1); }
TEST(DomainInterior, HexMesh202) { parametrized_test<mfem::Geometry::CUBE, 2>(2); }
TEST(DomainInterior, HexMesh203) { parametrized_test<mfem::Geometry::CUBE, 2>(3); }
TEST(DomainInterior, HexMesh204) { parametrized_test<mfem::Geometry::CUBE, 2>(4); }
TEST(DomainInterior, HexMesh205) { parametrized_test<mfem::Geometry::CUBE, 2>(5); }
TEST(DomainInterior, HexMesh206) { parametrized_test<mfem::Geometry::CUBE, 2>(6); }
TEST(DomainInterior, HexMesh207) { parametrized_test<mfem::Geometry::CUBE, 2>(7); }
TEST(DomainInterior, HexMesh208) { parametrized_test<mfem::Geometry::CUBE, 2>(8); }
TEST(DomainInterior, HexMesh209) { parametrized_test<mfem::Geometry::CUBE, 2>(9); }
TEST(DomainInterior, HexMesh210) { parametrized_test<mfem::Geometry::CUBE, 2>(10); }
TEST(DomainInterior, HexMesh211) { parametrized_test<mfem::Geometry::CUBE, 2>(11); }
TEST(DomainInterior, HexMesh212) { parametrized_test<mfem::Geometry::CUBE, 2>(12); }
TEST(DomainInterior, HexMesh213) { parametrized_test<mfem::Geometry::CUBE, 2>(13); }
TEST(DomainInterior, HexMesh214) { parametrized_test<mfem::Geometry::CUBE, 2>(14); }
TEST(DomainInterior, HexMesh215) { parametrized_test<mfem::Geometry::CUBE, 2>(15); }
TEST(DomainInterior, HexMesh216) { parametrized_test<mfem::Geometry::CUBE, 2>(16); }
TEST(DomainInterior, HexMesh217) { parametrized_test<mfem::Geometry::CUBE, 2>(17); }
TEST(DomainInterior, HexMesh218) { parametrized_test<mfem::Geometry::CUBE, 2>(18); }
TEST(DomainInterior, HexMesh219) { parametrized_test<mfem::Geometry::CUBE, 2>(19); }
TEST(DomainInterior, HexMesh220) { parametrized_test<mfem::Geometry::CUBE, 2>(20); }
TEST(DomainInterior, HexMesh221) { parametrized_test<mfem::Geometry::CUBE, 2>(21); }
TEST(DomainInterior, HexMesh222) { parametrized_test<mfem::Geometry::CUBE, 2>(22); }
TEST(DomainInterior, HexMesh223) { parametrized_test<mfem::Geometry::CUBE, 2>(23); }

TEST(DomainInterior, HexMesh300) { parametrized_test<mfem::Geometry::CUBE, 3>(0); }
TEST(DomainInterior, HexMesh301) { parametrized_test<mfem::Geometry::CUBE, 3>(1); }
TEST(DomainInterior, HexMesh302) { parametrized_test<mfem::Geometry::CUBE, 3>(2); }
TEST(DomainInterior, HexMesh303) { parametrized_test<mfem::Geometry::CUBE, 3>(3); }
TEST(DomainInterior, HexMesh304) { parametrized_test<mfem::Geometry::CUBE, 3>(4); }
TEST(DomainInterior, HexMesh305) { parametrized_test<mfem::Geometry::CUBE, 3>(5); }
TEST(DomainInterior, HexMesh306) { parametrized_test<mfem::Geometry::CUBE, 3>(6); }
TEST(DomainInterior, HexMesh307) { parametrized_test<mfem::Geometry::CUBE, 3>(7); }
TEST(DomainInterior, HexMesh308) { parametrized_test<mfem::Geometry::CUBE, 3>(8); }
TEST(DomainInterior, HexMesh309) { parametrized_test<mfem::Geometry::CUBE, 3>(9); }
TEST(DomainInterior, HexMesh310) { parametrized_test<mfem::Geometry::CUBE, 3>(10); }
TEST(DomainInterior, HexMesh311) { parametrized_test<mfem::Geometry::CUBE, 3>(11); }
TEST(DomainInterior, HexMesh312) { parametrized_test<mfem::Geometry::CUBE, 3>(12); }
TEST(DomainInterior, HexMesh313) { parametrized_test<mfem::Geometry::CUBE, 3>(13); }
TEST(DomainInterior, HexMesh314) { parametrized_test<mfem::Geometry::CUBE, 3>(14); }
TEST(DomainInterior, HexMesh315) { parametrized_test<mfem::Geometry::CUBE, 3>(15); }
TEST(DomainInterior, HexMesh316) { parametrized_test<mfem::Geometry::CUBE, 3>(16); }
TEST(DomainInterior, HexMesh317) { parametrized_test<mfem::Geometry::CUBE, 3>(17); }
TEST(DomainInterior, HexMesh318) { parametrized_test<mfem::Geometry::CUBE, 3>(18); }
TEST(DomainInterior, HexMesh319) { parametrized_test<mfem::Geometry::CUBE, 3>(19); }
TEST(DomainInterior, HexMesh320) { parametrized_test<mfem::Geometry::CUBE, 3>(20); }
TEST(DomainInterior, HexMesh321) { parametrized_test<mfem::Geometry::CUBE, 3>(21); }
TEST(DomainInterior, HexMesh322) { parametrized_test<mfem::Geometry::CUBE, 3>(22); }
TEST(DomainInterior, HexMesh323) { parametrized_test<mfem::Geometry::CUBE, 3>(23); }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
