#include <gtest/gtest.h>

#include "serac/numerics/functional/domain.hpp"

using namespace serac;

std::string mesh_dir = SERAC_REPO_DIR "/data/meshes/";

mfem::Mesh import_mesh(std::string meshfile)
{
  mfem::named_ifgzstream imesh(mesh_dir + meshfile);

  if (!imesh) {
    serac::logger::flush();
    std::string err_msg = axom::fmt::format("Can not open mesh file: '{0}'", mesh_dir + meshfile);
    SLIC_ERROR_ROOT(err_msg);
  }

  mfem::Mesh mesh(imesh, 1, 1, true);
  mesh.EnsureNodes();
  return mesh;
}

template <int dim>
tensor<double, dim> average(std::vector<tensor<double, dim> >& positions)
{
  tensor<double, dim> total{};
  for (auto x : positions) {
    total += x;
  }
  return total / double(positions.size());
}

TEST(domain, of_edges)
{
  {
    auto   mesh = import_mesh("onehex.mesh");
    Domain d0   = Domain::ofEdges(mesh, std::function([](std::vector<vec3> x) {
                                  return (0.5 * (x[0][0] + x[1][0])) < 0.25;  // x coordinate of edge midpoint
                                }));
    auto d0_edges = d0.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d0_edges.size(), 4);
    EXPECT_EQ(d0.dim(), 1);

    Domain d1 = Domain::ofEdges(mesh, std::function([](std::vector<vec3> x) {
                                  return (0.5 * (x[0][1] + x[1][1])) < 0.25;  // y coordinate of edge midpoint
                                }));
    auto d1_edges = d1.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d1_edges.size(), 4);
    EXPECT_EQ(d1.dim(), 1);

    Domain d2 = d0 | d1;
    auto d2_edges = d2.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d2_edges.size(), 7);
    EXPECT_EQ(d2.dim(), 1);

    Domain d3 = d0 & d1;
    auto d3_edges = d3.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d3_edges.size(), 1);
    EXPECT_EQ(d3.dim(), 1);

    // note: by_attr doesn't apply to edge sets in 3D, since
    //       mfem doesn't have the notion of edge attributes
    // Domain d4 = Domain::ofEdges(mesh, by_attr<dim>(3));
  }

  {
    auto   mesh = import_mesh("onetet.mesh");
    Domain d0   = Domain::ofEdges(mesh, std::function([](std::vector<vec3> x) {
                                  return (0.5 * (x[0][0] + x[1][0])) < 0.25;  // x coordinate of edge midpoint
                                }));
    auto d0_edges = d0.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d0_edges.size(), 3);
    EXPECT_EQ(d0.dim(), 1);

    Domain d1 = Domain::ofEdges(mesh, std::function([](std::vector<vec3> x) {
                                  return (0.5 * (x[0][1] + x[1][1])) < 0.25;  // y coordinate of edge midpoint
                                }));
    auto d1_edges = d1.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d1_edges.size(), 3);
    EXPECT_EQ(d1.dim(), 1);

    Domain d2 = d0 | d1;
    auto d2_edges = d2.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d2_edges.size(), 5);
    EXPECT_EQ(d2.dim(), 1);

    Domain d3 = d0 & d1;
    auto d3_edges = d3.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d3_edges.size(), 1);
    EXPECT_EQ(d3.dim(), 1);

    // note: by_attr doesn't apply to edge sets in 3D, since
    //       mfem doesn't have the notion of edge attributes
    // Domain d4 = Domain::ofEdges(mesh, by_attr<dim>(3));
  }

  {
    constexpr int dim  = 2;
    auto          mesh = import_mesh("beam-quad.mesh");
    mesh.FinalizeQuadMesh(true);
    Domain d0 = Domain::ofEdges(mesh, std::function([](std::vector<vec2> x, int /* bdr_attr */) {
                                  return (0.5 * (x[0][0] + x[1][0])) < 0.25;  // x coordinate of edge midpoint
                                }));
    auto d0_edges = d0.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d0_edges.size(), 1);
    EXPECT_EQ(d0.dim(), 1);

    Domain d1 = Domain::ofEdges(mesh, std::function([](std::vector<vec2> x, int /* bdr_attr */) {
                                  return (0.5 * (x[0][1] + x[1][1])) < 0.25;  // y coordinate of edge midpoint
                                }));
    auto d1_edges = d1.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d1_edges.size(), 8);
    EXPECT_EQ(d1.dim(), 1);

    Domain d2 = d0 | d1;
    auto d2_edges = d2.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d2_edges.size(), 9);
    EXPECT_EQ(d2.dim(), 1);

    Domain d3 = d0 & d1;
    auto d3_edges = d3.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d3_edges.size(), 0);
    EXPECT_EQ(d3.dim(), 1);

    // check that by_attr compiles
    Domain d4 = Domain::ofEdges(mesh, by_attr<dim>(3));

    Domain d5 = Domain::ofBoundaryElements(mesh, [](std::vector<vec2>, int) { return true; });
    auto d5_edges = d5.get(mfem::Geometry::SEGMENT);
    EXPECT_EQ(d5_edges.size(), 18);  // 1x8 row of quads has 18 boundary edges
  }
}

TEST(domain, of_faces)
{
  {
    constexpr int dim  = 3;
    auto          mesh = import_mesh("onehex.mesh");
    Domain        d0   = Domain::ofFaces(mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
                                  return average(vertices)[0] < 0.25;  // x coordinate of face center
                                }));
    auto d0_quads = d0.get(mfem::Geometry::SQUARE);
    EXPECT_EQ(d0_quads.size(), 1);
    EXPECT_EQ(d0.dim(), 2);

    Domain d1 = Domain::ofFaces(mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
                                  return average(vertices)[1] < 0.25;  // y coordinate of face center
                                }));
    auto d1_quads = d1.get(mfem::Geometry::SQUARE);
    EXPECT_EQ(d1_quads.size(), 1);
    EXPECT_EQ(d1.dim(), 2);

    Domain d2 = d0 | d1;
    auto d2_quads = d2.get(mfem::Geometry::SQUARE);
    EXPECT_EQ(d2_quads.size(), 2);
    EXPECT_EQ(d2.dim(), 2);

    Domain d3 = d0 & d1;
    auto d3_quads = d3.get(mfem::Geometry::SQUARE);
    EXPECT_EQ(d3_quads.size(), 0);
    EXPECT_EQ(d3.dim(), 2);

    // check that by_attr compiles
    Domain d4 = Domain::ofFaces(mesh, by_attr<dim>(3));

    Domain d5 = Domain::ofBoundaryElements(mesh, [](std::vector<vec3>, int) { return true; });
    auto d5_quads = d5.get(mfem::Geometry::SQUARE);
    EXPECT_EQ(d5_quads.size(), 6);
  }

  {
    constexpr int dim  = 3;
    auto          mesh = import_mesh("onetet.mesh");
    Domain        d0   = Domain::ofFaces(mesh, std::function([](std::vector<vec3> vertices, int /* bdr_attr */) {
                                  // accept face if it contains a vertex whose x coordinate is less than 0.1
                                  for (auto v : vertices) {
                                    if (v[0] < 0.1) return true;
                                  }
                                  return false;
                                }));
    auto d0_tris = d0.get(mfem::Geometry::TRIANGLE);
    EXPECT_EQ(d0_tris.size(), 4);
    EXPECT_EQ(d0.dim(), 2);

    Domain d1 = Domain::ofFaces(
        mesh, std::function([](std::vector<vec3> x, int /* bdr_attr */) { return average(x)[1] < 0.1; }));
    auto d1_tris = d1.get(mfem::Geometry::TRIANGLE);
    EXPECT_EQ(d1_tris.size(), 1);
    EXPECT_EQ(d1.dim(), 2);

    Domain d2 = d0 | d1;
    auto d2_tris = d2.get(mfem::Geometry::TRIANGLE);
    EXPECT_EQ(d2_tris.size(), 4);
    EXPECT_EQ(d2.dim(), 2);

    Domain d3 = d0 & d1;
    auto d3_tris = d3.get(mfem::Geometry::TRIANGLE);
    EXPECT_EQ(d3_tris.size(), 1);
    EXPECT_EQ(d3.dim(), 2);

    // check that by_attr compiles
    Domain d4 = Domain::ofFaces(mesh, by_attr<dim>(3));

    Domain d5 = Domain::ofBoundaryElements(mesh, [](std::vector<vec3>, int) { return true; });
    auto d5_tris = d5.get(mfem::Geometry::TRIANGLE);
    EXPECT_EQ(d5_tris.size(), 4);
  }

  {
    constexpr int dim  = 2;
    auto          mesh = import_mesh("beam-quad.mesh");
    Domain        d0   = Domain::ofFaces(mesh, std::function([](std::vector<vec2> vertices, int /* attr */) {
                                  return average(vertices)[0] < 2.25;  // x coordinate of face center
                                }));
    auto d0_quads = d0.get(mfem::Geometry::SQUARE);
    EXPECT_EQ(d0_quads.size(), 2);
    EXPECT_EQ(d0.dim(), 2);

    Domain d1 = Domain::ofFaces(mesh, std::function([](std::vector<vec2> vertices, int /* attr */) {
                                  return average(vertices)[1] < 0.55;  // y coordinate of face center
                                }));
    auto d1_quads = d1.get(mfem::Geometry::SQUARE);
    EXPECT_EQ(d1_quads.size(), 8);
    EXPECT_EQ(d1.dim(), 2);

    Domain d2 = d0 | d1;
    auto d2_quads = d2.get(mfem::Geometry::SQUARE);
    EXPECT_EQ(d2_quads.size(), 8);
    EXPECT_EQ(d2.dim(), 2);

    Domain d3 = d0 & d1;
    auto d3_quads = d3.get(mfem::Geometry::SQUARE);
    EXPECT_EQ(d3_quads.size(), 2);
    EXPECT_EQ(d3.dim(), 2);

    // check that by_attr compiles
    Domain d4 = Domain::ofFaces(mesh, by_attr<dim>(3));
  }
}

TEST(domain, of_elements)
{
  {
    constexpr int dim  = 3;
    auto          mesh = import_mesh("patch3D_tets_and_hexes.mesh");
    Domain        d0   = Domain::ofElements(mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
                                     return average(vertices)[0] < 0.7;  // x coordinate of face center
                                   }));

    EXPECT_EQ(d0.get(mfem::Geometry::TETRAHEDRON).size(), 0);
    EXPECT_EQ(d0.get(mfem::Geometry::CUBE).size(), 1);
    EXPECT_EQ(d0.dim(), 3);

    Domain d1 = Domain::ofElements(mesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
                                     return average(vertices)[1] < 0.75;  // y coordinate of face center
                                   }));
    EXPECT_EQ(d1.get(mfem::Geometry::TETRAHEDRON).size(), 6);
    EXPECT_EQ(d1.get(mfem::Geometry::CUBE).size(), 1);
    EXPECT_EQ(d1.dim(), 3);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.get(mfem::Geometry::TETRAHEDRON).size(), 6);
    EXPECT_EQ(d2.get(mfem::Geometry::CUBE).size(), 2);
    EXPECT_EQ(d2.dim(), 3);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.get(mfem::Geometry::TETRAHEDRON).size(), 0);
    EXPECT_EQ(d3.get(mfem::Geometry::CUBE).size(), 0);
    EXPECT_EQ(d3.dim(), 3);

    // check that by_attr works
    Domain d4 = Domain::ofElements(mesh, by_attr<dim>(3));
  }

  {
    constexpr int dim  = 2;
    auto          mesh = import_mesh("patch2D_tris_and_quads.mesh");
    Domain        d0   = Domain::ofElements(
                 mesh, std::function([](std::vector<vec2> vertices, int /* attr */) { return average(vertices)[0] < 0.45; }));
    EXPECT_EQ(d0.get(mfem::Geometry::TRIANGLE).size(), 1);
    EXPECT_EQ(d0.get(mfem::Geometry::SQUARE).size(), 1);
    EXPECT_EQ(d0.dim(), 2);

    Domain d1 = Domain::ofElements(
        mesh, std::function([](std::vector<vec2> vertices, int /* attr */) { return average(vertices)[1] < 0.45; }));
    EXPECT_EQ(d1.get(mfem::Geometry::TRIANGLE).size(), 1);
    EXPECT_EQ(d1.get(mfem::Geometry::SQUARE).size(), 1);
    EXPECT_EQ(d1.dim(), 2);

    Domain d2 = d0 | d1;
    EXPECT_EQ(d2.get(mfem::Geometry::TRIANGLE).size(), 2);
    EXPECT_EQ(d2.get(mfem::Geometry::SQUARE).size(), 2);
    EXPECT_EQ(d2.dim(), 2);

    Domain d3 = d0 & d1;
    EXPECT_EQ(d3.get(mfem::Geometry::TRIANGLE).size(), 0);
    EXPECT_EQ(d3.get(mfem::Geometry::SQUARE).size(), 0);
    EXPECT_EQ(d3.dim(), 2);

    // check that by_attr compiles
    Domain d4 = Domain::ofElements(mesh, by_attr<dim>(3));
  }
}

TEST(domain, entireDomain2d)
{
  constexpr int dim  = 2;
  constexpr int p    = 1;
  auto          mesh = import_mesh("patch2D_tris_and_quads.mesh");

  Domain d0 = EntireDomain(mesh);

  EXPECT_EQ(d0.dim(), 2);
  EXPECT_EQ(d0.get(mfem::Geometry::TRIANGLE).size(), 2);
  EXPECT_EQ(d0.get(mfem::Geometry::SQUARE).size(), 4);

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::FiniteElementSpace(&mesh, &fec);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);
  EXPECT_EQ(dof_indices.Size(), 8);
}

TEST(domain, entireDomain3d)
{
  constexpr int dim  = 3;
  constexpr int p    = 1;
  auto          mesh = import_mesh("patch3D_tets_and_hexes.mesh");

  Domain d0 = EntireDomain(mesh);

  EXPECT_EQ(d0.dim(), 3);
  EXPECT_EQ(d0.get(mfem::Geometry::TETRAHEDRON).size(), 12);
  EXPECT_EQ(d0.get(mfem::Geometry::CUBE).size(), 7);

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::FiniteElementSpace(&mesh, &fec);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);
  EXPECT_EQ(dof_indices.Size(), 25);
}

TEST(domain, of2dElementsFindsDofs)
{
  constexpr int dim  = 2;
  constexpr int p    = 2;
  auto          mesh = import_mesh("patch2D_tris_and_quads.mesh");

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::FiniteElementSpace(&mesh, &fec);

  auto find_element_0 = [](std::vector<vec2> vertices, int /* attr */) {
    auto centroid = average(vertices);
    return (centroid[0] < 0.5) && (centroid[1] < 0.25);
  };

  Domain d0 = Domain::ofElements(mesh, find_element_0);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 9);

  ///////////////////////////////////////

  auto find_element_4 = [](std::vector<vec2> vertices, int) {
    auto              centroid = average(vertices);
    tensor<double, 2> target{{0.533, 0.424}};
    return norm(centroid - target) < 1e-2;
  };
  Domain d1 = Domain::ofElements(mesh, find_element_4);

  Domain elements_0_and_4 = d0 | d1;

  dof_indices = elements_0_and_4.dof_list(&fes);
  EXPECT_EQ(dof_indices.Size(), 12);

  ///////////////////////////////////////

  Domain d2 = EntireDomain(mesh) - elements_0_and_4;

  dof_indices = d2.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 22);
}

TEST(domain, of3dElementsFindsDofs)
{
  constexpr int dim  = 3;
  constexpr int p    = 2;
  auto          mesh = import_mesh("patch3D_tets_and_hexes.mesh");

  auto fec = mfem::H1_FECollection(p, dim);
  auto fes = mfem::FiniteElementSpace(&mesh, &fec);

  auto find_element_0 = [](std::vector<vec3> vertices, int /* attr */) {
    auto centroid = average(vertices);
    vec3 target{{3.275, 0.7, 1.225}};
    return norm(centroid - target) < 1e-2;
  };

  Domain d0 = Domain::ofElements(mesh, find_element_0);

  mfem::Array<int> dof_indices = d0.dof_list(&fes);

  // element 0 is a P2 tetrahedron, so it should have 10 dofs
  EXPECT_EQ(dof_indices.Size(), 10);

  ///////////////////////////////////////

  auto find_element_1 = [](std::vector<vec3> vertices, int) {
    auto centroid = average(vertices);
    vec3 target{{3.275, 1.2, 0.725}};
    return norm(centroid - target) < 1e-2;
  };
  Domain d1 = Domain::ofElements(mesh, find_element_1);

  Domain elements_0_and_1 = d0 | d1;

  dof_indices = elements_0_and_1.dof_list(&fes);

  // Elements 0 and 1 are P2 tets that share one face -> 14 dofs
  EXPECT_EQ(dof_indices.Size(), 14);

  /////////////////////////////////////////

  Domain d2 = EntireDomain(mesh) - elements_0_and_1;

  dof_indices = d2.dof_list(&fes);

  EXPECT_EQ(dof_indices.Size(), 113);
}

int main(int argc, char* argv[])
{
  int num_procs, myid;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
