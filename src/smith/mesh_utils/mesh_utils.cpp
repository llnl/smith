// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/mesh_utils/mesh_utils.hpp"

#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "axom/core.hpp"
#include "axom/fmt.hpp"

#include "smith/infrastructure/logger.hpp"

namespace smith {

mfem::Mesh buildMeshFromFile(const std::string& mesh_file)
{
  // Open the mesh
  std::string msg = axom::fmt::format("Opening mesh file: '{0}'", mesh_file);
  SLIC_INFO_ROOT(msg);

  // Ensure correctness
  smith::logger::flush();
  if (!axom::utilities::filesystem::pathExists(mesh_file)) {
    msg = axom::fmt::format("Given mesh file does not exist: '{0}'", mesh_file);
    SLIC_ERROR_ROOT(msg);
  }

  // This inherits from std::ifstream, and will work the same way as a std::ifstream,
  // but is required for Exodus meshes
  mfem::named_ifgzstream imesh(mesh_file);

  if (!imesh) {
    smith::logger::flush();
    std::string err_msg = axom::fmt::format("Can not open mesh file: '{0}'", mesh_file);
    SLIC_ERROR_ROOT(err_msg);
  }

  return mfem::Mesh{imesh, 1, 1, true};
}

/**
 * @brief a transformation from the unit disk/sphere (in L1 norm) to a unit disk/sphere (in L2 norm)
 *
 * @param mesh The mesh to transform
 */
void squish(mfem::Mesh& mesh)
{
  int num_vertices = mesh.GetNV();
  int dim = mesh.SpaceDimension();

  mfem::Vector vertices;
  mesh.GetVertices(vertices);
  mfem::Vector vertex(dim);
  for (int i = 0; i < num_vertices; i++) {
    for (int d = 0; d < dim; d++) {
      vertex(d) = vertices[d * num_vertices + i];
    }

    double L1_norm = vertex.Norml1();
    double L2_norm = vertex.Norml2();
    vertex *= (L2_norm < 1.0e-6) ? 0.0 : (L1_norm / L2_norm);

    for (int d = 0; d < dim; d++) {
      vertices[d * num_vertices + i] = vertex(d);
    }
  }
  mesh.SetVertices(vertices);
}

mfem::Mesh buildDiskMesh(int approx_number_of_elements)
{
  static constexpr int dim = 2;
  static constexpr int num_elems = 4;
  static constexpr int num_vertices = 5;
  static constexpr int num_boundary_elements = 4;

  static constexpr double vertices[num_vertices][dim] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  static constexpr int triangles[num_elems][3] = {{1, 2, 0}, {2, 3, 0}, {3, 4, 0}, {4, 1, 0}};
  static constexpr int segments[num_elems][2] = {{1, 2}, {2, 3}, {3, 4}, {4, 1}};

  auto mesh = mfem::Mesh(dim, num_vertices, num_elems, num_boundary_elements);

  for (auto vertex : vertices) {
    mesh.AddVertex(vertex);
  }
  for (auto triangle : triangles) {
    mesh.AddTriangle(triangle);
  }
  for (auto segment : segments) {
    mesh.AddBdrSegment(segment);
  }
  mesh.FinalizeTriMesh();

  while (mesh.GetNE() < (0.5 * approx_number_of_elements)) {
    mesh.UniformRefinement();
  }

  squish(mesh);

  return mesh;
}

mfem::Mesh buildBallMesh(int approx_number_of_elements)
{
  static constexpr int dim = 3;
  static constexpr int num_elems = 8;
  static constexpr int num_vertices = 7;
  static constexpr int num_boundary_elements = 8;

  static constexpr double vertices[num_vertices][dim] = {{0, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, 0, -1},
                                                         {0, 0, 1}, {0, -1, 0}, {1, 0, 0}};
  static constexpr int triangles[num_elems][3] = {{4, 5, 6}, {4, 6, 2}, {4, 2, 1}, {4, 1, 5},
                                                  {5, 1, 3}, {5, 3, 6}, {3, 1, 2}, {6, 3, 2}};
  static constexpr int tetrahedra[num_elems][4] = {{0, 4, 5, 6}, {0, 4, 6, 2}, {0, 4, 2, 1}, {0, 4, 1, 5},
                                                   {0, 5, 1, 3}, {0, 5, 3, 6}, {0, 3, 1, 2}, {0, 6, 3, 2}};

  auto mesh = mfem::Mesh(dim, num_vertices, num_elems, num_boundary_elements);

  for (auto vertex : vertices) {
    mesh.AddVertex(vertex);
  }
  for (auto tetrahedron : tetrahedra) {
    mesh.AddTet(tetrahedron);
  }
  for (auto triangle : triangles) {
    mesh.AddBdrTriangle(triangle);
  }

  mesh.FinalizeTetMesh();

  while (mesh.GetNE() < (0.25 * approx_number_of_elements)) {
    mesh.UniformRefinement();
  }

  squish(mesh);

  return mesh;
}

mfem::Mesh buildRectangleMesh(int elements_in_x, int elements_in_y, double size_x, double size_y)
{
  return mfem::Mesh::MakeCartesian2D(elements_in_x, elements_in_y, mfem::Element::QUADRILATERAL, true, size_x, size_y);
}

mfem::Mesh buildCuboidMesh(int elements_in_x, int elements_in_y, int elements_in_z, double size_x, double size_y,
                           double size_z)
{
  return mfem::Mesh::MakeCartesian3D(elements_in_x, elements_in_y, elements_in_z, mfem::Element::HEXAHEDRON, size_x,
                                     size_y, size_z);
}

mfem::Mesh buildCylinderMesh(int radial_refinement, int elements_lengthwise, double radius, double height)
{
  static constexpr int dim = 2;
  static constexpr int num_vertices = 17;
  static constexpr int num_elems = 12;
  static constexpr int num_boundary_elements = 8;

  // a == 1.0 produces a mesh of a cylindrical "core" surrounded by
  // a cylindrical "shell", but makes some of the elements in the "core" section
  // nearly degenerate
  //
  // a > 1 makes the "core" section no longer a cylinder, but its elements
  // are no longer nearly degenerate
  constexpr double a = 1.3;
  static constexpr double vertices[num_vertices][dim] = {{0.0000000000000000, 0.0000000000000000},
                                                         {0.5773502691896258, 0.0000000000000000},
                                                         {0.4082482904638630 * a, 0.4082482904638630 * a},
                                                         {0.0000000000000000, 0.5773502691896258},
                                                         {-0.4082482904638630 * a, 0.4082482904638630 * a},
                                                         {-0.5773502691896258, 0.0000000000000000},
                                                         {-0.4082482904638630 * a, -0.4082482904638630 * a},
                                                         {0.0000000000000000, -0.5773502691896258},
                                                         {0.4082482904638630 * a, -0.4082482904638630 * a},
                                                         {1.000000000000000, 0.0000000000000000},
                                                         {0.7071067811865475, 0.7071067811865475},
                                                         {0.0000000000000000, 1.000000000000000},
                                                         {-0.707106781186548, 0.7071067811865475},
                                                         {-1.000000000000000, 0.0000000000000000},
                                                         {-0.707106781186548, -0.707106781186548},
                                                         {0.0000000000000000, -1.000000000000000},
                                                         {0.7071067811865475, -0.707106781186548}};

  static constexpr int elems[num_elems][4] = {{0, 1, 2, 3},   {0, 3, 4, 5},   {0, 5, 6, 7},   {0, 7, 8, 1},
                                              {1, 9, 10, 2},  {2, 10, 11, 3}, {3, 11, 12, 4}, {4, 12, 13, 5},
                                              {5, 13, 14, 6}, {6, 14, 15, 7}, {7, 15, 16, 8}, {8, 16, 9, 1}};

  static constexpr int boundary_elems[num_boundary_elements][2] = {{9, 10},  {10, 11}, {11, 12}, {12, 13},
                                                                   {13, 14}, {14, 15}, {15, 16}, {16, 9}};

  auto mesh = mfem::Mesh(dim, num_vertices, num_elems, num_boundary_elements);

  for (auto vertex : vertices) {
    mesh.AddVertex(vertex);
  }
  for (auto elem : elems) {
    mesh.AddQuad(elem);
  }
  for (auto boundary_elem : boundary_elems) {
    mesh.AddBdrSegment(boundary_elem);
  }

  for (int i = 0; i < radial_refinement; i++) {
    mesh.UniformRefinement();
  }

  // the coarse mesh is actually a filled octagon
  // this deforms the vertices slightly to make it
  // into filled disk instead
  {
    int n = mesh.GetNV();

    mfem::Vector new_vertices;
    mesh.GetVertices(new_vertices);
    mfem::Vector vertex(dim);
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        vertex(d) = new_vertices[d * n + i];
      }

      // stretch the octagonal shape into a circle of the appropriate radius
      double theta = atan2(vertex(1), vertex(0));
      double phi = fmod(theta + M_PI, M_PI_4);
      vertex *= radius * (cos(phi) + (-1.0 + sqrt(2.0)) * sin(phi));

      for (int d = 0; d < dim; d++) {
        new_vertices[d * n + i] = vertex(d);
      }
    }
    mesh.SetVertices(new_vertices);
  }

  return mfem::Mesh(*mfem::Extrude2D(&mesh, elements_lengthwise, height));
}

/// @brief Constructs a 2D MFEM mesh of a ring
mfem::Mesh buildRing(int radial_refinement, double inner_radius, double outer_radius, double total_angle, int sectors)
{
  using index_type = int;
  using size_type = std::vector<index_type>::size_type;

  static constexpr int dim = 2;

  SLIC_ASSERT_MSG(total_angle > 0., "only positive angles supported");

  // ensure total_angle is (0, 2 * pi]
  total_angle = std::min(total_angle, 2. * M_PI);
  const double angle = total_angle / sectors;

  auto num_elems = static_cast<size_type>(sectors);
  auto num_vertices_ring = static_cast<size_type>((total_angle == 2. * M_PI) ? sectors : sectors + 1);
  auto num_vertices = num_vertices_ring * 2;
  auto num_boundary_elements = num_elems * 2;

  SLIC_ERROR_ROOT_IF(outer_radius <= inner_radius,
                     "Outer radius is smaller than inner radius while building a cylinder mesh.");

  std::vector<std::vector<double>> vertices(static_cast<size_type>(num_vertices), std::vector<double>(dim, 0.));
  for (size_type i = 0; i < num_vertices_ring; i++) {
    double s = sin(angle * static_cast<double>(i));
    double c = cos(angle * static_cast<double>(i));
    vertices[i][0] = inner_radius * c;
    vertices[i][1] = inner_radius * s;

    vertices[i + num_vertices_ring][0] = outer_radius * c;
    vertices[i + num_vertices_ring][1] = outer_radius * s;
  }

  std::vector<std::vector<index_type>> elems(static_cast<size_type>(num_elems), std::vector<index_type>(4, 0));
  std::vector<std::vector<index_type>> boundary_elems(static_cast<size_type>(num_boundary_elements),
                                                      std::vector<index_type>(2, 0));
  for (size_type i = 0; i < num_elems; i++) {
    elems[i][0] = static_cast<index_type>(i);
    elems[i][1] = static_cast<index_type>(num_vertices_ring + i);
    elems[i][2] = static_cast<index_type>(num_vertices_ring + (i + 1) % (num_vertices_ring));
    elems[i][3] = static_cast<index_type>((i + 1) % num_vertices_ring);

    // inner boundary
    boundary_elems[i][0] = elems[i][3];
    boundary_elems[i][1] = elems[i][0];

    // outer boundary
    boundary_elems[i + num_elems][0] = elems[i][1];
    boundary_elems[i + num_elems][1] = elems[i][2];
  }

  mfem::Mesh mesh(dim, static_cast<int>(num_vertices), static_cast<int>(num_elems),
                  static_cast<int>(num_boundary_elements));

  for (auto vertex : vertices) {
    mesh.AddVertex(vertex.data());
  }
  for (auto elem : elems) {
    mesh.AddQuad(elem[0], elem[1], elem[2], elem[3]);
  }
  for (auto boundary_elem : boundary_elems) {
    mesh.AddBdrSegment(boundary_elem[0], boundary_elem[1]);
  }

  for (int i = 0; i < radial_refinement; i++) {
    mesh.UniformRefinement();
  }

  // the coarse mesh is actually a filled octagon
  // this deforms the vertices slightly to make it
  // into filled disk instead
  {
    int n = mesh.GetNV();

    mfem::Vector new_vertices;
    mesh.GetVertices(new_vertices);
    mfem::Vector vertex(dim);
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        vertex(d) = new_vertices[d * n + i];
      }

      // stretch the polygonal shape into a cylinder
      // phi is the angle to the closest multiple of a sector angle
      double theta = atan2(vertex(1), vertex(0));
      double phi = fmod(theta + 2. * M_PI, angle);

      // this calculation assumes the 0 <= phi <= angle
      // the distance from the center of the cylinder to the midpoint of the radial edge is known
      // the midpoint can also be used to form a right triangle to phi where
      // the angle is given by abs(0.5 * angle - phi)
      double factor = cos(fabs(0.5 * angle - phi)) / cos(0.5 * angle);
      vertex *= factor;

      for (int d = 0; d < dim; d++) {
        new_vertices[d * n + i] = vertex(d);
      }
    }
    mesh.SetVertices(new_vertices);
  }

  return mesh;
}

mfem::Mesh buildRingMesh(int radial_refinement, double inner_radius, double outer_radius, double total_angle,
                         int sectors)
{
  return buildRing(radial_refinement, inner_radius, outer_radius, total_angle, sectors);
}

mfem::Mesh buildHollowCylinderMesh(int radial_refinement, int elements_lengthwise, double inner_radius,
                                   double outer_radius, double height, double total_angle, int sectors)
{
  auto mesh = buildRing(radial_refinement, inner_radius, outer_radius, total_angle, sectors);
  return mfem::Mesh(*mfem::Extrude2D(&mesh, elements_lengthwise, height));
}

/// @brief Constructs an MFEM mesh of a hollow cylinder restricted to the first orthant
mfem::Mesh build_hollow_quarter_cylinder(std::size_t radial_divisions, std::size_t angular_divisions,
                                         std::size_t vertical_divisions, double inner_radius, double outer_radius,
                                         double height)
{
  constexpr int dim = 3;

  // start with a structured mesh of a cube
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(static_cast<int>(radial_divisions), static_cast<int>(angular_divisions),
                                                static_cast<int>(vertical_divisions), mfem::Element::HEXAHEDRON);

  int num_vertices = mesh.GetNV();
  mfem::Vector new_vertices;
  mesh.GetVertices(new_vertices);
  mfem::Vector vertex(dim);
  for (int i = 0; i < num_vertices; i++) {
    for (int d = 0; d < dim; d++) {
      vertex(d) = new_vertices[d * num_vertices + i];
    }

    // transform the vertices to make it into a cylindrical shell
    double r = inner_radius + (outer_radius - inner_radius) * vertex[0];
    double theta = vertex[1] * M_PI_2;
    vertex(0) = r * cos(theta);
    vertex(1) = r * sin(theta);
    vertex(2) = vertex(2) * height;

    for (int d = 0; d < dim; d++) {
      new_vertices[d * num_vertices + i] = vertex(d);
    }
  }
  mesh.SetVertices(new_vertices);

  return mesh;
}
namespace mesh {

void InputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Refinement levels
  container.addInt("ser_ref_levels", "Number of times to refine the mesh uniformly in serial.").defaultValue(0);
  container.addInt("par_ref_levels", "Number of times to refine the mesh uniformly in parallel.").defaultValue(0);

  // Types of meshes we support
  container.addString("type", "Type of mesh").required().validValues({"ball", "box", "disk", "file"});

  // `file` type mesh options
  container.addString("mesh", "Path to Mesh file");

  // `box` type mesh generation options
  auto& elements = container.addStruct("elements");
  // TODO: Can these be specified as required if elements is defined?
  elements.addInt("x", "x-dimension");
  elements.addInt("y", "y-dimension");
  elements.addInt("z", "z-dimension");

  auto& size = container.addStruct("size");
  // TODO: Can these be specified as required if elements is defined?
  size.addDouble("x", "Size in the x-dimension");
  size.addDouble("y", "Size in the y-dimension");
  size.addDouble("z", "Size in the z-dimension");

  // `ball` and `disk` mesh generation options
  container.addInt("approx_elements", "Approximate number of elements in an n-ball mesh");
}

std::unique_ptr<mfem::ParMesh> buildParallelMesh(const InputOptions& options, const MPI_Comm comm)
{
  std::optional<mfem::Mesh> serial_mesh;

  if (const auto file_opts = std::get_if<FileInputOptions>(&options.extra_options)) {
    SLIC_ERROR_ROOT_IF(file_opts->absolute_mesh_file_name.empty(),
                       "Absolute path to mesh file was not configured, did you forget to call findMeshFilePath?");
    serial_mesh.emplace(buildMeshFromFile(file_opts->absolute_mesh_file_name));
  } else if (const auto box_opts = std::get_if<BoxInputOptions>(&options.extra_options)) {
    const auto& elems = box_opts->elements;
    const auto& sizes = box_opts->overall_size;
    if (elems.size() == 2) {
      serial_mesh.emplace(buildRectangleMesh(elems.at(0), elems.at(1), sizes.at(0), sizes.at(1)));
    } else {
      serial_mesh.emplace(
          buildCuboidMesh(elems.at(0), elems.at(1), elems.at(2), sizes.at(0), sizes.at(1), sizes.at(2)));
    }
  } else if (const auto ball_opts = std::get_if<NBallInputOptions>(&options.extra_options)) {
    if (ball_opts->dimension == 2) {
      serial_mesh.emplace(buildDiskMesh(ball_opts->approx_elements));
    } else {
      serial_mesh.emplace(buildBallMesh(ball_opts->approx_elements));
    }
  }

  SLIC_ERROR_ROOT_IF(!serial_mesh, "Mesh input options were invalid");
  return refineAndDistribute(std::move(*serial_mesh), options.ser_ref_levels, options.par_ref_levels, comm);
}

std::unique_ptr<mfem::ParMesh> refineAndDistribute(mfem::Mesh&& serial_mesh, const int refine_serial,
                                                   const int refine_parallel, const MPI_Comm comm)
{
  // Serial refinement first
  for (int lev = 0; lev < refine_serial; lev++) {
    serial_mesh.UniformRefinement();
  }

  // Then create the parallel mesh and apply parallel refinement
  auto parallel_mesh = std::make_unique<mfem::ParMesh>(comm, serial_mesh);
  for (int lev = 0; lev < refine_parallel; lev++) {
    parallel_mesh->UniformRefinement();
  }

  parallel_mesh->EnsureNodes();
  parallel_mesh->ExchangeFaceNbrData();

  return parallel_mesh;
}

}  // namespace mesh
}  // namespace smith

smith::mesh::InputOptions FromInlet<smith::mesh::InputOptions>::operator()(const axom::inlet::Container& base)
{
  int ser_ref = base["ser_ref_levels"];
  int par_ref = base["par_ref_levels"];

  // This is for cuboid/rectangular meshes
  std::string mesh_type = base["type"];
  if (mesh_type == "box") {
    auto elements_input = base["elements"];
    bool z_present = elements_input.contains("z");

    std::vector<int> elements(z_present ? 3 : 2);
    elements[0] = elements_input["x"];
    elements[1] = elements_input["y"];
    if (z_present) elements[2] = elements_input["z"];

    std::vector<double> overall_size;
    if (base.contains("size")) {
      auto size_input = base["size"];
      overall_size.push_back(size_input["x"]);
      overall_size.push_back(size_input["y"]);

      if (size_input.contains("z")) {
        overall_size.push_back(size_input["z"]);
      }
    } else {
      overall_size = std::vector<double>(elements.size(), 1.);
    }

    return {smith::mesh::BoxInputOptions{elements, overall_size}, ser_ref, par_ref};
  } else if (mesh_type == "disk" || mesh_type == "ball") {
    int approx_elements = base["approx_elements"];
    int dim = 3;
    if (mesh_type == "disk") {
      dim = 2;
    }
    return {smith::mesh::NBallInputOptions{approx_elements, dim}, ser_ref, par_ref};
  } else if (mesh_type == "file") {  // This is for file-based meshes
    std::string mesh_path = base["mesh"];
    return {smith::mesh::FileInputOptions{mesh_path}, ser_ref, par_ref};
  }

  // If it reaches here, we haven't found a supported type
  smith::logger::flush();
  std::string err_msg = axom::fmt::format("Specified type not supported: '{0}'", mesh_type);
  SLIC_ERROR_ROOT(err_msg);
  return {};
}
