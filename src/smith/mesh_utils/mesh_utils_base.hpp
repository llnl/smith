// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file mesh_utils_base.hpp
 *
 * @brief This file contains helper functions for importing and managing
 *        various mesh objects.
 */

#pragma once

#include <memory>
#include <variant>

#include "mfem.hpp"

#include "smith/infrastructure/input.hpp"

namespace smith {

/**
 * @brief Constructs an MFEM mesh from a file
 *
 * This opens and reads an external mesh file and constructs a serial
 * MFEM Mesh object.
 *
 * @param[in] mesh_file The mesh file to open
 * @return A serial mesh object
 */
mfem::Mesh buildMeshFromFile(const std::string& mesh_file);

/**
 * @brief Constructs a 2D MFEM mesh of a unit disk, centered at the origin
 *
 * This routine creates a mesh by refining a coarse disk mesh until the
 * number of elements is as close as possible to the user-specified number of elements
 *
 * @param[in] approx_number_of_elements The appoximate number of elements
 * @return The constructed mesh
 */
mfem::Mesh buildDiskMesh(int approx_number_of_elements);

/**
 * @brief Constructs a 3D MFEM mesh of a unit ball, centered at the origin
 *
 * This routine creates a mesh by refining a coarse ball mesh until the
 * number of elements is as close as possible to the user-specified number of elements
 *
 * @param[in] approx_number_of_elements Approximate number of elements
 * @return The constructed mesh
 */
mfem::Mesh buildBallMesh(int approx_number_of_elements);

/**
 * @brief Constructs a 2D MFEM mesh of a rectangle
 *
 * @param[in] elements_in_x the number of elements in the x-direction
 * @param[in] elements_in_y the number of elements in the y-direction
 * @param[in] size_x Overall size in the x-direction
 * @param[in] size_y Overall size in the y-direction
 * @return The constructed serial mesh
 */
mfem::Mesh buildRectangleMesh(int elements_in_x, int elements_in_y, double size_x = 1., double size_y = 1.);

/**
 * @brief Constructs a 3D MFEM mesh of a cuboid
 *
 * @param[in] elements_in_x the number of elements in the x-direction
 * @param[in] elements_in_y the number of elements in the y-direction
 * @param[in] elements_in_z the number of elements in the z-direction
 * @param[in] size_x Overall size in the x-direction
 * @param[in] size_y Overall size in the y-direction
 * @param[in] size_z Overall size in the z-direction
 * @return The constructed serial mesh
 */
mfem::Mesh buildCuboidMesh(int elements_in_x, int elements_in_y, int elements_in_z, double size_x = 1.,
                           double size_y = 1., double size_z = 1.);

/**
 * @brief Constructs a 3D MFEM mesh of a cylinder
 *
 * @param[in] radial_refinement the number of times to apply uniform mesh refinement to the cross section
 * @param[in] elements_lengthwise the number of elements in the z-direction
 * @param[in] radius the radius of the cylinder
 * @param[in] height the number of elements in the z-direction
 *
 * @return The constructed mesh
 */
mfem::Mesh buildCylinderMesh(int radial_refinement, int elements_lengthwise, double radius, double height);

/**
 * @brief Constructs a 3D MFEM mesh of a hollow cylinder
 *
 * @param[in] radial_refinement the number of times to apply uniform mesh refinement to the cross section
 * @param[in] elements_lengthwise the number of elements in the z-direction
 * @param[in] inner_radius inner radius the radius of the cylindrical shell
 * @param[in] outer_radius ouer radius the radius of the cylindrical shell
 * @param[in] height the number of elements in the z-direction
 * @param[in] total_angle the angle in radians over which to generate the portion of an extruded cylinder
 * @param[in] sectors the number of starting sectors in the hollow cylinder
 *
 * @return A unique_ptr containing the constructed mesh
 */
mfem::Mesh buildHollowCylinderMesh(int radial_refinement, int elements_lengthwise, double inner_radius,
                                   double outer_radius, double height, double total_angle = M_PI, int sectors = 8);

/**
 * @brief Constructs an MFEM mesh of a hollow cylinder restricted to the first orthant
 *
 * @param radial_divisions number of elements in the radial direction
 * @param angular_divisions number of elements in the theta direction
 * @param vertical_divisions number of elements in the z-direction
 * @param inner_radius the radius of the inner wall of the cylinder
 * @param outer_radius the radius of the outer wall of the cylinder
 * @param height the height of the top surface of the cylinder
 *
 * @note the cylinder's axis of symmetry is along the z-direction
 */
mfem::Mesh build_hollow_quarter_cylinder(std::size_t radial_divisions, std::size_t angular_divisions,
                                         std::size_t vertical_divisions, double inner_radius, double outer_radius,
                                         double height);

/**
 * @brief Constructs a 2D MFEM mesh of a ring
 *
 * @param[in] radial_refinement the number of times to apply uniform mesh refinement to the cross section
 * @param[in] inner_radius inner radius the radius of the cylindrical shell
 * @param[in] outer_radius ouer radius the radius of the cylindrical shell
 * @param[in] total_angle the angle in radians over which to generate the portion of an extruded cylinder
 * @param[in] sectors the number of starting sectors in the hollow cylinder
 *
 * @return A unique_ptr containing the constructed mesh
 */
mfem::Mesh buildRingMesh(int radial_refinement, double inner_radius, double outer_radius, double total_angle = M_PI,
                         int sectors = 8);

/**
 * @brief Mesh related input options
 *
 */
namespace mesh {

/**
 * @brief Input options for meshes read from files
 *
 */
struct FileInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet container on which the input schema will be defined
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief The relative path for the mesh file
   */
  std::string relative_mesh_file_name;

  /**
   * @brief The absolute path for the mesh file, intended to be populated by the user directly
   */
  mutable std::string absolute_mesh_file_name{};
};

/**
 * @brief Input options for generated meshes
 *
 */
struct BoxInputOptions {
  /**
   * @brief Input file parameters for mesh generation
   *
   * @param[in] container Inlet container on which the input schema will be defined
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief The number of elements in each direction
   *
   */
  std::vector<int> elements;

  /**
   * @brief The physical size in each direction
   *
   */
  std::vector<double> overall_size;
};

/**
 * @brief Input options for generated meshes
 *
 */
struct NBallInputOptions {
  /**
   * @brief The approximate total number of desired elements
   */
  int approx_elements;

  /**
   * @brief The space dimension of the n-ball
   */
  int dimension;
};

/**
 * @brief Finalizes a serial mesh into a refined parallel mesh
 *
 * @param[in] serial_mesh The "base" serial mesh
 * @param[in] refine_serial The number of serial refinements
 * @param[in] refine_parallel The number of parallel refinements
 * @param[in] comm The MPI communicator
 *
 * @return A unique_ptr containing the constructed mesh
 *
 * @note It is sometimes required to refine serially first if your number of processors
 * is less than the original number of mesh elements
 */
std::unique_ptr<mfem::ParMesh> refineAndDistribute(mfem::Mesh&& serial_mesh, const int refine_serial = 0,
                                                   const int refine_parallel = 0, const MPI_Comm comm = MPI_COMM_WORLD);

}  // namespace mesh

}  // namespace smith
