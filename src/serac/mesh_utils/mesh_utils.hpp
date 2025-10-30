// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file mesh_utils.hpp
 *
 * @brief This file contains helper functions for importing and managing
 *        various mesh objects.
 */

#pragma once

#include <memory>
#include <variant>

#include "serac/mesh_utils/mesh_utils_base.hpp"

/**
 * The Serac namespace
 */
namespace serac {
namespace mesh {

/**
 * @brief Container for the mesh input options
 */
struct InputOptions {
  /**
   * @brief Input file parameters for mesh generation
   *
   * @param[in] container Inlet container on which the input schema will be defined
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief The mesh input options (either file or generated)
   *
   */
  std::variant<FileInputOptions, BoxInputOptions, NBallInputOptions> extra_options;

  /**
   * @brief The number of serial refinement levels
   *
   */
  int ser_ref_levels;

  /**
   * @brief The number of parallel refinement levels
   *
   */
  int par_ref_levels;
};

/**
 * @brief Constructs an MFEM parallel mesh from a set of input options
 *
 * @param[in] options The options used to construct the mesh
 * @param[in] comm The MPI communicator to use with the parallel mesh
 *
 * @return A unique_ptr containing the constructed mesh
 */
std::unique_ptr<mfem::ParMesh> buildParallelMesh(const InputOptions& options, const MPI_Comm comm = MPI_COMM_WORLD);

}  // namespace mesh
}  // namespace serac

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::mesh::InputOptions> {
  /// @brief Returns created object from Inlet container
  serac::mesh::InputOptions operator()(const axom::inlet::Container& base);
};
