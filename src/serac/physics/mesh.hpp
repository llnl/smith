// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file common.hpp
 *
 * @brief A file defining some enums and structs that are used by the different physics modules
 */
#pragma once

#include "mfem.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac {

// Forward declare
struct Domain;

/**
 * @brief Helper class for constructing a mesh consistent with serac
 */
class Mesh {
 public:
  /// @brief Construct from existing serial mfem mesh
  /// @param mesh serial mfem mesh
  /// @param meshtag string tag name for mesh
  /// @param serial_refine number of serial refinements
  /// @param parallel_refine number of parallel refinements
  Mesh(mfem::Mesh&& mesh, const std::string& meshtag, int serial_refine = 0, int parallel_refine = 0);

  /// @brief Construct from existing parallel mfem mesh
  /// @param mesh parallel mfem mesh
  /// @param meshtag string tag name for mesh
  Mesh(mfem::ParMesh& mesh, const std::string& meshtag);

  /// @brief Construct from path to mesh (typically .g or .mesh)
  /// @param meshfile path and name of mesh to read in
  /// @param meshtag string tag name for mesh
  /// @param serial_refine number of serial refinements
  /// @param parallel_refine number of parallel refinements
  Mesh(const std::string& meshfile, const std::string& meshtag, int serial_refine = 0, int parallel_refine = 0);

  /// @brief Returns string tag for mesh
  const std::string& tag() const { return mesh_tag; }

  /// @brief Returns parallel mfem mesh, ignores const for mfem interfaces
  mfem::ParMesh& mfemParMesh() const { return *mfem_mesh; }

  /// @brief Returns parallel communicator
  MPI_Comm getComm() const;

  /// @brief Returns domain corresponding to the entire mesh
  serac::Domain& entireDomain() const;

  /// @brief Returns registered domain with specified name
  serac::Domain& domain(const std::string& domain_name) const;

  /// @brief create domain of 3D boundary elements with specified name
  /// The second argument is a function taking a std::vector<vec3> corresponding
  /// to the nodal coordinates of the boundary element as well as an integer corresponding to the attribute id
  serac::Domain& addDomainOfBoundaryElements(const std::string& domain_name,
                                             std::function<bool(std::vector<vec3>, int)> func);

  /// @brief create domain of 2D boundary elements with specified name
  /// The second argument is a function taking a std::vector<vec2> corresponding
  /// to the nodal coordinates of the boundary element as well as an integer corresponding to the attribute id
  serac::Domain& addDomainOfBoundaryElements(const std::string& domain_name,
                                             std::function<bool(std::vector<vec2>, int)> func);

 private:
  /// @brief
  void createDomains();

  /// @brief
  std::string mesh_tag;

  /// @brief
  mfem::ParMesh* mfem_mesh;

  /// @brief
  mutable std::map<std::string, serac::Domain> domains_;
};

}  // namespace serac
