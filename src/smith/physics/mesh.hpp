// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file mesh.hpp
 *
 * @brief Smith mesh class which assists in constructing the appropriate parallel mfem meshes
 * and registering and accessing Domains for use in smith::Functional operations
 */
#pragma once

#include <memory>
#include <string>
#include <functional>
#include <map>
#include <vector>

#include "mpi.h"

#include "mfem.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/domain.hpp"

namespace smith {

// Forward declare
struct Domain;
class FiniteElementState;
class FiniteElementDual;

/**
 * @brief Helper class for constructing a mesh consistent with Smith
 */
class Mesh {
 public:
  /// @brief Construct from existing serial mfem mesh
  /// @param mesh serial mfem mesh
  /// @param meshtag string tag name for mesh
  /// @param serial_refine number of serial refinements
  /// @param parallel_refine number of parallel refinements
  /// @param comm the communicator that the parallel form of @p mesh should be made with
  Mesh(mfem::Mesh&& mesh, const std::string& meshtag, int serial_refine = 0, int parallel_refine = 0,
       MPI_Comm comm = MPI_COMM_WORLD);

  /// @brief Construct from existing parallel mfem mesh
  /// @param mesh parallel mfem mesh
  /// @param meshtag string tag name for mesh
  Mesh(mfem::ParMesh&& mesh, const std::string& meshtag);

  /// @brief Construct from path to mesh (typically .g or .mesh)
  /// @param meshfile path and name of mesh to read in
  /// @param meshtag string tag name for mesh
  /// @param serial_refine number of serial refinements
  /// @param parallel_refine number of parallel refinements
  /// @param comm the communicator that the parallel form of @p mesh should be made with
  Mesh(const std::string& meshfile, const std::string& meshtag, int serial_refine = 0, int parallel_refine = 0,
       MPI_Comm comm = MPI_COMM_WORLD);

  /// @brief Returns string tag for mesh
  const std::string& tag() const { return mesh_tag_; }

  /// @brief Returns const parallel mfem mesh
  const mfem::ParMesh& mfemParMesh() const { return *mfem_mesh_; }

  /// @brief Returns parallel mfem mesh
  mfem::ParMesh& mfemParMesh() { return *mfem_mesh_; }

  /// @brief Returns parallel communicator
  MPI_Comm getComm() const;

  /// @brief  Returns string, name used to access the entire domain body
  static std::string entireBodyName() { return "entire_body"; }

  /// @brief Returns domain corresponding to the entire mesh
  smith::Domain& entireBody() const;

  /// @brief  Returns string, name used to access the entire boundary
  static std::string entireBoundaryName() { return "entire_boundary"; }

  /// @brief Returns domain boundary corresponding to the entire mesh
  smith::Domain& entireBoundary() const;

  /// @brief  Returns string, name used to access the internal boundary elements
  static std::string internalBoundaryName() { return "internal_boundary"; }

  /// @brief Returns domain boundary corresponding to the internal boundary elements
  smith::Domain& internalBoundary() const;

  /// @brief Insert a domain onto mesh
  void insertDomain(const std::string& domain_name, const Domain& domain);

  /// @brief Returns registered domain with specified name
  smith::Domain& domain(const std::string& domain_name) const;

  /// @brief create domain of 3D boundary elements with specified name
  /// The second argument is a function taking a std::vector<vec3> corresponding
  /// to the nodal coordinates of the boundary element as well as an integer corresponding to the attribute id
  smith::Domain& addDomainOfBoundaryElements(const std::string& domain_name,
                                             std::function<bool(std::vector<vec3>, int)> func);

  /// @brief create domain of 2D boundary elements with specified name
  /// The second argument is a function taking a std::vector<vec2> corresponding
  /// to the nodal coordinates of the boundary element as well as an integer corresponding to the attribute id
  smith::Domain& addDomainOfBoundaryElements(const std::string& domain_name,
                                             std::function<bool(std::vector<vec2>, int)> func);

  /// @brief create domain of 3D elements with specified name
  /// The second argument is a function taking a std::vector<vec3> corresponding
  /// to the nodal coordinates of the element as well as an integer corresponding to the attribute id
  smith::Domain& addDomainOfBodyElements(const std::string& domain_name,
                                         std::function<bool(std::vector<vec3>, int)> func);

  /// @brief create domain of 2D boundary elements with specified name
  /// The second argument is a function taking a std::vector<vec2> corresponding
  /// to the nodal coordinates of the element as well as an integer corresponding to the attribute id
  smith::Domain& addDomainOfBodyElements(const std::string& domain_name,
                                         std::function<bool(std::vector<vec2>, int)> func);

  /// @brief get space associated with shape displacement
  const mfem::ParFiniteElementSpace& shapeDisplacementSpace();

  /// @brief create new shape displacement
  smith::FiniteElementState newShapeDisplacement();

  /// @brief create new shape displacement sensitivity
  smith::FiniteElementDual newShapeDisplacementDual();

 private:
  /// @brief Helper function used to throw exception if the size of the mesh on the local rank is 0
  void errorIfRankHasNoElements() const;

  /// @brief Sets up some initial domains: entire domain, entire boundary, and interior faces. Eventually we can read
  /// off names/blocks/attributes from the mesh and create default domains.
  void createDomains();

  /// @brief Helper function used to check if a domain name already exists
  void errorIfDomainExists(const std::string& domain_name) const;

  /// @brief string identifying mesh in the state manager
  std::string mesh_tag_;

  /// @brief parallel mfem mesh
  mfem::ParMesh* mfem_mesh_;

  /// @brief map from registered domain name to the domain instance
  mutable std::map<std::string, smith::Domain> domains_;
};

}  // namespace smith
