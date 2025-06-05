// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file mesh.hpp
 *
 * @brief Serac mesh class which assists in constructing the appropriate parallel mfem meshes
 * and registering and accessing Domains for use in serac::Functional operations
 */
#pragma once

#include <memory>
#include <string>
#include "mfem.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac {

// Forward declare
struct Domain;
class FiniteElementState;
class FiniteElementDual;

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
  serac::Domain& entireBody() const;

  /// @brief  Returns string, name used to access the entire boundary
  static std::string entireBoundaryName() { return "entire_boundary"; }

  /// @brief Returns domain boundary corresponding to the entire mesh
  serac::Domain& entireBoundary() const;

  /// @brief  Returns string, name used to access the internal boundary elements
  static std::string internalBoundaryName() { return "internal_boundary"; }

  /// @brief Returns domain boundary corresponding to the internal boundary elements
  serac::Domain& internalBoundary() const;

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

  /// @brief create domain of 3D elements with specified name
  /// The second argument is a function taking a std::vector<vec3> corresponding
  /// to the nodal coordinates of the element as well as an integer corresponding to the attribute id
  serac::Domain& addDomainOfBodyElements(const std::string& domain_name,
                                         std::function<bool(std::vector<vec3>, int)> func);

  /// @brief create domain of 2D boundary elements with specified name
  /// The second argument is a function taking a std::vector<vec2> corresponding
  /// to the nodal coordinates of the element as well as an integer corresponding to the attribute id
  serac::Domain& addDomainOfBodyElements(const std::string& domain_name,
                                         std::function<bool(std::vector<vec2>, int)> func);

  /// @brief get non-const shape displacement
  serac::FiniteElementState& shape_displacement();

  /// @brief get const shape displacement
  const serac::FiniteElementState& shape_displacement() const;

  /// @brief get non-const shape displacement dual
  serac::FiniteElementDual& shape_displacement_dual();

  /// @brief get const shape displacement dual
  const serac::FiniteElementDual& shape_displacement_dual() const;

 private:
  /// @brief Sets up some initial domains, for now just the 'entire_domain', but eventually we can read of
  /// names/blocks/attributes from the mesh and create default domains.
  void createDomains();

  /// @brief string identifying mesh in the state manager
  std::string mesh_tag_;

  /// @brief parallel mfem mesh
  mfem::ParMesh* mfem_mesh_;

  /// @brief map from registered domain name to the domain instance
  mutable std::map<std::string, serac::Domain> domains_;

  /// @brief shape_displacement
  std::shared_ptr<serac::FiniteElementState> shape_displacement_;

  /// @brief shape_displacement dual
  std::shared_ptr<serac::FiniteElementDual> shape_displacement_dual_;
};

}  // namespace serac
