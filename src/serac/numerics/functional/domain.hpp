// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>
#include "mfem.hpp"

#include "serac/numerics/functional/tensor.hpp"

namespace serac {

/**
 * @brief a class for representing a geometric region that can be used for integration
 *
 * This region can be an entire mesh or some subset of its elements
 */
struct Domain {
  /// @brief enum describing what kind of elements are included in a Domain
  enum Type
  {
    Elements,
    BoundaryElements
  };

  static constexpr int num_types = 2;  ///< the number of entries in the Type enum

  /// @brief the underyling mesh for this domain
  const mfem::Mesh& mesh_;

  /// @brief the geometric dimension of the domain
  int dim_;

  /// @brief whether the elements in this domain are on the boundary or not
  Type type_;

  /// note: only lists with appropriate dimension (see dim_) will be populated
  ///       for example, a 2D Domain may have `tri_ids_` and `quad_ids_` non-nonempty,
  ///       but all other lists will be empty
  ///
  /// these lists hold indices into the "E-vector" of the appropriate geometry
  ///
  /// @cond
  std::vector<int> vertex_ids_;
  std::vector<int> edge_ids_;
  std::vector<int> tri_ids_;
  std::vector<int> quad_ids_;
  std::vector<int> tet_ids_;
  std::vector<int> hex_ids_;

  std::vector<int> mfem_edge_ids_;
  std::vector<int> mfem_tri_ids_;
  std::vector<int> mfem_quad_ids_;
  std::vector<int> mfem_tet_ids_;
  std::vector<int> mfem_hex_ids_;
  /// @endcond

  Domain(const mfem::Mesh& m, int d, Type type = Domain::Type::Elements) : mesh_(m), dim_(d), type_(type) {}

  /**
   * @brief create a domain from some subset of the vertices in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which vertices will be
   * included in this domain. The function's argument is the spatial position of the vertex.
   */
  static Domain ofVertices(const mfem::Mesh& mesh, std::function<bool(vec2)> func);

  /// @overload
  static Domain ofVertices(const mfem::Mesh& mesh, std::function<bool(vec3)> func);

  /**
   * @brief create a domain from some subset of the edges in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which edges will be
   * included in this domain. The function's arguments are the list of vertex coordinates and
   * an attribute index (if appropriate).
   */
  static Domain ofEdges(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofEdges(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>)> func);

  /**
   * @brief create a domain from some subset of the faces in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which faces will be
   * included in this domain. The function's arguments are the list of vertex coordinates and
   * an attribute index (if appropriate).
   */
  static Domain ofFaces(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofFaces(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /**
   * @brief create a domain from some subset of the elements (spatial dim == geometry dim) in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which elements will be
   * included in this domain. The function's arguments are the list of vertex coordinates and
   * an attribute index (if appropriate).
   */
  static Domain ofElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /**
   * @brief create a domain from some subset of the boundary elements (spatial dim == geometry dim + 1) in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which boundary elements will be included in this domain
   */
  static Domain ofBoundaryElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofBoundaryElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /// @brief get elements by geometry type
  const std::vector<int>& get(mfem::Geometry::Type geom) const
  {
    if (geom == mfem::Geometry::POINT) return vertex_ids_;
    if (geom == mfem::Geometry::SEGMENT) return edge_ids_;
    if (geom == mfem::Geometry::TRIANGLE) return tri_ids_;
    if (geom == mfem::Geometry::SQUARE) return quad_ids_;
    if (geom == mfem::Geometry::TETRAHEDRON) return tet_ids_;
    if (geom == mfem::Geometry::CUBE) return hex_ids_;

    exit(1);
  }

  /// @brief get mfem degree of freedom list for a given FiniteElementSpace
  mfem::Array<int> dof_list(mfem::FiniteElementSpace* fes) const;
};

/// @brief constructs a domain from all the elements in a mesh
template <int dim>
Domain EntireDomain(const mfem::Mesh& mesh)
{
  auto tautological_predicate = [](std::vector<tensor<double, dim>>, int) { return true; };
  return Domain::ofElements(mesh, tautological_predicate);
}

/// @brief constructs a domain from all the boundary elements in a mesh
template <int dim>
Domain EntireBoundary(const mfem::Mesh& mesh)
{
  auto tautological_predicate = [](std::vector<tensor<double, dim>>, int) { return true; };
  return Domain::ofBoundaryElements(mesh, tautological_predicate);
}

/// @brief create a new domain that is the union of `a` and `b`
Domain operator|(const Domain& a, const Domain& b);

/// @brief create a new domain that is the intersection of `a` and `b`
Domain operator&(const Domain& a, const Domain& b);

/// @brief create a new domain that is the set difference of `a` and `b`
Domain operator-(const Domain& a, const Domain& b);

/// @brief convenience predicate for creating domains by attribute
template <int dim>
inline auto by_attr(int value)
{
  return [value](std::vector<tensor<double, dim> >, int attr) { return attr == value; };
}

}  // namespace serac
