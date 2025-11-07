// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstdlib>
#include <vector>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <set>

#include "mfem.hpp"

#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/element_restriction.hpp"
#include "smith/numerics/functional/typedefs.hpp"

namespace smith {

struct BlockElementRestriction;

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
    BoundaryElements,
    InteriorFaces
  };

  static constexpr int num_types = 3;  ///< the number of entries in the Type enum

  /// @brief the underyling mesh for this domain
  const mesh_t& mesh_;

  /// @brief the geometric dimension of the domain
  int dim_;

  /// @brief whether the elements in this domain are on the boundary or not
  Type type_;

  ///@{
  /// @name ElementIds
  /// Indices of elements contained in the domain.
  /// The first set, (edge_ids_, tri_ids, ...) hold the index of an element in
  /// this Domain in the set of all elements of like geometry in the mesh.
  /// For example, if edge_ids_[0] = 5, then element 0 in this domain is element
  /// 5 in the grouping of all edges in the mesh. In other words, these lists
  /// hold indices into the "E-vector" of the appropriate geometry. These are
  /// used primarily for identifying elements in the domain for participation
  /// in integrals.
  ///
  /// The second set, (mfem_edge_ids_, mfem_tri_ids_, ...), gives the ids of
  /// elements in this domain in the global mfem::Mesh data structure. These
  /// maps are needed to find the dofs that live on a Domain.
  ///
  /// Instances of Domain are meant to be homogeneous: only lists with
  /// appropriate dimension (see dim_) will be populated by the factory
  /// functions. For example, a 2D Domain may have `tri_ids_` and `quad_ids_`
  /// non-empty, but all other lists will be empty.
  ///
  /// @note For every entry in the first group (say, edge_ids_), there should
  /// be a corresponding entry into the second group (mfem_edge_ids_). This
  /// is an intended invariant of the class, but it's not enforced by the data
  /// structures. Prefer to use the factory methods (eg, \ref ofElements(...))
  /// to populate these lists automatically, as they repsect this invariant and
  /// are tested. Otherwise, use the \ref addElements(...) or addElements(...)
  /// methods to add new entities, as this requires you to add both entries and
  /// keep the corresponding lists in sync. You are discouraged from
  /// manipulating these lists directly.
  ///@}

  /// @cond
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

  /**
   * @brief a collection of restriction operators for the different test/trial spaces appearing in
   *        integrals evaluated over this Domain. These are stored on the Domain itself to avoid duplicating
   *        these restriction operators in each Integral over a given Domain.
   */
  std::map<FunctionSpace, BlockElementRestriction> restriction_operators;

  /// @brief Ids of interior faces that lie on the boundary shared by two processors
  std::vector<int> shared_interior_face_ids_;

  /**
   * @brief empty Domain constructor, with connectivity info to be populated later
   */
  Domain(const mesh_t& m, int d, Type type) : mesh_(m), dim_(d), type_(type) {}

  /**
   * @brief create a domain from some subset of the vertices in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which vertices will be
   * included in this domain. The function's argument is the spatial position of the vertex.
   */
  static Domain ofVertices(const mesh_t& mesh, std::function<bool(vec2)> func);

  /// @overload
  static Domain ofVertices(const mesh_t& mesh, std::function<bool(vec3)> func);

  /**
   * @brief create a domain from some subset of the edges in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which edges will be
   * included in this domain. The function's arguments are the list of vertex coordinates and
   * an attribute index (if appropriate).
   */
  static Domain ofEdges(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofEdges(const mesh_t& mesh, std::function<bool(std::vector<vec3>)> func);

  /**
   * @brief create a domain from some subset of the faces in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which faces will be
   * included in this domain. The function's arguments are the list of vertex coordinates and
   * an attribute index (if appropriate).
   */
  static Domain ofFaces(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofFaces(const mesh_t& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /**
   * @brief create a domain from some subset of the elements (spatial dim == geometry dim) in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which elements will be
   * included in this domain. The function's arguments are the list of vertex coordinates and
   * an attribute index (if appropriate).
   */
  static Domain ofElements(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofElements(const mesh_t& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /**
   * @brief create a domain from some subset of the boundary elements (spatial dim == geometry dim + 1) in an mfem::Mesh
   * @param mesh the entire mesh
   * @param func predicate function for determining which boundary elements will be included in this domain
   */
  static Domain ofBoundaryElements(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func);

  /// @overload
  static Domain ofBoundaryElements(const mesh_t& mesh, std::function<bool(std::vector<vec3>, int)> func);

  /// @brief get elements by geometry type
  const std::vector<int>& get(mfem::Geometry::Type geom) const
  {
    if (geom == mfem::Geometry::SEGMENT) return edge_ids_;
    if (geom == mfem::Geometry::TRIANGLE) return tri_ids_;
    if (geom == mfem::Geometry::SQUARE) return quad_ids_;
    if (geom == mfem::Geometry::TETRAHEDRON) return tet_ids_;
    if (geom == mfem::Geometry::CUBE) return hex_ids_;

    exit(1);
  }

  /// @brief get elements by geometry type
  const std::vector<int>& get_mfem_ids(mfem::Geometry::Type geom) const
  {
    if (geom == mfem::Geometry::SEGMENT) return mfem_edge_ids_;
    if (geom == mfem::Geometry::TRIANGLE) return mfem_tri_ids_;
    if (geom == mfem::Geometry::SQUARE) return mfem_quad_ids_;
    if (geom == mfem::Geometry::TETRAHEDRON) return mfem_tet_ids_;
    if (geom == mfem::Geometry::CUBE) return mfem_hex_ids_;

    exit(1);
  }

  /**
   * @brief returns how many elements of any type belong to this domain
   */
  int total_elements() const
  {
    return int(edge_ids_.size() + tri_ids_.size() + quad_ids_.size() + tet_ids_.size() + hex_ids_.size());
  }

  /**
   * @brief returns an array of the prefix sum of element counts belonging to this domain.
   *        Primarily intended to be used in mfem::BlockVector::Update(double * data, mfem::Array<int> bOffsets);
   */
  mfem::Array<int> bOffsets() const
  {
    mfem::Array<int> offsets(mfem::Geometry::NUM_GEOMETRIES + 1);

    int total = 0;
    offsets[mfem::Geometry::POINT] = total;
    total += 0;  // vertices;
    offsets[mfem::Geometry::SEGMENT] = total;
    total += int(edge_ids_.size());
    offsets[mfem::Geometry::TRIANGLE] = total;
    total += int(tri_ids_.size());
    offsets[mfem::Geometry::SQUARE] = total;
    total += int(quad_ids_.size());
    offsets[mfem::Geometry::TETRAHEDRON] = total;
    total += int(tet_ids_.size());
    offsets[mfem::Geometry::CUBE] = total;
    total += int(hex_ids_.size());
    offsets[mfem::Geometry::PRISM] = total;
    offsets[mfem::Geometry::PYRAMID] = total;
    offsets[mfem::Geometry::NUM_GEOMETRIES] = total;

    return offsets;
  }

  /// @brief get mfem degree of freedom list for a given FiniteElementSpace
  mfem::Array<int> dof_list(const fes_t* fes) const;

  /**
   * @brief create a restriction operator over this domain, using its FunctionSpace as a key
   * @note if a restriction for the given key (i.e. FunctionSpace) already exists, this function does nothing
   */
  void insert_restriction(const fes_t* fes, FunctionSpace space);

  /// @brief getter for accessing a restriction operator by its function space
  const BlockElementRestriction& get_restriction(FunctionSpace space);

  /// @brief Add an element to the domain
  ///
  /// This is meant for internal use on the class. Prefer to use the factory
  /// methods (ofElements, ofBoundaryElements, etc) to create domains and
  /// thereby populate the element lists.
  void addElement(int geom_id, int elem_id, mfem::Geometry::Type element_geometry);

  /// @brief Add a batch of elements to the domain
  ///
  /// This is meant for internal use on the class. Prefer to use the factory
  /// methods (ofElements, ofBoundaryElements, etc) to create domains and
  /// thereby populate the element lists.
  void addElements(const std::vector<int>& geom_id, const std::vector<int>& elem_id,
                   mfem::Geometry::Type element_geometry);

  /**
   * @brief Find the list of interior faces shared by two processors to make sure
   *        these faces are only integrated once.
   */
  void insert_shared_interior_face_list();
};

/// @brief constructs a domain from all the elements in a mesh
Domain EntireDomain(const mesh_t& mesh);

/// @brief constructs a domain from all the boundary elements in a mesh
Domain EntireBoundary(const mesh_t& mesh);

/// @brief constructs a domain from all the interior face elements in a mesh
Domain InteriorFaces(const mesh_t& mesh);

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
  return [value](std::vector<tensor<double, dim>>, int attr) { return value == attr; };
}

/// @brief convenience predicate for creating domains by a set of attributes
template <int dim>
inline auto by_attr(std::set<int> values)
{
  return [values](std::vector<tensor<double, dim>>, int attr) { return values.find(attr) != values.end(); };
}

/**
 * @brief count the number of elements of each geometry in a domain
 * @param domain the domain to count
 */
inline std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> geometry_counts(const Domain& domain)
{
  std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> counts{};

  constexpr std::array<mfem::Geometry::Type, 5> geometries = {mfem::Geometry::SEGMENT, mfem::Geometry::TRIANGLE,
                                                              mfem::Geometry::SQUARE, mfem::Geometry::TETRAHEDRON,
                                                              mfem::Geometry::CUBE};
  for (auto geom : geometries) {
    counts[uint32_t(geom)] = uint32_t(domain.get(geom).size());
  }
  return counts;
}

/**
 * @brief convenience function for computing the arithmetic mean of some list of vectors
 */
template <int dim>
inline tensor<double, dim> average(std::vector<tensor<double, dim>>& positions)
{
  tensor<double, dim> total{};
  for (auto x : positions) {
    total += x;
  }
  return total / double(positions.size());
}

}  // namespace smith
