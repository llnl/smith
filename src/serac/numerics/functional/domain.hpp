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

struct SetOperation {
  /// @cond
  using c_iter = std::vector<int>::const_iterator;
  using b_iter = std::back_insert_iterator<std::vector<int>>;
  using set_op = std::function<b_iter(c_iter, c_iter, c_iter, c_iter, b_iter)>;
  /// @endcond
};

/**
 * @brief a class for representing a geometric region that can be used for integration
 *
 * This region can be an entire mesh or some subset of its elements
 */
class Domain {
 public:


  /// @brief enum describing what kind of elements are included in a Domain
  enum Type
  {
    Elements,
    BoundaryElements
  };

  static constexpr int num_types = 2;  ///< the number of entries in the Type enum

  /// @brief the underyling mesh for this domain
  inline const mfem::Mesh& mesh() const { return mesh_; };

  /// @brief the geometric dimension of the domain
  inline int dim() const { return dim_; };

  /// @brief whether the elements in this domain are on the boundary or not
  inline Type type() const { return type_; };

  /// @brief construct an "empty" domain, to later be populated later with addElement member functions
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
    if (geom == mfem::Geometry::SEGMENT) return edge_ids_;
    if (geom == mfem::Geometry::TRIANGLE) return tri_ids_;
    if (geom == mfem::Geometry::SQUARE) return quad_ids_;
    if (geom == mfem::Geometry::TETRAHEDRON) return tet_ids_;
    if (geom == mfem::Geometry::CUBE) return hex_ids_;

    exit(1);
  }

  /// @brief get mfem degree of freedom list for a given FiniteElementSpace
  mfem::Array<int> dof_list(mfem::FiniteElementSpace* fes) const;

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

 private:

  /// @brief the underyling mesh for this domain
  const mfem::Mesh& mesh_;

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

  friend Domain set_operation(SetOperation::set_op op, const Domain& a, const Domain& b);

  friend Domain operator|(const Domain& a, const Domain& b);
  friend Domain operator&(const Domain& a, const Domain& b);
  friend Domain operator-(const Domain& a, const Domain& b);
};

/// @brief constructs a domain from all the elements in a mesh
Domain EntireDomain(const mfem::Mesh& mesh);

/// @brief constructs a domain from all the boundary elements in a mesh
Domain EntireBoundary(const mfem::Mesh& mesh);

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
