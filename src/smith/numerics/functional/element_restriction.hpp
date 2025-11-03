// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstdint>
#include <vector>
#include <map>

#include "mfem.hpp"
#include "axom/core.hpp"
#include "geometry.hpp"
#include "domain.hpp"

#include "smith/infrastructure/memory.hpp"
#include "smith/numerics/functional/typedefs.hpp"

inline bool isH1(const mfem::FiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::CONTINUOUS);
}

inline bool isHcurl(const mfem::FiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::TANGENTIAL);
}

inline bool isDG(const mfem::FiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::DISCONTINUOUS);
}

enum class FaceType
{
  BOUNDARY,
  INTERIOR
};

/// a struct of metadata (index, sign, orientation) associated with a degree of freedom
struct DoF {
  // sam: I wanted to use a bitfield for this type, but a 10+ year-old GCC bug
  // makes it practically impossible to assign to bitfields without warnings/errors
  // with -Wconversion enabled, see: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=39170
  // So, we resort to masks and bitshifting instead.

  static constexpr uint64_t sign_mask = 0x8000'0000'0000'0000;         ///< bits for sign field
  static constexpr uint64_t orientation_mask = 0x7000'0000'0000'0000;  ///< bits for orientation field
  static constexpr uint64_t index_mask = 0x0000'FFFF'FFFF'FFFF'FFFF;   ///< bits for the index field

  static constexpr uint64_t sign_shift = 63;         ///< number of trailing zeros in `sign_mask`
  static constexpr uint64_t orientation_shift = 60;  ///< number of trailing zeros in `orientation_mask`
  static constexpr uint64_t index_shift = 0;         ///< number of trailing zeros in `index_mask`

  /**
   * @brief a 64-bit word encoding the following metadata (laid out from MSB to LSB);
   *
   * - 1 sign bit
   * - 3 orientation bits
   * - 12 currently unused bits that may be used in the future
   * - 48 index bits
   *
   * all values are immutable unsigned integers
   */
  uint64_t bits;

  /// default ctor
  DoF() : bits{} {}

  /// copy ctor
  DoF(const DoF& other) : bits{other.bits} {}

  /// create a `DoF` from the given index, sign and orientation values
  DoF(uint64_t index, uint64_t sign = 0, uint64_t orientation = 0)
      : bits((sign & 0x1ULL << sign_shift) + ((orientation & 0x7ULL) << orientation_shift) + index)
  {
  }

  /// copy assignment operator
  void operator=(const DoF& other) { bits = other.bits; }

  /// get the sign field of this `DoF`
  int sign() const { return (bits & sign_mask) ? -1 : 1; }

  /// get the orientation field of this `DoF`
  uint64_t orientation() const { return ((bits & orientation_mask) >> orientation_shift); }

  /// get the index field of this `DoF`
  uint64_t index() const { return (bits & index_mask); }
};

/// a small struct used to enable range-based for loops in `Array2D`
template <typename T>
struct Range {
  T* begin() { return ptr[0]; }  ///< the beginning of the range
  T* end() { return ptr[1]; }    ///< the end of the range
  T* ptr[2];                     ///< the beginning and end of the range
};

/**
 * @brief a 2D array
 *
 * @tparam T
 */
template <typename T>
struct Array2D {
  Array2D() : values{}, dim{} {};

  /// create an uninitialized m-by-n two-dimensional array
  Array2D(uint64_t m, uint64_t n) : values(m * n, 0), dim{m, n} {};

  /// create an m-by-n two-dimensional array initialized with the values in `data` (assuming row-major)
  Array2D(std::vector<T>&& data, uint64_t m, uint64_t n) : values(data), dim{m, n} {};

  /// access a mutable "row" of this Array2D
  Range<T> operator()(uint64_t i) { return Range<T>{&values[i * dim[1]], &values[(i + 1) * dim[1]]}; }

  /// access an immutable "row" of this Array2D
  Range<const T> operator()(uint64_t i) const { return Range<const T>{&values[i * dim[1]], &values[(i + 1) * dim[1]]}; }

  /// access a mutable element of this Array2D
  T& operator()(uint64_t i, uint64_t j) { return values[i * dim[1] + j]; }

  /// access an immutable element of this Array2D
  const T& operator()(uint64_t i, uint64_t j) const { return values[i * dim[1] + j]; }

  /// @overload
  Array2D(int m, int n) : values(uint64_t(m) * uint64_t(n), 0), dim{uint64_t(m), uint64_t(n)} {}

  /// @overload
  Array2D(std::vector<T>&& data, int m, int n) : values(data), dim{uint64_t(m), uint64_t(n)} {}

  /// @overload
  Range<T> operator()(int i) { return Range<T>{&values[uint64_t(i) * dim[1]], &values[uint64_t(i + 1) * dim[1]]}; }

  /// @overload
  Range<const T> operator()(int i) const
  {
    return Range<const T>{&values[uint64_t(i) * dim[1]], &values[uint64_t(i + 1) * dim[1]]};
  }

  /// @overload
  T& operator()(int i, int j) { return values[uint64_t(i) * dim[1] + uint64_t(j)]; }

  /// @overload
  const T& operator()(int i, int j) const { return values[uint64_t(i) * dim[1] + uint64_t(j)]; }

  std::vector<T> values;  ///< the values of each element in the array
  uint64_t dim[2];        ///< the number of rows and columns in the array, respectively
};

namespace smith {

struct Domain;

/// a more complete version of mfem::ElementRestriction that works with {H1, Hcurl, L2} spaces (including on the
/// boundary)
struct ElementRestriction {
  /// default ctor leaves this object uninitialized
  ElementRestriction() {}

  /// ctor from a list of elements (e.g. from a smith::Domain)
  ElementRestriction(const fes_t* fes, mfem::Geometry::Type elem_geom, const std::vector<int>& domain_elements);

  /// the size of the "E-vector" associated with this restriction operator
  uint64_t ESize() const;

  /// the size of the "L-vector" associated with this restriction operator
  uint64_t LSize() const;

  /**
   * @brief Get a list of DoFs for element `i`
   *
   * @param i the index of the element
   * @param dofs (output) the DoFs associated with element `i`
   */
  void GetElementVDofs(int i, std::vector<DoF>& dofs) const;

  /// get the dof information for a given node / component
  DoF GetVDof(DoF node, uint64_t component) const;

  /// "L->E" in mfem parlance, each element gathers the values that belong to it, and stores them in the "E-vector"
  void Gather(const mfem::Vector& L_vector, mfem::Vector& E_vector) const;

  /// "E->L" in mfem parlance, each element scatter-adds its local vector into the appropriate place in the "L-vector"
  void ScatterAdd(const mfem::Vector& E_vector, mfem::Vector& L_vector) const;

  /// the size of the "E-vector"
  uint64_t esize;

  /// the size of the "L-vector"
  uint64_t lsize;

  /// the number of components at each node
  uint64_t components;

  /// the total number of nodes in the mesh
  uint64_t num_nodes;

  /// the number of elements of the given geometry
  uint64_t num_elements;

  /// the number of nodes in each element
  uint64_t nodes_per_elem;

  /// a 2D array (num_elements-by-nodes_per_elem) holding the dof info extracted from the finite element space
  axom::Array<DoF, 2, smith::detail::host_memory_space> dof_info;

  /// whether the underlying dofs are arranged "byNodes" or "byVDim"
  mfem::Ordering::Type ordering;
};

/**
 * @brief a generalization of mfem::ElementRestriction that works with multiple kinds of element geometries.
 * Instead of doing the "E->L" (gather) and "L->E" (scatter) operations for only one element geometry, this
 * class does them with block "E-vectors", where each element geometry is a separate block.
 */
struct BlockElementRestriction {
  /// default ctor leaves this object uninitialized
  BlockElementRestriction() {}

  /// create a BlockElementRestriction for the elements in a given domain
  BlockElementRestriction(const fes_t* fes, const Domain& domain);

  /// the size of the "E-vector" associated with this restriction operator
  uint64_t ESize() const;

  /// the size of the "L-vector" associated with this restriction operator
  uint64_t LSize() const;

  /// block offsets used when constructing mfem::HypreParVectors
  mfem::Array<int> bOffsets() const;

  /// "L->E" in mfem parlance, each element gathers the values that belong to it, and stores them in the "E-vector"
  void Gather(const mfem::Vector& L_vector, mfem::BlockVector& E_block_vector) const;

  /// "E->L" in mfem parlance, each element scatter-adds its local vector into the appropriate place in the "L-vector"
  void ScatterAdd(const mfem::BlockVector& E_block_vector, mfem::Vector& L_vector) const;

  /// the individual ElementRestriction operators for each element geometry
  std::map<mfem::Geometry::Type, ElementRestriction> restrictions;
};

}  // namespace smith

/**
 * @brief Get the list of dofs for each element (of the specified geometry) from the mfem::FiniteElementSpace
 *
 * @param fes the finite element space containing the dof information
 * @param geom the kind of element geometry
 */
axom::Array<DoF, 2, smith::detail::host_memory_space> GetElementDofs(const smith::fes_t* fes,
                                                                     mfem::Geometry::Type geom);

/**
 * @brief Get the list of dofs for each face element (of the specified geometry) from the fes_t
 *
 * @param fes the finite element space containing the dof information
 * @param geom the kind of element geometry
 * @param type whether the face is of interior or boundary type
 */
axom::Array<DoF, 2, smith::detail::host_memory_space> GetFaceDofs(const smith::fes_t* fes,
                                                                  mfem::Geometry::Type face_geom, FaceType type);

/// @overload
axom::Array<DoF, 2, smith::detail::host_memory_space> GetFaceDofs(const smith::fes_t* fes,
                                                                  mfem::Geometry::Type face_geom,
                                                                  const std::vector<int>& mfem_face_ids);
