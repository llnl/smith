// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrature_data.hpp
 *
 * @brief This file contains the declaration of the structures that manage quadrature point data
 */

#pragma once

#include "mfem.hpp"

#include "axom/core.hpp"

#include "serac/serac_config.hpp"

#include "serac/infrastructure/accelerator.hpp"

namespace serac {

/**
 * @brief these classes are a little confusing. These two
 * special types represent the similar (but different) cases of:
 *
 * Nothing: for qfunctions that have no notion of quadrature data (e.g. body forces).
 *          QuadratureData<Nothing> will store no data, and `Nothing` will never appear
 *          as an argument to a q-function (it will be omitted)
 *
 * Empty: for qfunctions associated with material models (where quadrature data is part of
 *        the interface) that do not actually need to store internal variables. QuadratureData<Empty>
 *        will also store no data, but it will still appear as an argument to the q-function
 *        (to make the material model interface consistent).
 */
struct Nothing {};

/**
 * @brief see `Nothing` for a complete description of this class and when to use it
 */
struct Empty {};

template <typename T>
struct QuadratureData;

}  // namespace serac

// we define these specializations to make it so that materials
// without state variables can use the same interface, without
// actually storing/accessing any data
namespace axom {

/// @cond
template <>
class Array<serac::Nothing, 2, MemorySpace::Dynamic> {
 public:
  Array() {}
  Array(uint32_t, uint32_t) {}
};

template <>
class ArrayView<serac::Nothing, 2, MemorySpace::Dynamic> {
 public:
  ArrayView(Array<serac::Nothing, 2, MemorySpace::Dynamic> /* unused */) {}

  /// dummy accessor to satisfy interface requirements
  SERAC_HOST_DEVICE serac::Nothing& operator()(const size_t, const size_t) { return data; }

  /// dummy accessor to satisfy interface requirements
  SERAC_HOST_DEVICE const serac::Nothing& operator()(const size_t, const size_t) const { return data; }

  serac::Nothing data;
};

template <>
class Array<serac::Empty, 2, MemorySpace::Dynamic> {
 public:
  Array() {}
  Array(uint32_t, uint32_t) {}
};

template <>
class ArrayView<serac::Empty, 2, MemorySpace::Dynamic> {
 public:
  ArrayView(Array<serac::Empty, 2, MemorySpace::Dynamic> /* unused */) {}

  /// dummy accessor to satisfy interface requirements
  SERAC_HOST_DEVICE serac::Empty& operator()(const size_t, const size_t) { return data; }

  /// dummy accessor to satisfy interface requirements
  SERAC_HOST_DEVICE const serac::Empty& operator()(const size_t, const size_t) const { return data; }

  serac::Empty data;
};
/// @endcond

}  // namespace axom

namespace serac {

/**
 * @brief A class for storing and access user-defined types at quadrature points
 *
 * @tparam the data type to be stored at each quadrature point
 *
 * @note users are not intended to create these objects directly, instead
 *       they should use the PhysicsModule::createQuadratureDataBuffer()
 */
template <typename T>
struct QuadratureData {
  /// @brief a list of integers, one associated with each type of mfem::Geometry
  using geom_array_t = std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES>;

  /**
   * @brief Initialize a new quadrature data buffer, optionally with some initial value
   *
   * @param elements the number of elements of each geometry
   * @param qpts_per_element how many quadrature points are present in each kind of element
   * @param value (optional) value used to initialize the buffer
   */
  QuadratureData(geom_array_t elements, geom_array_t qpts_per_element, T value = T{})
  {
    constexpr std::array geometries = {mfem::Geometry::SEGMENT, mfem::Geometry::TRIANGLE, mfem::Geometry::SQUARE,
                                       mfem::Geometry::TETRAHEDRON, mfem::Geometry::CUBE};

    for (auto geom : geometries) {
      if (elements[uint32_t(geom)] > 0) {
        data[geom] = axom::Array<T, 2>(elements[uint32_t(geom)], qpts_per_element[uint32_t(geom)]);
        data[geom].fill(value);
      }
    }
  }

  /**
   * @brief return the 2D array of quadrature point values for elements of the specified geometry
   * @param geom which element geometry's data to return
   */
  axom::ArrayView<T, 2> operator[](mfem::Geometry::Type geom) { return axom::ArrayView<T, 2>(data.at(geom)); }

  /// @brief a 3D array indexed by (which geometry, which element, which quadrature point)
  std::map<mfem::Geometry::Type, axom::Array<T, 2> > data;
};

/// @cond
template <>
struct QuadratureData<Nothing> {
  using geom_array_t = std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES>;

  QuadratureData() {}

  axom::ArrayView<Nothing, 2> operator[](mfem::Geometry::Type) { return axom::ArrayView<Nothing, 2>(data); }

  axom::Array<Nothing, 2, axom::MemorySpace::Dynamic> data;
};

template <>
struct QuadratureData<Empty> {
  using geom_array_t = std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES>;

  QuadratureData() {}

  axom::ArrayView<Empty, 2> operator[](mfem::Geometry::Type) { return axom::ArrayView<Empty, 2>(data); }

  axom::Array<Empty, 2, axom::MemorySpace::Dynamic> data;
};
/// @endcond

/// these values exist to serve as default arguments for materials without material state
extern std::shared_ptr<QuadratureData<Nothing> > NoQData;
extern std::shared_ptr<QuadratureData<Empty> > EmptyQData;

}  // namespace serac
