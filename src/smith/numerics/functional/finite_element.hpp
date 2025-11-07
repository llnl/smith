// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element.hpp
 *
 * @brief This file contains helper traits and enumerations for classifying
 * finite elements
 */
#pragma once

#include "tuple.hpp"
#include "tensor.hpp"
#include "geometry.hpp"
#include "polynomials.hpp"

namespace smith {

/**
 * @brief a convenience class for generating information about tensor product
 * integration rules from the underlying 1D rule.
 *
 * @tparam q how many quadrature points per dimension
 */
template <int q>
struct TensorProductQuadratureRule {
  tensor<double, q> weights1D;  ///< the weights of the underlying 1D quadrature rule
  tensor<double, q> points1D;   ///< the abscissae of the underlying 1D quadrature rule

  /// @brief return the quadrature weight for a quadrilateral
  SMITH_HOST_DEVICE double weight(int ix, int iy) const { return weights1D[ix] * weights1D[iy]; }

  /// @brief return the quadrature weight for a hexahedron
  SMITH_HOST_DEVICE double weight(int ix, int iy, int iz) const
  {
    return weights1D[ix] * weights1D[iy] * weights1D[iz];
  }
};

template <auto val>
struct CompileTimeValue {};

/**
 * @brief this struct is used to look up mfem's memory layout of
 * the quadrature point jacobian matrices
 *
 * @tparam g the element geometry
 * @tparam q the number of quadrature points per dimension
 */
template <mfem::Geometry::Type g, int q>
struct batched_jacobian;

/// @overload
template <int q>
struct batched_jacobian<mfem::Geometry::CUBE, q> {
  /// the data layout for this geometry and quadrature rule
  using type = tensor<double, 3, 3, q * q * q>;
};

/// @overload
template <int q>
struct batched_jacobian<mfem::Geometry::SQUARE, q> {
  /// the data layout for this geometry and quadrature rule
  using type = tensor<double, 2, 2, q * q>;
};

/// @overload
template <int q>
struct batched_jacobian<mfem::Geometry::TRIANGLE, q> {
  /// the data layout for this geometry and quadrature rule
  using type = tensor<double, 2, 2, q*(q + 1) / 2>;
};

/// @overload
template <int q>
struct batched_jacobian<mfem::Geometry::TETRAHEDRON, q> {
  /// the data layout for this geometry and quadrature rule
  using type = tensor<double, 3, 3, (q * (q + 1) * (q + 2)) / 6>;
};

/**
 * @brief this struct is used to look up mfem's memory layout of the
 * quadrature point position vectors
 *
 * @tparam g the element geometry
 * @tparam q the number of quadrature points per dimension
 */
template <mfem::Geometry::Type g, int q>
struct batched_position;

/// @overload
template <int q>
struct batched_position<mfem::Geometry::CUBE, q> {
  /// the data layout for this geometry and quadrature rule
  using type = tensor<double, 3, q * q * q>;
};

/// @overload
template <int q>
struct batched_position<mfem::Geometry::SQUARE, q> {
  /// the data layout for this geometry and quadrature rule
  using type = tensor<double, 2, q * q>;
};

/// @overload
template <int q>
struct batched_position<mfem::Geometry::TRIANGLE, q> {
  /// the data layout for this geometry and quadrature rule
  using type = tensor<double, 2, q*(q + 1) / 2>;
};

/// @overload
template <int q>
struct batched_position<mfem::Geometry::TETRAHEDRON, q> {
  /// the data layout for this geometry and quadrature rule
  using type = tensor<double, 3, (q * (q + 1) * (q + 2)) / 6>;
};

/// @overload
template <int q>
struct batched_position<mfem::Geometry::SEGMENT, q> {
  /// the data layout for this geometry and quadrature rule
  using type = tensor<double, q>;
};

/**
 * @brief this function returns information about how many elements
 * should be processed by a single thread block in CUDA (note: the optimal
 * values are hardware and problem specific, but these values are still significantly
 * faster than naively allocating only 1 element / block)
 *
 * @tparam g the element geometry
 * @param q the number of quadrature points per dimension
 * @return how many elements each thread block should process
 */
template <mfem::Geometry::Type g>
SMITH_HOST_DEVICE constexpr int elements_per_block(int q)
{
  if (g == mfem::Geometry::CUBE) {
    switch (q) {
      case 1:
        return 64;
      case 2:
        return 16;
      case 3:
        return 4;
      default:
        return 1;
    }
  }

  if (g == mfem::Geometry::SQUARE) {
    switch (q) {
      case 1:
        return 128;
      case 2:
        return 32;
      case 3:
        return 16;
      case 4:
        return 8;
      default:
        return 1;
    }
  }
}

/**
 * @brief Element conformity
 *
 * QOI   denotes a "quantity of interest", implying integration with the test function "1"
 * H1    denotes a function space where values are continuous across element boundaries
 * HCURL denotes a vector-valued function space where only the tangential component is continuous across element
 * boundaries HDIV  denotes a vector-valued function space where only the normal component is continuous across element
 * boundaries L2    denotes a function space where values are discontinuous across element boundaries
 */
enum class Family
{
  QOI,
  H1,
  HCURL,
  HDIV,
  L2
};

/**
 * @brief H1 elements of order @p p
 * @tparam p The order of the elements
 * @tparam c The vector dimension
 */
template <int p, int c = 1>
struct H1 {
  static constexpr int order = p;               ///< the polynomial order of the elements
  static constexpr int components = c;          ///< the number of components at each node
  static constexpr Family family = Family::H1;  ///< the family of the basis functions
};

/**
 * @brief H(curl) elements of order @p p
 * @tparam p The order of the elements
 * @tparam c The vector dimension
 */
template <int p, int c = 1>
struct Hcurl {
  static constexpr int order = p;                  ///< the polynomial order of the elements
  static constexpr int components = c;             ///< the number of components at each node
  static constexpr Family family = Family::HCURL;  ///< the family of the basis functions
};

/**
 * @brief Discontinuous elements of order @p p
 * @tparam p The order of the elements
 * @tparam c The vector dimension
 */
template <int p, int c = 1>
struct L2 {
  static constexpr int order = p;               ///< the polynomial order of the elements
  static constexpr int components = c;          ///< the number of components at each node
  static constexpr Family family = Family::L2;  ///< the family of the basis functions
};

/**
 * @brief "Quantity of Interest" elements (i.e. elements with a single shape function, 1)
 */
struct QOI {
  static constexpr int order = 0;                ///< the polynomial order of the elements
  static constexpr int components = 1;           ///< the number of components at each node
  static constexpr Family family = Family::QOI;  ///< the family of the basis functions
};

/**
 * @brief a small POD class for tracking function space metadata
 */
struct FunctionSpace {
  Family family;   ///< either H1, Hcurl, L2
  int order;       ///< polynomial order
  int components;  ///< how many values are stored at each node

  /**
   * @brief return the data contained in this struct as a tuple
   * @note the main point of this conversion is to take advantage of std::tuple's automatic
   *       lexicographic-ordering comparison operators
   */
  std::tuple<int, int, int> as_tuple() const { return std::tuple<int, int, int>(int(family), order, components); }

  /**
   * @brief defines an ordering over FunctionSpaces, to enable use in containers like std::map
   */
  bool operator<(FunctionSpace other) const { return this->as_tuple() < other.as_tuple(); }
};

/**
 * @brief transform information in the parent space  (i.e. values and derivatives w.r.t {xi, eta, zeta})
 * into the physical space (i.e. values and derivatives w.r.t. {x, y, z})
 *
 * @tparam f the element family, used to determine which kind of transformation to apply
 * @tparam T the types of quantities to be transformed
 * @tparam q how many values need to be transformed
 * @tparam dim the spatial dimension
 * @param qf_input the values to be transformed from parent to physical space
 * @param jacobians the jacobians of the isoparametric map from parent to physical space of each quadrature point
 */
template <Family f, typename T, int q, int dim>
SMITH_HOST_DEVICE void parent_to_physical(tensor<T, q>& qf_input, const tensor<double, dim, dim, q>& jacobians)
{
  [[maybe_unused]] constexpr int VALUE = 0;
  [[maybe_unused]] constexpr int DERIVATIVE = 1;

  for (int k = 0; k < q; k++) {
    tensor<double, dim, dim> J;
    for (int row = 0; row < dim; row++) {
      for (int col = 0; col < dim; col++) {
        J[row][col] = jacobians(col, row, k);
      }
    }

    if constexpr (f == Family::H1 || f == Family::L2) {
      // note: no transformation necessary for the values of H1-field
      get<DERIVATIVE>(qf_input[k]) = dot(get<DERIVATIVE>(qf_input[k]), inv(J));
    }

    if constexpr (f == Family::HCURL) {
      get<VALUE>(qf_input[k]) = dot(get<VALUE>(qf_input[k]), inv(J));
      get<DERIVATIVE>(qf_input[k]) = get<DERIVATIVE>(qf_input[k]) / det(J);
      if constexpr (dim == 3) {
        get<DERIVATIVE>(qf_input[k]) = dot(get<DERIVATIVE>(qf_input[k]), transpose(J));
      }
    }
  }
}

/**
 * @brief transform information in the physical space  (i.e. sources and fluxes w.r.t {x, y, z})
 * back to the parent space (i.e. values and derivatives w.r.t. {xi, eta, zeta}). Note: this also
 * multiplies by the outputs by the determinant of the quadrature point Jacobian.
 *
 * @tparam f the element family, used to determine which kind of transformation to apply
 * @tparam T the types of quantities to be transformed
 * @tparam q how many values need to be transformed
 * @tparam dim the spatial dimension
 * @param qf_output the values to be transformed from physical back to parent space
 * @param jacobians the jacobians of the isoparametric map from parent to physical space of each quadrature point
 */
template <Family f, typename T, int q, int dim>
SMITH_HOST_DEVICE void physical_to_parent(tensor<T, q>& qf_output, const tensor<double, dim, dim, q>& jacobians)
{
  [[maybe_unused]] constexpr int SOURCE = 0;
  [[maybe_unused]] constexpr int FLUX = 1;

  for (int k = 0; k < q; k++) {
    tensor<double, dim, dim> J_T;
    for (int row = 0; row < dim; row++) {
      for (int col = 0; col < dim; col++) {
        J_T[row][col] = jacobians(row, col, k);
      }
    }

    auto dv = det(J_T);

    if constexpr (f == Family::H1 || f == Family::L2) {
      get<SOURCE>(qf_output[k]) = get<SOURCE>(qf_output[k]) * dv;
      get<FLUX>(qf_output[k]) = dot(get<FLUX>(qf_output[k]), inv(J_T)) * dv;
    }

    // note: the flux term here is usually divided by detJ, but
    // physical_to_parent also multiplies every quadrature-point value by det(J)
    // so that part cancels out
    if constexpr (f == Family::HCURL) {
      get<SOURCE>(qf_output[k]) = dot(get<SOURCE>(qf_output[k]), inv(J_T)) * dv;
      if constexpr (dim == 3) {
        get<FLUX>(qf_output[k]) = dot(get<FLUX>(qf_output[k]), transpose(J_T));
      }
    }

    if constexpr (f == Family::QOI) {
      qf_output[k] = qf_output[k] * dv;
    }
  }
}

/**
 * @brief Template prototype for finite element implementations
 * @tparam g The geometry of the element
 * @tparam family The continuity of the element
 * the implementations of the different finite element specializations
 * are in .inl files in the detail/ directory.
 *
 * In each of these files, the finite_element specialization
 * should implement the following concept:
 *
 * struct finite_element< some_geometry, some_space > > {
 *   static constexpr mfem::Geometry::Type geometry = ...; ///< one of Triangle, Quadrilateral, etc
 *   static constexpr Family family     = ...; ///< one of H1, HCURL, HDIV, etc
 *   static constexpr int  components   = ...; ///< how many components per node
 *   static constexpr int  dim          = ...; ///< number of parent element coordinates
 *   static constexpr int  ndof         = ...; ///< how many degrees of freedom for an element with 1 component per node
 *
 *   /// implement the way this element type interpolates the solution on the interior of the element
 *   static constexpr auto shape_functions(tensor<double, dim> xi) { ... }
 *
 *   /// implement the derivatives of this element's shape functions w.r.t. parent element coordinates
 *   static constexpr auto shape_function_derivatives(tensor<double, dim> xi) { ... }
 * };
 *
 */
template <mfem::Geometry::Type g, typename family>
struct finite_element;

#include "detail/segment_H1.inl"
#include "detail/segment_Hcurl.inl"
#include "detail/segment_L2.inl"

#include "detail/triangle_H1.inl"
#include "detail/triangle_L2.inl"

#include "detail/quadrilateral_H1.inl"
#include "detail/quadrilateral_Hcurl.inl"
#include "detail/quadrilateral_L2.inl"

#include "detail/tetrahedron_H1.inl"
#include "detail/tetrahedron_L2.inl"

#include "detail/hexahedron_H1.inl"
#include "detail/hexahedron_Hcurl.inl"
#include "detail/hexahedron_L2.inl"

#include "detail/qoi.inl"

}  // namespace smith
