// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstddef>

#include "smith/numerics/functional/element_restriction.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "mfem.hpp"

namespace smith {

/**
 * @brief a class that computes and stores positions and jacobians at each quadrature point
 * @note analogous to mfem::GeometricFactors, except that it implements the position/jacobian
 *       calculations on boundary elements and on simplex elements
 */
struct GeometricFactors {
  /// @brief default ctor, leaving this object uninitialized
  GeometricFactors() {};

  /**
   * @brief calculate positions and jacobians for quadrature points belonging to
   * elements with the specified geometry, belonging to the provided mesh.
   *
   * @param domain the domain of integration
   * @param q a parameter controlling the number of quadrature points per element
   * @param elem_geom which kind of element geometry to select
   */
  GeometricFactors(const Domain& domain, int q, mfem::Geometry::Type elem_geom);

  // descriptions copied from mfem

  /// Mapped (physical) coordinates of all quadrature points.
  /** This array uses a column-major layout with dimensions (NQ x SDIM x NE)
      where
      - NQ = number of quadrature points per element,
      - SDIM = space dimension of the mesh = mesh.SpaceDimension(), and
      - NE = number of elements in the mesh. */
  mfem::Vector X;

  /// Jacobians of the element transformations at all quadrature points.
  /** This array uses a column-major layout with dimensions (NQ x SDIM x DIM x
      NE) where
      - NQ = number of quadrature points per element,
      - SDIM = space dimension of the mesh = mesh.SpaceDimension(),
      - DIM = dimension of the mesh = mesh.Dimension(), and
      - NE = number of elements in the mesh. */
  mfem::Vector J;

  /// the number of elements in the domain
  std::size_t num_elements;
};

}  // namespace smith
