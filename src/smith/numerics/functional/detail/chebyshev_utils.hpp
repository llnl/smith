// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file chebyshev_utils.hpp
 *
 * @brief Chebyshev polynomial evaluation utilities for simplex Hdiv elements
 *
 * These are used by the Vandermonde-based shape function construction in
 * triangle_Hdiv.inl and tetrahedron_Hdiv.inl.
 */

#pragma once

// NOTE: This header is included from within namespace smith {} in finite_element.hpp,
// so no namespace wrapper is needed here.

/// Evaluate Chebyshev polynomials T_0(2x-1), ..., T_order(2x-1) at x
SMITH_HOST_DEVICE inline constexpr void chebyshev_eval(int order, double x, double* u)
{
  u[0] = 1.0;
  if (order < 1) return;
  double z = 2.0 * x - 1.0;
  u[1] = z;
  for (int i = 1; i < order; i++) {
    u[i + 1] = 2.0 * z * u[i] - u[i - 1];
  }
}

/// Evaluate Chebyshev polynomials and their derivatives (w.r.t. x, not z)
SMITH_HOST_DEVICE inline constexpr void chebyshev_eval_d(int order, double x, double* u, double* d)
{
  u[0] = 1.0;
  d[0] = 0.0;
  if (order < 1) return;
  double z = 2.0 * x - 1.0;
  u[1] = z;
  d[1] = 2.0;
  for (int i = 1; i < order; i++) {
    u[i + 1] = 2.0 * z * u[i] - u[i - 1];
    d[i + 1] = double(i + 1) * (z * d[i] / double(i) + 2.0 * u[i]);
  }
}
