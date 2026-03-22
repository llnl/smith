// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tensor_product_basis.hpp
 *
 * @brief Shared 1D basis evaluation matrices for tensor-product Hcurl/Hdiv elements
 *
 * B1: Gauss-Legendre (open) interpolation at quadrature points
 * B2: Gauss-Lobatto (closed) interpolation at quadrature points
 * G2: Gauss-Lobatto (closed) derivative at quadrature points
 *
 * Used by quadrilateral_Hdiv.inl and hexahedron_Hdiv.inl (and their Hcurl
 * counterparts use identical logic).
 */

#pragma once

// NOTE: This header is included from within namespace smith {} in finite_element.hpp.
// We use namespace basis_detail to avoid name collisions with member functions
// of the same name inside finite_element specializations.

namespace basis_detail {

template <int p, bool apply_weights, int q>
constexpr auto calculate_B1()
{
  constexpr auto points1D  = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
  [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
  tensor<double, q, p> B1{};
  for (int i = 0; i < q; i++) {
    B1[i] = GaussLegendreInterpolation<p>(points1D[i]);
    if constexpr (apply_weights) B1[i] = B1[i] * weights1D[i];
  }
  return B1;
}

template <int p, bool apply_weights, int q>
constexpr auto calculate_B2()
{
  constexpr auto points1D  = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
  [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
  tensor<double, q, p + 1> B2{};
  for (int i = 0; i < q; i++) {
    B2[i] = GaussLobattoInterpolation<p + 1>(points1D[i]);
    if constexpr (apply_weights) B2[i] = B2[i] * weights1D[i];
  }
  return B2;
}

template <int p, bool apply_weights, int q>
constexpr auto calculate_G2()
{
  constexpr auto points1D  = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
  [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
  tensor<double, q, p + 1> G2{};
  for (int i = 0; i < q; i++) {
    G2[i] = GaussLobattoInterpolationDerivative<p + 1>(points1D[i]);
    if constexpr (apply_weights) G2[i] = G2[i] * weights1D[i];
  }
  return G2;
}

}  // namespace basis_detail
