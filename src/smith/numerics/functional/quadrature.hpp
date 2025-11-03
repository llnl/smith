// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrature.hpp
 *
 * @brief Definitions of quadrature rules for quads and hexes
 */

#pragma once

#include "tensor.hpp"
#include "polynomials.hpp"
#include "finite_element.hpp"

namespace smith {

/**
 * @brief A rule for numerical quadrature (set of points and weights)
 * Can be thought of as a compile-time analogue of @p mfem::IntegrationRule
 * @tparam n The number of points in the rule
 * @tparam dim The spatial dimension of integration
 */
template <int n, int dim>
struct QuadratureRule {
  /// @brief The scalar weights of each point
  tensor<double, n> weights;

  /// @brief The coordinates in reference space for each quadrature point
  tensor<double, n, dim> points;

  /// @brief Returns the number of points in the rule
  SMITH_HOST_DEVICE constexpr std::size_t size() const { return n; }
};

/**
 * @brief Returns the Gauss-Legendre quadrature rule for an element and order
 * @tparam The shape of the element to produce a quadrature rule for
 * @tparam Q the number of quadrature points per dimension
 */
template <mfem::Geometry::Type g, int Q>
SMITH_HOST_DEVICE constexpr auto GaussQuadratureRule()
{
  auto x = GaussLegendreNodes<Q, mfem::Geometry::SEGMENT>();
  auto w = GaussLegendreWeights<Q, mfem::Geometry::SEGMENT>();

  if constexpr (g == mfem::Geometry::SEGMENT) {
    return QuadratureRule<Q, 1>{w, make_tensor<Q, 1>([&x](int i, int /*j*/) { return x[i]; })};
  }

  if constexpr (g == mfem::Geometry::SQUARE) {
    QuadratureRule<Q * Q, 2> rule{};
    int count = 0;
    for (int j = 0; j < Q; j++) {
      for (int i = 0; i < Q; i++) {
        rule.points[count] = {x[i], x[j]};
        rule.weights[count++] = w[i] * w[j];
      }
    }
    return rule;
  }

  if constexpr (g == mfem::Geometry::CUBE) {
    QuadratureRule<Q * Q * Q, 3> rule{};
    int count = 0;
    for (int k = 0; k < Q; k++) {
      for (int j = 0; j < Q; j++) {
        for (int i = 0; i < Q; i++) {
          rule.points[count] = {x[i], x[j], x[k]};
          rule.weights[count++] = w[i] * w[j] * w[k];
        }
      }
    }
    return rule;
  }
}

}  // namespace smith
