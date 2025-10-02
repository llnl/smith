// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file extrapolation_operator.hpp
 *
 * @brief Implements an extrapolation operator for time-dependent fields using
 * zero-, first-, or second-order extrapolation.
 */

#pragma once

#include "serac/infrastructure/logger.hpp"

namespace serac {

/**
 * @brief Implements a cycle-aware extrapolator for time-dependent fields.
 *
 * This operator handles extrapolation of any time-dependent field (e.g., pressure, velocity)
 * using a compile-time specified maximum order and a runtime cycle count to determine
 * how many states are available.
 *
 * @tparam r_order The desired extrapolation order (0 = zero, 1 = constant, 2 = linear).
 */
template <int r_order>
struct ExtrapolationOperator {

  static_assert(
    r_order >= 0 && r_order <= 2,
    "Supported extrapolation orders are 0 (zero), 1 (constant), and 2 (linear)."
  );

  /**
   * @brief Constructs an extrapolation operator over the given finite element space.
   *
   * This operator enables time-extrapolation of fields defined in the specified finite element space,
   * such as velocity or pressure, using a compile-time extrapolation order.
   *
   * The extrapolated field φ^{*,k+1} will live in the same space as the provided input states φᵏ, φᵏ⁻¹, etc.
   *
   * @param space The finite element space in which the extrapolated fields reside.
   */
  ExtrapolationOperator(
    mfem::ParFiniteElementSpace *space
  ) : space_(space) {}

  /**
   * @brief Extrapolates a field value φ^{*,k+1} from previous states.
   *
   * The extrapolated value is computed as:
   *   φ^{*,r,k+1} = ∑ₘ γₘ · φ^{k−m}, for m = 0..r−1
   * where r = min(r_order, k+1), and γₘ are extrapolation weights.
   *
   * The input vector previous_states is ordered from newest (φᵏ) to oldest (φ^{k−(r−1)}).
   *
   * @param previous_states  Vector of past states φᵏ, φᵏ⁻¹, … ordered newest to oldest.
   * @param cycle            Current simulation cycle (corresponds to time step k+1).
   * @return                 Extrapolated field φ^{*,k+1}.
   */
  ::serac::FiniteElementState
  operator()(
    ::std::vector<::serac::FiniteElementState> const &previous_states,
    int cycle
  ) const {

    SLIC_ERROR_ROOT_IF(cycle <= 0, "Extrapolation requires at least one completed cycle.");

    int const available_order = std::min(cycle, r_order);  // Effective order.

    SLIC_ERROR_ROOT_IF(
      static_cast<int>(previous_states.size()) != available_order,
      axom::fmt::format("Expected at least {} previous states, but got {}.", available_order, previous_states.size())
    );

    ::serac::FiniteElementState result(*space_);  // Initialize from most recent state (φ^k).

    if (available_order == 0) {
      result = 0.0;  // Zero extrapolation.
    } else {
      auto gammas = this->compute_gammas(available_order);
      result  = previous_states[0];
      result *= gammas[0];  // result = γ₀ · φ^k
      for (int j = 1; j < available_order; ++j) {
        result.Add(gammas[j], previous_states[j]);  // result += γⱼ · φ^{k−j}
      }
    }

    return result;

  }

  /**
   * @brief Applies the derivative ∂φ^{*,j} / ∂φᵗ to a vector.
   *
   * Given a target extrapolated field φ^{*,j} and a source field φᵗ, this method computes:
   *   ∂φ^{*,j} / ∂φᵗ · v = γₘ · v,
   * where m = j − 1 − t, and γₘ is the corresponding extrapolation weight.
   *
   * If m is outside the valid range [0, r−1], where r = min(r_order, j), the result is zero.
   *
   * @tparam VectorType A vector-like type supporting assignment and scalar multiplication.
   * @param j      The target cycle index (for φ^{*,j}).
   * @param t      The source time index (φᵗ).
   * @param vec    The vector v to scale.
   * @return       The scaled vector γₘ · v, or zero if m is out of bounds.
   */
  template <typename VectorType>
  VectorType
  apply_derivative(
    int const &j,
    int const &t,
    VectorType const &vec
  ) const {

    int const r = std::min(j, r_order);  // Effective order.
    int const m = (j - 1) - t;           // Index difference m = j−1−t

    VectorType result(vec);  // Copy the input vector v.

    if (m < 0 || m >= r) {
      result = 0.0;  // Zero contribution if out of bounds.
    } else {
      auto gammas = this->compute_gammas(r);
      result *= gammas[m];  // Scale by γₘ.
    }

    return result;

  }

private:

  /// @brief The finite element space in which the extrapolated fields are defined.
  mfem::ParFiniteElementSpace *space_;

  /**
   * @brief Compute extrapolation weights γₘ for φ^{*,k+1} given an extrapolation order.
   *
   * These weights define the extrapolated value:
   *   φ^{*,k+1} = ∑ₘ γₘ · φ^{k−m}, for m = 0..order−1.
   *
   * @param order The extrapolation order (must be 0, 1, or 2).
   * @return A vector of γₘ weights indexed by m.
   */
  std::vector<double>
  compute_gammas(
    int order
  ) const {

    SLIC_ERROR_ROOT_IF(
      order < 0 || order > 2,
      axom::fmt::format("Unsupported extrapolation order = {}. Only orders 0, 1, and 2 are supported.", order)
    );

    switch (order) {
      case 0:
        return {};           // Zero extrapolation → φ^{*,k+1} = 0
      case 1:
        return {1.0};        // Constant extrapolation → φ^{*,k+1} = φ^k
      case 2:
        return {2.0, -1.0};  // Linear extrapolation → φ^{*,k+1} = 2φ^k − φ^{k−1}
      default:
        // Should be unreachable due to earlier guard.
        SLIC_ERROR_ROOT("Unreachable case in switch for extrapolation weights.");
        return {};
    }

  }

};  // struct ExtrapolationOperator

}  // namespace serac
