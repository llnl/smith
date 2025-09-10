// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file fixed_step_bdf_operator.hpp
 *
 * @brief Implements a Backward Differentiation Formula (BDF) time derivative
 * operator with fixed timestep and dynamic order selection.
 */

#include "serac/infrastructure/logger.hpp"

namespace serac {

/**
 * @brief Implements a fixed time step Backward Differentiation Formula (BDF) operator.
 *
 * This class supports BDF1 through BDF6 schemes with constant timestep. It computes
 * the BDF coefficients dynamically based on how many previous states are available.
 *
 * @tparam s_order The maximum order of the BDF method (e.g., 1 to 6).
 */
template <int s_order = 2>
struct FixedStepBDFOperator {

  /**
   * @brief Computes the BDF time derivative: Dφ^{k+1} = (β₀ * φ^{k+1} + weighted_sum_previous) / Δt
   *
   * The BDF method approximates the time derivative at time step k+1 using a weighted
   * combination of current and past states. The `weighted_sum_previous` argument is assumed
   * to already include the correct signs for β₁ through β_q.
   *
   * As a result, this function uses a '+' to combine β₀ * φ^{k+1} with the weighted sum.
   *
   * @tparam T1 Type of the current state φ^{k+1}, includes AD support (e.g., dual numbers).
   * @tparam T2 Type of the weighted sum of previous states (typically plain).
   *
   * @param phi_k_plus_1          The current state at time step k+1.
   * @param weighted_sum_previous The precomputed sum β₁ * φ^k + β₂ * φ^{k−1} + … (signs included).
   * @param dt                    The time step size Δt.
   * @param k_plus_1              Current time step index (i.e., solving for time step \( k + 1 \))
   *
   * @return The time derivative Dφ^{k+1}
   */
  template <typename T1, typename T2>
  auto
  time_derivative(
    T1 phi_k_plus_1,          // e.g., serac::tensor<serac::dual<...>, 2>
    T2 weighted_sum_previous, // e.g., serac::tensor<double, 2>
    double dt,
    int k_plus_1
  ) const {

    int available_order = std::min(k_plus_1, s_order);
    auto beta = compute_beta(available_order);

    SLIC_ERROR_ROOT_IF(
      static_cast<int>(beta.size()) < 1,
      "Insufficient BDF coefficients computed. Cannot evaluate time derivative."
    );

    // Compute the full BDF expression: (β₀ * φ^{k+1} + Σ_j β_j * φ^{k-j}) / Δt
    return ((beta[0] * phi_k_plus_1) + weighted_sum_previous) / dt;

  }

  /**
   * @brief Computes the weighted sum of previous states using dynamic BDF coefficients.
   *
   * @param previous_states A container of previous states order from newest (`u^k`) to oldest (`u^{k-q+1}`).
   * @param k_plus_1        Current time step index (i.e., solving for time step \( k + 1 \))
   *
   * @return The weighted sum of previous values.
   */
  ::serac::FiniteElementState
  weighted_sum(
    std::vector<::serac::FiniteElementState> const &previous_states,
    int k_plus_1
  ) const {

    SLIC_ERROR_ROOT_IF(
      k_plus_1 == 0 || previous_states.empty(),
      "No previous states available to compute weighted sum."
    );

    int available_order = std::min(k_plus_1, s_order);

    SLIC_ERROR_ROOT_IF(
      static_cast<int>(previous_states.size()) != available_order,
      ::axom::fmt::format("Expected {} previous states, but got {}.", available_order, previous_states.size())
    );

    auto const beta = compute_beta(available_order);

    ::serac::FiniteElementState result = previous_states[0];
    result *= beta[1]; // Apply first beta coefficient (skip beta[0], which is for u_{k+1})

    for (int i = 1; i < available_order; ++i) {
      result.Add(beta[i + 1], previous_states[i]);
    }

    return result;

  }

  /**
   * @brief Retrieves the BDF coefficient β_j for a given cycle and index.
   *
   * @param k_plus_1 Current time step index (i.e., solving for time step \( k + 1 \))
   * @param j        Index of the coefficient (0 = next-step, 1 = previous, etc.).
   * @return         The β_j coefficient for the current order.
   */
  double
  compute_beta(
    int k_plus_1,
    int j
  ) const {

    SLIC_ERROR_ROOT_IF(k_plus_1 <= 0, "Cannot compute BDF coefficient: cycle must be greater than zero.");

    int available_order = std::min(k_plus_1, s_order);
    auto betas = compute_beta(available_order);

    SLIC_ERROR_ROOT_IF(
      j < 0 || j >= static_cast<int>(betas.size()),
      ::axom::fmt::format("Index {} out of range for BDF coefficients (available: {}).", j, betas.size())
    );

    return betas[j];

  }

  /**
   * @brief Applies the derivative ∂(Dφ^j)/∂φ^t to a vector.
   *
   * This computes the action of the BDF derivative’s Jacobian with respect to φ^t:
   *   ∂/∂φ^t [ (β₀ * φ^j + ∑ₘ βₘ * φ^{j−m}) / Δt ] = (βₘ / Δt) · I
   * where m = j − t and t ∈ {j, j−1, ..., j−(r−1)}.
   *
   * If t is outside this range, the result is zero.
   *
   * @tparam VectorType A vector-like type supporting assignment and scalar multiplication.
   * @param j   The current step index (computing ∂/∂φ^t of Dφ^j).
   * @param t   The time index of the derivative target φ^t.
   * @param vec The vector v to scale.
   * @param dt  The time step Δt.
   * @return    A scaled copy of vec, or zero if out of valid range.
   */
  template <typename VectorType>
  VectorType
  apply_derivative(
    int j,
    int t,
    VectorType const &vec
  ) const {

    int r = std::min(j, s_order);  // Effective BDF order.
    int m = j - t;

    VectorType result(vec);  // Copy input vector.

    if (m < 0 || m >= r + 1) {
      result = 0.0;
    } else {
      auto beta = compute_beta(r);  // Get β₀ through β_r.
      result *= beta[m];
    }

    return result;

  }

  /**
   * @brief Returns the BDF order actually in use.
   */
  static int constexpr order() { return s_order; }

private:

  /**
   * @brief Internal function to compute all β_j coefficients for a given order.
   *
   * @param order The order of the BDF method (≤ s_order).
   * @return      Vector of BDF coefficients {β₀, β₁, ..., β_order}.
   */
  std::vector<double>
  compute_beta(
    int order
  ) const {

    SLIC_ERROR_ROOT_IF(
      order < 1 || order > 6,
      ::axom::fmt::format("Unsupported BDF order: {} (valid range is 1 to 6).", order)
    );

    switch (order) {
      case 1: return {1.0, -1.0};
      case 2: return {3.0 / 2.0, -2.0, 0.5};
      case 3: return {11.0 / 6.0, -3.0, 1.5, -1.0 / 3.0};
      case 4: return {25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 0.25};
      case 5: return {137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 1.25, -0.2};
      case 6: return {147.0 / 60.0, -6.0, 7.5, -20.0 / 3.0, 3.75, -1.2, 1.0 / 6.0};
      default:
      return {};
    }

  }

};  // struct FixedStepBDFOperator

} // namespace serac
