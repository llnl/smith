/**
 * @file fixed_step_bdf_operator.hpp
 *
 * @brief Implements a Backward Differentiation Formula (BDF) time derivative
 * operator with fixed timestep and dynamic order selection.
 */

#pragma once

#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/temporal_stencil.hpp"

namespace smith {

/**
 * @brief Fixed-step Backward Differentiation Formula (BDF) operator.
 *
 * Implements constant time-step BDF schemes of order 1 through 6.
 *
 * For a maximum configured order s_order_, the effective order at cycle k+1 is
 *
 *     r_eff = min(s_order_, k_plus_1).
 *
 * The operator provides:
 *
 *  - Evaluation of the BDF time derivative
 *      Dφ^{k+1} = (β₀ φ^{k+1} + Σ_{j=1}^{r_eff} β_j φ^{k+1-j}) / Δt
 *
 *  - Construction of the weighted lag-state sum
 *      Σ_{j=1}^{r_eff} β_j φ^{k+1-j}
 *
 * Lag states are obtained from a TemporalStencil and are assumed to be
 * ordered newest → oldest:
 *
 *      { φ^k, φ^{k-1}, φ^{k-2}, ... }.
 *
 * The operator does not own or modify the stencil.
 */
struct FixedStepBDFOperator {

  /**
   * @brief Construct a BDF operator with a maximum order.
   *
   * @param s_order Maximum BDF order (valid range: 1–6).
   */
  explicit
  FixedStepBDFOperator(
    int s_order = 2
  ) : s_order_(s_order) {

    SLIC_ERROR_ROOT_IF(
      s_order_ < 1 || s_order_ > 6,
      axom::fmt::format("Unsupported BDF order: {} (valid range is 1 to 6).", s_order_)
    );

  }

  /**
   * @brief Evaluate the BDF time derivative at cycle k+1.
   *
   * Computes
   *
   *     Dφ^{k+1} = (β₀ φ^{k+1} + weighted_sum_previous) / Δt,
   *
   * where weighted_sum_previous represents
   *
   *     Σ_{j=1}^{r_eff} β_j φ^{k+1-j}.
   *
   * The effective order is r_eff = min(s_order_, k_plus_1).
   *
   * @tparam T1 Type of φ^{k+1} (may include automatic differentiation).
   * @tparam T2 Type of the weighted lag-state sum.
   *
   * @param phi_k_plus_1          Current state φ^{k+1}.
   * @param weighted_sum_previous Precomputed lag-state combination.
   * @param dt                    Time step size Δt.
   * @param k_plus_1              Current cycle index.
   *
   * @return Approximation of ∂φ/∂t at time k+1.
   */
  template <typename T1, typename T2>
  auto
  time_derivative(
    T1 phi_k_plus_1,          // e.g., serac::tensor<serac::dual<...>, 2>
    T2 weighted_sum_previous, // e.g., serac::tensor<double, 2>
    double dt,
    int k_plus_1
  ) const {

    auto available_order = std::min(k_plus_1, s_order_);
    auto beta = this->compute_beta(available_order);

    SLIC_ERROR_ROOT_IF(
      beta.size() < 1,
      "Insufficient BDF coefficients computed. Cannot evaluate time derivative."
    );

    // Compute the full BDF expression: (β₀ * φ^{k+1} + Σ_j β_j * φ^{k-j}) / Δt
    return ((beta[0] * phi_k_plus_1) + weighted_sum_previous) / dt;

  }

  /**
   * @brief Compute the weighted sum of lag states required by the BDF scheme.
   *
   * For effective order r_eff = min(s_order_, k_plus_1), this computes
   *
   *     Σ_{j=1}^{r_eff} β_j φ^{k+1-j}.
   *
   * The stencil must contain at least r_eff lag states ordered newest → oldest.
   *
   * β₀ is excluded because it multiplies φ^{k+1} in the full time-derivative expression.
   *
   * @param stencil  Temporal stencil storing lag FieldState objects.
   * @param k_plus_1 Current cycle index.
   *
   * @return FiniteElementState representing the weighted lag-state sum.
   */
  smith::FiniteElementState
  weighted_sum(
    smith::TemporalStencil<smith::FieldState> const &stencil,
    int k_plus_1
  ) const {

    SLIC_ERROR_ROOT_IF(
      k_plus_1 <= 0,
      "No previous states available to compute weighted sum."
    );

    // Effective order for this cycle
    int available_order = std::min(k_plus_1, s_order_);

    // Extract required lag states (newest → oldest)
    auto history = stencil.view(available_order);

    SLIC_ERROR_ROOT_IF(
      history.size() != static_cast<size_t>(available_order),
      axom::fmt::format(
        "Expected {} previous states, but got {}.",
        available_order,
        history.size()
      )
    );

    auto const beta = this->compute_beta(available_order);

    // Initialize with β₁ φ^k
    smith::FiniteElementState result = *history[0].get();
    result *= beta[1]; // Apply first beta coefficient (skip beta[0], which is for u_{k+1}).

    // Add remaining β_j φ^{k-j}.
    for (size_t j = 1; j < available_order; ++j) {
      result.Add(beta[j + 1], *history[static_cast<std::size_t>(j)].get());
    }

    return result;

  }

  /**
   * @brief Return a specific BDF coefficient β_j for the current cycle.
   *
   * The effective order is r_eff = min(s_order_, k_plus_1). Valid indices are j = 0,…,r_eff.
   *
   * @param k_plus_1 Current cycle index.
   * @param j        Coefficient index.
   *
   * @return β_j for the effective BDF order.
   */
  double
  compute_beta(
    int k_plus_1,
    int j
  ) const {

    SLIC_ERROR_ROOT_IF(
      k_plus_1 <= 0,
      "Cannot compute BDF coefficient: cycle must be greater than zero."
    );

    auto available_order = std::min(k_plus_1, s_order_);
    auto const betas = this->compute_beta(available_order);

    SLIC_ERROR_ROOT_IF(
      j < 0 || j >= betas.size(),
      axom::fmt::format(
        "Index {} out of range for BDF coefficients (available: {}).",
        j,
        betas.size()
      )
    );

    return betas[j];

  }

   /**
   * @brief Return the configured maximum BDF order.
   */
  int order() const { return s_order_; }

private:

  /**
   * @brief Compute all BDF coefficients for a given order.
   *
   * Returns
   *
   *     { β₀, β₁, …, β_order }.
   *
   * @param order Effective BDF order (1–6).
   *
   * @return Vector of BDF coefficients.
   */
  std::vector<double>
  compute_beta(
    int order
  ) const {

    SLIC_ERROR_ROOT_IF(
      order < 1 || order > 6,
      axom::fmt::format(
        "Unsupported BDF order: {} (valid range is 1 to 6).",
        order
      )
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

  /**
   * @brief Maximum BDF order configured for this operator.
   */
  int s_order_;

};  // struct FixedStepBDFOperator

}  // namespace smith
