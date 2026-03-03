/**
 * @file extrapolation_operator.hpp
 *
 * @brief Implements an extrapolation operator for time-dependent fields using
 * zero-, first-, or second-order extrapolation.
 */

#pragma once

#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/temporal_stencil.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

/**
 * @brief Implements a cycle-aware extrapolator for time-dependent fields.
 *
 * This operator handles extrapolation of any time-dependent field (e.g., pressure, velocity)
 * using a runtime-specified maximum order and a runtime cycle count to determine
 * how many states are available.
 */
template <typename fes_t>
class ExtrapolationOperator {

public:

  /**
   * @brief Constructs an extrapolation operator for time-dependent finite element fields.
   *
   * This operator computes explicit extrapolated fields from a sequence of previous
   * finite element states using a runtime-selected extrapolation order.
   *
   * For extrapolation orders greater than zero, the result is formed directly from
   * the provided history states. For zero-order extrapolation, the operator allocates
   * a new field on the given mesh and initializes it to zero.
   *
   * The extrapolated field φ^{*,k+1} always lives in the same finite element space
   * as the input history states.
   *
   * @param mesh  Mesh used to allocate a new field when no history is available
   *              (zero-order extrapolation).
   * @param order The desired extrapolation order
   *              (0 = zero, 1 = constant, 2 = linear).
   */
  ExtrapolationOperator(
    smith::Mesh &mesh,
    int order
  ) : mesh_(mesh)
    , order_(order) {

    SLIC_ERROR_ROOT_IF(
      order_ < 0 || order_ > 2,
      axom::fmt::format(
        "Unsupported extrapolation order = {}. Only orders 0, 1, and 2 are supported.",
        order_
      )
    );

  }

  /**
   * @brief Extrapolate a finite element field φ^{*,k+1} from a temporal stencil.
   *
   * This operator evaluates an explicit extrapolation of order r_eff,
   *
   *     φ^{*,k+1} = ∑_{m=0}^{r_eff-1} γ_m φ^{k-m},
   *
   * where
   *
   *     r_eff = min(order_, cycle).
   *
   * The stencil stores lag states ordered from newest to oldest:
   *
   *     { φ^k, φ^{k-1}, φ^{k-2}, ... }.
   *
   * The operator determines internally how many states are required based on the
   * configured extrapolation order and the current cycle. The stencil is treated
   * as a read-only storage container; it is neither modified nor owned by this operator.
   *
   * For zero effective order (r_eff = 0), a zero field is returned in the appropriate
   * finite element space.
   *
   * @param stencil Temporal stencil storing lag FieldState objects ordered newest → oldest.
   * @param cycle   Current simulation cycle (corresponding to time step k+1).
   *
   * @return A new FiniteElementState representing φ^{*,k+1}.
   */
  smith::FiniteElementState
  operator()(
    TemporalStencil<smith::FieldState> const &stencil,
    int cycle
  ) const {

    SLIC_ERROR_ROOT_IF(
      cycle <= 0,
      "Extrapolation requires at least one completed cycle."
    );

    // Determine effective extrapolation order for this cycle.
    int const available_order = std::min(cycle, order_);

    // Extract the required lag states (newest → oldest).
    auto history = stencil.view(available_order);

    // Allocate result in the correct finite element space.
    smith::FiniteElementState result(mesh_.mfemParMesh(), fes_t{}, "extrapolated");

    if (available_order == 0) {

      // Zero-order extrapolation: return zero field.
      result = 0.0;

    } else {

      // Compute extrapolation weights γ_m.
      auto gammas = this->compute_gammas(available_order);

      // Initialize with γ_0 φ^k.
      result  = *history[0].get();
      result *= gammas[0];  // γ₀ · φ^k

      // Accumulate remaining contributions γ_m φ^{k-m}.
      for (int j = 1; j < available_order; ++j) {
        result.Add(gammas[j], *history[static_cast<std::size_t>(j)].get());
      }

    }

    return result;

  }

private:

  /// @brief Mesh used to construct new finite element states when extrapolation
  ///        requires allocation (e.g., zero-order extrapolation).
  smith::Mesh &mesh_;

  /// @brief Maximum extrapolation order supported by this operator.
  int order_;

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
      axom::fmt::format(
        "Unsupported extrapolation order = {}. Only orders 0, 1, and 2 are supported.",
        order
      )
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

}  // namespace smith
