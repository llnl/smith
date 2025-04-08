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
   * @brief Extrapolates a value based on available previous states.
   *
   * @param previous_states A container of previous states, ordered from newest (`p^k`) to oldest (`p^{k-r+1}`).
   * @param cycle           Current simulation cycle (used to determine how many previous states are available).
   *
   * @return The extrapolated field value.
   */
  ::serac::FiniteElementState
  operator()(
    ::std::vector<::serac::FiniteElementState> const &previous_states,
    int cycle
  ) const {

    SLIC_ERROR_ROOT_IF(cycle <= 0, "Extrapolation requires at least one completed cycle.");

    int available_order = std::min(cycle, r_order);

    SLIC_ERROR_ROOT_IF(
      static_cast<int>(previous_states.size()) < available_order,
      axom::fmt::format("Expected at least {} previous states, but got {}.", available_order, previous_states.size())
    );

    ::serac::FiniteElementState result = previous_states[0];  // Initialize from most recent state (`p^k`).

    if (available_order == 0) {
      result = 0.0; // Zero extrapolation.
    } else if (available_order == 1) {
      // Constant extrapolation: do nothing.
    } else if (available_order == 2) {
      // Linear extrapolation: 2*u_k - u_{k-1}
      result.Add(+1.0, previous_states[0]);  // result = 2 * u_k
      result.Add(-1.0, previous_states[1]);  // result -= u_{k-1}
    } else {
      SLIC_ERROR_ROOT(
        ::axom::fmt::format(
          "Unexpected extrapolation order: {}. Supported values are 0, 1, 2. This should be unreachable due to static_assert above.",
          available_order
        )
      );
    }

    return result;

  }

};  // struct ExtrapolationOperator

}  // namespace serac
