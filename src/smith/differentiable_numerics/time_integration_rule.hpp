// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file time_integration_rule.hpp
 *
 * @brief Provides templated implementations for discretizing values, velocities and accelerations from current and
 * previous states
 */

#pragma once

#include "smith/physics/common.hpp"

namespace smith {

/// @brief encodes rules for time discretizing second order odes (involving first and second time derivatives).
/// When solving f(u, u_dot, u_dot_dot, t) = 0
/// this class provides the current discrete approximation for u, u_dot, and u_dot_dot as a function of
/// (u^{n+1},u^n,u_dot^n,u_dot_dot^n).
struct SecondOrderTimeIntegrationRule {
  /// @brief
  /// @param theta weighting on current vs previous state value.
  SecondOrderTimeIntegrationRule(double theta = 1.0) : theta_(theta) {}

  /// @brief evaluate value of the ode state as used by the integration rule
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto value([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                               [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                               [[maybe_unused]] const T4& accel_old) const
  {
    return field_new;
  }

  /// @brief evaluate time derivative discretization of the ode state as used by the integration rule
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto derivative([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                                    [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                                    [[maybe_unused]] const T4& accel_old) const
  {
    auto v_np5 = (1.0 / t.dt()) * (field_new - field_old);
    auto v_n = velo_old;
    return (2.0 * v_np5) - v_n;  // (2.0 / t.dt()) * (field_new - field_old) - velo_old;
    // return v_fd;
  }

  /// @brief evaluate time derivative discretization of the ode state as used by the integration rule
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto second_derivative([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                                           [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                                           [[maybe_unused]] const T4& accel_old) const
  {
    auto v_np5 = (1.0 / t.dt()) * (field_new - field_old);
    auto v_n = velo_old;
    auto a_np25 = (1.0 / t.dt()) * (v_np5 - v_n);
    return (4.0 * a_np25) - accel_old;
  }

  double alpha_v0 = 1.0;

  double theta_;  ///< parameter specifying the particular integration rule for integrating second order systems with
                  ///< two steps
};

}  // namespace smith