// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file time_integration_rule.hpp
 *
 * @brief Provides templated implementations for discretizing values, velocities and accelerations from current and previous
 * states
 */

#pragma once

#include "smith/physics/common.hpp"

namespace smith {

/// @brief encodes rules for 2-point integration rules for a first order ode discretization.
/// When solving u_dot = f(u, t),
/// this discretizes as derivative(u,u_old) = f(value(u,u_old,t)).
struct SecondOrderTimeIntegrationRule {
  /// @brief
  /// @param theta weighting on current vs previous state value.
  SecondOrderTimeIntegrationRule(double theta = 1.0) : theta_(theta) {}

  /// @brief evaluate value of the ode state as used by the integration rule
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto value(const TimeInfo& /*t*/, const T1& field_new, [[maybe_unused]] const T2& field_old,
                               [[maybe_unused]] const T3& velo_old, [[maybe_unused]] const T4& accel_old) const
  {
    return field_new;
  }

  /// @brief evaluate time derivative discretization of the ode state as used by the integration rule
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto derivative(const TimeInfo& t, const T1& field_new, const T2& field_old,
                                    [[maybe_unused]] const T3& velo_old, [[maybe_unused]] const T4& accel_old) const
  {
    return (1.0 / t.dt()) * (field_new - field_old);
  }

  /// @brief evaluate time derivative discretization of the ode state as used by the integration rule
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto second_derivative([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                                           [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                                           [[maybe_unused]] const T4& accel_old) const
  {
    return accel_old;
  }

  double theta_;  ///< parameter specifying the particular integration rule for first order, two step integrator
};

}  // namespace smith