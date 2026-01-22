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

/// @brief encodes rules for time discretizing first order odes (involving first time derivatives).
/// When solving f(u, u_dot, t) = 0
/// this class provides the current discrete approximation for u and u_dot as a function of
/// (u^{n+1}, u^n).
struct BackwardEulerFirstOrderTimeIntegrationRule {
  /// @brief Constructor
  BackwardEulerFirstOrderTimeIntegrationRule() {}

  /// @brief evaluate value of the ode state as used by the integration rule
  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto value(const TimeInfo& /*t*/, const T1& field_new, const T2& /*field_old*/) const
  {
    return field_new;
  }

  /// @brief evaluate time derivative discretization of the ode state as used by the integration rule
  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto derivative(const TimeInfo& t, const T1& field_new, const T2& field_old) const
  {
    return (1.0 / t.dt()) * (field_new - field_old);
  }
};

/// @brief Options for second order time integration methods
enum class SecondOrderTimeIntegrationMethod
{
  IMPLICIT_NEWMARK,  /// implicit newmark discretization
  QUASI_STATIC  /// quasi-static, specifies current field, velocity is central difference (for quasi-static artificial
                /// viscosity), and acceleration is lagged for cases where it is set to some fixed value in time.
};

/// @brief encodes rules for time discretizing second order odes (involving first and second time derivatives).
/// When solving f(u, u_dot, u_dot_dot, t) = 0
/// this class provides the current discrete approximation for u, u_dot, and u_dot_dot as a function of
/// (u^{n+1},u^n,u_dot^n,u_dot_dot^n).
struct SecondOrderTimeIntegrationRule {
  /// @brief Constructor
  SecondOrderTimeIntegrationRule(SecondOrderTimeIntegrationMethod method) : method_(method) {}

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
    return (2.0 / t.dt()) * (field_new - field_old) - velo_old;
  }

  /// @brief evaluate time derivative discretization of the ode state as used by the integration rule
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto second_derivative([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                                           [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                                           [[maybe_unused]] const T4& accel_old) const
  {
    auto dt = t.dt();
    return (4.0 / (dt * dt)) * (field_new - field_old) - (4.0 / dt) * velo_old - accel_old;
  }

  SecondOrderTimeIntegrationMethod method_;  ///< method specifying time integration rule to inject into the q-function.
};

}  // namespace smith
