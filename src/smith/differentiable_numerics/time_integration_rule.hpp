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
#include "smith/differentiable_numerics/field_state.hpp"

namespace smith {

class TimeIntegrationRule {
 public:
  virtual ~TimeIntegrationRule() {}
  virtual FieldState corrected_value(const TimeInfo& t, const std::vector<FieldState>& states) const = 0;

  virtual FieldState corrected_dot(const TimeInfo& t, const std::vector<FieldState>& states) const = 0;

  virtual FieldState corrected_ddot(const TimeInfo& t, const std::vector<FieldState>& states) const = 0;
};

/// @brief encodes rules for time discretizing first order odes (involving first time derivatives).
/// When solving f(u, u_dot, t) = 0
/// this class provides the current discrete approximation for u and u_dot as a function of
/// (u^{n+1}, u^n).
class BackwardEulerFirstOrderTimeIntegrationRule : public TimeIntegrationRule {
 public:
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
  SMITH_HOST_DEVICE auto dot(const TimeInfo& t, const T1& field_new, const T2& field_old) const
  {
    return (1.0 / t.dt()) * (field_new - field_old);
  }

  FieldState corrected_value(const TimeInfo& t, const std::vector<FieldState>& states) const override
  {
    return value(t, states[0], states[1]);
  }

  FieldState corrected_dot(const TimeInfo& t, const std::vector<FieldState>& states) const override
  {
    return dot(t, states[0], states[1]);
  }

  FieldState corrected_ddot(const TimeInfo& /*t*/, const std::vector<FieldState>& states) const override
  {
    SLIC_ERROR("BackwardEulerFirstOrderTimeIntegrationRule does not support second derivatives.");
    return states[0];
  }
};

/// Altenative name for Backward Euler which makes sense when restricting what are typically second order odes,
/// for example transient solid mechanics, to the quasi-static approximation.  It happens that the implementation is
/// identical to backward-Euler applied to first order systems as we want to be able to capture current velocity
/// dependencies.
using QuasiStaticFirstOrderTimeIntegrationRule = BackwardEulerFirstOrderTimeIntegrationRule;

/// @brief encodes rules for time discretizing second order odes (involving first and second time derivatives).
/// When solving f(u, u_dot, u_dot_dot, t) = 0
/// this class provides the current discrete approximation for u, u_dot, and u_dot_dot as a function of
/// (u^{n+1},u^n,u_dot^n,u_dot_dot^n).
struct ImplicitNewmarkSecondOrderTimeIntegrationRule : public TimeIntegrationRule {
 public:
  /// @brief Constructor
  ImplicitNewmarkSecondOrderTimeIntegrationRule() {}

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
  SMITH_HOST_DEVICE auto dot([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                             [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                             [[maybe_unused]] const T4& accel_old) const
  {
    return (2.0 / t.dt()) * (field_new - field_old) - velo_old;
  }

  /// @brief evaluate time derivative discretization of the ode state as used by the integration rule
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto ddot([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                              [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                              [[maybe_unused]] const T4& accel_old) const
  {
    auto dt = t.dt();
    return (4.0 / (dt * dt)) * (field_new - field_old) - (4.0 / dt) * velo_old - accel_old;
  }

  FieldState corrected_value(const TimeInfo& t, const std::vector<FieldState>& states) const override
  {
    return value(t, states[0], states[1], states[2], states[3]);
  }

  FieldState corrected_dot(const TimeInfo& t, const std::vector<FieldState>& states) const override
  {
    return dot(t, states[0], states[1], states[2], states[3]);
  }

  FieldState corrected_ddot(const TimeInfo& t, const std::vector<FieldState>& states) const override
  {
    return ddot(t, states[0], states[1], states[2], states[3]);
  }
};

/// @brief encodes rules for time discretizing second order odes (involving first and second time derivatives).
/// When solving f(u, u_dot, u_dot_dot, t) = 0
/// this class provides the current discrete approximation for u, u_dot, and u_dot_dot as a function of
/// (u^{n+1},u^n,u_dot^n,u_dot_dot^n).
struct QuasiStaticSecondOrderTimeIntegrationRule : public TimeIntegrationRule {
 public:
  /// @brief Constructor
  QuasiStaticSecondOrderTimeIntegrationRule() {}

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
  SMITH_HOST_DEVICE auto dot([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                             [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                             [[maybe_unused]] const T4& accel_old) const
  {
    return (1.0 / t.dt()) * (field_new - field_old);
  }

  /// @brief evaluate time derivative discretization of the ode state as used by the integration rule
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto ddot([[maybe_unused]] const TimeInfo& t, [[maybe_unused]] const T1& field_new,
                              [[maybe_unused]] const T2& field_old, [[maybe_unused]] const T3& velo_old,
                              [[maybe_unused]] const T4& accel_old) const
  {
    return accel_old;
  }

  FieldState corrected_value(const TimeInfo& t, const std::vector<FieldState>& states) const override
  {
    return value(t, states[0], states[1], states[2], states[3]);
  }

  FieldState corrected_dot(const TimeInfo& t, const std::vector<FieldState>& states) const override
  {
    return dot(t, states[0], states[1], states[2], states[3]);
  }

  FieldState corrected_ddot(const TimeInfo& t, const std::vector<FieldState>& states) const override
  {
    return ddot(t, states[0], states[1], states[2], states[3]);
  }
};

}  // namespace smith
