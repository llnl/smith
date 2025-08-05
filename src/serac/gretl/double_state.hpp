// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file double_state.hpp
 */

#pragma once

#include "vector_state.hpp"

namespace gretl {

/// @brief returns a * x + b * y, where x and y are double states
inline State<double> axpby(double a, const State<double>& x, double b, const State<double>& y)
{
  auto z = x.clone({x, y});

  z.set_eval([a, b](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const double& X = upstreams[0].get<double>();
    const double& Y = upstreams[1].get<double>();
    double Z = a * X + b * Y;
    downstream.set(Z);
  });

  z.set_vjp([a, b](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const double& Z_dual = downstream.get_dual<double>();
    double& X_dual = upstreams[0].get_dual<double>();
    double& Y_dual = upstreams[1].get_dual<double>();
    X_dual += a * Z_dual;
    Y_dual += b * Z_dual;
  });

  return z.finalize();
}

/// @brief returns a * x + b, where x is a double state
inline State<double> axpb(double a, const State<double>& x, double b)
{
  auto z = x.clone({x});

  z.set_eval([a, b](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const double& X = upstreams[0].get<double>();
    double Z = a * X + b;
    downstream.set(Z);
  });

  z.set_vjp([a](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const double& Z_dual = downstream.get_dual<double>();
    double& X_dual = upstreams[0].get_dual<double>();
    X_dual += a * Z_dual;
  });

  return z.finalize();
}

/// @brief add a double state with a double state
inline State<double> operator+(const State<double>& x, const State<double>& y) { return axpby(1.0, x, 1.0, y); }

/// @brief subtract a double state from a double state
inline State<double> operator-(const State<double>& x, const State<double>& y) { return axpby(1.0, x, -1.0, y); }

/// @brief add a double state with a double
inline State<double> operator+(const State<double>& x, double b) { return axpb(1.0, x, b); }

/// @brief add a double with a double state
inline State<double> operator+(double b, const State<double>& x) { return axpb(1.0, x, b); }

/// @brief subtract a double from a double state
inline State<double> operator-(const State<double>& x, double b) { return axpb(1.0, x, -b); }

/// @brief subtract a double state from a double
inline State<double> operator-(double a, const State<double>& x) { return axpb(-1.0, x, a); }

/// @brief multiply a double state with a double
inline State<double> operator*(double a, const State<double>& x) { return axpb(a, x, 0.0); }

/// @brief multiply a double state with a double
inline State<double> operator*(const State<double>& x, double a) { return axpb(a, x, 0.0); }

/// @brief divide a double state by a double
inline State<double> operator/(const State<double>& x, double a) { return axpb(1.0 / a, x, 0.0); }

/// @brief divide a double by a double double state
inline State<double> operator/(double a, const State<double>& x)
{
  auto z = x.clone({x});

  z.set_eval([a](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const double& X = upstreams[0].get<double>();
    downstream.set(a / X);
  });

  z.set_vjp([a](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const double& Z_dual = downstream.get_dual<double>();
    const double& X = upstreams[0].get<double>();
    upstreams[0].get_dual<double>() -= a * Z_dual / (X * X);
  });

  return z.finalize();
}

}  // namespace gretl