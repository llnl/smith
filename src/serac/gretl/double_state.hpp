#pragma once

#include "vector_state.hpp"
#include "serac/gretl/create_state.hpp"

namespace gretl {

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
    const double& Z_dual = downstream.get_dual<double, double>();
    double& X_dual = upstreams[0].get_dual<double, double>();
    double& Y_dual = upstreams[1].get_dual<double, double>();
    X_dual += a * Z_dual;
    Y_dual += b * Z_dual;
  });

  return z.finalize();
}

inline State<double> axpb(double a, const State<double>& x, double b)
{
  auto z = x.clone({x});

  z.set_eval([a, b](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const double& X = upstreams[0].get<double>();
    double Z = a * X + b;
    downstream.set(Z);
  });

  z.set_vjp([a](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const double& Z_dual = downstream.get_dual<double, double>();
    double& X_dual = upstreams[0].get_dual<double, double>();
    X_dual += a * Z_dual;
  });

  return z.finalize();
}

inline State<double> operator+(const State<double>& x, const State<double>& y) { return axpby(1.0, x, 1.0, y); }

inline State<double> operator-(const State<double>& x, const State<double>& y) { return axpby(1.0, x, -1.0, y); }

inline State<double> operator+(const State<double>& x, double b) { return axpb(1.0, x, b); }

inline State<double> operator+(double b, const State<double>& x) { return axpb(1.0, x, b); }

inline State<double> operator-(const State<double>& x, double b) { return axpb(1.0, x, -b); }

inline State<double> operator-(double a, const State<double>& x) { return axpb(-1.0, x, a); }

inline State<double> operator*(double a, const State<double>& x) { return axpb(a, x, 0.0); }

inline State<double> operator*(const State<double>& x, double a) { return axpb(a, x, 0.0); }

inline State<double> operator/(const State<double>& x, double a) { return axpb(1.0 / a, x, 0.0); }

inline State<double> operator/(double a, const State<double>& x)
{
  auto z = x.clone({x});

  z.set_eval([a](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const double& X = upstreams[0].get<double>();
    downstream.set(a / X);
  });

  z.set_vjp([a](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const double& Z_dual = downstream.get_dual<double, double>();
    const double& X = upstreams[0].get<double>();
    upstreams[0].get_dual<double, double>() -= a * Z_dual / (X * X);
  });

  return z.finalize();
}

inline State<double> operator*(const State<double>& x, const State<double>& y)
{
  return clone_state([](const double& a, const double& b) { return a * b; },
                     [](const double& a, const double& b, const double&, double& a_, double& b_, const double& c_) {
                       a_ += b * c_;
                       b_ += a * c_;
                     },
                     x, y);
}

}  // namespace gretl