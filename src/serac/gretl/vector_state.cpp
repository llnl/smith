// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "vector_state.hpp"
#include <iostream>

namespace gretl {

VectorState testing_update(const VectorState& a)
{
  VectorState b = a.clone({a});

  b.set_eval([](const UpstreamStates& upstreams, DownstreamState& downstream) {
    const auto& a_ = upstreams[0];
    auto& b_ = downstream;

    const Vector& A = a_.get<Vector>();
    size_t sz = A.size();
    Vector B(sz);
    for (size_t i = 0; i < sz; ++i) {
      B[i] = A[i] / 3.0 + 2.0;
    }
    b_.set(std::move(B));
  });

  b.set_vjp([](UpstreamStates& upstreams, const DownstreamState& downstream) {
    Vector& Abar = upstreams[0].get_dual<Vector, Vector>();
    const Vector& Bbar = downstream.get_dual<Vector, Vector>();
    const size_t sz = Bbar.size();

    for (size_t i = 0; i < sz; ++i) {
      Abar[i] += Bbar[i] / 3.0;
    }
  });

  return b.finalize();
}

VectorState operator+(const VectorState& a, const VectorState& b)
{
  VectorState c = a.clone({a, b});

  c.set_eval([](const UpstreamStates& upstreams, DownstreamState& downstream) {
    Vector C = upstreams[0].get<Vector>();  // just making a copy
    const Vector& B = upstreams[1].get<Vector>();
    size_t sz = C.size();
    for (size_t i = 0; i < sz; ++i) {
      C[i] += B[i];
    }
    downstream.set(std::move(C));
  });

  c.set_vjp([](UpstreamStates& upstreams, const DownstreamState& downstream) {
    const Vector& Cbar = downstream.get_dual<Vector, Vector>();
    size_t sz = Cbar.size();

    Vector& Abar = upstreams[0].get_dual<Vector, Vector>();
    for (size_t i = 0; i < sz; ++i) {
      Abar[i] += Cbar[i];
    }

    Vector& Bbar = upstreams[1].get_dual<Vector, Vector>();
    for (size_t i = 0; i < sz; ++i) {
      Bbar[i] += Cbar[i];
    }
  });

  return c.finalize();
}

VectorState operator*(const VectorState& a, double b)
{
  VectorState c = a.clone({a});

  c.set_eval([b](const UpstreamStates& upstreams, DownstreamState& downstream) {
    Vector C = upstreams[0].get<Vector>();
    for (auto&& v : C) {
      v *= b;
    }
    downstream.set(std::move(C));
  });

  c.set_vjp([b](UpstreamStates& upstreams, const DownstreamState& downstream) {
    const Vector& Cbar = downstream.get_dual<Vector, Vector>();
    Vector& Abar = upstreams[0].get_dual<Vector, Vector>();
    for (size_t i = 0; i < Abar.size(); ++i) {
      Abar[i] += b * Cbar[i];
    }
  });

  return c.finalize();
}

VectorState operator*(double b, const VectorState& a) { return a * b; }

State<double> inner_product(const VectorState& a, const VectorState& b)
{
  State<double> c = a.create_state<double>({a, b});

  c.set_eval([](const UpstreamStates& upstreams, DownstreamState& downstream) {
    double prod = 0.0;
    auto A = upstreams[0].get<Vector>();
    auto B = upstreams[1].get<Vector>();
    size_t sz = get_same_size<double>({&A, &B});
    for (size_t i = 0; i < sz; ++i) {
      prod += A[i] * B[i];
    }
    downstream.set(prod);
  });

  c.set_vjp([](UpstreamStates& upstreams, const DownstreamState& downstream) {
    double Cbar = downstream.get_dual<double, double>();

    auto a_ = upstreams[0];
    auto b_ = upstreams[1];

    const Vector& A = a_.get<Vector>();
    const Vector& B = b_.get<Vector>();
    size_t sz = get_same_size<double>({&A, &B});

    Vector& Abar = a_.get_dual<Vector, Vector>();
    for (size_t i = 0; i < sz; ++i) {
      Abar[i] += B[i] * Cbar;
    }

    Vector& Bbar = b_.get_dual<Vector, Vector>();
    for (size_t i = 0; i < sz; ++i) {
      Bbar[i] += A[i] * Cbar;
    }
  });

  return c.finalize();
}

VectorState operator*(const VectorState& a, const VectorState& b)
{
  VectorState c = a.clone({a, b});

  c.set_eval([](const UpstreamStates& upstreams, DownstreamState& downstream) {
    Vector C = upstreams[0].get<Vector>();
    const Vector& B = upstreams[1].get<Vector>();
    size_t sz = get_same_size<double>({&B, &C});
    for (size_t i = 0; i < sz; ++i) {
      C[i] *= B[i];
    }
    downstream.set(std::move(C));
  });

  c.set_vjp([](UpstreamStates& upstreams, const DownstreamState& downstream) {
    const Vector& Cbar = downstream.get_dual<Vector, Vector>();

    auto a_ = upstreams[0];
    auto b_ = upstreams[1];

    const Vector& A = a_.get<Vector>();
    const Vector& B = b_.get<Vector>();
    size_t sz = get_same_size<double>({&A, &B});

    Vector& Abar = a_.get_dual<Vector, Vector>();
    for (size_t i = 0; i < sz; ++i) {
      Abar[i] += B[i] * Cbar[i];
    }

    Vector& Bbar = b_.get_dual<Vector, Vector>();
    for (size_t i = 0; i < sz; ++i) {
      Bbar[i] += A[i] * Cbar[i];
    }
  });

  return c.finalize();
}

VectorState copy(const VectorState& a)
{
  VectorState b = a.clone({a});

  b.set_eval(
      [](const UpstreamStates& upstreams, DownstreamState& downstream) { downstream.set(upstreams[0].get<Vector>()); });

  b.set_vjp([](UpstreamStates& upstreams, const DownstreamState& downstream) {
    const Vector& Bbar = downstream.get_dual<Vector, Vector>();
    auto a_ = upstreams[0];
    Vector& Abar = a_.get_dual<Vector, Vector>();
    for (size_t i = 0; i < Abar.size(); ++i) {
      Abar[i] += Bbar[i];
    }
  });

  return b.finalize();
}

}  // namespace gretl
