// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "vector_state.hpp"

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

    assert(B.size() == 0);
  });

  b.set_vjp([](UpstreamStates& upstreams, const DownstreamState& downstream) {
    auto& a_ = upstreams[0];
    const auto& b_ = downstream;

    const Vector& Bbar = b_.get_dual<Vector>();
    size_t sz = Bbar.size();

    if (a_.dual_valid()) {
      Vector& Abar = a_.get_dual<Vector>();
      for (size_t i = 0; i < sz; ++i) {
        Abar[i] += Bbar[i] / 3.0;
      }
    } else {
      Vector Abar(sz);
      for (size_t i = 0; i < sz; ++i) {
        Abar[i] = Bbar[i] / 3.0;
      }
      a_.set_dual(std::move(Abar));

      assert(Abar.size() == 0);
    }
  });

  return b.finalize();
}

VectorState operator+(const VectorState& a, const VectorState& b)
{
  VectorState c = a.clone({a, b});

  c.set_eval([](const UpstreamStates& upstreams, DownstreamState& downstream) {
    auto a_ = upstreams[0];
    auto b_ = upstreams[1];
    Vector C = a_.get<Vector>();  // copy from a
    const Vector& B = b_.get<Vector>();
    size_t sz = C.size();
    for (size_t i = 0; i < sz; ++i) {
      C[i] += B[i];
    }
    downstream.set(std::move(C));
  });

  c.set_vjp([](UpstreamStates& upstreams, const DownstreamState& downstream) {
    const Vector& Cbar = downstream.get_dual<Vector>();
    auto a_ = upstreams[0];
    auto b_ = upstreams[1];

    if (a_.dual_valid()) {
      Vector& Abar = a_.get_dual<Vector>();
      for (size_t i = 0; i < Abar.size(); ++i) {
        Abar[i] += Cbar[i];
      }
    } else {
      a_.set_dual(Cbar);
    }

    if (b_.dual_valid()) {
      Vector& Bbar = b_.get_dual<Vector>();
      for (size_t i = 0; i < Bbar.size(); ++i) {
        Bbar[i] += Cbar[i];
      }
    } else {
      b_.set_dual(Cbar);
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
    const Vector& Cbar = downstream.get_dual<Vector>();
    auto& a_ = upstreams[0];
    if (a_.dual_valid()) {
      Vector& Abar = a_.get_dual<Vector>();
      for (size_t i = 0; i < Abar.size(); ++i) {
        Abar[i] += b * Cbar[i];
      }
    } else {
      Vector Abar = Cbar;
      for (auto&& v : Abar) {
        v *= b;
      }
      a_.set_dual(std::move(Abar));
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
    double Cbar = downstream.get_dual<double>();

    auto& a_ = upstreams[0];
    auto& b_ = upstreams[1];

    const Vector& A = a_.get<Vector>();
    const Vector& B = b_.get<Vector>();
    size_t sz = get_same_size<double>({&A, &B});

    if (a_.dual_valid()) {
      Vector& Abar = a_.get_dual<Vector>();
      for (size_t i = 0; i < sz; ++i) {
        Abar[i] += B[i] * Cbar;
      }
    } else {
      Vector Abar(sz);
      for (size_t i = 0; i < sz; ++i) {
        Abar[i] = B[i] * Cbar;
      }
      a_.set_dual(std::move(Abar));
      assert(Abar.empty());
    }

    if (b_.dual_valid()) {
      Vector& Bbar = b_.get_dual<Vector>();
      for (size_t i = 0; i < sz; ++i) {
        Bbar[i] += A[i] * Cbar;
      }
    } else {
      Vector Bbar(sz);
      for (size_t i = 0; i < sz; ++i) {
        Bbar[i] = A[i] * Cbar;
      }
      b_.set_dual(std::move(Bbar));
      assert(Bbar.empty());
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
    const Vector& Bbar = downstream.get_dual<Vector>();
    auto& a_ = upstreams[0];
    if (a_.dual_valid()) {
      Vector& Abar = a_.get_dual<Vector>();
      for (size_t i = 0; i < Abar.size(); ++i) {
        Abar[i] += Bbar[i];
      }
    } else {
      a_.set_dual(Bbar);
    }
  });

  return b.finalize();
}

namespace vec {
Vector zero_clone::operator()(const Vector& from) { return Vector(from.size(), 0.0); }
};  // namespace vec

}  // namespace gretl