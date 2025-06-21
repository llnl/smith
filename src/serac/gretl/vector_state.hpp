// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file upstream_state.hpp
 */

#pragma once

#include <vector>
#include "state.hpp"

namespace gretl {

using Vector = std::vector<double>;
using VectorState = State<Vector>;

VectorState testing_update(const VectorState& a);

VectorState copy(const VectorState& a);

VectorState operator+(const VectorState& a, const VectorState& b);
VectorState operator*(const VectorState& a, double b);
VectorState operator*(double b, const VectorState& a);
VectorState operator*(const VectorState& a, const VectorState& b);
// VectorState operator/(const VectorState& a, const VectorState& b);

State<double> inner_product(const VectorState& a, const VectorState& b);

namespace vec {

static gretl::InitializeZeroDual<Vector, Vector> initialize_zero_dual = [](const Vector& from) {
  Vector to(from.size(), 0.0);
  return to;
};

}  // namespace vec

template <typename T>
size_t get_same_size(const std::vector<const std::vector<T>*>& vs)
{
  size_t size = vs[0]->size();
  for (size_t n = 1; n < vs.size(); ++n) {
    gretl_assert(size == vs[n]->size());
  }
  return size;
}
}  // namespace gretl