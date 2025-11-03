// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file vector_state.hpp
 */

#pragma once

#include <vector>
#include "state.hpp"

namespace gretl {

using Vector = std::vector<double>;  ///< using for gretl::Vector
using VectorState = State<Vector>;   ///< using for gretl::VectorState

VectorState testing_update(const VectorState& a);  ///< an arbitrary evaluation function which takes a VectorState and
                                                   ///< nonlinearly returns another VectorState

VectorState copy(const VectorState& a);  ///< copies an existing VectorState

VectorState operator+(const VectorState& a, const VectorState& b);  ///< addition operator
VectorState operator*(const VectorState& a, double b);              ///< multiplication operator
VectorState operator*(double b, const VectorState& a);              ///< multiplication operator
VectorState operator*(const VectorState& a, const VectorState& b);  ///< compinent-wise multiplication operator

State<double> inner_product(const VectorState& a, const VectorState& b);  ///< inner product between VectorStates

namespace vec {

/// @brief default InitializeZeroDual for VectorState
static gretl::InitializeZeroDual<Vector, Vector> initialize_zero_dual = [](const Vector& from) {
  Vector to(from.size(), 0.0);
  return to;
};

}  // namespace vec

/// @brief gets size of the first vector in a vector of vectors and checks that all inner vectors have the same size
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
