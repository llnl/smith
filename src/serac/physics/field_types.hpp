// Copyright Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file residual_types.hpp
 *
 * @brief Defines common types and helper functions for using the residual and scalar_objective classes
 */

#pragma once

#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

namespace serac {

/// @brief using
using FieldPtr = FiniteElementState*;

/// @brief using
using DualFieldPtr = FiniteElementDual*;

/// @brief using
using ConstFieldPtr = FiniteElementState const*;

/// @brief using
using ConstDualFieldPtr = FiniteElementDual const*;

  template <typename T>
auto residualPointers(std::vector<std::shared_ptr<T>>& states, std::vector<std::shared_ptr<T>>& params)
{
  std::vector<T*> pointers;
  for (auto& t : states) {
    pointers.push_back(t.get());
  }
  for (auto& t : params) {
    pointers.push_back(t.get());
  }
  return pointers;
}

template <typename T>
auto residualPointers(std::vector<T>& states, std::vector<T>& params)
{
  std::vector<T*> pointers;
  for (auto& t : states) {
    pointers.push_back(&t);
  }
  for (auto& t : params) {
    pointers.push_back(&t);
  }
  return pointers;
}

template <typename T>
auto residualPointers(T& state)
{
  return std::vector<T*>{&state};
}

template <typename T>
auto constResidualPointers(std::vector<std::shared_ptr<T>>& states, std::vector<std::shared_ptr<T>>& params)
{
  std::vector<T const*> pointers;
  for (auto& t : states) {
    pointers.push_back(t.get());
  }
  for (auto& t : params) {
    pointers.push_back(t.get());
  }
  return pointers;
}

template <typename T>
auto constResidualPointers(const std::vector<T>& states, const std::vector<T>& params = {})
{
  std::vector<T const*> pointers;
  for (auto& t : states) {
    pointers.push_back(&t);
  }
  for (auto& t : params) {
    pointers.push_back(&t);
  }
  return pointers;
}

template <typename T>
auto constResidualPointers(const T& state)
{
  return std::vector<T const*>{&state};
}

}