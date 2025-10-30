// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file field_types.hpp
 *
 * @brief Defines common types and helper functions for using the residual and scalar_objective classes
 */

#pragma once

#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_dual.hpp"

namespace smith {

/// Polynomial order used to discretize the shape displacement field
constexpr int SHAPE_ORDER = 1;

/// Function space for shape displacement on dimension 2 meshes
constexpr H1<SHAPE_ORDER, 2> SHAPE_DIM_2;

/// Function space for shape displacement on dimension 2 meshes
constexpr H1<SHAPE_ORDER, 3> SHAPE_DIM_3;

/// @brief using
using FieldPtr = FiniteElementState*;

/// @brief using
using DualFieldPtr = FiniteElementDual*;

/// @brief using
using ConstFieldPtr = FiniteElementState const*;

/// @brief using
using ConstDualFieldPtr = FiniteElementDual const*;

/// @brief Get a vector of FieldPtr or DualFieldPtr from a vector of shared_pointers to FiniteElementState or
/// FiniteElementDual
template <typename T>
auto getFieldPointers(std::vector<std::shared_ptr<T>>& states, std::vector<std::shared_ptr<T>>& params)
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  std::vector<T*> pointers;
  for (auto& t : states) {
    pointers.push_back(t.get());
  }
  for (auto& t : params) {
    pointers.push_back(t.get());
  }
  return pointers;
}

/// @brief Get a vector of FieldPtr or DualFieldPtr from a vector of shared_pointers to FiniteElementState or
/// FiniteElementDual
template <typename T>
auto getFieldPointers(std::vector<std::shared_ptr<T>>& states)
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  std::vector<T*> pointers;
  for (auto& t : states) {
    pointers.push_back(t.get());
  }
  return pointers;
}

/// @brief Get a vector of FieldPtr or DualFieldPtr from a vector of FiniteElementState or FiniteElementDual
template <typename T>
auto getFieldPointers(std::vector<T>& states, std::vector<T>& params)
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  std::vector<T*> pointers;
  for (auto& t : states) {
    pointers.push_back(&t);
  }
  for (auto& t : params) {
    pointers.push_back(&t);
  }
  return pointers;
}

/// @brief Get a vector of FieldPtr or DualFieldPtr from a vector of FiniteElementState or FiniteElementDual
template <typename T>
auto getFieldPointers(std::vector<T>& states)
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  std::vector<T*> pointers;
  for (auto& t : states) {
    pointers.push_back(&t);
  }
  return pointers;
}

/// @brief Get a vector of FieldPtr or DualFieldPtr from a single shared_ptr<FiniteElementState> or
/// shared_ptr<FiniteElementDual>
template <typename T>
auto getFieldPointers(std::shared_ptr<T>& state)
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  return std::vector<T*>{state.get()};
}

/// @brief Get a vector of FieldPtr or DualFieldPtr from a single FiniteElementState or FiniteElementDual
template <typename T>
auto getFieldPointers(T& state)
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  return std::vector<T*>{&state};
}

/// @brief Get a vector of ConstFieldPtr or ConstDualFieldPtr from a vector of shared_pointers to FiniteElementState or
/// FiniteElementDual
template <typename T>
auto getConstFieldPointers(const std::vector<std::shared_ptr<T>>& states,
                           const std::vector<std::shared_ptr<T>>& params = {})
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  std::vector<T const*> pointers;
  for (auto& t : states) {
    pointers.push_back(t.get());
  }
  for (auto& t : params) {
    pointers.push_back(t.get());
  }
  return pointers;
}

/// @brief Get a vector of ConstFieldPtr or ConstDualFieldPtr from a vector of shared_pointers to FiniteElementState or
/// FiniteElementDual
template <typename T>
auto getConstFieldPointers(const std::vector<T*>& states, const std::vector<T*>& params = {})
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  std::vector<T const*> pointers;
  for (auto& t : states) {
    pointers.push_back(t);
  }
  for (auto& t : params) {
    pointers.push_back(t);
  }
  return pointers;
}

/// @brief Get a vector of ConstFieldPtr or ConstDualFieldPtr from a vector of FiniteElementState or FiniteElementDual
template <typename T>
auto getConstFieldPointers(const std::vector<T>& states, const std::vector<T>& params = {})
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  std::vector<T const*> pointers;
  for (auto& t : states) {
    pointers.push_back(&t);
  }
  for (auto& t : params) {
    pointers.push_back(&t);
  }
  return pointers;
}

/// @brief Get a vector of ConstFieldPtr or ConstDualFieldPtr from a single shared_ptr<FiniteElementState> or
/// shared_ptr<FiniteElementDual>
template <typename T>
auto getConstFieldPointers(const std::shared_ptr<T>& state)
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  return std::vector<T const*>{state.get()};
}

/// @brief Get a vector of ConstFieldPtr or ConstDualFieldPtr from a single FiniteElementState or FiniteElementDual
template <typename T>
auto getConstFieldPointers(const T& state)
{
  static_assert(std::is_same_v<T, FiniteElementState> || std::is_same_v<T, FiniteElementDual>,
                "Type must be either FiniteElementState or FiniteElementDual");
  return std::vector<T const*>{&state};
}

}  // namespace smith
