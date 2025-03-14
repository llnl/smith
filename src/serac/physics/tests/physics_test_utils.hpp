// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file utils_physics_tests.hpp
 *
 * @brief Common functions used in multiple physics unit tests
 */

#pragma once

#include <vector>
#include "serac/physics/state/finite_element_dual.hpp"
#include "serac/physics/state/finite_element_state.hpp"

template <typename T>
auto getPointers(std::vector<T>& values)
{
  std::vector<T*> pointers;
  for (auto& t : values) {
    pointers.push_back(&t);
  }
  return pointers;
}

template <typename T>
auto getPointers(std::vector<T>& states, std::vector<T>& params)
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
auto getPointers(T& v)
{
  return std::vector<T*>{&v};
}

void pseudoRand(serac::FiniteElementState& dual)
{
  int sz = dual.Size();
  for (int i = 0; i < sz; ++i) {
    dual(i) = -1.2 + 2.02 * (double(i) / sz);
  }
}

void pseudoRand(serac::FiniteElementDual& dual)
{
  int sz = dual.Size();
  for (int i = 0; i < sz; ++i) {
    dual(i) = -1.2 + 2.02 * (double(i) / sz);
  }
}