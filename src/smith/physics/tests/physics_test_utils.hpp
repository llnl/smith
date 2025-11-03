// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file physics_test_utils.hpp
 *
 * @brief Common functions used in multiple physics unit tests
 */

#pragma once

#include <vector>
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

void pseudoRand(smith::FiniteElementVector& dual)
{
  int sz = dual.Size();
  for (int i = 0; i < sz; ++i) {
    dual(i) = -1.2 + 2.02 * (double(i) / sz);
  }
}
