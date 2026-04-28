// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"
#include "mfem.hpp"

// In these tests std::algorithms should only be used once -
// naive for-loops should be used everywhere else in a given
// test to ensure only the single use of the algorithm is tested

TEST(ArrayAlgo, StdTransform)
{
  constexpr int size = 10;
  mfem::Array<int> input(size);
  for (int i = 0; i < size; i++) {
    input[i] = i;
  }
  mfem::Array<int> output(size);
  std::transform(input.begin(), input.end(), output.begin(), [](int arg) { return 2 * arg; });
  for (int i = 0; i < size; i++) {
    ASSERT_EQ(output[i], input[i] * 2);
  }
}

TEST(ArrayAlgo, StdAccumulate)
{
  constexpr int size = 10;
  mfem::Array<int> input(size);
  int sum = 0;
  for (int i = 0; i < size; i++) {
    input[i] = i;
    sum += i;
  }
  int accumulated = std::accumulate(input.begin(), input.end(), 0);
  ASSERT_EQ(sum, accumulated);
}
