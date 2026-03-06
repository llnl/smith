// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>

#include <cuda_runtime.h>
#include "gtest/gtest.h"

void vector_add(float* out, float* a, float* b, int n);

TEST(CudaSmoketest, VecAdd)
{
  constexpr int N = 10;

  float a[N];
  float b[N];
  float out[N];

  std::fill(a, a + N, 2.0f);
  std::fill(b, b + N, 4.0f);

  vector_add(out, a, b, N);

  std::for_each(out, out + N, [](const float f) { EXPECT_DOUBLE_EQ(f, 6.0); });
}
