// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

template <typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);

void LinearQuadBasis(const double* xi, double* phi)
{
  phi[0] = 0.25 * (1 - xi[0]) * (1 - xi[1]);
  phi[1] = 0.25 * (1 + xi[0]) * (1 - xi[1]);
  phi[2] = 0.25 * (1 + xi[0]) * (1 + xi[1]);
  phi[3] = 0.25 * (1 - xi[0]) * (1 + xi[1]);
}

void LinearQuadBasisDeriv(const double* xi, double* phi, double* dphi_dxi, double* dphi_deta)
{
  double xi_dot[2] = {1.0, 0.0};
  __enzyme_fwddiff<void>(reinterpret_cast<void*>(LinearQuadBasis), xi, xi_dot, phi, dphi_dxi);
  xi_dot[0] = 0.0;
  xi_dot[1] = 1.0;
  __enzyme_fwddiff<void>(reinterpret_cast<void*>(LinearQuadBasis), xi, xi_dot, phi, dphi_deta);
}

TEST(enzyme_smoke, basic_use)
{
  double xi[2] = {0.2, -0.4};
  double phi[4] = {0.0, 0.0, 0.0, 0.0};
  double dphi_dxi[4] = {0.0, 0.0, 0.0, 0.0};
  double dphi_deta[4] = {0.0, 0.0, 0.0, 0.0};

  LinearQuadBasisDeriv(xi, phi, dphi_dxi, dphi_deta);

  EXPECT_EQ(dphi_dxi[0], -0.25 * (1.0 - xi[1]));
  EXPECT_EQ(dphi_deta[0], -0.25 * (1.0 - xi[0]));
  EXPECT_EQ(dphi_dxi[1], 0.25 * (1.0 - xi[1]));
  EXPECT_EQ(dphi_deta[1], -0.25 * (1.0 + xi[0]));
  EXPECT_EQ(dphi_dxi[2], 0.25 * (1.0 + xi[1]));
  EXPECT_EQ(dphi_deta[2], 0.25 * (1.0 + xi[0]));
  EXPECT_EQ(dphi_dxi[3], -0.25 * (1.0 + xi[1]));
  EXPECT_EQ(dphi_deta[3], 0.25 * (1.0 - xi[0]));
}
