// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

namespace {

double jbarResidual(double dpsi_djbar, double p, double J, double Jbar, double penalty)
{
  return dpsi_djbar - p - penalty * (J - Jbar);
}

double jbarEnergy(double dpsi_djbar, double p, double J, double Jbar, double penalty)
{
  return dpsi_djbar * Jbar + p * (J - Jbar) + 0.5 * penalty * (J - Jbar) * (J - Jbar);
}

double explicitUzawaResidual(double J, double Jbar, double penalty)
{
  return -penalty * (J - Jbar);
}

}  // namespace

TEST(AugmentedLagrangianUpdate, JbarResidualIsPositiveEnergyDerivative)
{
  constexpr double dpsi_djbar = 3.2;
  constexpr double p = -0.7;
  constexpr double J = 1.13;
  constexpr double Jbar = 0.94;
  constexpr double penalty = 11.0;
  constexpr double eps = 1.0e-7;

  const double finite_difference =
      (jbarEnergy(dpsi_djbar, p, J, Jbar + eps, penalty) -
       jbarEnergy(dpsi_djbar, p, J, Jbar - eps, penalty)) /
      (2.0 * eps);

  EXPECT_NEAR(jbarResidual(dpsi_djbar, p, J, Jbar, penalty), finite_difference, 1.0e-8);
  EXPECT_GT((jbarResidual(dpsi_djbar, p, J, Jbar + eps, penalty) -
             jbarResidual(dpsi_djbar, p, J, Jbar - eps, penalty)) /
                (2.0 * eps),
            0.0);
}

TEST(AugmentedLagrangianUpdate, ExplicitPressureResidualIncrementsCurrentUzawaIterate)
{
  constexpr double p_current = 0.4;
  constexpr double J = 0.97;
  constexpr double Jbar = 1.05;
  constexpr double penalty = 8.0;

  const double residual = explicitUzawaResidual(J, Jbar, penalty);
  const double p_new = p_current - residual;

  EXPECT_NEAR(p_new, p_current + penalty * (J - Jbar), 1.0e-14);
}
