// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "smith/numerics/steihaug_toint_cg.hpp"

TEST(SteihaugTointCG, SolvesSPDInsideBoundary)
{
  int size = 2;
  mfem::Vector diag(size);
  diag[0] = 2.0;
  diag[1] = 4.0;
  mfem::SparseMatrix H(diag);

  mfem::Vector r0(size);
  r0[0] = 1.0;
  r0[1] = 1.0;

  smith::TrustRegionSettings settings;
  settings.cg_tol = 1e-10;
  settings.max_cg_iterations = 10;

  double trSize = 100.0;  // Huge trust region
  smith::TrustRegionResults results(size);

  mfem::Vector rCurrent(size);

  smith::steihaugTointCG(r0, rCurrent, H, nullptr, settings, trSize, results, r0.Norml2() * r0.Norml2(),
                         smith::dotMany);

  // Solution should be H^{-1} (-r0)
  // x = -0.5, y = -0.25
  EXPECT_NEAR(results.z[0], -0.5, 1e-9);
  EXPECT_NEAR(results.z[1], -0.25, 1e-9);
  EXPECT_EQ(results.interior_status, smith::TrustRegionResults::Status::Interior);
}

TEST(SteihaugTointCG, HitsBoundary)
{
  int size = 1;
  mfem::Vector diag(size);
  diag[0] = 1.0;
  mfem::SparseMatrix H(diag);

  mfem::Vector r0(size);
  r0[0] = 1.0;

  smith::TrustRegionSettings settings;
  settings.max_cg_iterations = 10;

  double trSize = 0.5;  // Small trust region, solution would be -1.0
  smith::TrustRegionResults results(size);

  mfem::Vector rCurrent(size);

  smith::steihaugTointCG(r0, rCurrent, H, nullptr, settings, trSize, results, r0.Norml2() * r0.Norml2(),
                         smith::dotMany);

  EXPECT_NEAR(results.z.Norml2(), 0.5, 1e-9);
  EXPECT_EQ(results.interior_status, smith::TrustRegionResults::Status::OnBoundary);
}

TEST(SteihaugTointCG, DetectsNegativeCurvature)
{
  int size = 1;
  mfem::Vector diag(size);
  diag[0] = -1.0;  // Negative curvature
  mfem::SparseMatrix H(diag);

  mfem::Vector r0(size);
  r0[0] = 1.0;

  smith::TrustRegionSettings settings;
  settings.max_cg_iterations = 10;

  double trSize = 2.0;
  smith::TrustRegionResults results(size);

  mfem::Vector rCurrent(size);

  smith::steihaugTointCG(r0, rCurrent, H, nullptr, settings, trSize, results, r0.Norml2() * r0.Norml2(),
                         smith::dotMany);

  // For negative curvature, it should go to boundary
  EXPECT_NEAR(results.z.Norml2(), 2.0, 1e-9);
  EXPECT_EQ(results.interior_status, smith::TrustRegionResults::Status::NegativeCurvature);
}
