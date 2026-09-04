// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <vector>

#include <gtest/gtest.h>
#include "smith/numerics/steihaug_toint_cg.hpp"

namespace {

class NegativeIdentityPreconditioner : public mfem::Solver {
 public:
  explicit NegativeIdentityPreconditioner(int size) : mfem::Solver(size) {}

  void SetOperator(const mfem::Operator& op) override { height = width = op.Height(); }

  void Mult(const mfem::Vector& input, mfem::Vector& output) const override
  {
    output = input;
    output *= -1.0;
  }
};

std::vector<double> localDotMany(const std::vector<smith::DotPair>& pairs)
{
  std::vector<double> products(pairs.size(), 0.0);
  for (size_t i = 0; i < pairs.size(); ++i) {
    products[i] = (*pairs[i].first) * (*pairs[i].second);
  }
  return products;
}

}  // namespace

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

  smith::steihaugTointCG(r0, rCurrent, H, nullptr, settings, trSize, results, r0.Norml2() * r0.Norml2(), localDotMany);

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

  smith::steihaugTointCG(r0, rCurrent, H, nullptr, settings, trSize, results, r0.Norml2() * r0.Norml2(), localDotMany);

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

  smith::steihaugTointCG(r0, rCurrent, H, nullptr, settings, trSize, results, r0.Norml2() * r0.Norml2(), localDotMany);

  // For negative curvature, it should go to boundary
  EXPECT_NEAR(results.z.Norml2(), 2.0, 1e-9);
  EXPECT_EQ(results.interior_status, smith::TrustRegionResults::Status::NegativeCurvature);
}

TEST(SteihaugTointCG, LowInitialResidualHonorsMinimumIterations)
{
  mfem::Vector diagonal(1);
  diagonal[0] = 1.0;
  mfem::SparseMatrix hessian(diagonal);

  smith::TrustRegionSettings settings;
  settings.cg_tol = 1.0;
  settings.max_cg_iterations = 10;

  double trust_region_size = 1.0;
  mfem::Vector current_residual(1);

  mfem::Vector low_residual(1);
  low_residual[0] = 1.0e-8;
  smith::TrustRegionResults early_exit_results(1);
  early_exit_results.z = 2.0;
  smith::steihaugTointCG(low_residual, current_residual, hessian, nullptr, settings, trust_region_size,
                         early_exit_results, low_residual * low_residual, localDotMany);

  EXPECT_EQ(early_exit_results.cg_iterations_count, 0);
  EXPECT_EQ(early_exit_results.interior_status, smith::TrustRegionResults::Status::Interior);
  EXPECT_DOUBLE_EQ(early_exit_results.z[0], 0.0);

  settings.min_cg_iterations = 1;
  smith::TrustRegionResults forced_iteration_results(1);
  smith::steihaugTointCG(low_residual, current_residual, hessian, nullptr, settings, trust_region_size,
                         forced_iteration_results, low_residual * low_residual, localDotMany);

  EXPECT_EQ(forced_iteration_results.cg_iterations_count, 1);
  EXPECT_EQ(forced_iteration_results.interior_status, smith::TrustRegionResults::Status::Interior);
}

TEST(SteihaugTointCG, FallsBackFromNonPositivePreconditioner)
{
  mfem::Vector diagonal(2);
  diagonal[0] = 2.0;
  diagonal[1] = 4.0;
  mfem::SparseMatrix hessian(diagonal);
  NegativeIdentityPreconditioner preconditioner(2);

  mfem::Vector initial_residual(2);
  initial_residual = 1.0;
  mfem::Vector current_residual(2);

  smith::TrustRegionSettings settings;
  settings.cg_tol = 1.0e-10;
  settings.max_cg_iterations = 10;

  double trust_region_size = 100.0;
  smith::TrustRegionResults results(2);
  smith::steihaugTointCG(initial_residual, current_residual, hessian, &preconditioner, settings, trust_region_size,
                         results, initial_residual * initial_residual, localDotMany);

  EXPECT_NEAR(results.z[0], -0.5, 1.0e-9);
  EXPECT_NEAR(results.z[1], -0.25, 1.0e-9);
  EXPECT_EQ(results.interior_status, smith::TrustRegionResults::Status::Interior);
}

TEST(TrustRegionDogleg, SelectsFeasibleNewtonPointWhenCauchyPointIsLonger)
{
  mfem::Vector cauchy_point(2);
  cauchy_point[0] = 2.0;
  cauchy_point[1] = 0.0;

  mfem::Vector newton_point(2);
  newton_point[0] = 0.5;
  newton_point[1] = 0.25;

  mfem::Vector step(2);
  smith::doglegStep(cauchy_point, newton_point, 10.0, step, localDotMany);

  EXPECT_DOUBLE_EQ(step[0], newton_point[0]);
  EXPECT_DOUBLE_EQ(step[1], newton_point[1]);
}
