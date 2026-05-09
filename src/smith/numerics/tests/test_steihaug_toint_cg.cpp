// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "smith/numerics/steihaug_toint_cg.hpp"

namespace {

class TestDelegate : public smith::SteihaugTointDelegate {
public:
  std::array<double, 4> dot_many_4(const mfem::Vector& a0, const mfem::Vector& b0,
                                   const mfem::Vector& a1, const mfem::Vector& b1,
                                   const mfem::Vector& a2, const mfem::Vector& b2,
                                   const mfem::Vector& a3, const mfem::Vector& b3) const override
  {
    return {a0 * b0, a1 * b1, a2 * b2, a3 * b3};
  }

  std::array<double, 2> dot_many_2(const mfem::Vector& a0, const mfem::Vector& b0,
                                   const mfem::Vector& a1, const mfem::Vector& b1) const override
  {
    return {a0 * b0, a1 * b1};
  }

  void projectToBoundaryWithCoefs(mfem::Vector& z, const mfem::Vector& d, double delta, double zz, double zd,
                                  double dd) const override
  {
    double deltadelta_m_zz = delta * delta - zz;
    if (deltadelta_m_zz <= 0) return;
    double tau = (std::sqrt(deltadelta_m_zz * dd + zd * zd) - zd) / dd;
    z.Add(tau, d);
  }
};

class DiagonalOperator : public mfem::Operator {
public:
  DiagonalOperator(const mfem::Vector& diag) : mfem::Operator(diag.Size()), diag_(diag) {}
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    for (int i = 0; i < height; ++i) {
      y[i] = diag_[i] * x[i];
    }
  }
private:
  const mfem::Vector& diag_;
};

} // namespace

TEST(SteihaugTointCG, SolvesSPDInsideBoundary)
{
  int size = 2;
  mfem::Vector diag(size);
  diag[0] = 2.0;
  diag[1] = 4.0;
  DiagonalOperator H(diag);

  mfem::Vector r0(size);
  r0[0] = 1.0;
  r0[1] = 1.0;

  smith::TrustRegionSettings settings;
  settings.cg_tol = 1e-10;
  settings.max_cg_iterations = 10;
  
  double trSize = 100.0; // Huge trust region
  smith::TrustRegionResults results(size);
  
  mfem::Vector rCurrent(size);
  TestDelegate delegate;
  
  smith::steihaugTointCG(r0, rCurrent, H, nullptr, settings, trSize, results, r0.Norml2() * r0.Norml2(), delegate);
  
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
  DiagonalOperator H(diag);

  mfem::Vector r0(size);
  r0[0] = 1.0;

  smith::TrustRegionSettings settings;
  settings.max_cg_iterations = 10;
  
  double trSize = 0.5; // Small trust region, solution would be -1.0
  smith::TrustRegionResults results(size);
  
  mfem::Vector rCurrent(size);
  TestDelegate delegate;
  
  smith::steihaugTointCG(r0, rCurrent, H, nullptr, settings, trSize, results, r0.Norml2() * r0.Norml2(), delegate);
  
  EXPECT_NEAR(results.z.Norml2(), 0.5, 1e-9);
  EXPECT_EQ(results.interior_status, smith::TrustRegionResults::Status::OnBoundary);
}

TEST(SteihaugTointCG, DetectsNegativeCurvature)
{
  int size = 1;
  mfem::Vector diag(size);
  diag[0] = -1.0; // Negative curvature
  DiagonalOperator H(diag);

  mfem::Vector r0(size);
  r0[0] = 1.0;

  smith::TrustRegionSettings settings;
  settings.max_cg_iterations = 10;
  
  double trSize = 2.0;
  smith::TrustRegionResults results(size);
  
  mfem::Vector rCurrent(size);
  TestDelegate delegate;
  
  smith::steihaugTointCG(r0, rCurrent, H, nullptr, settings, trSize, results, r0.Norml2() * r0.Norml2(), delegate);
  
  // For negative curvature, it should go to boundary
  EXPECT_NEAR(results.z.Norml2(), 2.0, 1e-9);
  EXPECT_EQ(results.interior_status, smith::TrustRegionResults::Status::NegativeCurvature);
}
