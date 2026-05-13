// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/steihaug_toint_cg.hpp"

namespace smith {

namespace {

void projectToBoundaryWithCoefs(mfem::Vector& z, const mfem::Vector& d, double delta, double zz, double zd, double dd)
{
  const double deltadelta_m_zz = delta * delta - zz;
  if (deltadelta_m_zz <= 0.0) return;
  const double tau = (std::sqrt(deltadelta_m_zz * dd + zd * zd) - zd) / dd;
  z.Add(tau, d);
}

}  // namespace

std::vector<double> dotMany(const std::vector<DotPair>& pairs)
{
  std::vector<double> products(pairs.size(), 0.0);
  for (size_t i = 0; i < pairs.size(); ++i) {
    MFEM_ASSERT(pairs[i].first->Size() == pairs[i].second->Size(), "Incompatible vector sizes.");
    products[i] = (*pairs[i].first) * (*pairs[i].second);
  }
  return products;
}

void steihaugTointCG(const mfem::Vector& r0, mfem::Vector& rCurrent, const mfem::Operator& H, const mfem::Solver* P,
                     const TrustRegionSettings& settings, double& trSize, TrustRegionResults& results,
                     double r0_norm_squared, const DotManyFunction& dot_many)
{
  // minimize r0@z + 0.5*z@J@z
  results.interior_status = TrustRegionResults::Status::Interior;
  results.cg_iterations_count = 0;

  auto& z = results.z;
  auto& cgIter = results.cg_iterations_count;
  auto& d = results.d;
  auto& Pr = results.Pr;
  auto& Hd = results.H_d;

  const double cg_tol_squared = settings.cg_tol * settings.cg_tol;

  if (r0_norm_squared <= cg_tol_squared && settings.min_cg_iterations == 0) {
    return;
  }

  rCurrent = r0;
  if (P) {
    P->Mult(rCurrent, Pr);
  } else {
    Pr = rCurrent;
  }

  // d = -Pr
  d = Pr;
  d *= -1.0;

  z = 0.0;
  double zz = 0.;

  // rPr = dot(rCurrent, Pr)
  double rPr = dot_many({{&rCurrent, &Pr}, {&rCurrent, &Pr}})[0];

  for (cgIter = 1; cgIter <= settings.max_cg_iterations; ++cgIter) {
    H.Mult(d, Hd);

    auto dots = dot_many({{&d, &rCurrent}, {&d, &Hd}, {&z, &d}, {&d, &d}});
    double descent_check = dots[0];
    double curvature = dots[1];
    double zd = dots[2];
    double dd = dots[3];

    if (descent_check > 0) {
      d *= -1;
      Hd *= -1;
      results.interior_status = TrustRegionResults::Status::NonDescentDirection;
      descent_check *= -1.0;
      curvature *= -1.0;
      zd *= -1.0;
    }

    const double alphaCg = curvature != 0.0 ? rPr / curvature : 0.0;
    const double zzNp1 = zz + 2.0 * alphaCg * zd + alphaCg * alphaCg * dd;

    const bool go_to_boundary = curvature <= 0 || zzNp1 >= trSize * trSize;
    if (go_to_boundary) {
      projectToBoundaryWithCoefs(z, d, trSize, zz, zd, dd);
      if (curvature <= 0) {
        results.interior_status = TrustRegionResults::Status::NegativeCurvature;
      } else {
        results.interior_status = TrustRegionResults::Status::OnBoundary;
      }
      return;
    }

    auto& zPred = Pr;
    zPred = z;
    zPred.Add(alphaCg, d);
    z = zPred;

    if (results.interior_status == TrustRegionResults::Status::NonDescentDirection) {
      return;
    }

    rCurrent.Add(alphaCg, Hd);

    if (P) {
      P->Mult(rCurrent, Pr);
    } else {
      Pr = rCurrent;
    }

    auto dots2 = dot_many({{&rCurrent, &Pr}, {&rCurrent, &rCurrent}});
    double rPrNp1 = dots2[0];
    double r_current_norm_squared = dots2[1];

    if (r_current_norm_squared <= cg_tol_squared && cgIter >= settings.min_cg_iterations) {
      return;
    }

    double beta = rPrNp1 / rPr;
    rPr = rPrNp1;
    d *= beta;
    d.Add(-1.0, Pr);

    zz = zzNp1;
  }
  cgIter--;
}

}  // namespace smith
