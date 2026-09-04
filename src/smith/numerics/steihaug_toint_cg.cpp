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
  const double deltadelta_m_zz = std::max(delta * delta - zz, 0.0);
  if (deltadelta_m_zz == 0.0) return;
  const double tau = (std::sqrt(deltadelta_m_zz * dd + zd * zd) - zd) / dd;
  z.Add(tau, d);
}

void projectToBoundaryBetweenWithCoefs(mfem::Vector& z, const mfem::Vector& y, double delta, double zz, double zy,
                                       double yy)
{
  const double dd = yy - 2.0 * zy + zz;
  const double zd = zy - zz;
  const double boundary_gap = std::max(delta * delta - zz, 0.0);
  if (boundary_gap == 0.0) return;
  const double tau = (std::sqrt(boundary_gap * dd + zd * zd) - zd) / dd;
  z.Add(-tau, z);
  z.Add(tau, y);
}

}  // namespace

void doglegStep(const mfem::Vector& cauchy_point, const mfem::Vector& newton_point, double trust_region_size,
                mfem::Vector& step, const DotManyFunction& dot_many)
{
  const auto dots = dot_many({{&cauchy_point, &cauchy_point}, {&newton_point, &newton_point}});
  const double cauchy_norm_squared = dots[0];
  const double newton_norm_squared = dots[1];
  const double trust_region_size_squared = trust_region_size * trust_region_size;

  if (newton_norm_squared <= trust_region_size_squared) {
    step = newton_point;
  } else if (cauchy_norm_squared >= trust_region_size_squared) {
    step = cauchy_point;
    step *= std::sqrt(trust_region_size_squared / cauchy_norm_squared);
  } else {
    step = cauchy_point;
    const double cauchy_newton = dot_many({{&cauchy_point, &newton_point}})[0];
    projectToBoundaryBetweenWithCoefs(step, newton_point, trust_region_size, cauchy_norm_squared, cauchy_newton,
                                      newton_norm_squared);
  }
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

  z = 0.0;
  const double cg_tol_squared = settings.cg_tol * settings.cg_tol;

  if (r0_norm_squared <= cg_tol_squared && settings.min_cg_iterations == 0) {
    return;
  }

  rCurrent = r0;
  const mfem::Solver* active_preconditioner = P;
  if (active_preconditioner) {
    active_preconditioner->Mult(rCurrent, Pr);
  } else {
    Pr = rCurrent;
  }

  // rPr = dot(rCurrent, Pr)
  double rPr = dot_many({{&rCurrent, &Pr}})[0];
  if (!(rPr > 0.0)) {
    active_preconditioner = nullptr;
    Pr = rCurrent;
    rPr = r0_norm_squared;
  }

  // d = -Pr
  d = Pr;
  d *= -1.0;

  double zz = 0.;

  for (cgIter = 1; cgIter <= settings.max_cg_iterations; ++cgIter) {
    H.Mult(d, Hd);

    auto dots = dot_many({{&d, &rCurrent}, {&d, &Hd}, {&z, &d}, {&d, &d}});
    double descent_check = dots[0];
    double curvature = dots[1];
    double zd = dots[2];
    double dd = dots[3];

    if (descent_check >= 0.0) {
      results.interior_status = TrustRegionResults::Status::NonDescentDirection;
      return;
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

    // Alias Pr as temporary workspace 'zPred' to avoid allocation
    auto& zPred = Pr;
    zPred = z;
    zPred.Add(alphaCg, d);
    z = zPred;

    if (results.interior_status == TrustRegionResults::Status::NonDescentDirection) {
      return;
    }

    rCurrent.Add(alphaCg, Hd);

    if (active_preconditioner) {
      active_preconditioner->Mult(rCurrent, Pr);
    } else {
      Pr = rCurrent;
    }

    auto dots2 = dot_many({{&rCurrent, &Pr}, {&rCurrent, &rCurrent}});
    double rPrNp1 = dots2[0];
    double r_current_norm_squared = dots2[1];

    if (!(rPrNp1 > 0.0)) {
      active_preconditioner = nullptr;
      Pr = rCurrent;
      rPrNp1 = r_current_norm_squared;
    }

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
