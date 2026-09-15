// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/steihaug_toint_cg.hpp"

#include <cmath>
#include <limits>

namespace smith {

namespace {

double projectToBoundaryWithCoefs(mfem::Vector& z, const mfem::Vector& d, double delta, double zz, double zd, double dd)
{
  const double deltadelta_m_zz = std::max(delta * delta - zz, 0.0);
  if (deltadelta_m_zz == 0.0) return 0.0;
  const double tau = (std::sqrt(deltadelta_m_zz * dd + zd * zd) - zd) / dd;
  z.Add(tau, d);
  return tau;
}

}  // namespace

std::optional<TrustRegionModelCandidate> bestTrustRegionModelCandidate(const std::array<double, 4>& model_objectives,
                                                                       const std::array<bool, 4>& valid_candidates)
{
  std::optional<TrustRegionModelCandidate> best_candidate;
  double best_objective = std::numeric_limits<double>::max();
  for (size_t i = 0; i < model_objectives.size(); ++i) {
    if (valid_candidates[i] && std::isfinite(model_objectives[i]) && model_objectives[i] < best_objective) {
      best_candidate = static_cast<TrustRegionModelCandidate>(i);
      best_objective = model_objectives[i];
    }
  }
  return best_candidate;
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
  auto& Hz = results.H_z;

  const double cg_tol_squared = settings.cg_tol * settings.cg_tol;

  z = 0.0;
  Hz = 0.0;

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

  double zz = 0.;

  // rPr = dot(rCurrent, Pr)
  double rPr = dot_many({{&rCurrent, &Pr}})[0];

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
      const double tau = projectToBoundaryWithCoefs(z, d, trSize, zz, zd, dd);
      Hz.Add(tau, Hd);
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
    Hz.Add(alphaCg, Hd);

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
