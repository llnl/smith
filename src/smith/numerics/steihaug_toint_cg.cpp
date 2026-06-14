// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/steihaug_toint_cg.hpp"

#include "mpi.h"

namespace smith {

namespace {

bool isDescentDirection(double directional_derivative) { return directional_derivative < 0.0; }

void projectToBoundaryWithCoefs(mfem::Vector& z, const mfem::Vector& d, double delta, double zz, double zd, double dd)
{
  const double deltadelta_m_zz = std::max(delta * delta - zz, 0.0);
  if (deltadelta_m_zz == 0.0) return;
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

bool isDescentDirection(const mfem::Vector& direction, const mfem::Vector& residual, const DotManyFunction& dot_many)
{
  return isDescentDirection(dot_many({{&direction, &residual}})[0]);
}

void steihaugTointCG(const mfem::Vector& r0, mfem::Vector& rCurrent, const mfem::Operator& H, const mfem::Solver* P,
                     const TrustRegionSettings& settings, double& trSize, TrustRegionResults& results,
                     double r0_norm_squared, const DotManyFunction& dot_many, CGProfile* profile)
{
  // minimize r0@z + 0.5*z@J@z
  results.interior_status = TrustRegionResults::Status::Interior;
  results.cg_iterations_count = 0;
  results.cg_hit_max_iters = false;
  results.cg_model_stagnated = false;

  auto& z = results.z;
  auto& cgIter = results.cg_iterations_count;
  auto& d = results.d;
  auto& Pr = results.Pr;
  auto& Hd = results.H_d;

  // Profiling wrappers — zero-cost when `profile` is nullptr (compiler folds the branch).
  auto timed_P = [&](const mfem::Vector& in, mfem::Vector& out) {
    if (!P) {
      out = in;
      return;
    }
    if (profile) {
      double t0 = MPI_Wtime();
      P->Mult(in, out);
      profile->P_mult_time += MPI_Wtime() - t0;
      ++profile->P_mult_count;
    } else {
      P->Mult(in, out);
    }
  };
  auto timed_H = [&](const mfem::Vector& in, mfem::Vector& out) {
    if (profile) {
      double t0 = MPI_Wtime();
      H.Mult(in, out);
      profile->H_mult_time += MPI_Wtime() - t0;
      ++profile->H_mult_count;
    } else {
      H.Mult(in, out);
    }
  };
  auto timed_dots = [&](const std::vector<DotPair>& pairs) {
    if (profile) {
      double t0 = MPI_Wtime();
      auto r = dot_many(pairs);
      profile->dots_time += MPI_Wtime() - t0;
      ++profile->dot_call_count;
      return r;
    }
    return dot_many(pairs);
  };

  const double cg_tol_squared = settings.cg_tol * settings.cg_tol;

  if (r0_norm_squared <= cg_tol_squared && settings.min_cg_iterations == 0) {
    return;
  }

  rCurrent = r0;
  timed_P(rCurrent, Pr);

  // d = -Pr
  d = Pr;
  d *= -1.0;

  z = 0.0;
  double zz = 0.;

  // rPr = dot(rCurrent, Pr)
  double rPr = timed_dots({{&rCurrent, &Pr}})[0];

  // d·r is maintained by recurrence to avoid a per-iteration reduction:
  //   d_{k+1}·r_{k+1} = -rPr_{k+1} + beta*(d_k·r_k + alpha*d_k·H d_k).
  // At iteration 1, d = -Pr so d·r = -rPr exactly.
  double descent_check = -rPr;

  // Quadratic-model value m_k = g^T z + 0.5 z^T H z; starts at 0 (z_0 = 0).
  // Per-iter decrement = 0.5 * alpha_k * rPr_k (standard PCG identity).
  double model_value = 0.0;
  size_t stagnant_count = 0;
  const bool stagnation_enabled = settings.model_stagnation_window > 0 && settings.model_energy_stagnation_reltol > 0.0;

  for (cgIter = 1; cgIter <= settings.max_cg_iterations; ++cgIter) {
    if (!isDescentDirection(descent_check)) {
      d *= -1.0;
      results.interior_status = TrustRegionResults::Status::NonDescentDirection;
    }

    timed_H(d, Hd);

    // One fused reduction covers the curvature plus the step-norm ingredients:
    // ||z + alpha*d||^2 = zz + 2*alpha*(z·d) + alpha^2*(d·d), so the candidate
    // norm and the boundary projection need no further reductions.
    auto dots1 = timed_dots({{&d, &Hd}, {&z, &d}, {&d, &d}});
    double curvature = dots1[0];
    double zd = dots1[1];
    double dd = dots1[2];
    const double alphaCg = curvature != 0.0 ? rPr / curvature : 0.0;

    auto& zPred = Pr;
    zPred = z;
    zPred.Add(alphaCg, d);
    double zzNp1 = zz + 2.0 * alphaCg * zd + alphaCg * alphaCg * dd;

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

    z = zPred;

    if (results.interior_status == TrustRegionResults::Status::NonDescentDirection) {
      return;
    }

    // Model-decrease stagnation termination. Per-iter decrement = 0.5*alpha*rPr;
    // stop when relative decrement / |cumulative model| stays below tol for
    // `window` consecutive iters. Cheap; no extra reductions.
    if (stagnation_enabled) {
      const double decrement = 0.5 * alphaCg * rPr;
      model_value -= decrement;
      const double abs_m = std::abs(model_value);
      if (abs_m > 0.0 && decrement < settings.model_energy_stagnation_reltol * abs_m) {
        ++stagnant_count;
        if (stagnant_count >= settings.model_stagnation_window) {
          results.cg_model_stagnated = true;
          return;
        }
      } else {
        stagnant_count = 0;
      }
    }

    rCurrent.Add(alphaCg, Hd);

    timed_P(rCurrent, Pr);

    auto dots2 = timed_dots({{&rCurrent, &Pr}, {&rCurrent, &rCurrent}});
    double rPrNp1 = dots2[0];
    double r_current_norm_squared = dots2[1];

    if (r_current_norm_squared <= cg_tol_squared && cgIter >= settings.min_cg_iterations) {
      return;
    }

    double beta = rPrNp1 / rPr;
    descent_check = -rPrNp1 + beta * (descent_check + alphaCg * curvature);
    rPr = rPrNp1;
    d *= beta;
    d.Add(-1.0, Pr);

    zz = zzNp1;
  }
  cgIter--;
  // Reached the iter cap with Interior status (boundary / negative-curvature
  // branches above all return early). Mark so callers can route to the
  // subspace path.
  if (results.interior_status == TrustRegionResults::Status::Interior) {
    results.cg_hit_max_iters = true;
  }
}

}  // namespace smith
