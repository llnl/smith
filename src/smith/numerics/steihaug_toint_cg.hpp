// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <array>
#include <functional>
#include <utility>
#include <vector>

#include "mfem.hpp"

namespace smith {

/// Internal structure for storing trust region settings
struct TrustRegionSettings {
  /// cg tol
  double cg_tol = 1e-8;
  /// min cg iters
  size_t min_cg_iterations = 0;  //
  /// max cg iters should be around # of system dofs
  size_t max_cg_iterations = 10000;  //
  /// Relative quadratic-model-decrease threshold below which a CG iter counts as "stagnant".
  /// 0 disables the check. Typical: 1e-3. Compared against per-iter decrement / |cumulative model|.
  double model_energy_stagnation_reltol = 0.0;
  /// Consecutive stagnant iters required to exit CG early. 0 disables.
  size_t model_stagnation_window = 0;
  /// max cumulative iterations
  size_t max_cumulative_iteration = 1;
  /// minimum trust region size
  double min_tr_size = 1e-13;
  /// trust region decrease factor
  double t1 = 0.25;
  /// trust region increase factor
  double t2 = 1.75;
  /// worse case energy drop ratio.  trust region accepted if energy drop is better than this.
  double eta1 = 1e-9;
  /// non-ideal energy drop ratio.  trust region decreases if energy drop is worse than this.
  double eta2 = 0.1;
  /// ideal energy drop ratio.  trust region increases if energy drop is better than this.
  double eta3 = 0.6;
  /// parameter limiting how fast the energy can drop relative to the prediction (in case the energy surrogate is poor)
  double eta4 = 4.2;
};

/// Internal structure for storing trust region stateful data
struct TrustRegionResults {
  /// Constructor takes the size of the solution vector
  TrustRegionResults(int size)
  {
    z.SetSize(size);
    H_z.SetSize(size);
    d.SetSize(size);
    H_d.SetSize(size);
    Pr.SetSize(size);
    cauchy_point.SetSize(size);
    H_cauchy_point.SetSize(size);
    z = 0.0;
    H_z = 0.0;
    d = 0.0;
    H_d = 0.0;
    Pr = 0.0;
    cauchy_point = 0.0;
    H_cauchy_point = 0.0;
  }

  /// resets trust region results for a new outer iteration
  void reset()
  {
    z = 0.0;
    cauchy_point = 0.0;
  }

  /// enumerates the possible final status of the trust region steps
  enum class Status
  {
    Interior,
    NegativeCurvature,
    OnBoundary,
    NonDescentDirection
  };

  /// step direction
  mfem::Vector z;
  /// action of hessian on current step z
  mfem::Vector H_z;
  /// incrementalCG direction
  mfem::Vector d;
  /// action of hessian on direction d
  mfem::Vector H_d;
  /// preconditioned residual
  mfem::Vector Pr;
  /// cauchy point
  mfem::Vector cauchy_point;
  /// action of hessian on direction of cauchy point
  mfem::Vector H_cauchy_point;
  /// specifies if step is interior, exterior, negative curvature, etc.
  Status interior_status = Status::Interior;
  /// iteration counter
  size_t cg_iterations_count = 0;
  /// true when the inner CG exhausted `max_cg_iterations` without converging or
  /// hitting the TR boundary. Distinct from `interior_status`, which the CG
  /// always leaves as Interior in that case.
  bool cg_hit_max_iters = false;
  /// true when CG exited because the quadratic-model decrease stagnated for
  /// `model_stagnation_window` consecutive iters. Status stays Interior.
  bool cg_model_stagnated = false;
};

using DotPair = std::pair<const mfem::Vector*, const mfem::Vector*>;                      ///< using
using DotManyFunction = std::function<std::vector<double>(const std::vector<DotPair>&)>;  ///< using

/// In-CG wall-time accumulators. Pass `nullptr` to skip profiling.
struct CGProfile {
  double H_mult_time = 0.0;  ///< wall time spent in H.Mult inside CG
  double P_mult_time = 0.0;  ///< wall time spent in P->Mult inside CG
  double dots_time = 0.0;    ///< wall time spent in dot_many invocations (incl. Allreduce)
  size_t H_mult_count = 0;
  size_t P_mult_count = 0;
  size_t dot_call_count = 0;  ///< number of dot_many invocations
};

/// compute local dot products for many vector pairs
std::vector<double> dotMany(const std::vector<DotPair>& pairs);

/// true when direction is locally downhill for the quadratic model's linear term
bool isDescentDirection(const mfem::Vector& direction, const mfem::Vector& residual, const DotManyFunction& dot_many);

/**
 * @brief Minimize quadratic sub-problem given residual vector, the action of the stiffness and a preconditioner
 *
 * This is a standard implementation of 'The Conjugate Gradient Method and Trust Regions in Large Scale Optimization'
 * by T. Steihaug. It is also called the Steihaug-Toint CG trust region algorithm (see also Trust Region Methods
 * by Conn, Gould, and Toint).
 */
void steihaugTointCG(const mfem::Vector& r0, mfem::Vector& rCurrent, const mfem::Operator& H, const mfem::Solver* P,
                     const TrustRegionSettings& settings, double& trSize, TrustRegionResults& results,
                     double r0_norm_squared, const DotManyFunction& dot_many, CGProfile* profile = nullptr);

}  // namespace smith
