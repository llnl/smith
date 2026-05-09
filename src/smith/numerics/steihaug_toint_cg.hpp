// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "mfem.hpp"
#include <array>

namespace smith {

/// Internal structure for storing trust region settings
struct TrustRegionSettings {
  /// cg tol
  double cg_tol = 1e-8;
  /// min cg iters
  size_t min_cg_iterations = 0;  //
  /// max cg iters should be around # of system dofs
  size_t max_cg_iterations = 10000;  //
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
    d_old.SetSize(size);
    H_d_old.SetSize(size);
    d.SetSize(size);
    H_d.SetSize(size);
    Pr.SetSize(size);
    cauchy_point.SetSize(size);
    H_cauchy_point.SetSize(size);
    z = 0.0;
    H_z = 0.0;
    d_old = 0.0;
    H_d_old = 0.0;
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
  /// old step direction
  mfem::Vector d_old;
  /// action of hessian on previous step z_old
  mfem::Vector H_d_old;
  /// true after at least one accepted line-search step has populated d_old
  bool has_d_old = false;
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
};

class SteihaugTointDelegate {
 public:
  virtual ~SteihaugTointDelegate() = default;

  virtual std::array<double, 4> dot_many_4(const mfem::Vector& a0, const mfem::Vector& b0, const mfem::Vector& a1,
                                           const mfem::Vector& b1, const mfem::Vector& a2, const mfem::Vector& b2,
                                           const mfem::Vector& a3, const mfem::Vector& b3) const = 0;

  virtual std::array<double, 2> dot_many_2(const mfem::Vector& a0, const mfem::Vector& b0, const mfem::Vector& a1,
                                           const mfem::Vector& b1) const = 0;

  virtual void projectToBoundaryWithCoefs(mfem::Vector& z, const mfem::Vector& d, double delta, double zz, double zd,
                                          double dd) const = 0;
};

/**
 * @brief Minimize quadratic sub-problem given residual vector, the action of the stiffness and a preconditioner
 *
 * This is a standard implementation of 'The Conjugate Gradient Method and Trust Regions in Large Scale Optimization'
 * by T. Steihaug. It is also called the Steihaug-Toint CG trust region algorithm (see also Trust Region Methods
 * by Conn, Gould, and Toint).
 */
void steihaugTointCG(const mfem::Vector& r0, mfem::Vector& rCurrent, const mfem::Operator& H, const mfem::Solver* P,
                     const TrustRegionSettings& settings, double& trSize, TrustRegionResults& results,
                     double r0_norm_squared, const SteihaugTointDelegate& delegate);

}  // namespace smith
