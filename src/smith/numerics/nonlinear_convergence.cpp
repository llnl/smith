// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/nonlinear_convergence.hpp"

#include <algorithm>
#include <cmath>

#include "axom/fmt.hpp"
#include "axom/slic.hpp"

namespace smith {

namespace {

double residualNorm(const mfem::Vector& residual, MPI_Comm comm)
{
#ifdef MFEM_USE_MPI
  return mfem::ParNormlp(residual, 2.0, comm);
#else
  static_cast<void>(comm);
  return residual.Norml2();
#endif
}

}  // namespace

void NonlinearConvergenceContext::reset() { initial_global_norm = -1.0; }

std::vector<double> computeResidualBlockNorms(const std::vector<mfem::Vector>& residuals, MPI_Comm comm)
{
  std::vector<double> block_norms(residuals.size(), 0.0);
  for (size_t i = 0; i < residuals.size(); ++i) {
    block_norms[i] = residualNorm(residuals[i], comm);
  }
  return block_norms;
}

ConvergenceStatus evaluateResidualConvergence(double tolerance_multiplier, double abs_tol, double rel_tol,
                                              const std::vector<double>& block_norms,
                                              NonlinearConvergenceContext& context)
{
  ConvergenceStatus status;
  status.block_norms = block_norms;

  double global_norm_squared = 0.0;
  for (double block_norm : block_norms) {
    global_norm_squared += block_norm * block_norm;
  }

  status.global_norm = std::sqrt(global_norm_squared);
  if (context.initial_global_norm < 0.0) {
    context.initial_global_norm = status.global_norm;
  }
  status.global_goal = std::max(abs_tol, rel_tol * context.initial_global_norm);
  status.global_converged = status.global_norm <= tolerance_multiplier * status.global_goal;
  status.converged = status.global_converged;
  return status;
}

EquationSolverConvergenceManager::EquationSolverConvergenceManager(MPI_Comm comm, double abs_tol, double rel_tol)
    : comm_(comm), abs_tol_(abs_tol), rel_tol_(rel_tol)
{
}

void EquationSolverConvergenceManager::reset() const { context_.reset(); }

void EquationSolverConvergenceManager::setTolerances(double abs_tol, double rel_tol) const
{
  abs_tol_ = abs_tol;
  rel_tol_ = rel_tol;
}

ConvergenceStatus EquationSolverConvergenceManager::evaluate(double tolerance_multiplier,
                                                             const mfem::Vector& residual) const
{
  return evaluateResidualConvergence(tolerance_multiplier, abs_tol_, rel_tol_, {residualNorm(residual, comm_)},
                                     context_);
}

}  // namespace smith
