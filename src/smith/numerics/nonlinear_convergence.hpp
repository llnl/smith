// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>

#include "mpi.h"
#include "mfem.hpp"

namespace smith {

/// @brief Detailed status from evaluating nonlinear residual convergence.
struct ConvergenceStatus {
  bool global_converged = false;         ///< True when the scalar global residual criterion passes.
  bool converged = false;                ///< True when the scalar global residual criterion passes.
  double global_norm = 0.0;              ///< Current scalar global residual norm.
  double global_goal = 0.0;              ///< Scalar convergence threshold used for the global check.
  std::vector<double> block_norms = {};  ///< Residual norms used to form the scalar global norm.
};

/// @brief Stores initial residual norms used for relative nonlinear convergence checks.
struct NonlinearConvergenceContext {
  double initial_global_norm = -1.0;  ///< Initial scalar global residual norm for the current solve.

  /// @brief Clear all stored initial norms for a new solve.
  /// The next convergence evaluation seeds a fresh relative-tolerance baseline from its current residual norms.
  void reset();
};

/// @brief Compute one L2 norm per residual block.
/// @param residuals Residual vectors, one per logical block.
/// @param comm MPI communicator used for parallel norm reduction.
/// @return L2 norm of each residual block.
std::vector<double> computeResidualBlockNorms(const std::vector<mfem::Vector>& residuals, MPI_Comm comm);

/// @brief Evaluate scalar nonlinear residual convergence.
/// @param tolerance_multiplier Extra scale factor applied to the computed convergence goals.
/// @param abs_tol Scalar absolute tolerance.
/// @param rel_tol Scalar relative tolerance.
/// @param block_norms Current residual norm for each block.
/// @param context Stored initial norms for relative tolerance evaluation.
/// @return Full convergence status for the current residual snapshot.
ConvergenceStatus evaluateResidualConvergence(double tolerance_multiplier, double abs_tol, double rel_tol,
                                              const std::vector<double>& block_norms,
                                              NonlinearConvergenceContext& context);

/// @brief Owns nonlinear convergence state for an inner `EquationSolver` solve.
class EquationSolverConvergenceManager {
 public:
  /// @brief Construct a convergence manager for one nonlinear solver.
  /// @param comm MPI communicator used for block norm reductions.
  /// @param abs_tol Scalar absolute tolerance.
  /// @param rel_tol Scalar relative tolerance.
  EquationSolverConvergenceManager(MPI_Comm comm, double abs_tol, double rel_tol);

  /// @brief Reset stored initial residual norms.
  void reset() const;

  /// @brief Set scalar tolerances.
  void setTolerances(double abs_tol, double rel_tol) const;

  /// @brief Evaluate convergence for the current monolithic residual vector.
  ConvergenceStatus evaluate(double tolerance_multiplier, const mfem::Vector& residual) const;

 private:
  MPI_Comm comm_;
  mutable double abs_tol_;
  mutable double rel_tol_;
  mutable NonlinearConvergenceContext context_ = {};
};

/// @brief Small interface for nonlinear solvers that can accept Smith-managed convergence state.
class ConvergenceManagedNonlinearSolver {
 public:
  virtual ~ConvergenceManagedNonlinearSolver() = default;

  /// @brief Attach the shared convergence manager used to evaluate nonlinear stopping criteria.
  virtual void setConvergenceManager(std::shared_ptr<EquationSolverConvergenceManager> convergence_manager) = 0;
};

}  // namespace smith
