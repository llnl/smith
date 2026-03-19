// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <string>
#include <vector>

#include "mpi.h"
#include "mfem.hpp"

#include "smith/numerics/solver_config.hpp"

namespace smith {

/// @brief Detailed status from evaluating nonlinear residual convergence.
struct ConvergenceStatus {
  bool global_converged = false;         ///< True when the scalar global residual criterion passes.
  bool block_converged = false;          ///< True when all per-block residual criteria pass.
  bool converged = false;                ///< True when either the global or block criterion passes.
  bool block_path_enabled = false;       ///< True when per-block tolerances are active for this evaluation.
  double global_norm = 0.0;              ///< Current scalar global residual norm.
  double global_goal = 0.0;              ///< Scalar convergence threshold used for the global check.
  std::vector<double> block_norms = {};  ///< Current residual norm for each block.
  std::vector<double> block_goals = {};  ///< Convergence threshold used for each residual block.
};

/// @brief Stores initial residual norms used for relative nonlinear convergence checks.
struct NonlinearConvergenceContext {
  double initial_global_norm = -1.0;             ///< Initial scalar global residual norm for the current solve.
  std::vector<double> initial_block_norms = {};  ///< Initial residual norm for each block for the current solve.

  /// @brief Clear all stored initial norms for a new solve.
  void reset();
};

/// @brief Expand an optional per-block tolerance vector to the requested block count.
/// @param block_tols User-provided per-block tolerance vector, possibly empty.
/// @param num_blocks Number of residual blocks expected.
/// @param empty_value Fill value to use when @p block_tols is empty.
/// @param tol_name Name used in validation error messages.
/// @return A per-block tolerance vector of length @p num_blocks.
std::vector<double> expandPerBlockTolerances(const std::vector<double>& block_tols, size_t num_blocks,
                                             double empty_value, const std::string& tol_name);

/// @brief Compute one L2 norm per residual block.
/// @param residuals Residual vectors, one per logical block.
/// @param comm MPI communicator used for parallel norm reduction.
/// @return L2 norm of each residual block.
std::vector<double> computeResidualBlockNorms(const std::vector<mfem::Vector>& residuals, MPI_Comm comm);

/// @brief Compute one L2 norm per residual block from a monolithic residual vector.
/// @param residual Monolithic residual vector.
/// @param block_offsets Offsets delimiting residual blocks within @p residual.
/// @param comm MPI communicator used for parallel norm reduction.
/// @return L2 norm of each residual block.
std::vector<double> computeResidualBlockNorms(const mfem::Vector& residual, const std::vector<int>& block_offsets,
                                              MPI_Comm comm);

/// @brief Evaluate scalar, per-block, and combined nonlinear residual convergence.
/// @param tolerance_multiplier Extra scale factor applied to the computed convergence goals.
/// @param abs_tol Scalar absolute tolerance.
/// @param rel_tol Scalar relative tolerance.
/// @param block_abs_tols Per-block absolute tolerances.
/// @param block_rel_tols Per-block relative tolerances.
/// @param block_path_enabled Whether the per-block path should participate in convergence.
/// @param block_norms Current residual norm for each block.
/// @param context Stored initial norms for relative tolerance evaluation.
/// @return Full convergence status for the current residual snapshot.
ConvergenceStatus evaluateResidualConvergence(double tolerance_multiplier, double abs_tol, double rel_tol,
                                              const std::vector<double>& block_abs_tols,
                                              const std::vector<double>& block_rel_tols, bool block_path_enabled,
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

  /// @brief Configure block offsets and per-block tolerances for a monolithic residual.
  void setBlockData(const std::vector<int>& block_offsets, BlockConvergenceTolerances block_tolerances) const;

  /// @brief Return true when per-block tolerances are active.
  bool blockPathEnabled() const;

  /// @brief Evaluate convergence for the current monolithic residual vector.
  ConvergenceStatus evaluate(double tolerance_multiplier, const mfem::Vector& residual) const;

 private:
  MPI_Comm comm_;
  double abs_tol_;
  double rel_tol_;
  mutable std::vector<int> block_offsets_ = {};
  mutable BlockConvergenceTolerances block_tolerances_ = {};
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
