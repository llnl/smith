// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_solver.hpp
 *
 * @brief This file contains the declaration of the DifferentiableSolver interface
 */

#pragma once

#include <memory>
#include <functional>
#include <optional>
#include <vector>
#include <mpi.h>

namespace mfem {
class Solver;
class Vector;
class HypreParMatrix;
class BlockOperator;
}  // namespace mfem

namespace smith {

class EquationSolver;
class BoundaryConditionManager;
class FiniteElementState;
class FiniteElementDual;
class Mesh;
struct NonlinearSolverOptions;
struct LinearSolverOptions;

/// @brief Abstract interface to DifferentiableBlockSolver interface. Each differentiable block solve should provide
/// both its forward solve and an adjoint solve
class DifferentiableBlockSolver {
 public:
  /// @brief destructor
  virtual ~DifferentiableBlockSolver() {}

  using FieldT = FiniteElementState;                        ///< using
  using FieldPtr = std::shared_ptr<FieldT>;                 ///< using
  using FieldD = FiniteElementDual;                         ///< using
  using DualPtr = std::shared_ptr<FieldD>;                  ///< using
  using MatrixPtr = std::unique_ptr<mfem::HypreParMatrix>;  ///< using

  /// @brief Required for certain solvers/preconditions, e.g. when multigrid algorithms want a near null-space
  /// For these cases, it should be called before solve
  virtual void completeSetup(const std::vector<FieldT>& us) = 0;

  /// @brief Solve a set of equations with a vector of FiniteElementState as unknown
  /// @param u_guesses initial guess for solver
  /// @param residuals std::vector<std::function> for equations to be solved
  /// @param jacobians std::vector<std::vector>> of std::function for evaluating the linearized Jacobians about the
  /// current solution
  /// @return std::vector of solution vectors (FiniteElementState)
  virtual std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses,
      std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)> residuals,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)> jacobians) const = 0;

  /// @brief Solve the (linear) adjoint set of equations with a vector of FiniteElementState as unknown
  /// @param u_bars std::vector of right hand sides (rhs) for the solve
  /// @param jacobian_transposed std::vector<std::vector>> of evaluated linearized adjoint space matrices
  /// @return The adjoint vector of solution field
  virtual std::vector<FieldPtr> solveAdjoint(const std::vector<DualPtr>& u_bars,
                                             std::vector<std::vector<MatrixPtr>>& jacobian_transposed) const = 0;

  /// @brief Check whether the stage residuals satisfy convergence for this solver's configured tolerance.
  /// @param tolerance_multiplier Multiplier applied to the configured tolerance (e.g. 1.0 for the final check,
  ///        or a smaller value for a stricter intermediate check)
  /// @param residuals The current residual vectors for each block in this solver
  /// @return true if all residuals satisfy the convergence criterion
  virtual bool checkConvergence(double tolerance_multiplier, const std::vector<mfem::Vector>& residuals) const = 0;

  /// @brief Reset internal convergence tracking state (e.g. the stored initial residual norm used for relative
  /// tolerance). Call this at the start of each new solve sequence.
  virtual void resetConvergenceState() const {}

  /// @brief Interface option to clear memory between solves to avoid high-water mark memory usage.
  virtual void clearMemory() const {}
};

/// @brief Implementation of the DifferentiableBlockSolver interface for the special case of nonlinear solves with
/// linear adjoint solves
class DifferentiableSolver : public DifferentiableBlockSolver {
 public:
  /// @brief Construct from a nonlinear equation solver.
  /// @note The caller is responsible for choosing inner vs outer tolerance when using this
  /// constructor directly.  The builder function buildDifferentiableSolver
  /// applies a 0.6x inner-tolerance factor automatically.
  DifferentiableSolver(std::unique_ptr<EquationSolver> s, MPI_Comm comm, double abs_tol = 1e-12,
                                     double rel_tol = 1e-8);

  /// @overload
  void completeSetup(const std::vector<FieldT>& us) override;

  /// @overload
  bool checkConvergence(double tolerance_multiplier, const std::vector<mfem::Vector>& residuals) const override;

  /// @overload
  void resetConvergenceState() const override;

  /// @overload
  std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses,
      std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)> residuals,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)> jacobians) const override;

  /// @overload
  std::vector<FieldPtr> solveAdjoint(const std::vector<DualPtr>& u_bars,
                                     std::vector<std::vector<MatrixPtr>>& jacobian_transposed) const override;

  mutable std::unique_ptr<mfem::BlockOperator>
      block_jac_;  ///< Need to hold an instance of a block operator to work with the mfem solver interface
  mutable std::vector<std::vector<MatrixPtr>>
      matrix_of_jacs_;  ///< Holding vectors of block matrices to that do not going out of scope before the mfem solver
                        ///< is done with using them in the block_jac_

  mutable std::unique_ptr<EquationSolver>
      nonlinear_solver_;  ///< the nonlinear equation solver used for the forward pass

  MPI_Comm comm_;                                        ///< MPI communicator for parallel norm computation
  double abs_tol_;                                       ///< absolute residual tolerance for convergence check
  double rel_tol_;                                       ///< relative residual tolerance for convergence check
  mutable std::optional<double> initial_residual_norm_;  ///< residual norm at first convergence check (for rel tol)
};

/// @brief Create a differentiable nonlinear block solver
/// @param nonlinear_opts nonlinear options struct
/// @param linear_opts linear options struct
/// @param mesh mesh
std::shared_ptr<DifferentiableSolver> buildDifferentiableSolver(
    NonlinearSolverOptions nonlinear_opts, LinearSolverOptions linear_opts, const smith::Mesh& mesh);

}  // namespace smith
