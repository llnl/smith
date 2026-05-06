// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_block_solver.hpp
 *
 * @brief This file contains nonlinear block solver interfaces and helpers
 */

#pragma once

#include <memory>
#include <functional>
#include <optional>
#include <vector>
#include <mpi.h>

#include "smith/numerics/solver_config.hpp"
#include "smith/numerics/nonlinear_convergence.hpp"
#include "smith/differentiable_numerics/bc_ramp_options.hpp"

namespace mfem {
class Solver;
class Vector;
class HypreParMatrix;
class BlockOperator;
template <class T>
class Array;
}  // namespace mfem

namespace smith {

class EquationSolver;
class BoundaryConditionManager;
class FiniteElementState;
class FiniteElementDual;
class Mesh;
struct NonlinearSolverOptions;
struct LinearSolverOptions;

/// @brief Abstract interface for nonlinear block solvers that provide both forward and adjoint solves.
class NonlinearBlockSolverBase {
 public:
  /// @brief destructor
  virtual ~NonlinearBlockSolverBase() {}

  using FieldT = FiniteElementState;                        ///< using
  using FieldPtr = std::shared_ptr<FieldT>;                 ///< using
  using FieldD = FiniteElementDual;                         ///< using
  using DualPtr = std::shared_ptr<FieldD>;                  ///< using
  using MatrixPtr = std::unique_ptr<mfem::HypreParMatrix>;  ///< using

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

  /// @brief Evaluate convergence for this solver's configured tolerance using solver-owned convergence state.
  ConvergenceStatus convergenceStatus(double tolerance_multiplier, const std::vector<mfem::Vector>& residuals) const
  {
    return convergenceStatus(tolerance_multiplier, residuals, convergence_context_);
  }

  /// @brief Evaluate convergence with externally owned convergence state.
  virtual ConvergenceStatus convergenceStatus(double tolerance_multiplier, const std::vector<mfem::Vector>& residuals,
                                              NonlinearConvergenceContext& context) const = 0;

  /// @brief Initialize a convergence context from a residual snapshot without relying on a fake convergence test.
  virtual void primeConvergenceContext(const std::vector<mfem::Vector>& residuals,
                                       NonlinearConvergenceContext& context) const = 0;

  /// @brief Check whether the current residuals satisfy the convergence criterion.
  bool checkConvergence(double tolerance_multiplier, const std::vector<mfem::Vector>& residuals) const
  {
    return convergenceStatus(tolerance_multiplier, residuals).converged;
  }

  /// @brief Reset internal convergence tracking state (e.g. the stored initial residual norm used for relative
  /// tolerance). Call this at the start of each new solve sequence.
  virtual void resetConvergenceState() const { convergence_context_.reset(); }

  /// @brief Interface option to clear memory between solves to avoid high-water mark memory usage.
  virtual void clearMemory() const {}

  /// @brief Set an inner-solve tolerance multiplier, e.g. for staggered solves.
  virtual void setInnerToleranceMultiplier(double multiplier) = 0;

  /// @brief Apply or clear relaxed intermediate-solve policy.
  virtual void setIntermediateTolerancePolicy(bool enabled, double abs_tol_factor, double rel_tol_floor,
                                              int max_iterations) const
  {
    static_cast<void>(enabled);
    static_cast<void>(abs_tol_factor);
    static_cast<void>(rel_tol_floor);
    static_cast<void>(max_iterations);
  }

  /// @brief Effective nonlinear solver print level for user-facing diagnostics.
  virtual int printLevel() const { return 0; }

  /// @brief Returns true if the most recent @ref solve call satisfied its convergence criterion.
  /// Subclasses must write @c last_solve_converged_ at the end of their solve path.
  /// True before first call (optimistic default — ramp predicate sees only the NaN check then).
  bool lastSolveConverged() const { return last_solve_converged_; }

  /// @brief Configure BC cutback behavior for block_solve when this solver is in use.
  void setBcRampOptions(const BcRampOptions& options) { bc_ramp_options_ = options; }

  /// @brief Read current BC ramp options.
  const BcRampOptions& bcRampOptions() const { return bc_ramp_options_; }

  /// @brief Linearized warm-start step from last-converged state to target BCs.
  ///
  /// Solves the partitioned linear system K_ff du_f = -K_fc du_c (with du_c the
  /// BC delta at constrained dofs) and returns u_prev + s*du with s ∈ (0, 1]
  /// scaled down until @p residual_finite(u_prev + s*du, s) is true. Single-row
  /// only. Default returns success=false; subclasses with a linear solver may
  /// override.
  struct WarmStart {
    bool success = false;
    double alpha = 1.0;      ///< accepted scale s; aligns with BC ramp alpha
    FieldPtr initial_guess;  ///< u_prev + s*du
  };
  virtual WarmStart linearWarmStart(const FieldPtr& /*u_prev*/, const FieldPtr& /*target_bc*/,
                                    mfem::HypreParMatrix& /*K_raw_at_u_prev*/,
                                    const mfem::Array<int>& /*constrained_tdofs*/,
                                    const std::function<bool(const FieldPtr&, double)>& /*residual_finite*/) const
  {
    return {};
  }

  /// @brief Whether this solver should attempt linearWarmStart in block_solve.
  virtual bool warmStartEnabled() const { return false; }

 protected:
  mutable bool is_setup_ = false;             ///< Records if this block solver has its preconditioner initialized.
  mutable bool last_solve_converged_ = true;  ///< Set by subclasses after each solve.
  mutable NonlinearConvergenceContext convergence_context_ = {};  ///< Solver-owned convergence state for one solve.
  BcRampOptions bc_ramp_options_{};                               ///< BC ramp configuration (default: disabled).
};

/// @brief Nonlinear block solver backed by an EquationSolver forward solve and linear adjoint solves.
class NonlinearBlockSolver : public NonlinearBlockSolverBase {
 public:
  using NonlinearBlockSolverBase::convergenceStatus;

  /// @brief Construct from a nonlinear equation solver.
  /// @note The caller is responsible for choosing inner vs outer tolerance when using this
  /// constructor directly.  The builder function buildNonlinearBlockSolver
  /// applies a 0.6x inner-tolerance factor automatically.
  NonlinearBlockSolver(std::unique_ptr<EquationSolver> s, MPI_Comm comm, double abs_tol = 1e-12, double rel_tol = 1e-8,
                       std::optional<NonlinearSolverOptions> retained_nonlinear_options = std::nullopt,
                       std::optional<LinearSolverOptions> retained_linear_options = std::nullopt);

  /// @overload
  ConvergenceStatus convergenceStatus(double tolerance_multiplier, const std::vector<mfem::Vector>& residuals,
                                      NonlinearConvergenceContext& context) const override;

  /// @overload
  void primeConvergenceContext(const std::vector<mfem::Vector>& residuals,
                               NonlinearConvergenceContext& context) const override;

  /// @overload
  std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses,
      std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)> residuals,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)> jacobians) const override;

  /// @overload
  std::vector<FieldPtr> solveAdjoint(const std::vector<DualPtr>& u_bars,
                                     std::vector<std::vector<MatrixPtr>>& jacobian_transposed) const override;

  /// @brief Initialize the preconditioner in case of vector problems
  void completeSetup(const std::vector<FieldPtr>& us) const;

  /// @brief Set the inner tolerance multiplier.
  void setInnerToleranceMultiplier(double multiplier) override { inner_tol_multiplier_ = multiplier; }

  /// @overload
  void setIntermediateTolerancePolicy(bool enabled, double abs_tol_factor, double rel_tol_floor,
                                      int max_iterations) const override
  {
    use_intermediate_tolerances_ = enabled;
    intermediate_abs_tol_factor_ = abs_tol_factor;
    intermediate_rel_tol_floor_ = rel_tol_floor;
    intermediate_max_iterations_ = max_iterations;
  }

  /// @brief Build a fresh solver instance from retained config.
  std::shared_ptr<NonlinearBlockSolver> cloneFresh() const;

  /// @overload
  int printLevel() const override;

  /// @overload
  bool warmStartEnabled() const override
  {
    return retained_nonlinear_options_ && retained_nonlinear_options_->warm_start;
  }

  /// @overload
  WarmStart linearWarmStart(const FieldPtr& u_prev, const FieldPtr& target_bc, mfem::HypreParMatrix& K_raw_at_u_prev,
                            const mfem::Array<int>& constrained_tdofs,
                            const std::function<bool(const FieldPtr&, double)>& residual_finite) const override;

  mutable std::unique_ptr<mfem::BlockOperator>
      block_jac_;  ///< Need to hold an instance of a block operator to work with the mfem solver interface
  mutable std::vector<std::vector<MatrixPtr>>
      matrix_of_jacs_;  ///< Holding vectors of block matrices to that do not going out of scope before the mfem solver
                        ///< is done with using them in the block_jac_

  mutable std::unique_ptr<EquationSolver>
      nonlinear_solver_;  ///< the nonlinear equation solver used for the forward pass

  MPI_Comm comm_;                                     ///< MPI communicator for parallel norm computation
  double abs_tol_;                                    ///< absolute residual tolerance for convergence check
  double rel_tol_;                                    ///< relative residual tolerance for convergence check
  double inner_tol_multiplier_ = 1.0;                 ///< multiplier for tolerances during inner solves
  mutable bool use_intermediate_tolerances_ = false;  ///< whether to relax cutback solves
  mutable double intermediate_abs_tol_factor_ = 1.0;  ///< abs_tol multiplier for cutback solves
  mutable double intermediate_rel_tol_floor_ = 0.0;   ///< rel_tol floor for cutback solves
  mutable int intermediate_max_iterations_ = 0;       ///< max nonlinear iterations for cutback solves
  std::optional<NonlinearSolverOptions> retained_nonlinear_options_ = std::nullopt;  ///< retained nonlinear config
  std::optional<LinearSolverOptions> retained_linear_options_ = std::nullopt;        ///< retained linear config
};

/// @brief Create an equation-backed nonlinear block solver.
/// @param nonlinear_opts nonlinear options struct
/// @param linear_opts linear options struct
/// @param mesh mesh
std::shared_ptr<NonlinearBlockSolver> buildNonlinearBlockSolver(NonlinearSolverOptions nonlinear_opts,
                                                                LinearSolverOptions linear_opts,
                                                                const smith::Mesh& mesh);

}  // namespace smith
