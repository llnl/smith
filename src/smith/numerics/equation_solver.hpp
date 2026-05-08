// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file equation_solver.hpp
 *
 * @brief This file contains the declaration of an equation solver wrapper
 */

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>
#include <utility>

#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/input.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/numerics/petsc_solvers.hpp"

namespace smith {

/**
 * @brief Solver-facing interface for Jacobian operations.
 *
 * A JacobianOperator represents the operations available on J(x) after differentiating a residual but before
 * necessarily assembling a sparse matrix. Concrete implementations may support matrix-free products, sparse assembly,
 * diagonal extraction, or all of them. Unsupported operations should throw.
 */
class JacobianOperator : public mfem::Operator {
 public:
  using mfem::Operator::Operator;

  /// Assemble the sparse Jacobian representation.
  virtual std::unique_ptr<mfem::HypreParMatrix> assemble()
  {
    SLIC_ERROR("This JacobianOperator does not support sparse assembly.");
    return nullptr;
  }

  /// Assemble the scalar true-dof diagonal of the Jacobian.
  virtual void assembleDiagonal(mfem::Vector&) const
  {
    SLIC_ERROR("This JacobianOperator does not support diagonal assembly.");
  }
};

/**
 * @brief Adapter from a smith::functional Gradient object to the solver-facing JacobianOperator interface.
 */
template <typename Gradient>
class FunctionalJacobianOperator : public JacobianOperator {
  using GradientT = std::remove_reference_t<Gradient>;

 public:
  explicit FunctionalJacobianOperator(GradientT& gradient)
      : JacobianOperator(gradient.Height(), gradient.Width()), gradient_(&gradient)
  {
  }

  explicit FunctionalJacobianOperator(GradientT&& gradient)
      : JacobianOperator(gradient.Height(), gradient.Width()),
        owned_gradient_(std::make_unique<GradientT>(std::move(gradient))),
        gradient_(owned_gradient_.get())
  {
  }

  void Mult(const mfem::Vector& dx, mfem::Vector& y) const override { gradient_->Mult(dx, y); }

  void AddMult(const mfem::Vector& dx, mfem::Vector& y, const double a = 1.0) const override
  {
    gradient_->AddMult(dx, y, a);
  }

  std::unique_ptr<mfem::HypreParMatrix> assemble() override { return gradient_->assemble(); }

  void assembleDiagonal(mfem::Vector& diag) const override { gradient_->assembleDiagonal(diag); }

 private:
  std::unique_ptr<GradientT> owned_gradient_;
  GradientT* gradient_;
};

/**
 * @brief Matrix-free tangent action callback.
 *
 * The callback evaluates y = J(x) dx for the current nonlinear state x
 * without requiring EquationSolver to assemble J.
 */
using MatrixFreeTangentAction = std::function<void(const mfem::Vector& x, const mfem::Vector& dx, mfem::Vector& y)>;

/**
 * @brief Callback that evaluates and returns a JacobianOperator at the supplied nonlinear state.
 */
using JacobianOperatorFactory = std::function<std::unique_ptr<JacobianOperator>(const mfem::Vector& x)>;

/// Diagnostic counters for the nonlinear PCG-block solver
struct PcgBlockDiagnostics {
  /// Number of nonlinear residual evaluations
  size_t num_residuals = 0;
  /// Number of assembled Jacobian-vector products
  size_t num_hess_vecs = 0;
  /// Number of preconditioner applications
  size_t num_preconds = 0;
  /// Number of assembled Jacobians
  size_t num_jacobian_assembles = 0;
  /// Number of solver-facing JacobianOperator evaluations
  size_t num_jacobian_operator_evals = 0;
  /// Number of direct diagonal assemblies
  size_t num_diagonal_assembles = 0;
  /// Number of preconditioner operator updates
  size_t num_preconditioner_updates = 0;
  /// Number of accepted prefix blocks
  size_t num_prefix_accepts = 0;
  /// Number of momentum resets
  size_t num_momentum_resets = 0;
  /// Number of steps with nonzero PCG beta
  size_t num_nonzero_beta = 0;
  /// Number of steps with zero PCG beta
  size_t num_zero_beta = 0;
  /// Number of accepted blocks
  size_t num_blocks = 0;
  /// Number of rejected blocks
  size_t num_block_rejects = 0;
  /// Number of Powell restarts
  size_t num_powell_restarts = 0;
  /// Number of descent-guard restarts
  size_t num_descent_restarts = 0;
  /// Number of non-positive curvature directions
  size_t num_negative_curvature = 0;
  /// Number of line-search backtracks
  size_t num_line_search_backtracks = 0;
  /// Number of positive-curvature steps capped by the trust radius
  size_t num_trust_capped_steps = 0;
  /// Number of accepted inner PCG steps
  size_t num_accepted_steps = 0;
  /// Number of trial inner PCG steps
  size_t num_trial_steps = 0;
  /// Time spent evaluating nonlinear residuals
  double residual_seconds = 0.0;
  /// Time spent applying Jacobian-vector products
  double hess_vec_seconds = 0.0;
  /// Time spent applying JacobianOperator products
  double jacobian_operator_hess_vec_seconds = 0.0;
  /// Time spent applying assembled Jacobian products
  double assembled_hess_vec_seconds = 0.0;
  /// Time spent applying legacy matrix-free tangent products
  double matrix_free_hess_vec_seconds = 0.0;
  /// Time spent applying preconditioners
  double preconditioner_seconds = 0.0;
  /// Time spent evaluating JacobianOperator factories
  double jacobian_operator_eval_seconds = 0.0;
  /// Time spent assembling sparse Jacobians
  double jacobian_assembly_seconds = 0.0;
  /// Time spent directly assembling diagonals
  double diagonal_assembly_seconds = 0.0;
  /// Time spent inverting direct diagonals
  double diagonal_invert_seconds = 0.0;
  /// Time spent refreshing preconditioner data
  double preconditioner_update_seconds = 0.0;
  /// Time spent in preconditioner SetOperator calls
  double preconditioner_setup_seconds = 0.0;
  /// Last trust scale used by the solver
  double final_h_scale = 1.0;
  /// Last accepted block trust ratio
  double last_trust_ratio = 0.0;
};

/// Diagnostic counters for the TrustRegion nonlinear solver
struct TrustRegionDiagnostics {
  /// Number of nonlinear residual evaluations
  size_t num_residuals = 0;
  /// Number of Jacobian-vector products
  size_t num_hess_vecs = 0;
  /// Number of Hessian-vector products in model CG solves
  size_t num_model_hess_vecs = 0;
  /// Number of Hessian-vector products in Cauchy-point construction
  size_t num_cauchy_hess_vecs = 0;
  /// Number of Hessian-vector products in line-search model checks
  size_t num_line_search_hess_vecs = 0;
  /// Number of preconditioner applications
  size_t num_preconds = 0;
  /// Number of assembled Jacobians
  size_t num_jacobian_assembles = 0;
  /// Number of solver-facing JacobianOperator evaluations
  size_t num_jacobian_operator_evals = 0;
  /// Number of direct diagonal assemblies
  size_t num_diagonal_assembles = 0;
  /// Number of trust-region model CG iterations
  size_t num_cg_iterations = 0;
  /// Number of subspace solves
  size_t num_subspace_solves = 0;
  /// Number of retained-leftmost Hessian-vector products for subspace solves
  size_t num_subspace_leftmost_hess_vecs = 0;
  /// Number of batched Hessian-vector groups used for subspace solves
  size_t num_subspace_hess_vec_batches = 0;
  /// Number of Hessian-vector products inside subspace batches
  size_t num_subspace_batched_hess_vecs = 0;
  /// Number of accepted-step history vectors added to subspace solves
  size_t num_subspace_past_step_vectors = 0;
  /// Number of Hessian-vector products for accepted-step history vectors
  size_t num_subspace_past_step_hess_vecs = 0;
  /// Number of nonlinear-solve-start directions added to subspace solves
  size_t num_subspace_solve_start_vectors = 0;
  /// Number of Hessian-vector products for nonlinear-solve-start directions
  size_t num_subspace_solve_start_hess_vecs = 0;
  /// Number of quadratic subspace backend solves
  size_t num_quadratic_subspace_solves = 0;
  /// Number of cubic subspace backend attempts
  size_t num_cubic_subspace_attempts = 0;
  /// Number of cubic subspace attempts that used the cubic candidate
  size_t num_cubic_subspace_uses = 0;
  /// Number of cubic subspace attempts that fell back to the quadratic candidate
  size_t num_cubic_subspace_quadratic_fallbacks = 0;
  /// Number of preconditioner operator updates
  size_t num_preconditioner_updates = 0;
  /// Number of nonmonotone accepted TrustRegion steps based on work surrogate
  size_t num_nonmonotone_work_accepts = 0;
  /// Number of accepted TrustRegion work-surrogate steps that monotone acceptance would have rejected
  size_t num_monotone_work_would_reject = 0;
  /// Time spent evaluating nonlinear residuals
  double residual_seconds = 0.0;
  /// Time spent applying Jacobian-vector products
  double hess_vec_seconds = 0.0;
  /// Time spent applying Hessian-vector products in model CG solves
  double model_hess_vec_seconds = 0.0;
  /// Time spent applying Hessian-vector products in Cauchy-point construction
  double cauchy_hess_vec_seconds = 0.0;
  /// Time spent applying Hessian-vector products in line-search model checks
  double line_search_hess_vec_seconds = 0.0;
  /// Time spent applying JacobianOperator products
  double jacobian_operator_hess_vec_seconds = 0.0;
  /// Time spent evaluating JacobianOperator factories
  double jacobian_operator_eval_seconds = 0.0;
  /// Time spent directly assembling diagonals
  double diagonal_assembly_seconds = 0.0;
  /// Time spent inverting direct diagonals
  double diagonal_invert_seconds = 0.0;
  /// Time spent applying preconditioners
  double preconditioner_seconds = 0.0;
  /// Total time spent in the nonlinear solve
  double total_seconds = 0.0;
  /// Time spent solving trust-region model problems
  double model_solve_seconds = 0.0;
  /// Total time spent in trust-region subspace solves
  double subspace_seconds = 0.0;
  /// Time spent building/applying retained leftmost directions for subspace solves
  double subspace_leftmost_seconds = 0.0;
  /// Time spent in subspace Hessian-vector batches
  double subspace_hess_vec_batch_seconds = 0.0;
  /// Time spent removing dependent directions before subspace solves
  double subspace_filter_seconds = 0.0;
  /// Time spent in dense subspace backend assembly/solve work
  double subspace_backend_seconds = 0.0;
  /// Time spent projecting dense subspace Hessian
  double subspace_project_A_seconds = 0.0;
  /// Time spent projecting dense subspace Gram matrix
  double subspace_project_gram_seconds = 0.0;
  /// Time spent projecting dense subspace gradient
  double subspace_project_b_seconds = 0.0;
  /// Time spent building dense subspace orthonormal basis
  double subspace_basis_seconds = 0.0;
  /// Time spent forming reduced dense Hessian
  double subspace_reduced_A_seconds = 0.0;
  /// Time spent in dense subspace eigensystems
  double subspace_dense_eigensystem_seconds = 0.0;
  /// Time spent in dense trust-region solve outside eigensystems
  double subspace_dense_trust_solve_seconds = 0.0;
  /// Time spent reconstructing full-space subspace solution
  double subspace_reconstruct_solution_seconds = 0.0;
  /// Time spent reconstructing retained leftmost vectors
  double subspace_reconstruct_leftmost_seconds = 0.0;
  /// Time spent in subspace postprocessing and model-energy comparison
  double subspace_finalize_seconds = 0.0;
  /// Time spent building the Cauchy point
  double cauchy_point_seconds = 0.0;
  /// Time spent in dogleg step construction
  double dogleg_seconds = 0.0;
  /// Time spent in line-search and trust-radius acceptance logic
  double line_search_seconds = 0.0;
  /// Time spent in TrustRegion dot products
  double dot_seconds = 0.0;
  /// Number of TrustRegion dot products
  size_t num_dot_products = 0;
  /// Number of TrustRegion dot batches/reductions
  size_t num_dot_reductions = 0;
  /// Number of dot products in trust-region model solves
  size_t num_model_dot_products = 0;
  /// Number of dot products in Cauchy-point construction
  size_t num_cauchy_dot_products = 0;
  /// Number of dot products in dogleg construction
  size_t num_dogleg_dot_products = 0;
  /// Number of dot products in line-search and acceptance logic
  size_t num_line_search_dot_products = 0;
  /// Number of setup dot products outside the main per-step kernels
  size_t num_setup_dot_products = 0;
  /// Time spent in trust-region model-solve dot products
  double model_dot_seconds = 0.0;
  /// Time spent in Cauchy-point dot products
  double cauchy_dot_seconds = 0.0;
  /// Time spent in dogleg dot products
  double dogleg_dot_seconds = 0.0;
  /// Time spent in line-search dot products
  double line_search_dot_seconds = 0.0;
  /// Time spent in setup dot products
  double setup_dot_seconds = 0.0;
  /// Time spent in TrustRegion vector add/update operations
  double vector_update_seconds = 0.0;
  /// Time spent in TrustRegion vector copies and scaling operations
  double vector_copy_scale_seconds = 0.0;
  /// Time spent in TrustRegion boundary projection operations
  double projection_seconds = 0.0;
  /// Time spent assembling sparse Jacobians
  double jacobian_assembly_seconds = 0.0;
  /// Time spent refreshing preconditioner data
  double preconditioner_update_seconds = 0.0;
  /// Time spent in preconditioner SetOperator calls
  double preconditioner_setup_seconds = 0.0;
  /// Last TrustRegion accumulated work-surrogate level used by nonmonotone acceptance
  double last_work_objective = 0.0;
  /// Last nonmonotone reference work-surrogate level
  double last_nonmonotone_work_reference = 0.0;
};

/**
 * @brief This class manages the objects typically required to solve a nonlinear set of equations arising from
 * discretization of a PDE of the form F(x) = 0. Specifically, it has
 *
 *   1. An @a mfem::NewtonSolver containing the nonlinear solution operator
 *   2. An @a mfem::Solver containing a linear solver that is used by the nonlinear solution operator and adjoint
 * solvers
 *   3. An optional @a mfem::Solver containing a preconditioner for the linear solution operator
 *
 * This @a EquationSolver manages these objects together to ensure they all exist when called by their associated
 * physics simulation module.
 *
 * An equation solver can either be constructed by supplying pre-built nonlinear and linear solvers with a
 * preconditioner, or it can be constructed using @a smith::NonlinearSolverOptions and @a smith::LinearSolverOptions
 * structs with the @a smith::mfem_ext::buildEquationSolver factory method.
 */
class EquationSolver {
 public:
  // _equationsolver_constructor_start
  /**
   * Constructs a new nonlinear equation solver
   * @param[in] nonlinear_solver A constructed nonlinear solver
   * @param[in] linear_solver A constructed linear solver to be called by the nonlinear algorithm and adjoint
   * equation solves
   * @param[in] preconditioner An optional constructed precondition to aid the linear solver
   */
  EquationSolver(std::unique_ptr<mfem::NewtonSolver> nonlinear_solver, std::unique_ptr<mfem::Solver> linear_solver,
                 std::unique_ptr<mfem::Solver> preconditioner = nullptr);
  // _equationsolver_constructor_end

  // _build_equationsolver_start
  /**
   * @brief Construct an equation solver object using nonlinear and linear solver option structs.
   *
   * @param nonlinear_opts The options to configure the nonlinear solution scheme
   * @param lin_opts The options to configure the underlying linear solution scheme to be used by the nonlinear solver
   * @param comm The MPI communicator for the supplied nonlinear operators and HypreParVectors
   */
  EquationSolver(NonlinearSolverOptions nonlinear_opts = {}, LinearSolverOptions lin_opts = {},
                 MPI_Comm comm = MPI_COMM_WORLD);
  // _build_equationsolver_end

  /**
   * Updates the solver with the provided operator
   * @param[in] op The operator (nonlinear system of equations) to use, "F" in F(x) = 0
   * @note This operator is required to return an @a mfem::HypreParMatrix from its @a GetGradient method. This is
   * due to the use of Hypre-based linear solvers.
   */
  void setOperator(const mfem::Operator& op);

  /**
   * @brief Sets an optional matrix-free tangent action for nonlinear solvers that can use J(x) dx directly.
   *
   * Solvers that do not support matrix-free tangent actions ignore this callback. Supported solvers retain their
   * assembled-gradient fallback when no callback is set.
   *
   * @param[in] tangent_action Callback evaluating y = J(x) dx.
   */
  void setMatrixFreeTangentAction(MatrixFreeTangentAction tangent_action);

  /**
   * @brief Sets an optional JacobianOperator factory for nonlinear solvers that can use matrix-free Jacobian products.
   *
   * This is the preferred replacement for the narrower matrix-free tangent-action callback. During migration,
   * PCG-block uses this callback first when it is registered and otherwise falls back to MatrixFreeTangentAction or
   * assembled gradients.
   *
   * @param[in] jacobian_operator Callback evaluating and returning J(x).
   */
  void setJacobianOperator(JacobianOperatorFactory jacobian_operator);

  /**
   * Solves the system F(x) = 0
   * @param[in,out] x Solution to the system of nonlinear equations
   * @note The input value of @a x will be used as an initial guess for iterative nonlinear solution methods
   */
  void solve(mfem::Vector& x) const;

  /**
   * Returns the underlying solver object
   * @return A non-owning reference to the underlying nonlinear solver
   */
  mfem::NewtonSolver& nonlinearSolver() { return *nonlin_solver_; }

  /**
   * @overload
   */
  const mfem::NewtonSolver& nonlinearSolver() const { return *nonlin_solver_; }

  /**
   * Returns diagnostic counters when the nonlinear solver is PcgBlock.
   * @return Optional PCG-block diagnostics; empty for other nonlinear solvers
   */
  std::optional<PcgBlockDiagnostics> pcgBlockDiagnostics() const;

  /**
   * Returns diagnostic counters when the nonlinear solver is TrustRegion.
   * @return Optional TrustRegion diagnostics; empty for other nonlinear solvers
   */
  std::optional<TrustRegionDiagnostics> trustRegionDiagnostics() const;

  /**
   * Returns the underlying linear solver object
   * @return A non-owning reference to the underlying linear solver
   */
  mfem::Solver& linearSolver() { return *lin_solver_; }

  /**
   * @overload
   */
  const mfem::Solver& linearSolver() const { return *lin_solver_; }

  /**
   * Returns the underlying preconditioner
   * @return A pointer to the underlying preconditioner
   * @note This may be null if a preconditioner is not given
   */
  mfem::Solver& preconditioner() { return *preconditioner_; }

  /**
   * @overload
   */
  const mfem::Solver& preconditioner() const { return *preconditioner_; }

  /**
   * Input file parameters specific to this class
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

 private:
  /**
   * @brief The optional preconditioner (used for an iterative solver only)
   */
  std::unique_ptr<mfem::Solver> preconditioner_;

  /**
   * @brief The linear solver object, either custom, direct (SuperLU), or iterative
   */
  std::unique_ptr<mfem::Solver> lin_solver_;

  /**
   * @brief The nonlinear solver object
   */
  std::unique_ptr<mfem::NewtonSolver> nonlin_solver_;

  /**
   * @brief Whether the solver (linear solver) has been configured with the nonlinear solver
   * @note This is a workaround as some nonlinear solvers require SetOperator to be called
   * before SetSolver
   */
  bool nonlin_solver_set_solver_called_ = false;
};

/**
 * @brief A wrapper class for using the MFEM SuperLU solver with a HypreParMatrix
 */
class SuperLUSolver : public mfem::Solver {
 public:
  /**
   * @brief Constructs a wrapper over an mfem::SuperLUSolver
   * @param[in] comm The MPI communicator used by the vectors and matrices in the solve
   * @param[in] print_level The verbosity level for the mfem::SuperLUSolver
   */
  SuperLUSolver(int print_level, MPI_Comm comm) : superlu_solver_(comm)
  {
    superlu_solver_.SetColumnPermutation(mfem::superlu::PARMETIS);
    if (print_level == 0) {
      superlu_solver_.SetPrintStatistics(false);
    }
  }

  /**
   * @brief Factor and solve the linear system y = Op^{-1} x using DSuperLU
   *
   * @param input The input RHS vector
   * @param output The output solution vector
   */
  void Mult(const mfem::Vector& input, mfem::Vector& output) const;

  /**
   * @brief Set the underlying matrix operator to use in the solution algorithm
   *
   * @param op The matrix operator to factorize with SuperLU
   * @pre This operator must be an assembled HypreParMatrix or a BlockOperator
   * with all blocks either null or HypreParMatrixs for compatibility with
   * SuperLU
   */
  void SetOperator(const mfem::Operator& op);

 private:
  /**
   * @brief The owner of the SuperLU matrix for the gradient, stored
   * as a member variable for lifetime purposes
   */
  mutable std::unique_ptr<mfem::SuperLURowLocMatrix> superlu_mat_;

  /**
   * @brief The underlying MFEM-based SuperLU solver. It requires a special
   * SuperLU matrix type which we store in this object. This enables compatibility
   * with HypreParMatrix when used as an input.
   */
  mfem::SuperLUSolver superlu_solver_;
};

#ifdef MFEM_USE_STRUMPACK
/**
 * @brief A wrapper class for using the MFEM Strumpack solver with a HypreParMatrix
 */
class StrumpackSolver : public mfem::Solver {
 public:
  /**
   * @brief Constructs a wrapper over an mfem::STRUMPACKSolver
   * @param[in] comm The MPI communicator used by the vectors and matrices in the solve
   * @param[in] print_level The verbosity level for the mfem::STRUMPACKSolver
   */
  StrumpackSolver(int print_level, MPI_Comm comm) : strumpack_solver_(comm)
  {
    strumpack_solver_.SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
    strumpack_solver_.SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);

    if (print_level == 1) {
      strumpack_solver_.SetPrintFactorStatistics(true);
      strumpack_solver_.SetPrintSolveStatistics(true);
    }
  }

  /**
   * @brief Factor and solve the linear system y = Op^{-1} x using Strumpack
   *
   * @param input The input RHS vector
   * @param output The output solution vector
   */
  void Mult(const mfem::Vector& input, mfem::Vector& output) const;

  /**
   * @brief Set the underlying matrix operator to use in the solution algorithm
   *
   * @param op The matrix operator to factorize with Strumpack
   * @pre This operator must be an assembled HypreParMatrix for compatibility with Strumpack
   */
  void SetOperator(const mfem::Operator& op);

 private:
  /**
   * @brief The owner of the Strumpack matrix for the gradient, stored
   * as a member variable for lifetime purposes
   */
  mutable std::unique_ptr<mfem::STRUMPACKRowLocMatrix> strumpack_mat_;

  /**
   * @brief The underlying MFEM-based Strumpack solver. It requires a special
   * Strumpack matrix type which we store in this object. This enables compatibility
   * with HypreParMatrix when used as an input.
   */
  mfem::STRUMPACKSolver strumpack_solver_;
};

#endif

/**
 * @brief Function for building a monolithic parallel Hypre matrix from a block system of smaller Hypre matrices
 *
 * @param block_operator The block system of HypreParMatrices
 * @return The assembled monolithic HypreParMatrix
 *
 * @pre @a block_operator must have assembled HypreParMatrices for its sub-blocks
 */
std::unique_ptr<mfem::HypreParMatrix> buildMonolithicMatrix(const mfem::BlockOperator& block_operator);

/**
 * @brief Build a nonlinear solver using the nonlinear option struct
 *
 * @param nonlinear_opts The options to configure the nonlinear solution scheme
 * @param linear_opts The options to configure the linear solution scheme
 * @param preconditioner A preconditioner to help with either linear or nonlinear solves
 * @param comm The MPI communicator for the supplied nonlinear operators and HypreParVectors
 * @return The constructed nonlinear solver
 */
std::unique_ptr<mfem::NewtonSolver> buildNonlinearSolver(NonlinearSolverOptions nonlinear_opts,
                                                         const LinearSolverOptions& linear_opts,
                                                         mfem::Solver& preconditioner, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Build the linear solver and its associated preconditioner given a linear options struct
 *
 * @param linear_opts The options to configure the linear solver and preconditioner
 * @param comm The MPI communicator for the supplied HypreParMatrix and HypreParVectors
 * @return A pair containing the constructed linear solver and preconditioner objects
 */
std::pair<std::unique_ptr<mfem::Solver>, std::unique_ptr<mfem::Solver>> buildLinearSolverAndPreconditioner(
    LinearSolverOptions linear_opts = {}, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Build a preconditioner from the available options
 * @param linear_opts The options to configure the linear solver and preconditioner
 * @param comm The communicator for the underlying operator and HypreParVectors
 * @return A constructed preconditioner based on the input option
 */
std::unique_ptr<mfem::Solver> buildPreconditioner(LinearSolverOptions linear_opts,
                                                  [[maybe_unused]] MPI_Comm comm = MPI_COMM_WORLD);

#ifdef MFEM_USE_AMGX
/**
 * @brief Build an AMGX preconditioner
 *
 * @param options The options used to construct the AMGX preconditioner
 * @param comm The communicator for the underlying operator and HypreParVectors
 * @return The constructed AMGX preconditioner
 */
std::unique_ptr<mfem::AmgXSolver> buildAMGX(const AMGXOptions& options, const MPI_Comm comm);
#endif

}  // namespace smith

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<smith::LinearSolverOptions> {
  /// @brief Returns created object from Inlet container
  smith::LinearSolverOptions operator()(const axom::inlet::Container& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<smith::NonlinearSolverOptions> {
  /// @brief Returns created object from Inlet container
  smith::NonlinearSolverOptions operator()(const axom::inlet::Container& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<smith::EquationSolver> {
  /// @brief Returns created object from Inlet container
  smith::EquationSolver operator()(const axom::inlet::Container& base);
};
