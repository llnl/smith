// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/equation_solver.hpp"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#include "smith/smith_config.hpp"
#include "smith/infrastructure/profiling.hpp"
#include "smith/numerics/trust_region_solver.hpp"
#include "smith/infrastructure/logger.hpp"

namespace smith {

namespace {

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start)
{
  return std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - start).count();
}

}  // namespace

/// Newton solver with a 2-way line-search.  Reverts to regular Newton if max_line_search_iterations is set to 0.
class NewtonSolver : public mfem::NewtonSolver {
 protected:
  /// initial solution vector to do line-search off of
  mutable mfem::Vector x0;

  /// nonlinear solver options
  NonlinearSolverOptions nonlinear_options;

  /// reconstructed smith print level
  mutable size_t print_level = 0;

 public:
  /// constructor
  NewtonSolver(const NonlinearSolverOptions& nonlinear_opts) : nonlinear_options(nonlinear_opts) {}

#ifdef MFEM_USE_MPI
  /// parallel constructor
  NewtonSolver(MPI_Comm comm_, const NonlinearSolverOptions& nonlinear_opts)
      : mfem::NewtonSolver(comm_), nonlinear_options(nonlinear_opts)
  {
  }
#endif

  /// Evaluate the residual, put in rOut and return its norm.
  double evaluateNorm(const mfem::Vector& x, mfem::Vector& rOut) const
  {
    SMITH_MARK_FUNCTION;
    double normEval = std::numeric_limits<double>::max();
    try {
      oper->Mult(x, rOut);
      normEval = Norm(rOut);
    } catch (const std::exception&) {
      normEval = std::numeric_limits<double>::max();
    }
    return normEval;
  }

  /// assemble the jacobian
  void assembleJacobian(const mfem::Vector& x) const
  {
    SMITH_MARK_FUNCTION;
    grad = &oper->GetGradient(x);
    if (nonlinear_options.force_monolithic) {
      auto* grad_blocked = dynamic_cast<mfem::BlockOperator*>(grad);
      if (grad_blocked) grad = buildMonolithicMatrix(*grad_blocked).release();
    }
  }

  /// set the preconditioner for the linear solver
  void setPreconditioner() const
  {
    SMITH_MARK_FUNCTION;
    prec->SetOperator(*grad);
  }

  /// solve the linear system
  void solveLinearSystem(const mfem::Vector& r_, mfem::Vector& c_) const
  {
    SMITH_MARK_FUNCTION;
    prec->Mult(r_, c_);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
  }

  /// @overload
  void Mult(const mfem::Vector&, mfem::Vector& x) const
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

    print_level = static_cast<size_t>(std::max(nonlinear_options.print_level, 0));
    print_level = print_options.iterations ? std::max<size_t>(1, print_level) : print_level;
    print_level = print_options.summary ? std::max<size_t>(2, print_level) : print_level;

    using real_t = mfem::real_t;

    real_t norm, norm_goal = 0;
    norm = initial_norm = evaluateNorm(x, r);
    if (norm == 0.0) return;

    if (print_level == 1) {
      mfem::out << "Newton iteration " << std::setw(3) << 0 << " : ||r|| = " << std::setw(13) << norm << "\n";
    }

    norm_goal = std::max(rel_tol * initial_norm, abs_tol);
    prec->iterative_mode = false;

    int it = 0;
    for (; true; it++) {
      MFEM_ASSERT(mfem::IsFinite(norm), "norm = " << norm);
      if (print_level >= 2) {
        mfem::out << "Newton iteration " << std::setw(3) << it << " : ||r|| = " << std::setw(13) << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << std::setw(13) << (initial_norm != 0.0 ? norm / initial_norm : norm);
        }
        mfem::out << '\n';
      }

      if ((print_level >= 1) && (norm != norm)) {
        mfem::out << "Initial residual for Newton iteration is undefined/nan.\n";
        mfem::out << "Newton: No convergence!\n";
        return;
      }

      if (norm <= norm_goal && it >= nonlinear_options.min_iterations) {
        converged = true;
        break;
      } else if (it >= max_iter) {
        converged = false;
        break;
      }

      real_t norm_nm1 = norm;

      assembleJacobian(x);
      setPreconditioner();
      solveLinearSystem(r, c);

      // there must be a better way to do this?
      x0.SetSize(x.Size());
      x0 = 0.0;
      x0.Add(1.0, x);

      real_t stepScale = 1.0;
      add(x0, -stepScale, c, x);
      norm = evaluateNorm(x, r);

      const int max_ls_iters = nonlinear_options.max_line_search_iterations;
      static constexpr real_t reduction = 0.5;

      const double sufficientDecreaseParam = 0.0;  // 1e-15;
      const double cMagnitudeInR = sufficientDecreaseParam != 0.0 ? std::abs(Dot(c, r)) / norm_nm1 : 0.0;

      auto is_improved = [=](real_t currentNorm, real_t c_scale) {
        return currentNorm < norm_nm1 - sufficientDecreaseParam * c_scale * cMagnitudeInR;
      };

      // back-track linesearch
      int ls_iter = 0;
      int ls_iter_sum = 0;
      for (; !is_improved(norm, stepScale) && ls_iter < max_ls_iters; ++ls_iter, ++ls_iter_sum) {
        stepScale *= reduction;
        add(x0, -stepScale, c, x);
        norm = evaluateNorm(x, r);
      }

      // try the opposite direction and linesearch back from there
      if (max_ls_iters > 0 && ls_iter == max_ls_iters && !is_improved(norm, stepScale)) {
        stepScale = 1.0;
        add(x0, stepScale, c, x);
        norm = evaluateNorm(x, r);

        ls_iter = 0;
        for (; !is_improved(norm, stepScale) && ls_iter < max_ls_iters; ++ls_iter, ++ls_iter_sum) {
          stepScale *= reduction;
          add(x0, stepScale, c, x);
          norm = evaluateNorm(x, r);
        }

        // ok, the opposite direction was also terrible, lets go back, cut in half 1 last time and accept it hoping for
        // the best
        if (ls_iter == max_ls_iters && !is_improved(norm, stepScale)) {
          ++ls_iter_sum;
          stepScale *= reduction;
          add(x0, -stepScale, c, x);
          norm = evaluateNorm(x, r);
        }
      }

      if (ls_iter_sum) {
        if (print_level >= 2) {
          mfem::out << "Number of line search steps taken = " << ls_iter_sum << std::endl;
        }
        if (print_level >= 2 && (ls_iter_sum == 2 * max_ls_iters + 1)) {
          mfem::out << "The maximum number of line search cut back have occurred, the resulting residual may not have "
                       "decreased. "
                    << std::endl;
        }
      }
    }

    final_iter = it;
    final_norm = norm;

    if (print_level == 1) {
      mfem::out << "Newton iteration " << std::setw(3) << final_iter << " : ||r|| = " << std::setw(13) << norm << '\n';
    }
    if (!converged && print_level >= 1) {  // (print_options.summary || print_options.warnings)) {
      mfem::out << "Newton: No convergence!\n";
    }
  }
};

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
    H_d_old_at_accept.SetSize(size);
    d.SetSize(size);
    H_d.SetSize(size);
    Pr.SetSize(size);
    cauchy_point.SetSize(size);
    H_cauchy_point.SetSize(size);
    z = 0.0;
    H_z = 0.0;
    d_old = 0.0;
    H_d_old = 0.0;
    H_d_old_at_accept = 0.0;
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
  /// action of previous accepted hessian on previous step z_old
  mfem::Vector H_d_old_at_accept;
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

/// trust region printing utility function
void printTrustRegionInfo(double realWork, double modelObjective, size_t cgIters, double trSize, bool willAccept)
{
  mfem::out << "real work = " << std::setw(13) << realWork << ", model energy = " << std::setw(13)
            << modelObjective << ", cg iter = " << std::setw(7) << cgIters << ", next tr size = " << std::setw(8)
            << trSize << ", accepting = " << willAccept << std::endl;
}

/**
 * @brief Equation solver class based on a standard preconditioned trust-region algorithm
 *
 * This is a fairly standard implementation of 'The Conjugate Gradient Method and Trust Regions in Large Scale
 * Optimization' by T. Steihaug It is also called the Steihaug-Toint CG trust region algorithm (see also Trust Region
 * Methods by Conn, Gould, and Toint). One important difference is we do not compute an explicit energy.  Instead we
 * rely on an incremental work approximation: 0.5 (f^n + f^{n+1}) dot (u^{n+1} - u^n).  While less theoretically sound,
 * it appears to be very effective in practice.
 */
class TrustRegion : public mfem::NewtonSolver {
 protected:
  /// predicted solution
  mutable mfem::Vector x_pred;
  /// predicted residual
  mutable mfem::Vector r_pred;
  /// scratch
  mutable mfem::Vector scratch;
  /// left most eigenvectors
  mutable std::vector<std::shared_ptr<mfem::Vector>> left_mosts;
  /// the action of the stiffness/hessian (H) on the left most eigenvectors
  mutable std::vector<std::shared_ptr<mfem::Vector>> H_left_mosts;
  /// previous accepted-iteration Hessian actions on the retained left most eigenvectors
  mutable std::vector<std::shared_ptr<mfem::Vector>> previous_H_left_mosts;
  /// accepted TrustRegion steps, newest first
  mutable std::vector<std::shared_ptr<mfem::Vector>> accepted_step_history;
  /// initial state for this nonlinear solve, used as an optional history direction
  mutable mfem::Vector solve_start_x;
  mutable mfem::Vector min_residual_x;
  mutable double min_residual_norm = -1.0;

  /// nonlinear solution options
  NonlinearSolverOptions nonlinear_options;
  /// linear solution options
  LinearSolverOptions linear_options;

  /// handle to the preconditioner used by the trust region, it ignores the linear solver as a SPD preconditioner is
  /// currently required
  Solver& tr_precond;

  /// reconstructed smith print level
  mutable size_t print_level = 0;

 public:
  /// internal counter for hess-vecs
  mutable size_t num_hess_vecs = 0;
  /// internal counter for model CG hess-vecs
  mutable size_t num_model_hess_vecs = 0;
  /// internal counter for Cauchy-point hess-vecs
  mutable size_t num_cauchy_hess_vecs = 0;
  /// internal counter for line-search hess-vecs
  mutable size_t num_line_search_hess_vecs = 0;
  /// internal counter for preconditions
  mutable size_t num_preconds = 0;
  /// internal counter for residuals
  mutable size_t num_residuals = 0;
  /// internal counter for subspace solves
  mutable size_t num_subspace_solves = 0;
  /// internal counter for retained-leftmost Hessian-vector products used by subspace solves
  mutable size_t num_subspace_leftmost_hess_vecs = 0;
  /// internal counter for batched Hessian-vector groups used by subspace solves
  mutable size_t num_subspace_hess_vec_batches = 0;
  /// internal counter for Hessian-vector products inside subspace batches
  mutable size_t num_subspace_batched_hess_vecs = 0;
  /// internal counter for accepted-step history vectors added to subspace solves
  mutable size_t num_subspace_past_step_vectors = 0;
  /// internal counter for accepted-step history Hessian-vector products
  mutable size_t num_subspace_past_step_hess_vecs = 0;
  /// internal counter for nonlinear-solve-start directions added to subspace solves
  mutable size_t num_subspace_solve_start_vectors = 0;
  /// internal counter for nonlinear-solve-start Hessian-vector products
  mutable size_t num_subspace_solve_start_hess_vecs = 0;
  /// internal counter for quadratic subspace backend solves
  mutable size_t num_quadratic_subspace_solves = 0;
  /// internal counter for cubic subspace backend attempts
  mutable size_t num_cubic_subspace_attempts = 0;
  /// internal counter for cubic subspace candidates used
  mutable size_t num_cubic_subspace_uses = 0;
  /// internal counter for cubic attempts that returned quadratic candidate
  mutable size_t num_cubic_subspace_quadratic_fallbacks = 0;
  /// internal counter for matrix assembles
  mutable size_t num_jacobian_assembles = 0;
  /// internal counter for JacobianOperator evaluations
  mutable size_t num_jacobian_operator_evals = 0;
  /// internal counter for direct diagonal assemblies
  mutable size_t num_diagonal_assembles = 0;
  /// internal counter for model CG iterations
  mutable size_t num_cg_iterations = 0;
  /// internal counter for preconditioner operator updates
  mutable size_t num_preconditioner_updates = 0;
  /// internal counter for nonmonotone accepted steps
  mutable size_t num_nonmonotone_work_accepts = 0;
  /// internal counter for accepted steps that monotone acceptance would reject
  mutable size_t num_monotone_work_would_reject = 0;
  /// time spent evaluating residuals
  mutable double residual_seconds = 0.0;
  /// time spent applying Hessian-vector products
  mutable double hess_vec_seconds = 0.0;
  /// time spent applying model CG Hessian-vector products
  mutable double model_hess_vec_seconds = 0.0;
  /// time spent applying Cauchy-point Hessian-vector products
  mutable double cauchy_hess_vec_seconds = 0.0;
  /// time spent applying line-search Hessian-vector products
  mutable double line_search_hess_vec_seconds = 0.0;
  /// time spent applying JacobianOperator Hessian-vector products
  mutable double jacobian_operator_hess_vec_seconds = 0.0;
  /// time spent evaluating JacobianOperator factories
  mutable double jacobian_operator_eval_seconds = 0.0;
  /// time spent directly assembling diagonals
  mutable double diagonal_assembly_seconds = 0.0;
  /// time spent inverting direct diagonals
  mutable double diagonal_invert_seconds = 0.0;
  /// time spent applying preconditioners
  mutable double preconditioner_seconds = 0.0;
  /// total time spent in the nonlinear solve
  mutable double total_seconds = 0.0;
  /// time spent solving trust-region model problems
  mutable double model_solve_seconds = 0.0;
  /// total time spent in trust-region subspace solves
  mutable double subspace_seconds = 0.0;
  /// time spent building retained leftmost subspace directions
  mutable double subspace_leftmost_seconds = 0.0;
  /// time spent in subspace Hessian-vector batches
  mutable double subspace_hess_vec_batch_seconds = 0.0;
  /// time spent removing dependent directions for subspace solves
  mutable double subspace_filter_seconds = 0.0;
  /// time spent in dense subspace backend assembly/solve work
  mutable double subspace_backend_seconds = 0.0;
  /// time spent in subspace postprocessing and model-energy comparison
  mutable double subspace_finalize_seconds = 0.0;
  /// time spent building the Cauchy point
  mutable double cauchy_point_seconds = 0.0;
  /// time spent constructing dogleg steps
  mutable double dogleg_seconds = 0.0;
  /// time spent in line-search and trust-radius acceptance logic
  mutable double line_search_seconds = 0.0;
  /// time spent in dot products
  mutable double dot_seconds = 0.0;
  /// number of dot products
  mutable size_t num_dot_products = 0;
  /// number of dot product batches/reductions
  mutable size_t num_dot_reductions = 0;
  /// number of dot products in trust-region model solves
  mutable size_t num_model_dot_products = 0;
  /// number of dot products in Cauchy-point construction
  mutable size_t num_cauchy_dot_products = 0;
  /// number of dot products in dogleg construction
  mutable size_t num_dogleg_dot_products = 0;
  /// number of dot products in line-search and acceptance logic
  mutable size_t num_line_search_dot_products = 0;
  /// number of setup dot products outside the main per-step kernels
  mutable size_t num_setup_dot_products = 0;
  /// time spent in trust-region model-solve dot products
  mutable double model_dot_seconds = 0.0;
  /// time spent in Cauchy-point dot products
  mutable double cauchy_dot_seconds = 0.0;
  /// time spent in dogleg dot products
  mutable double dogleg_dot_seconds = 0.0;
  /// time spent in line-search dot products
  mutable double line_search_dot_seconds = 0.0;
  /// time spent in setup dot products
  mutable double setup_dot_seconds = 0.0;
  /// time spent in vector add/update operations
  mutable double vector_update_seconds = 0.0;
  /// time spent in vector copies and scaling operations
  mutable double vector_copy_scale_seconds = 0.0;
  /// time spent in boundary projection operations
  mutable double projection_seconds = 0.0;
  /// time spent assembling Jacobians
  mutable double jacobian_assembly_seconds = 0.0;
  /// time spent refreshing preconditioners
  mutable double preconditioner_update_seconds = 0.0;
  /// time spent in preconditioner SetOperator calls
  mutable double preconditioner_setup_seconds = 0.0;
  /// current accumulated actual work-surrogate level for nonmonotone acceptance
  mutable double current_work_objective = 0.0;
  /// last nonmonotone reference work surrogate
  mutable double last_nonmonotone_work_reference = 0.0;
  /// Optional JacobianOperator factory
  JacobianOperatorFactory jacobian_operator_factory;
  /// Cached JacobianOperator for current TrustRegion iteration
  mutable std::unique_ptr<JacobianOperator> current_jacobian_operator;
  /// Inverted scalar diagonal preconditioner for JacobianOperator mode
  mutable mfem::Vector inverse_diagonal_preconditioner;
  /// Current assembled Hessian clone used to preserve a valid previous Hessian
  mutable std::unique_ptr<mfem::Operator> current_hessian;
  /// Previous assembled Hessian used for cubic finite-difference subspace models
  mutable std::unique_ptr<mfem::Operator> previous_hessian;

#ifdef MFEM_USE_MPI
  /// constructor
  TrustRegion(MPI_Comm comm_, const NonlinearSolverOptions& nonlinear_opts, const LinearSolverOptions& linear_opts,
              Solver& tPrec)
      : mfem::NewtonSolver(comm_), nonlinear_options(nonlinear_opts), linear_options(linear_opts), tr_precond(tPrec)
  {
  }
#endif

  /// Timed dot product with global and grouped accounting.
  double timedDot(const mfem::Vector& a, const mfem::Vector& b, size_t& group_count, double& group_seconds) const
  {
    auto start = Clock::now();
    const double value = Dot(a, b);
    const double seconds = secondsSince(start);
    ++num_dot_products;
    ++num_dot_reductions;
    ++group_count;
    dot_seconds += seconds;
    group_seconds += seconds;
    return value;
  }

  /// Timed pair of dot products with one local vector pass and one MPI reduction when possible.
  std::pair<double, double> timedDot2(const mfem::Vector& a0, const mfem::Vector& b0, const mfem::Vector& a1,
                                      const mfem::Vector& b1, size_t& group_count, double& group_seconds) const
  {
    if (dot_oper) {
      return {timedDot(a0, b0, group_count, group_seconds), timedDot(a1, b1, group_count, group_seconds)};
    }

    MFEM_ASSERT(a0.Size() == b0.Size(), "Incompatible vector sizes.");
    MFEM_ASSERT(a1.Size() == b1.Size(), "Incompatible vector sizes.");

    auto start = Clock::now();
    mfem::real_t products[2] = {0.0, 0.0};
    if (a0.Size() == a1.Size()) {
      for (int i = 0; i < a0.Size(); ++i) {
        products[0] += a0[i] * b0[i];
        products[1] += a1[i] * b1[i];
      }
    } else {
      for (int i = 0; i < a0.Size(); ++i) {
        products[0] += a0[i] * b0[i];
      }
      for (int i = 0; i < a1.Size(); ++i) {
        products[1] += a1[i] * b1[i];
      }
    }

#ifdef MFEM_USE_MPI
    const MPI_Comm dot_comm = GetComm();
    if (dot_comm != MPI_COMM_NULL) {
      mfem::real_t global_products[2] = {0.0, 0.0};
      MPI_Allreduce(products, global_products, 2, MFEM_MPI_REAL_T, MPI_SUM, dot_comm);
      products[0] = global_products[0];
      products[1] = global_products[1];
    }
#endif

    const double seconds = secondsSince(start);
    num_dot_products += 2;
    ++num_dot_reductions;
    group_count += 2;
    dot_seconds += seconds;
    group_seconds += seconds;
    return {products[0], products[1]};
  }

  struct Dot4Result {
    double v0 = 0.0;
    double v1 = 0.0;
    double v2 = 0.0;
    double v3 = 0.0;
  };

  /// Timed four-dot batch with one local vector pass and one MPI reduction when possible.
  Dot4Result timedDot4(const mfem::Vector& a0, const mfem::Vector& b0, const mfem::Vector& a1, const mfem::Vector& b1,
                       const mfem::Vector& a2, const mfem::Vector& b2, const mfem::Vector& a3,
                       const mfem::Vector& b3, size_t& group_count, double& group_seconds) const
  {
    if (dot_oper) {
      return {.v0 = timedDot(a0, b0, group_count, group_seconds),
              .v1 = timedDot(a1, b1, group_count, group_seconds),
              .v2 = timedDot(a2, b2, group_count, group_seconds),
              .v3 = timedDot(a3, b3, group_count, group_seconds)};
    }

    MFEM_ASSERT(a0.Size() == b0.Size(), "Incompatible vector sizes.");
    MFEM_ASSERT(a1.Size() == b1.Size(), "Incompatible vector sizes.");
    MFEM_ASSERT(a2.Size() == b2.Size(), "Incompatible vector sizes.");
    MFEM_ASSERT(a3.Size() == b3.Size(), "Incompatible vector sizes.");
    MFEM_ASSERT(a0.Size() == a1.Size() && a0.Size() == a2.Size() && a0.Size() == a3.Size(),
                "timedDot4 currently requires equal vector sizes.");

    auto start = Clock::now();
    mfem::real_t products[4] = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < a0.Size(); ++i) {
      products[0] += a0[i] * b0[i];
      products[1] += a1[i] * b1[i];
      products[2] += a2[i] * b2[i];
      products[3] += a3[i] * b3[i];
    }

#ifdef MFEM_USE_MPI
    const MPI_Comm dot_comm = GetComm();
    if (dot_comm != MPI_COMM_NULL) {
      mfem::real_t global_products[4] = {0.0, 0.0, 0.0, 0.0};
      MPI_Allreduce(products, global_products, 4, MFEM_MPI_REAL_T, MPI_SUM, dot_comm);
      for (int i = 0; i < 4; ++i) {
        products[i] = global_products[i];
      }
    }
#endif

    const double seconds = secondsSince(start);
    num_dot_products += 4;
    ++num_dot_reductions;
    group_count += 4;
    dot_seconds += seconds;
    group_seconds += seconds;
    return {.v0 = products[0], .v1 = products[1], .v2 = products[2], .v3 = products[3]};
  }

  template <typename HessVecFunc>
  void batchedSubspaceHessVec(HessVecFunc hess_vec_func, const std::vector<const mfem::Vector*>& inputs,
                              const std::vector<mfem::Vector*>& outputs) const
  {
    MFEM_VERIFY(inputs.size() == outputs.size(), "Subspace Hessian-vector batch input/output size mismatch");
    if (inputs.empty()) {
      return;
    }

    auto start = Clock::now();
    ++num_subspace_hess_vec_batches;
    num_subspace_batched_hess_vecs += inputs.size();
    for (size_t i = 0; i < inputs.size(); ++i) {
      hess_vec_func(*inputs[i], *outputs[i]);
    }
    subspace_hess_vec_batch_seconds += secondsSince(start);
  }

  template <typename HessVecFunc>
  void timedModelHessVec(HessVecFunc hess_vec_func, const mfem::Vector& input, mfem::Vector& output) const
  {
    auto start = Clock::now();
    hess_vec_func(input, output);
    model_hess_vec_seconds += secondsSince(start);
    ++num_model_hess_vecs;
  }

  template <typename HessVecFunc>
  void timedCauchyHessVec(HessVecFunc hess_vec_func, const mfem::Vector& input, mfem::Vector& output) const
  {
    auto start = Clock::now();
    hess_vec_func(input, output);
    cauchy_hess_vec_seconds += secondsSince(start);
    ++num_cauchy_hess_vecs;
  }

  template <typename HessVecFunc>
  void timedLineSearchHessVec(HessVecFunc hess_vec_func, const mfem::Vector& input, mfem::Vector& output) const
  {
    auto start = Clock::now();
    hess_vec_func(input, output);
    line_search_hess_vec_seconds += secondsSince(start);
    ++num_line_search_hess_vecs;
  }

  double nonmonotoneWorkReference(const std::vector<double>& work_objective_history) const
  {
    if (work_objective_history.empty()) {
      return current_work_objective;
    }
    return *std::max_element(work_objective_history.begin(), work_objective_history.end());
  }

  void pushWorkObjectiveHistory(std::vector<double>& work_objective_history, double objective) const
  {
    const int window = nonlinear_options.trust_nonmonotone_window;
    if (window <= 0) {
      return;
    }
    work_objective_history.push_back(objective);
    while (work_objective_history.size() > static_cast<size_t>(window)) {
      work_objective_history.erase(work_objective_history.begin());
    }
  }

  void pushAcceptedStepHistory(const mfem::Vector& step) const
  {
    if (nonlinear_options.trust_num_past_steps <= 0) {
      accepted_step_history.clear();
      return;
    }

    accepted_step_history.insert(accepted_step_history.begin(), std::make_shared<mfem::Vector>(step));
    const size_t max_size = static_cast<size_t>(nonlinear_options.trust_num_past_steps) + 1;
    while (accepted_step_history.size() > max_size) {
      accepted_step_history.pop_back();
    }
  }

  /// finds tau s.t. (z + tau*d)^2 = trSize^2
  void projectToBoundaryWithCoefs(mfem::Vector& z, const mfem::Vector& d, double delta, double zz, double zd,
                                  double dd) const
  {
    auto start = Clock::now();
    // find z + tau d
    double deltadelta_m_zz = delta * delta - zz;
    if (deltadelta_m_zz == 0) return;  // already on boundary
    double tau = (std::sqrt(deltadelta_m_zz * dd + zd * zd) - zd) / dd;
    z.Add(tau, d);
    projection_seconds += secondsSince(start);
  }

  /// solve the exact trust-region subspace problem with directions ds, and the leftmosts
  template <typename HessVecFunc>
  void solveTheSubspaceProblem([[maybe_unused]] mfem::Vector& z, [[maybe_unused]] const HessVecFunc& hess_vec_func,
                               [[maybe_unused]] const std::vector<const mfem::Vector*> ds,
                               [[maybe_unused]] const std::vector<const mfem::Vector*> Hds,
                               [[maybe_unused]] const mfem::Vector& g, [[maybe_unused]] double delta,
                               [[maybe_unused]] int num_leftmost,
                               [[maybe_unused]] std::vector<std::shared_ptr<mfem::Vector>>& candidate_left_mosts,
                               [[maybe_unused]] const mfem::Vector& previous_step,
                               [[maybe_unused]] const mfem::Vector* previous_H_previous_step,
                               [[maybe_unused]] bool allow_cubic_subspace) const
  {
    SMITH_MARK_FUNCTION;
    auto subspace_start = Clock::now();
    ++num_subspace_solves;

    std::vector<const mfem::Vector*> directions;
    for (auto& d : ds) {
      directions.emplace_back(d);
    }
    for (auto& left : left_mosts) {
      directions.emplace_back(left.get());
    }

    std::vector<const mfem::Vector*> H_directions;
    for (auto& Hd : Hds) {
      H_directions.emplace_back(Hd);
    }
    for (auto& H_left : H_left_mosts) {
      H_directions.emplace_back(H_left.get());
    }

    mfem::Vector b(g);
    b *= -1;

    mfem::Vector sol;
    std::vector<std::shared_ptr<mfem::Vector>> leftvecs;
    std::vector<double> leftvals;
    double energy_change;

    try {
      auto backend_start = Clock::now();
      if (nonlinear_options.trust_use_cubic_subspace && allow_cubic_subspace && previous_hessian) {
        std::vector<mfem::Vector> previous_H_vectors;
        std::vector<const mfem::Vector*> previous_H_directions;
        previous_H_vectors.reserve(directions.size());
        previous_H_directions.reserve(directions.size());
        for (const auto* direction : directions) {
          previous_H_vectors.emplace_back(direction->Size());
          previous_hessian->Mult(*direction, previous_H_vectors.back());
          previous_H_directions.emplace_back(&previous_H_vectors.back());
        }
        ++num_cubic_subspace_attempts;
        bool used_cubic = false;
        std::tie(sol, leftvecs, leftvals, energy_change) = solveCubicSubspaceProblemMfem(
            directions, H_directions, previous_H_directions, previous_step, b, delta, num_leftmost, &used_cubic);
        if (used_cubic) {
          ++num_cubic_subspace_uses;
        } else {
          ++num_cubic_subspace_quadratic_fallbacks;
          ++num_quadratic_subspace_solves;
        }
      } else {
        ++num_quadratic_subspace_solves;
        std::tie(sol, leftvecs, leftvals, energy_change) =
            solveSubspaceProblem(directions, H_directions, b, delta, num_leftmost);
      }
      subspace_backend_seconds += secondsSince(backend_start);
    } catch (const std::exception& e) {
      if (print_level >= 1) {
        mfem::out << "subspace solve failed with " << e.what() << std::endl;
      }
      subspace_seconds += secondsSince(subspace_start);
      return;
    }

    auto finalize_start = Clock::now();
    candidate_left_mosts.clear();
    for (auto& lv : leftvecs) {
      candidate_left_mosts.emplace_back(std::move(lv));
    }

    double base_energy = computeEnergy(g, hess_vec_func, z);
    double subspace_energy = computeEnergy(g, hess_vec_func, sol);

    if (print_level >= 2) {
      double leftval = leftvals.size() ? leftvals[0] : 1.0;
      mfem::out << "Energy using subspace solver from: " << base_energy << ", to: " << subspace_energy << " / "
                << energy_change << ".  Min eig: " << leftval << std::endl;
    }

    if (subspace_energy < base_energy) {
      z = sol;
    }
    subspace_finalize_seconds += secondsSince(finalize_start);
    subspace_seconds += secondsSince(subspace_start);
  }

  /// finds tau s.t. (z + tau*(y-z))^2 = trSize^2
  void projectToBoundaryBetweenWithCoefs(mfem::Vector& z, const mfem::Vector& y, double trSize, double zz, double zy,
                                         double yy) const
  {
    auto start = Clock::now();
    double dd = yy - 2 * zy + zz;
    double zd = zy - zz;
    double tau = (std::sqrt((trSize * trSize - zz) * dd + zd * zd) - zd) / dd;
    z.Add(-tau, z);
    z.Add(tau, y);
    projection_seconds += secondsSince(start);
  }

  /// take a dogleg step in direction s, solution norm must be within trSize
  void doglegStep(const mfem::Vector& cp, const mfem::Vector& newtonP, double trSize, mfem::Vector& s) const
  {
    SMITH_MARK_FUNCTION;
    auto [cc, nn] = timedDot2(cp, cp, newtonP, newtonP, num_dogleg_dot_products, dogleg_dot_seconds);
    double tt = trSize * trSize;

    auto update_start = Clock::now();
    s = 0.0;
    vector_copy_scale_seconds += secondsSince(update_start);
    if (cc >= tt) {
      update_start = Clock::now();
      add(s, std::sqrt(tt / cc), cp, s);
      vector_update_seconds += secondsSince(update_start);
    } else if (cc > nn) {
      if (print_level >= 2) {
        mfem::out << "cp outside newton, preconditioner likely inaccurate\n";
      }
      update_start = Clock::now();
      add(s, 1.0, cp, s);
      vector_update_seconds += secondsSince(update_start);
    } else if (nn > tt) {  // on the dogleg (we have nn >= cc, and tt >= cc)
      update_start = Clock::now();
      add(s, 1.0, cp, s);
      vector_update_seconds += secondsSince(update_start);
      double cn = timedDot(cp, newtonP, num_dogleg_dot_products, dogleg_dot_seconds);
      projectToBoundaryBetweenWithCoefs(s, newtonP, trSize, cc, cn, nn);
    } else {
      update_start = Clock::now();
      s = newtonP;
      vector_copy_scale_seconds += secondsSince(update_start);
    }
  }

  /// compute the energy of the linearized system for a given solution vector z
  template <typename HessVecFunc>
  double computeEnergy(const mfem::Vector& r_local, const HessVecFunc& H, const mfem::Vector& z) const
  {
    SMITH_MARK_FUNCTION;
    double rz = timedDot(r_local, z, num_line_search_dot_products, line_search_dot_seconds);
    mfem::Vector tmp(r_local);
    tmp = 0.0;
    H(z, tmp);
    return rz + 0.5 * timedDot(z, tmp, num_line_search_dot_products, line_search_dot_seconds);
  }

  /// Minimize quadratic sub-problem given residual vector, the action of the stiffness and a preconditioner
  template <typename HessVecFunc, typename PrecondFunc>
  void solveTrustRegionModelProblem(const mfem::Vector& r0, mfem::Vector& rCurrent, HessVecFunc hess_vec_func,
                                    PrecondFunc precond, const TrustRegionSettings& settings, double& trSize,
                                    TrustRegionResults& results, double r0_norm_squared) const
  {
    SMITH_MARK_FUNCTION;
    // minimize r0@z + 0.5*z@J@z
    results.interior_status = TrustRegionResults::Status::Interior;
    results.cg_iterations_count = 0;

    auto& z = results.z;
    auto& cgIter = results.cg_iterations_count;
    auto& d = results.d;
    auto& Pr = results.Pr;
    auto& Hd = results.H_d;

    const double cg_tol_squared = settings.cg_tol * settings.cg_tol;

    if (r0_norm_squared <= cg_tol_squared && settings.min_cg_iterations == 0) {
      if (print_level >= 2) {
        mfem::out << "Trust region solution state within tolerance on first iteration."
                  << "\n";
      }
      return;
    }

    auto copy_start = Clock::now();
    rCurrent = r0;
    vector_copy_scale_seconds += secondsSince(copy_start);
    precond(rCurrent, Pr);

    // d = -Pr
    copy_start = Clock::now();
    d = Pr;
    d *= -1.0;

    z = 0.0;
    vector_copy_scale_seconds += secondsSince(copy_start);
    double zz = 0.;
    double rPr = timedDot(rCurrent, Pr, num_model_dot_products, model_dot_seconds);

    // std::cout << "initial energy = " << computeEnergy(r0, hess_vec_func, z) << std::endl;

    for (cgIter = 1; cgIter <= settings.max_cg_iterations; ++cgIter) {
      hess_vec_func(d, Hd);
      const auto dots = timedDot4(d, rCurrent, d, Hd, z, d, d, d, num_model_dot_products, model_dot_seconds);
      double descent_check = dots.v0;
      double curvature = dots.v1;
      double zd = dots.v2;
      double dd = dots.v3;
      if (descent_check > 0) {
        copy_start = Clock::now();
        d *= -1;
        Hd *= -1;
        vector_copy_scale_seconds += secondsSince(copy_start);
        results.interior_status = TrustRegionResults::Status::NonDescentDirection;
        descent_check *= -1.0;
        curvature *= -1.0;
        zd *= -1.0;
      }

      const double alphaCg = curvature != 0.0 ? rPr / curvature : 0.0;
      const double zzNp1 = zz + 2.0 * alphaCg * zd + alphaCg * alphaCg * dd;

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

      auto& zPred = Pr;  // re-use Pr memory.
                         // This predicted step will no longer be used by the time Pr is, so we can avoid an extra
                         // vector floating around
      auto update_start = Clock::now();
      add(z, alphaCg, d, zPred);
      vector_update_seconds += secondsSince(update_start);

      copy_start = Clock::now();
      z = zPred;
      vector_copy_scale_seconds += secondsSince(copy_start);

      if (results.interior_status == TrustRegionResults::Status::NonDescentDirection) {
        if (print_level >= 2) {
          mfem::out << "Found a non descent direction\n";
        }
        return;
      }

      update_start = Clock::now();
      add(rCurrent, alphaCg, Hd, rCurrent);
      vector_update_seconds += secondsSince(update_start);

      precond(rCurrent, Pr);
      auto [rPrNp1, r_current_norm_squared] =
          timedDot2(rCurrent, Pr, rCurrent, rCurrent, num_model_dot_products, model_dot_seconds);
      if (r_current_norm_squared <= cg_tol_squared && cgIter >= settings.min_cg_iterations) {
        return;
      }

      double beta = rPrNp1 / rPr;
      rPr = rPrNp1;
      update_start = Clock::now();
      add(-1.0, Pr, beta, d, d);
      vector_update_seconds += secondsSince(update_start);

      zz = zzNp1;
    }
    cgIter--;  // if all cg iterations are taken, correct for output
  }

  std::unique_ptr<mfem::Operator> cloneAssembledOperator(const mfem::Operator& op) const
  {
    if (const auto* hypre_matrix = dynamic_cast<const mfem::HypreParMatrix*>(&op)) {
      return std::make_unique<mfem::HypreParMatrix>(*hypre_matrix);
    }
    if (const auto* sparse_matrix = dynamic_cast<const mfem::SparseMatrix*>(&op)) {
      return std::make_unique<mfem::SparseMatrix>(*sparse_matrix);
    }
    if (const auto* block_operator = dynamic_cast<const mfem::BlockOperator*>(&op)) {
      return buildMonolithicMatrix(*block_operator);
    }
    return nullptr;
  }

  /// assemble the jacobian
  void assembleJacobian(const mfem::Vector& x) const
  {
    SMITH_MARK_FUNCTION;
    auto start = Clock::now();
    ++num_jacobian_assembles;
    if (nonlinear_options.trust_use_cubic_subspace) {
      previous_hessian = std::move(current_hessian);
    }
    grad = &oper->GetGradient(x);
    if (nonlinear_options.force_monolithic) {
      auto* grad_blocked = dynamic_cast<mfem::BlockOperator*>(grad);
      if (grad_blocked) grad = buildMonolithicMatrix(*grad_blocked).release();
    }
    if (nonlinear_options.trust_use_cubic_subspace) {
      current_hessian = cloneAssembledOperator(*grad);
    }
    jacobian_assembly_seconds += secondsSince(start);
  }

  /// Set an optional JacobianOperator factory.
  void setJacobianOperator(JacobianOperatorFactory jacobian_operator)
  {
    jacobian_operator_factory = std::move(jacobian_operator);
  }

  /// Evaluate and cache the JacobianOperator at x.
  void updateJacobianOperator(const mfem::Vector& x) const
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(!jacobian_operator_factory, "No JacobianOperator factory is registered.");
    auto start = Clock::now();
    ++num_jacobian_operator_evals;
    current_jacobian_operator = jacobian_operator_factory(x);
    SLIC_ERROR_ROOT_IF(!current_jacobian_operator, "JacobianOperator factory returned a null operator.");
    jacobian_operator_eval_seconds += secondsSince(start);
  }

  /// Assemble and invert the scalar diagonal preconditioner from the current JacobianOperator.
  void updateDiagonalPreconditioner() const
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(!current_jacobian_operator, "Cannot build diagonal preconditioner without a JacobianOperator.");

    auto diagonal_start = Clock::now();
    current_jacobian_operator->assembleDiagonal(inverse_diagonal_preconditioner);
    diagonal_assembly_seconds += secondsSince(diagonal_start);
    ++num_diagonal_assembles;

    auto invert_start = Clock::now();
    double max_abs_diag = 0.0;
    for (int i = 0; i < inverse_diagonal_preconditioner.Size(); ++i) {
      max_abs_diag = std::max(max_abs_diag, std::abs(inverse_diagonal_preconditioner[i]));
    }

    const double floor = nonlinear_options.pcg_diagonal_floor * max_abs_diag;
    SLIC_ERROR_ROOT_IF(!(floor > 0.0), "Cannot invert a zero Jacobian diagonal for TrustRegion preconditioning.");
    for (int i = 0; i < inverse_diagonal_preconditioner.Size(); ++i) {
      inverse_diagonal_preconditioner[i] = 1.0 / std::max(std::abs(inverse_diagonal_preconditioner[i]), floor);
    }
    diagonal_invert_seconds += secondsSince(invert_start);
  }

  /// evaluate the nonlinear residual
  mfem::real_t computeResidual(const mfem::Vector& x_, mfem::Vector& r_) const
  {
    SMITH_MARK_FUNCTION;
    auto start = Clock::now();
    ++num_residuals;
    oper->Mult(x_, r_);
    const auto norm = Norm(r_);
    residual_seconds += secondsSince(start);
    return norm;
  }

  /// apply the action of the current Jacobian representation to a vector
  void hessVec(const mfem::Vector& x_, mfem::Vector& v_) const
  {
    SMITH_MARK_FUNCTION;
    auto start = Clock::now();
    ++num_hess_vecs;
    if (nonlinear_options.trust_use_jacobian_operator) {
      SLIC_ERROR_ROOT_IF(!current_jacobian_operator, "TrustRegion JacobianOperator mode has no current operator.");
      current_jacobian_operator->Mult(x_, v_);
      const double seconds = secondsSince(start);
      hess_vec_seconds += seconds;
      jacobian_operator_hess_vec_seconds += seconds;
    } else {
      grad->Mult(x_, v_);
      hess_vec_seconds += secondsSince(start);
    }
  }

  /// apply trust region specific preconditioner
  void precond(const mfem::Vector& x_, mfem::Vector& v_) const
  {
    SMITH_MARK_FUNCTION;
    auto start = Clock::now();
    ++num_preconds;
    if (nonlinear_options.trust_use_jacobian_operator) {
      SLIC_ERROR_ROOT_IF(inverse_diagonal_preconditioner.Size() != x_.Size(),
                         "TrustRegion JacobianOperator diagonal preconditioner is not initialized.");
      v_.SetSize(x_.Size());
      for (int i = 0; i < x_.Size(); ++i) {
        v_[i] = inverse_diagonal_preconditioner[i] * x_[i];
      }
    } else {
      tr_precond.Mult(x_, v_);
    }
    preconditioner_seconds += secondsSince(start);
  };

  /// Return solver diagnostic counters.
  TrustRegionDiagnostics diagnostics() const
  {
    return {.num_residuals = num_residuals,
            .num_hess_vecs = num_hess_vecs,
            .num_model_hess_vecs = num_model_hess_vecs,
            .num_cauchy_hess_vecs = num_cauchy_hess_vecs,
            .num_line_search_hess_vecs = num_line_search_hess_vecs,
            .num_preconds = num_preconds,
            .num_jacobian_assembles = num_jacobian_assembles,
            .num_jacobian_operator_evals = num_jacobian_operator_evals,
            .num_diagonal_assembles = num_diagonal_assembles,
            .num_cg_iterations = num_cg_iterations,
            .num_subspace_solves = num_subspace_solves,
            .num_subspace_leftmost_hess_vecs = num_subspace_leftmost_hess_vecs,
            .num_subspace_hess_vec_batches = num_subspace_hess_vec_batches,
            .num_subspace_batched_hess_vecs = num_subspace_batched_hess_vecs,
            .num_subspace_past_step_vectors = num_subspace_past_step_vectors,
            .num_subspace_past_step_hess_vecs = num_subspace_past_step_hess_vecs,
            .num_subspace_solve_start_vectors = num_subspace_solve_start_vectors,
            .num_subspace_solve_start_hess_vecs = num_subspace_solve_start_hess_vecs,
            .num_quadratic_subspace_solves = num_quadratic_subspace_solves,
            .num_cubic_subspace_attempts = num_cubic_subspace_attempts,
            .num_cubic_subspace_uses = num_cubic_subspace_uses,
            .num_cubic_subspace_quadratic_fallbacks = num_cubic_subspace_quadratic_fallbacks,
            .num_preconditioner_updates = num_preconditioner_updates,
            .num_nonmonotone_work_accepts = num_nonmonotone_work_accepts,
            .num_monotone_work_would_reject = num_monotone_work_would_reject,
            .residual_seconds = residual_seconds,
            .hess_vec_seconds = hess_vec_seconds,
            .model_hess_vec_seconds = model_hess_vec_seconds,
            .cauchy_hess_vec_seconds = cauchy_hess_vec_seconds,
            .line_search_hess_vec_seconds = line_search_hess_vec_seconds,
            .jacobian_operator_hess_vec_seconds = jacobian_operator_hess_vec_seconds,
            .jacobian_operator_eval_seconds = jacobian_operator_eval_seconds,
            .diagonal_assembly_seconds = diagonal_assembly_seconds,
            .diagonal_invert_seconds = diagonal_invert_seconds,
            .preconditioner_seconds = preconditioner_seconds,
            .total_seconds = total_seconds,
            .model_solve_seconds = model_solve_seconds,
            .subspace_seconds = subspace_seconds,
            .subspace_leftmost_seconds = subspace_leftmost_seconds,
            .subspace_hess_vec_batch_seconds = subspace_hess_vec_batch_seconds,
            .subspace_filter_seconds = subspace_filter_seconds,
            .subspace_backend_seconds = subspace_backend_seconds,
            .subspace_project_A_seconds = trustRegionSubspaceTimings().project_A_seconds,
            .subspace_project_gram_seconds = trustRegionSubspaceTimings().project_gram_seconds,
            .subspace_project_b_seconds = trustRegionSubspaceTimings().project_b_seconds,
            .subspace_basis_seconds = trustRegionSubspaceTimings().basis_seconds,
            .subspace_reduced_A_seconds = trustRegionSubspaceTimings().reduced_A_seconds,
            .subspace_dense_eigensystem_seconds = trustRegionSubspaceTimings().dense_eigensystem_seconds,
            .subspace_dense_trust_solve_seconds = trustRegionSubspaceTimings().dense_trust_solve_seconds,
            .subspace_reconstruct_solution_seconds = trustRegionSubspaceTimings().reconstruct_solution_seconds,
            .subspace_reconstruct_leftmost_seconds = trustRegionSubspaceTimings().reconstruct_leftmost_seconds,
            .subspace_finalize_seconds = subspace_finalize_seconds,
            .cauchy_point_seconds = cauchy_point_seconds,
            .dogleg_seconds = dogleg_seconds,
            .line_search_seconds = line_search_seconds,
            .dot_seconds = dot_seconds,
            .num_dot_products = num_dot_products,
            .num_dot_reductions = num_dot_reductions,
            .num_model_dot_products = num_model_dot_products,
            .num_cauchy_dot_products = num_cauchy_dot_products,
            .num_dogleg_dot_products = num_dogleg_dot_products,
            .num_line_search_dot_products = num_line_search_dot_products,
            .num_setup_dot_products = num_setup_dot_products,
            .model_dot_seconds = model_dot_seconds,
            .cauchy_dot_seconds = cauchy_dot_seconds,
            .dogleg_dot_seconds = dogleg_dot_seconds,
            .line_search_dot_seconds = line_search_dot_seconds,
            .setup_dot_seconds = setup_dot_seconds,
            .vector_update_seconds = vector_update_seconds,
            .vector_copy_scale_seconds = vector_copy_scale_seconds,
            .projection_seconds = projection_seconds,
            .jacobian_assembly_seconds = jacobian_assembly_seconds,
            .preconditioner_update_seconds = preconditioner_update_seconds,
            .preconditioner_setup_seconds = preconditioner_setup_seconds,
            .last_work_objective = current_work_objective,
            .last_nonmonotone_work_reference = last_nonmonotone_work_reference};
  }

  /// @overload
  void Mult(const mfem::Vector&, mfem::Vector& X) const
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");
    auto total_start = Clock::now();

    print_level = static_cast<size_t>(std::max(nonlinear_options.print_level, 0));
    print_level = print_options.iterations ? std::max<size_t>(1, print_level) : print_level;
    print_level = print_options.summary ? std::max<size_t>(2, print_level) : print_level;

    using real_t = mfem::real_t;

    num_hess_vecs = 0;
    num_model_hess_vecs = 0;
    num_cauchy_hess_vecs = 0;
    num_line_search_hess_vecs = 0;
    num_preconds = 0;
    num_residuals = 0;
    num_subspace_solves = 0;
    num_subspace_leftmost_hess_vecs = 0;
    num_subspace_hess_vec_batches = 0;
    num_subspace_batched_hess_vecs = 0;
    num_subspace_past_step_vectors = 0;
    num_subspace_past_step_hess_vecs = 0;
    num_subspace_solve_start_vectors = 0;
    num_subspace_solve_start_hess_vecs = 0;
    num_quadratic_subspace_solves = 0;
    num_cubic_subspace_attempts = 0;
    num_cubic_subspace_uses = 0;
    num_cubic_subspace_quadratic_fallbacks = 0;
    num_jacobian_assembles = 0;
    num_jacobian_operator_evals = 0;
    num_diagonal_assembles = 0;
    num_cg_iterations = 0;
    num_preconditioner_updates = 0;
    num_nonmonotone_work_accepts = 0;
    num_monotone_work_would_reject = 0;
    residual_seconds = 0.0;
    hess_vec_seconds = 0.0;
    model_hess_vec_seconds = 0.0;
    cauchy_hess_vec_seconds = 0.0;
    line_search_hess_vec_seconds = 0.0;
    jacobian_operator_hess_vec_seconds = 0.0;
    jacobian_operator_eval_seconds = 0.0;
    diagonal_assembly_seconds = 0.0;
    diagonal_invert_seconds = 0.0;
    preconditioner_seconds = 0.0;
    total_seconds = 0.0;
    model_solve_seconds = 0.0;
    subspace_seconds = 0.0;
    subspace_leftmost_seconds = 0.0;
    subspace_hess_vec_batch_seconds = 0.0;
    subspace_filter_seconds = 0.0;
    subspace_backend_seconds = 0.0;
    subspace_finalize_seconds = 0.0;
    cauchy_point_seconds = 0.0;
    dogleg_seconds = 0.0;
    line_search_seconds = 0.0;
    dot_seconds = 0.0;
    num_dot_products = 0;
    num_dot_reductions = 0;
    num_model_dot_products = 0;
    num_cauchy_dot_products = 0;
    num_dogleg_dot_products = 0;
    num_line_search_dot_products = 0;
    num_setup_dot_products = 0;
    model_dot_seconds = 0.0;
    cauchy_dot_seconds = 0.0;
    dogleg_dot_seconds = 0.0;
    line_search_dot_seconds = 0.0;
    setup_dot_seconds = 0.0;
    vector_update_seconds = 0.0;
    vector_copy_scale_seconds = 0.0;
    projection_seconds = 0.0;
    jacobian_assembly_seconds = 0.0;
    preconditioner_update_seconds = 0.0;
    preconditioner_setup_seconds = 0.0;
    current_work_objective = 0.0;
    last_nonmonotone_work_reference = 0.0;
    accepted_step_history.clear();
    resetTrustRegionSubspaceTimings();
    solve_start_x.SetSize(X.Size());
    solve_start_x = X;
    min_residual_x.SetSize(X.Size());
    min_residual_x = X;
    current_jacobian_operator.reset();
    inverse_diagonal_preconditioner.SetSize(0);
    previous_H_left_mosts.clear();
    current_hessian.reset();
    previous_hessian.reset();

    real_t norm, norm_goal = 0.0;
    norm = initial_norm = computeResidual(X, r);
    min_residual_norm = initial_norm;
    if (norm == 0.0) return;

    norm_goal = std::max(rel_tol * initial_norm, abs_tol);

    if (print_level == 1) {
      mfem::out << "TrustRegion iteration " << std::setw(3) << 0 << " : ||r|| = " << std::setw(13) << norm << "\n";
    }

    SLIC_ERROR_ROOT_IF(nonlinear_options.trust_nonmonotone_window < 0,
                       "TrustRegion requires trust_nonmonotone_window >= 0");
    std::vector<double> work_objective_history;
    pushWorkObjectiveHistory(work_objective_history, current_work_objective);

    prec->iterative_mode = false;
    tr_precond.iterative_mode = false;

    // local arrays
    x_pred.SetSize(X.Size());
    x_pred = 0.0;
    r_pred.SetSize(X.Size());
    r_pred = 0.0;
    scratch.SetSize(X.Size());
    scratch = 0.0;

    TrustRegionResults trResults(X.Size());
    TrustRegionSettings settings;
    settings.min_cg_iterations = static_cast<size_t>(nonlinear_options.min_iterations);
    settings.max_cg_iterations = static_cast<size_t>(linear_options.max_iterations);
    settings.cg_tol = 0.5 * norm_goal;

    int subspace_option = nonlinear_options.subspace_option;
    int num_leftmost = nonlinear_options.num_leftmost;

    auto copy_start = Clock::now();
    scratch = 1.0;
    vector_copy_scale_seconds += secondsSince(copy_start);
    double tr_size = nonlinear_options.trust_region_scaling *
                     std::sqrt(timedDot(scratch, scratch, num_setup_dot_products, setup_dot_seconds));
    size_t cumulative_cg_iters_from_last_precond_update = 0;

    int it = 0;
    for (; true; it++) {
      MFEM_ASSERT(mfem::IsFinite(norm), "norm = " << norm);
      if (print_level >= 2) {
        mfem::out << "TrustRegion iteration " << std::setw(3) << it << " : ||r|| = " << std::setw(13) << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << std::setw(13) << (initial_norm != 0.0 ? norm / initial_norm : norm);
          mfem::out << ", x_incr = " << std::setw(13) << trResults.d.Norml2();
        } else {
          mfem::out << ", norm goal = " << std::setw(13) << norm_goal;
        }
        mfem::out << '\n';
      }

      if (print_level >= 1 && (norm != norm)) {
        mfem::out << "Initial residual for trust-region iteration is undefined/nan." << std::endl;
        mfem::out << "TrustRegion: No convergence!\n";
        return;
      }

      if (norm <= norm_goal && it >= nonlinear_options.min_iterations) {
        converged = true;
        break;
      } else if (it >= max_iter) {
        converged = false;
        break;
      }

      if (nonlinear_options.trust_use_jacobian_operator) {
        SLIC_ERROR_ROOT_IF(!jacobian_operator_factory,
                           "TrustRegion JacobianOperator mode requires a registered JacobianOperator factory.");
        updateJacobianOperator(X);
        updateDiagonalPreconditioner();
        ++num_preconditioner_updates;
        cumulative_cg_iters_from_last_precond_update = 0;
      } else {
        assembleJacobian(X);

        if (it == 0 || (trResults.cg_iterations_count >= settings.max_cg_iterations ||
                        cumulative_cg_iters_from_last_precond_update >= settings.max_cumulative_iteration)) {
          auto preconditioner_update_start = Clock::now();
          auto preconditioner_setup_start = Clock::now();
          tr_precond.SetOperator(*grad);
          preconditioner_setup_seconds += secondsSince(preconditioner_setup_start);
          preconditioner_update_seconds += secondsSince(preconditioner_update_start);
          ++num_preconditioner_updates;
          cumulative_cg_iters_from_last_precond_update = 0;
        }
      }

      auto hess_vec_func = [&](const mfem::Vector& x_, mfem::Vector& v_) { hessVec(x_, v_); };
      auto precond_func = [&](const mfem::Vector& x_, mfem::Vector& v_) { precond(x_, v_); };

      double cauchyPointNormSquared = tr_size * tr_size;
      trResults.reset();

      {
        auto cauchy_start = Clock::now();
        timedCauchyHessVec(hess_vec_func, r, trResults.H_d);
        const double gKg = timedDot(r, trResults.H_d, num_cauchy_dot_products, cauchy_dot_seconds);
        const double residual_norm_squared = norm * norm;
        if (gKg > 0) {
          const double alphaCp = -residual_norm_squared / gKg;
          auto update_start = Clock::now();
          add(trResults.cauchy_point, alphaCp, r, trResults.cauchy_point);
          vector_update_seconds += secondsSince(update_start);
          cauchyPointNormSquared =
              timedDot(trResults.cauchy_point, trResults.cauchy_point, num_cauchy_dot_products, cauchy_dot_seconds);
        } else {
          const double alphaTr = -tr_size / norm;
          auto update_start = Clock::now();
          add(trResults.cauchy_point, alphaTr, r, trResults.cauchy_point);
          vector_update_seconds += secondsSince(update_start);
          if (print_level >= 2) {
            mfem::out << "Negative curvature un-preconditioned cauchy point direction found."
                      << "\n";
          }
        }
        cauchy_point_seconds += secondsSince(cauchy_start);
      }

      if (cauchyPointNormSquared >= tr_size * tr_size) {
        if (print_level >= 2) {
          mfem::out << "Un-preconditioned gradient cauchy point outside trust region, step size = "
                    << std::sqrt(cauchyPointNormSquared) << "\n";
        }
        trResults.cauchy_point *= (tr_size / std::sqrt(cauchyPointNormSquared));
        trResults.z = trResults.cauchy_point;

        trResults.cg_iterations_count = 1;
        trResults.interior_status = TrustRegionResults::Status::OnBoundary;
      } else {
        settings.cg_tol = std::max(0.5 * norm_goal, 5e-5 * norm);
        auto model_start = Clock::now();
        auto model_hess_vec_func = [&](const mfem::Vector& x_, mfem::Vector& v_) {
          timedModelHessVec(hess_vec_func, x_, v_);
        };
        solveTrustRegionModelProblem(r, scratch, model_hess_vec_func, precond_func, settings, tr_size, trResults,
                                     norm * norm);
        model_solve_seconds += secondsSince(model_start);
      }
      cumulative_cg_iters_from_last_precond_update += trResults.cg_iterations_count;
      num_cg_iterations += trResults.cg_iterations_count;

      bool have_computed_Hvs = false;
      bool have_computed_H_left_mosts = false;
      std::vector<std::shared_ptr<mfem::Vector>> candidate_left_mosts;

      int lineSearchIter = 0;
      while (lineSearchIter <= nonlinear_options.max_line_search_iterations) {
        auto line_search_start = Clock::now();
        ++lineSearchIter;

        auto dogleg_start = Clock::now();
        doglegStep(trResults.cauchy_point, trResults.z, tr_size, trResults.d);
        dogleg_seconds += secondsSince(dogleg_start);

        const bool check_subspace_boundary = subspace_option >= 1;
        const double d_norm =
            check_subspace_boundary
                ? std::sqrt(timedDot(trResults.d, trResults.d, num_line_search_dot_products, line_search_dot_seconds))
                : 0.0;
        bool use_with_option1 =
            (subspace_option >= 1) && (trResults.interior_status == TrustRegionResults::Status::NonDescentDirection ||
                                       trResults.interior_status == TrustRegionResults::Status::NegativeCurvature ||
                                       ((d_norm > (1.0 - 1.0e-6) * tr_size) && lineSearchIter > 1));
        bool use_with_option2 = (subspace_option >= 2) && (d_norm > (1.0 - 1.0e-6) * tr_size);
        bool use_with_option3 = (subspace_option >= 3);
        const bool allow_cubic_subspace =
            trResults.interior_status == TrustRegionResults::Status::NegativeCurvature || use_with_option2;

        if (use_with_option1 || use_with_option2 || use_with_option3) {
          if (!have_computed_Hvs) {
            have_computed_Hvs = true;

            std::vector<const mfem::Vector*> subspace_hess_inputs{&trResults.z, &trResults.cauchy_point};
            std::vector<mfem::Vector*> subspace_hess_outputs{&trResults.H_z, &trResults.H_cauchy_point};
            if (trResults.has_d_old) {
              subspace_hess_inputs.push_back(&trResults.d_old);
              subspace_hess_outputs.push_back(&trResults.H_d_old);
            }

            batchedSubspaceHessVec(hess_vec_func, subspace_hess_inputs, subspace_hess_outputs);
          }

          if (!have_computed_H_left_mosts) {
            have_computed_H_left_mosts = true;
            auto leftmost_start = Clock::now();
            previous_H_left_mosts = H_left_mosts;
            H_left_mosts.clear();
            std::vector<const mfem::Vector*> leftmost_inputs;
            std::vector<mfem::Vector*> leftmost_outputs;
            for (auto& left : left_mosts) {
              H_left_mosts.emplace_back(std::make_shared<mfem::Vector>(*left));
              leftmost_inputs.push_back(left.get());
              leftmost_outputs.push_back(H_left_mosts.back().get());
              ++num_subspace_leftmost_hess_vecs;
            }
            subspace_leftmost_seconds += secondsSince(leftmost_start);
            batchedSubspaceHessVec(hess_vec_func, leftmost_inputs, leftmost_outputs);
          }

          std::vector<const mfem::Vector*> ds{&trResults.z, &trResults.cauchy_point};
          std::vector<const mfem::Vector*> H_ds{&trResults.H_z, &trResults.H_cauchy_point};
          if (trResults.has_d_old) {
            ds.push_back(&trResults.d_old);
            H_ds.push_back(&trResults.H_d_old);
          }

          std::vector<mfem::Vector> H_past_steps;
          std::vector<const mfem::Vector*> past_step_inputs;
          std::vector<mfem::Vector*> past_step_outputs;
          const size_t max_past_steps = static_cast<size_t>(std::max(nonlinear_options.trust_num_past_steps, 0));
          const size_t num_past_steps =
              accepted_step_history.size() > 1 ? std::min(max_past_steps, accepted_step_history.size() - 1) : 0;
          H_past_steps.reserve(num_past_steps);
          past_step_inputs.reserve(num_past_steps);
          past_step_outputs.reserve(num_past_steps);
          for (size_t i = 0; i < num_past_steps; ++i) {
            const auto& past_step = accepted_step_history[i + 1];
            H_past_steps.emplace_back(past_step->Size());
            past_step_inputs.push_back(past_step.get());
            past_step_outputs.push_back(&H_past_steps.back());
          }
          if (!past_step_inputs.empty()) {
            num_subspace_past_step_vectors += past_step_inputs.size();
            num_subspace_past_step_hess_vecs += past_step_inputs.size();
            batchedSubspaceHessVec(hess_vec_func, past_step_inputs, past_step_outputs);
            for (size_t i = 0; i < past_step_inputs.size(); ++i) {
              ds.push_back(past_step_inputs[i]);
              H_ds.push_back(past_step_outputs[i]);
            }
          }

          mfem::Vector solve_start_direction;
          mfem::Vector H_solve_start_direction;
          if (nonlinear_options.trust_use_solve_start_direction && solve_start_x.Size() == X.Size()) {
            solve_start_direction.SetSize(X.Size());
            subtract(solve_start_x, X, solve_start_direction);
            if (solve_start_direction.Norml2() > 0.0) {
              H_solve_start_direction.SetSize(X.Size());
              std::vector<const mfem::Vector*> solve_start_inputs{&solve_start_direction};
              std::vector<mfem::Vector*> solve_start_outputs{&H_solve_start_direction};
              ++num_subspace_solve_start_vectors;
              ++num_subspace_solve_start_hess_vecs;
              batchedSubspaceHessVec(hess_vec_func, solve_start_inputs, solve_start_outputs);
              ds.push_back(&solve_start_direction);
              H_ds.push_back(&H_solve_start_direction);
            }
          }

          mfem::Vector min_residual_direction;
          mfem::Vector H_min_residual_direction;
          if (nonlinear_options.trust_use_min_residual_direction && min_residual_x.Size() == X.Size()) {
            min_residual_direction.SetSize(X.Size());
            subtract(min_residual_x, X, min_residual_direction);
            if (min_residual_direction.Norml2() > 0.0) {
              H_min_residual_direction.SetSize(X.Size());
              std::vector<const mfem::Vector*> min_res_inputs{&min_residual_direction};
              std::vector<mfem::Vector*> min_res_outputs{&H_min_residual_direction};
              // Reusing solve_start counters for now
              ++num_subspace_solve_start_vectors;
              ++num_subspace_solve_start_hess_vecs;
              batchedSubspaceHessVec(hess_vec_func, min_res_inputs, min_res_outputs);
              ds.push_back(&min_residual_direction);
              H_ds.push_back(&H_min_residual_direction);
            }
          }
          solveTheSubspaceProblem(trResults.d, hess_vec_func, ds, H_ds, r, tr_size, num_leftmost, candidate_left_mosts,
                                  trResults.d_old,
                                  trResults.has_d_old ? &trResults.H_d_old_at_accept : nullptr, allow_cubic_subspace);
        }

        static constexpr double roundOffTol = 0.0;  // 1e-14;

        timedLineSearchHessVec(hess_vec_func, trResults.d, trResults.H_d);
        const auto [dHd, rd] = timedDot2(trResults.d, trResults.H_d, r, trResults.d, num_line_search_dot_products,
                                         line_search_dot_seconds);
        double modelObjective = rd + 0.5 * dHd - roundOffTol;

        auto update_start = Clock::now();
        add(X, trResults.d, x_pred);
        vector_update_seconds += secondsSince(update_start);

        double realObjective = std::numeric_limits<double>::max();
        double normPred = std::numeric_limits<double>::max();
        try {
          normPred = computeResidual(x_pred, r_pred);
          if (normPred < min_residual_norm) {
            min_residual_norm = normPred;
            min_residual_x = x_pred;
          }
          double obj1 =
              0.5 * (rd + timedDot(r_pred, trResults.d, num_line_search_dot_products, line_search_dot_seconds)) -
              roundOffTol;
          realObjective = obj1;
        } catch (const std::exception&) {
          realObjective = std::numeric_limits<double>::max();
          normPred = std::numeric_limits<double>::max();
        }

        const double trial_work_objective = current_work_objective + realObjective;
        last_nonmonotone_work_reference = nonmonotoneWorkReference(work_objective_history);

        if (normPred <= norm_goal) {
          trResults.d_old = trResults.d;
          trResults.H_d_old_at_accept = trResults.H_d;
          trResults.has_d_old = true;
          pushAcceptedStepHistory(trResults.d);
          if (!candidate_left_mosts.empty()) {
            left_mosts = std::move(candidate_left_mosts);
          }
          copy_start = Clock::now();
          X = x_pred;
          r = r_pred;
          vector_copy_scale_seconds += secondsSince(copy_start);
          norm = normPred;
          current_work_objective = trial_work_objective;
          pushWorkObjectiveHistory(work_objective_history, current_work_objective);
          line_search_seconds += secondsSince(line_search_start);
          if (print_level >= 2) {
            printTrustRegionInfo(realObjective, modelObjective, trResults.cg_iterations_count, tr_size, true);
            trResults.cg_iterations_count =
                0;  // zero this output so it doesn't look like the linesearch is doing cg iterations
          }
          break;
        }

        double modelImprove = -modelObjective;
        double realImprove = -realObjective;

        double rho = realImprove / modelImprove;
        if (modelObjective > 0) {
          if (print_level >= 2) {
            mfem::out << "Found a positive model objective increase.  Debug if you see this.\n";
          }
          rho = realImprove / -modelImprove;
        }

        // std::cout << "rho , stuff = " << rho << " " << settings.eta3 << std::endl;
        // std::cout << "stat = "<< trResults.interior_status << std::endl;

        if (!(rho >= settings.eta2) ||
            rho > settings.eta4) {  // not enough progress, decrease trust region. write it this way to handle NaNs.
          tr_size *= settings.t1;
        } else if ((rho > settings.eta3 && rho <= settings.eta4 &&
                    trResults.interior_status == TrustRegionResults::Status::OnBoundary) ||
                   (rho > 0.95 && rho < 1.05 &&
                    trResults.interior_status ==
                        TrustRegionResults::Status::NegativeCurvature)) {  // good progress, on boundary, increase trust
                                                                           // region
          tr_size *= settings.t2;
        }

        // eventually extend to handle this case to handle occasional roundoff issues
        // modelRes = g + Jd
        // modelResNorm = np.linalg.norm(modelRes)
        // realResNorm = np.linalg.norm(gy)
        const bool monotoneAccept = rho >= settings.eta1 && rho <= settings.eta4;
        const bool nonmonotoneAccept =
            nonlinear_options.trust_nonmonotone_window > 0 && modelObjective < 0.0 && rho <= settings.eta4 &&
            trial_work_objective <= last_nonmonotone_work_reference + settings.eta1 * modelObjective;
        bool willAccept = monotoneAccept || nonmonotoneAccept;  // or (rho >= -0 and realResNorm <= gNorm)

        if (print_level >= 2) {
          printTrustRegionInfo(realObjective, modelObjective, trResults.cg_iterations_count, tr_size, willAccept);
          trResults.cg_iterations_count =
              0;  // zero this output so it doesn't look like the linesearch is doing cg iterations
        }

        if (willAccept) {
          trResults.d_old = trResults.d;
          trResults.H_d_old_at_accept = trResults.H_d;
          trResults.has_d_old = true;
          pushAcceptedStepHistory(trResults.d);
          if (!candidate_left_mosts.empty()) {
            left_mosts = std::move(candidate_left_mosts);
          }
          if (nonmonotoneAccept && !monotoneAccept) {
            ++num_nonmonotone_work_accepts;
            ++num_monotone_work_would_reject;
          }
          copy_start = Clock::now();
          X = x_pred;
          r = r_pred;
          vector_copy_scale_seconds += secondsSince(copy_start);
          norm = normPred;
          current_work_objective = trial_work_objective;
          pushWorkObjectiveHistory(work_objective_history, current_work_objective);
          line_search_seconds += secondsSince(line_search_start);
          break;
        }
        line_search_seconds += secondsSince(line_search_start);
      }
    }

    final_iter = it;
    final_norm = norm;

    if (print_level == 1) {
      mfem::out << "TrustRegion iteration " << std::setw(3) << final_iter << " : ||r|| = " << std::setw(13) << norm
                << '\n';
    }
    if (!converged && print_level >= 1) {  // (print_options.summary || print_options.warnings)) {
      mfem::out << "TrustRegion: No convergence!\n";
    }

    if (false && print_level >= 2) {
      mfem::out << "num hess vecs = " << num_hess_vecs << "\n";
      mfem::out << "num preconds = " << num_preconds << "\n";
      mfem::out << "num residuals = " << num_residuals << "\n";
      mfem::out << "num subspace solves = " << num_subspace_solves << "\n";
      mfem::out << "num jacobian_assembles = " << num_jacobian_assembles << "\n";
    }
    total_seconds = secondsSince(total_start);
  }
};

/**
 * @brief Skeleton for a nonlinear preconditioned conjugate-gradient block solver.
 *
 * The full algorithm is added in a follow-on chunk. This class establishes the Smith/MFEM integration points used by
 * that implementation: residual evaluation, Jacobian assembly, Hessian-vector products, preconditioning, counters, and
 * standard nonlinear convergence bookkeeping.
 */
class PcgBlockSolver : public mfem::NewtonSolver {
 protected:
  /// Trial solution vector
  mutable mfem::Vector x_trial;
  /// Trial residual vector
  mutable mfem::Vector r_trial;
  /// Scratch vector
  mutable mfem::Vector scratch;

  /// Nonlinear solution options
  NonlinearSolverOptions nonlinear_options;

  /// Preconditioner used by the PCG-block recurrence
  Solver& pcg_precond;

  /// Reconstructed Smith print level
  mutable size_t print_level = 0;

 public:
  /// Internal counter for hess-vecs
  mutable size_t num_hess_vecs = 0;
  /// Internal counter for preconditions
  mutable size_t num_preconds = 0;
  /// Internal counter for residuals
  mutable size_t num_residuals = 0;
  /// Internal counter for matrix assembles
  mutable size_t num_jacobian_assembles = 0;
  /// Internal counter for JacobianOperator evaluations
  mutable size_t num_jacobian_operator_evals = 0;
  /// Internal counter for direct diagonal assemblies
  mutable size_t num_diagonal_assembles = 0;
  /// Internal counter for preconditioner operator updates
  mutable size_t num_preconditioner_updates = 0;
  /// Internal counter for accepted prefix blocks
  mutable size_t num_prefix_accepts = 0;
  /// Internal counter for momentum resets
  mutable size_t num_momentum_resets = 0;
  /// Internal counter for nonzero PCG beta values
  mutable size_t num_nonzero_beta = 0;
  /// Internal counter for zero PCG beta values
  mutable size_t num_zero_beta = 0;
  /// Internal counter for accepted blocks
  mutable size_t num_blocks = 0;
  /// Internal counter for rejected blocks
  mutable size_t num_block_rejects = 0;
  /// Internal counter for Powell restarts
  mutable size_t num_powell_restarts = 0;
  /// Internal counter for descent-guard restarts
  mutable size_t num_descent_restarts = 0;
  /// Internal counter for non-positive curvature directions
  mutable size_t num_negative_curvature = 0;
  /// Internal counter for line-search backtracks
  mutable size_t num_line_search_backtracks = 0;
  /// Internal counter for positive-curvature steps capped by the trust radius
  mutable size_t num_trust_capped_steps = 0;
  /// Internal counter for accepted inner PCG steps
  mutable size_t num_accepted_steps = 0;
  /// Internal counter for trial inner PCG steps
  mutable size_t num_trial_steps = 0;
  /// Last trust scale used by the solver
  mutable double final_h_scale = 1.0;
  /// Last accepted block trust ratio
  mutable double last_trust_ratio = 0.0;
  /// Time spent evaluating residuals
  mutable double residual_seconds = 0.0;
  /// Time spent applying all Hessian-vector products
  mutable double hess_vec_seconds = 0.0;
  /// Time spent applying JacobianOperator Hessian-vector products
  mutable double jacobian_operator_hess_vec_seconds = 0.0;
  /// Time spent applying assembled Hessian-vector products
  mutable double assembled_hess_vec_seconds = 0.0;
  /// Time spent applying legacy matrix-free tangent products
  mutable double matrix_free_hess_vec_seconds = 0.0;
  /// Time spent applying preconditioners
  mutable double preconditioner_seconds = 0.0;
  /// Time spent evaluating JacobianOperator factories
  mutable double jacobian_operator_eval_seconds = 0.0;
  /// Time spent assembling sparse Jacobians
  mutable double jacobian_assembly_seconds = 0.0;
  /// Time spent directly assembling diagonals
  mutable double diagonal_assembly_seconds = 0.0;
  /// Time spent inverting direct diagonals
  mutable double diagonal_invert_seconds = 0.0;
  /// Time spent refreshing preconditioner data
  mutable double preconditioner_update_seconds = 0.0;
  /// Time spent in preconditioner SetOperator calls
  mutable double preconditioner_setup_seconds = 0.0;

  /// Optional matrix-free tangent action, y = J(x) dx
  MatrixFreeTangentAction matrix_free_tangent_action;
  /// Optional JacobianOperator factory
  JacobianOperatorFactory jacobian_operator_factory;
  /// Cached JacobianOperator for the current PCG block
  mutable std::unique_ptr<JacobianOperator> current_jacobian_operator;
  /// Owned sparse Jacobian assembled through the JacobianOperator fallback path
  mutable std::unique_ptr<mfem::HypreParMatrix> assembled_jacobian_from_operator;
  /// Inverted scalar diagonal preconditioner for the current PCG block
  mutable mfem::Vector inverse_diagonal_preconditioner;
  /// Whether the current PCG block should use the scalar diagonal preconditioner
  mutable bool use_inverse_diagonal_preconditioner = false;

#ifdef MFEM_USE_MPI
  /// Constructor
  PcgBlockSolver(MPI_Comm comm_, const NonlinearSolverOptions& nonlinear_opts, Solver& preconditioner)
      : mfem::NewtonSolver(comm_), nonlinear_options(nonlinear_opts), pcg_precond(preconditioner)
  {
  }
#endif

  /// Assemble the Jacobian at x.
  void assembleJacobian(const mfem::Vector& x) const
  {
    SMITH_MARK_FUNCTION;
    auto start = Clock::now();
    ++num_jacobian_assembles;
    grad = &oper->GetGradient(x);
    if (nonlinear_options.force_monolithic) {
      auto* grad_blocked = dynamic_cast<mfem::BlockOperator*>(grad);
      if (grad_blocked) grad = buildMonolithicMatrix(*grad_blocked).release();
    }
    jacobian_assembly_seconds += secondsSince(start);
  }

  /// Evaluate the nonlinear residual.
  mfem::real_t computeResidual(const mfem::Vector& x, mfem::Vector& residual) const
  {
    SMITH_MARK_FUNCTION;
    auto start = Clock::now();
    ++num_residuals;
    oper->Mult(x, residual);
    const auto norm = Norm(residual);
    residual_seconds += secondsSince(start);
    return norm;
  }

  /// Set an optional matrix-free tangent action.
  void setMatrixFreeTangentAction(MatrixFreeTangentAction tangent_action)
  {
    matrix_free_tangent_action = std::move(tangent_action);
  }

  /// Set an optional JacobianOperator factory.
  void setJacobianOperator(JacobianOperatorFactory jacobian_operator)
  {
    jacobian_operator_factory = std::move(jacobian_operator);
  }

  /// Evaluate and cache the JacobianOperator at x.
  void updateJacobianOperator(const mfem::Vector& x) const
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(!jacobian_operator_factory, "No JacobianOperator factory is registered.");
    auto start = Clock::now();
    ++num_jacobian_operator_evals;
    current_jacobian_operator = jacobian_operator_factory(x);
    SLIC_ERROR_ROOT_IF(!current_jacobian_operator, "JacobianOperator factory returned a null operator.");
    jacobian_operator_eval_seconds += secondsSince(start);
  }

  /// Assemble and invert the scalar diagonal preconditioner from the current JacobianOperator.
  void updateDiagonalPreconditioner() const
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(!current_jacobian_operator, "Cannot build diagonal preconditioner without a JacobianOperator.");

    auto diagonal_start = Clock::now();
    current_jacobian_operator->assembleDiagonal(inverse_diagonal_preconditioner);
    diagonal_assembly_seconds += secondsSince(diagonal_start);
    ++num_diagonal_assembles;

    auto invert_start = Clock::now();
    double max_abs_diag = 0.0;
    for (int i = 0; i < inverse_diagonal_preconditioner.Size(); ++i) {
      max_abs_diag = std::max(max_abs_diag, std::abs(inverse_diagonal_preconditioner[i]));
    }

    const double floor = nonlinear_options.pcg_diagonal_floor * max_abs_diag;
    SLIC_ERROR_ROOT_IF(!(floor > 0.0), "Cannot invert a zero Jacobian diagonal for PCG-block preconditioning.");
    for (int i = 0; i < inverse_diagonal_preconditioner.Size(); ++i) {
      inverse_diagonal_preconditioner[i] = 1.0 / std::max(std::abs(inverse_diagonal_preconditioner[i]), floor);
    }
    diagonal_invert_seconds += secondsSince(invert_start);

    use_inverse_diagonal_preconditioner = true;
  }

  /// Refresh the tangent and preconditioner used by the next PCG block attempt.
  void refreshBlockOperators(const mfem::Vector& x) const
  {
    auto refresh_start = Clock::now();
    if (jacobian_operator_factory) {
      updateJacobianOperator(x);
      ++num_preconditioner_updates;
      if (nonlinear_options.pcg_use_jacobian_diagonal_preconditioner) {
        updateDiagonalPreconditioner();
      } else {
        use_inverse_diagonal_preconditioner = false;
        auto assembly_start = Clock::now();
        ++num_jacobian_assembles;
        assembled_jacobian_from_operator = current_jacobian_operator->assemble();
        jacobian_assembly_seconds += secondsSince(assembly_start);
        grad = assembled_jacobian_from_operator.get();
        auto setup_start = Clock::now();
        pcg_precond.SetOperator(*grad);
        preconditioner_setup_seconds += secondsSince(setup_start);
      }
    } else {
      SLIC_ERROR_ROOT_IF(nonlinear_options.pcg_use_jacobian_diagonal_preconditioner,
                         "PCG-block diagonal preconditioning requires a registered JacobianOperator.");
      current_jacobian_operator.reset();
      use_inverse_diagonal_preconditioner = false;
      assembleJacobian(x);
      ++num_preconditioner_updates;
      auto setup_start = Clock::now();
      pcg_precond.SetOperator(*grad);
      preconditioner_setup_seconds += secondsSince(setup_start);
    }
    preconditioner_update_seconds += secondsSince(refresh_start);
  }

  /// Apply the tangent at x to dx.
  void hessVec(const mfem::Vector& x, const mfem::Vector& dx, mfem::Vector& y) const
  {
    SMITH_MARK_FUNCTION;
    auto start = Clock::now();
    ++num_hess_vecs;
    if (current_jacobian_operator) {
      current_jacobian_operator->Mult(dx, y);
      const double seconds = secondsSince(start);
      hess_vec_seconds += seconds;
      jacobian_operator_hess_vec_seconds += seconds;
    } else if (jacobian_operator_factory) {
      updateJacobianOperator(x);
      current_jacobian_operator->Mult(dx, y);
      const double seconds = secondsSince(start);
      hess_vec_seconds += seconds;
      jacobian_operator_hess_vec_seconds += seconds;
    } else if (matrix_free_tangent_action) {
      matrix_free_tangent_action(x, dx, y);
      const double seconds = secondsSince(start);
      hess_vec_seconds += seconds;
      matrix_free_hess_vec_seconds += seconds;
    } else {
      grad->Mult(dx, y);
      const double seconds = secondsSince(start);
      hess_vec_seconds += seconds;
      assembled_hess_vec_seconds += seconds;
    }
  }

  /// Apply the configured nonlinear PCG preconditioner.
  void precond(const mfem::Vector& x, mfem::Vector& v) const
  {
    SMITH_MARK_FUNCTION;
    auto start = Clock::now();
    ++num_preconds;
    if (use_inverse_diagonal_preconditioner) {
      SLIC_ERROR_ROOT_IF(inverse_diagonal_preconditioner.Size() != x.Size(),
                         "PCG-block diagonal preconditioner size does not match the residual vector.");
      v.SetSize(x.Size());
      for (int i = 0; i < x.Size(); ++i) {
        v[i] = inverse_diagonal_preconditioner[i] * x[i];
      }
    } else {
      pcg_precond.Mult(x, v);
    }
    preconditioner_seconds += secondsSince(start);
  }

  /// Return solver diagnostic counters.
  PcgBlockDiagnostics diagnostics() const
  {
    return {.num_residuals = num_residuals,
            .num_hess_vecs = num_hess_vecs,
            .num_preconds = num_preconds,
            .num_jacobian_assembles = num_jacobian_assembles,
            .num_jacobian_operator_evals = num_jacobian_operator_evals,
            .num_diagonal_assembles = num_diagonal_assembles,
            .num_preconditioner_updates = num_preconditioner_updates,
            .num_prefix_accepts = num_prefix_accepts,
            .num_momentum_resets = num_momentum_resets,
            .num_nonzero_beta = num_nonzero_beta,
            .num_zero_beta = num_zero_beta,
            .num_blocks = num_blocks,
            .num_block_rejects = num_block_rejects,
            .num_powell_restarts = num_powell_restarts,
            .num_descent_restarts = num_descent_restarts,
            .num_negative_curvature = num_negative_curvature,
            .num_line_search_backtracks = num_line_search_backtracks,
            .num_trust_capped_steps = num_trust_capped_steps,
            .num_accepted_steps = num_accepted_steps,
            .num_trial_steps = num_trial_steps,
            .residual_seconds = residual_seconds,
            .hess_vec_seconds = hess_vec_seconds,
            .jacobian_operator_hess_vec_seconds = jacobian_operator_hess_vec_seconds,
            .assembled_hess_vec_seconds = assembled_hess_vec_seconds,
            .matrix_free_hess_vec_seconds = matrix_free_hess_vec_seconds,
            .preconditioner_seconds = preconditioner_seconds,
            .jacobian_operator_eval_seconds = jacobian_operator_eval_seconds,
            .jacobian_assembly_seconds = jacobian_assembly_seconds,
            .diagonal_assembly_seconds = diagonal_assembly_seconds,
            .diagonal_invert_seconds = diagonal_invert_seconds,
            .preconditioner_update_seconds = preconditioner_update_seconds,
            .preconditioner_setup_seconds = preconditioner_setup_seconds,
            .final_h_scale = final_h_scale,
            .last_trust_ratio = last_trust_ratio};
  }

  /// @overload
  void Mult(const mfem::Vector&, mfem::Vector& X) const
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

    print_level = static_cast<size_t>(std::max(nonlinear_options.print_level, 0));
    print_level = print_options.iterations ? std::max<size_t>(1, print_level) : print_level;
    print_level = print_options.summary ? std::max<size_t>(2, print_level) : print_level;

    num_hess_vecs = 0;
    num_preconds = 0;
    num_residuals = 0;
    num_jacobian_assembles = 0;
    num_jacobian_operator_evals = 0;
    num_diagonal_assembles = 0;
    num_preconditioner_updates = 0;
    num_prefix_accepts = 0;
    num_momentum_resets = 0;
    num_nonzero_beta = 0;
    num_zero_beta = 0;
    num_blocks = 0;
    num_block_rejects = 0;
    num_powell_restarts = 0;
    num_descent_restarts = 0;
    num_negative_curvature = 0;
    num_line_search_backtracks = 0;
    num_trust_capped_steps = 0;
    num_accepted_steps = 0;
    num_trial_steps = 0;
    final_h_scale = nonlinear_options.pcg_h_scale_init;
    last_trust_ratio = 0.0;
    residual_seconds = 0.0;
    hess_vec_seconds = 0.0;
    jacobian_operator_hess_vec_seconds = 0.0;
    assembled_hess_vec_seconds = 0.0;
    matrix_free_hess_vec_seconds = 0.0;
    preconditioner_seconds = 0.0;
    jacobian_operator_eval_seconds = 0.0;
    jacobian_assembly_seconds = 0.0;
    diagonal_assembly_seconds = 0.0;
    diagonal_invert_seconds = 0.0;
    preconditioner_update_seconds = 0.0;
    preconditioner_setup_seconds = 0.0;
    current_jacobian_operator.reset();
    assembled_jacobian_from_operator.reset();
    inverse_diagonal_preconditioner.SetSize(0);
    use_inverse_diagonal_preconditioner = false;

    SLIC_ERROR_ROOT_IF(nonlinear_options.pcg_block_len <= 0, "PcgBlock requires pcg_block_len > 0");
    SLIC_ERROR_ROOT_IF(nonlinear_options.pcg_window <= 0, "PcgBlock requires pcg_window > 0");
    SLIC_ERROR_ROOT_IF(nonlinear_options.pcg_ls_max_backtracks < 0, "PcgBlock requires pcg_ls_max_backtracks >= 0");
    SLIC_ERROR_ROOT_IF(nonlinear_options.pcg_delta_avg_window <= 0, "PcgBlock requires pcg_delta_avg_window > 0");

    mfem::real_t norm = computeResidual(X, r);
    initial_norm = norm;
    if (norm == 0.0) {
      converged = true;
      final_iter = 0;
      final_norm = norm;
      return;
    }

    const mfem::real_t norm_goal = std::max(rel_tol * initial_norm, abs_tol);

    if (print_level == 1) {
      mfem::out << "PcgBlock iteration " << std::setw(3) << 0 << " : ||r|| = " << std::setw(13) << norm << "\n";
    }

    pcg_precond.iterative_mode = false;

    x_trial.SetSize(X.Size());
    x_trial = 0.0;
    r_trial.SetSize(X.Size());
    r_trial = 0.0;
    scratch.SetSize(X.Size());
    scratch = 0.0;

    mfem::Vector r_block(X.Size());
    mfem::Vector r_candidate(X.Size());
    mfem::Vector force(X.Size());
    mfem::Vector z(X.Size());
    mfem::Vector z_old(X.Size());
    mfem::Vector p(X.Size());
    mfem::Vector p_old(X.Size());
    mfem::Vector Hp(X.Size());
    mfem::Vector step(X.Size());
    mfem::Vector x_candidate(X.Size());

    bool have_momentum = false;
    double rho_old = 0.0;
    double h_scale = nonlinear_options.pcg_h_scale_init;
    int retries_remaining = nonlinear_options.pcg_max_block_retries;
    int it = 0;
    double cumulative_work = 0.0;
    std::vector<double> work_history{cumulative_work};
    std::vector<double> accepted_step_norms;

    auto append_bounded = [](std::vector<double>& history, double value, int max_size) {
      history.push_back(value);
      const auto bound = static_cast<size_t>(max_size);
      if (history.size() > bound) {
        const auto num_to_remove = static_cast<std::vector<double>::difference_type>(history.size() - bound);
        history.erase(history.begin(), history.begin() + num_to_remove);
      }
    };

    auto reset_momentum = [&]() {
      have_momentum = false;
      rho_old = 0.0;
      p_old = 0.0;
      z_old = 0.0;
      ++num_momentum_resets;
    };

    auto window_max = [&](const std::vector<double>& history) {
      const int window = nonlinear_options.pcg_window;
      const auto begin = history.size() > static_cast<size_t>(window) ? history.end() - window : history.begin();
      return *std::max_element(begin, history.end());
    };

    auto current_delta_ref = [&]() {
      if (accepted_step_norms.empty()) {
        return 0.0;
      }
      const int window = nonlinear_options.pcg_delta_avg_window;
      const auto begin = accepted_step_norms.size() > static_cast<size_t>(window) ? accepted_step_norms.end() - window
                                                                                  : accepted_step_norms.begin();
      double sum = 0.0;
      for (auto iter = begin; iter != accepted_step_norms.end(); ++iter) {
        sum += *iter;
      }
      return sum / static_cast<double>(accepted_step_norms.end() - begin);
    };

    for (; true;) {
      MFEM_ASSERT(mfem::IsFinite(norm), "norm = " << norm);
      if (print_level >= 2) {
        mfem::out << "PcgBlock iteration " << std::setw(3) << it << " : ||r|| = " << std::setw(13) << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << std::setw(13) << (initial_norm != 0.0 ? norm / initial_norm : norm);
        } else {
          mfem::out << ", norm goal = " << std::setw(13) << norm_goal;
        }
        mfem::out << '\n';
      }

      if (print_level >= 1 && (norm != norm)) {
        mfem::out << "Initial residual for PCG-block iteration is undefined/nan." << std::endl;
        mfem::out << "PcgBlock: No convergence!\n";
        converged = false;
        break;
      }

      if (norm <= norm_goal && it >= nonlinear_options.min_iterations) {
        converged = true;
        break;
      } else if (it >= max_iter) {
        converged = false;
        break;
      } else if (retries_remaining <= 0 || h_scale < nonlinear_options.pcg_min_h_scale) {
        converged = false;
        break;
      }

      refreshBlockOperators(X);

      r_block = r;
      const double norm_block = norm;
      bool block_finished = false;

      while (!block_finished) {
        x_trial = X;
        r = r_block;
        norm = norm_block;

        double block_predicted = 0.0;
        double block_actual = 0.0;
        double block_delta_ref = current_delta_ref();
        double block_trust_size = h_scale * (block_delta_ref > 0.0 ? block_delta_ref : 1.0);
        double trial_cumulative_work = cumulative_work;
        int trial_steps = 0;
        bool trial_failed = false;
        bool trial_ended_after_inner_failure = false;
        std::vector<double> trial_step_norms;
        auto trial_work_history = work_history;

        for (int block_it = 0; block_it < nonlinear_options.pcg_block_len && it + trial_steps < max_iter; ++block_it) {
          force = r;
          force *= -1.0;
          precond(force, z);
          ++num_trial_steps;

          const double rho = Dot(force, z);
          if (!mfem::IsFinite(rho) || rho <= 0.0) {
            trial_ended_after_inner_failure = trial_steps > 0;
            trial_failed = trial_steps == 0;
            break;
          }

          double beta = 0.0;
          if (have_momentum) {
            const double force_dot_z_old = Dot(force, z_old);
            beta = std::max(0.0, (rho - force_dot_z_old) / rho_old);
            if (std::abs(force_dot_z_old) > nonlinear_options.pcg_powell_eta * rho) {
              beta = 0.0;
              ++num_powell_restarts;
            }
          }

          p = z;
          if (have_momentum && beta != 0.0) {
            p.Add(beta, p_old);
          }

          double force_dot_p = Dot(force, p);
          if (force_dot_p <= nonlinear_options.pcg_eps_descent * rho) {
            beta = 0.0;
            p = z;
            force_dot_p = rho;
            ++num_descent_restarts;
          }
          if (beta == 0.0) {
            ++num_zero_beta;
          } else {
            ++num_nonzero_beta;
          }

          hessVec(X, p, Hp);
          const double pHp = Dot(p, Hp);

          double alpha = 0.0;
          double alpha_quad = std::numeric_limits<double>::quiet_NaN();
          const bool positive_curvature = pHp > 0.0 && mfem::IsFinite(pHp);
          if (positive_curvature) {
            alpha_quad = force_dot_p / pHp;
            alpha = alpha_quad;
          } else {
            ++num_negative_curvature;
          }

          const double p_norm = Norm(p);
          double delta_ref = current_delta_ref();
          if (delta_ref <= 0.0 && alpha > 0.0 && mfem::IsFinite(alpha) && p_norm > 0.0) {
            delta_ref = alpha * p_norm;
          } else if (delta_ref <= 0.0) {
            delta_ref = 1.0;
          }
          block_delta_ref = delta_ref;
          block_trust_size = h_scale * delta_ref;

          const bool apply_trust_cap = !positive_curvature || h_scale < nonlinear_options.pcg_h_scale_init;
          bool trust_capped = false;
          if (apply_trust_cap && p_norm > 0.0) {
            const double alpha_cap = h_scale * delta_ref / p_norm;
            if (alpha > 0.0 && mfem::IsFinite(alpha)) {
              if (alpha_cap < alpha) {
                ++num_trust_capped_steps;
                trust_capped = true;
              }
              alpha = std::min(alpha, alpha_cap);
            } else {
              alpha = alpha_cap;
              trust_capped = true;
            }
          }

          if (!(alpha > 0.0) || !mfem::IsFinite(alpha)) {
            trial_ended_after_inner_failure = trial_steps > 0;
            trial_failed = trial_steps == 0;
            break;
          }

          bool accepted_step = false;
          double accepted_work = 0.0;
          double accepted_predicted = 0.0;
          double accepted_step_norm = 0.0;
          int accepted_ls_count = 0;

          for (int ls = 0; ls <= nonlinear_options.pcg_ls_max_backtracks; ++ls) {
            step = p;
            step *= alpha;
            add(x_trial, step, x_candidate);

            const double norm_candidate = computeResidual(x_candidate, r_candidate);
            const double work = -0.5 * Dot(r, step) - 0.5 * Dot(r_candidate, step);
            const double cumulative_candidate = trial_cumulative_work + work;
            const double work_ref = window_max(trial_work_history);
            const bool finite_candidate = mfem::IsFinite(norm_candidate) && mfem::IsFinite(work);
            const bool sufficient_work =
                cumulative_candidate >= work_ref - nonlinear_options.pcg_ls_armijo_c * alpha * force_dot_p;

            if (finite_candidate && (sufficient_work || norm_candidate <= norm_goal)) {
              const double predicted = alpha * force_dot_p - 0.5 * alpha * alpha * pHp;
              accepted_predicted = std::max(predicted, 0.0);
              accepted_work = work;
              accepted_step_norm = Norm(step);
              accepted_ls_count = ls;
              norm = norm_candidate;
              accepted_step = true;
              break;
            }

            alpha *= nonlinear_options.pcg_ls_shrink;
          }

          if (!accepted_step) {
            trial_ended_after_inner_failure = trial_steps > 0;
            trial_failed = trial_steps == 0;
            break;
          }

          x_trial = x_candidate;
          r = r_candidate;
          trial_cumulative_work += accepted_work;
          append_bounded(trial_work_history, trial_cumulative_work, nonlinear_options.pcg_window);
          append_bounded(trial_step_norms, accepted_step_norm, nonlinear_options.pcg_delta_avg_window);
          block_predicted += accepted_predicted;
          block_actual += accepted_work;
          num_line_search_backtracks += static_cast<size_t>(accepted_ls_count);

          if (print_level >= 2) {
            mfem::out << "  PcgBlock step " << std::setw(3) << (it + trial_steps + 1) << " : alpha = " << std::setw(13)
                      << alpha << ", approx work = " << std::setw(13) << accepted_predicted
                      << ", achieved work = " << std::setw(13) << accepted_work << ", trust size = " << std::setw(13)
                      << block_trust_size << ", capped = " << trust_capped << ", ls = " << accepted_ls_count << '\n';
          }

          p_old = p;
          z_old = z;
          rho_old = rho;
          have_momentum = true;
          ++trial_steps;
          ++num_accepted_steps;

          if (norm <= norm_goal && it + trial_steps >= nonlinear_options.min_iterations) {
            break;
          }
        }

        double trust_ratio = 1.0;
        if (block_predicted > nonlinear_options.pcg_eps_descent) {
          trust_ratio = block_actual / block_predicted;
        } else if (block_actual < 0.0) {
          trust_ratio = -std::numeric_limits<double>::infinity();
        }

        const bool block_converged = norm <= norm_goal && it + trial_steps >= nonlinear_options.min_iterations;
        const bool accept_block =
            trial_steps > 0 && !trial_failed &&
            (block_converged || (block_actual >= 0.0 && trust_ratio >= nonlinear_options.pcg_trust_eta_bad));

        const double old_h_scale = h_scale;
        const bool prefix_accept = accept_block && trial_ended_after_inner_failure;
        bool reset_next_momentum = false;
        if (accept_block) {
          if (prefix_accept) {
            ++num_prefix_accepts;
          }
          X = x_trial;
          cumulative_work = trial_cumulative_work;
          work_history = std::move(trial_work_history);
          accepted_step_norms.insert(accepted_step_norms.end(), trial_step_norms.begin(), trial_step_norms.end());
          if (accepted_step_norms.size() > static_cast<size_t>(nonlinear_options.pcg_delta_avg_window)) {
            accepted_step_norms.erase(accepted_step_norms.begin(),
                                      accepted_step_norms.end() - nonlinear_options.pcg_delta_avg_window);
          }
          it += trial_steps;
          ++num_blocks;

          if (trust_ratio < nonlinear_options.pcg_trust_eta_bad) {
            h_scale = std::max(h_scale * nonlinear_options.pcg_shrink, nonlinear_options.pcg_min_h_scale);
            reset_momentum();
            reset_next_momentum = true;
          } else if (trial_ended_after_inner_failure) {
            reset_momentum();
            reset_next_momentum = true;
          } else if (trust_ratio >= nonlinear_options.pcg_trust_eta_good) {
            h_scale = std::min(h_scale * nonlinear_options.pcg_growth, nonlinear_options.pcg_h_scale_init);
          }
          const double next_trust_size = h_scale * block_delta_ref;

          if (print_level >= 2) {
            mfem::out << "PcgBlock block accepted: steps = " << std::setw(3) << trial_steps
                      << ", prefix = " << prefix_accept << ", approx work = " << std::setw(13) << block_predicted
                      << ", achieved work = " << std::setw(13) << block_actual << ", rho = " << std::setw(13)
                      << trust_ratio << ", h_scale = " << std::setw(13) << old_h_scale << " -> " << std::setw(13)
                      << h_scale << ", trust size = " << std::setw(13) << block_trust_size << " -> " << std::setw(13)
                      << next_trust_size << ", reset momentum = " << reset_next_momentum << '\n';
          }
          last_trust_ratio = trust_ratio;

          block_finished = true;
        } else {
          r = r_block;
          norm = norm_block;
          h_scale *= nonlinear_options.pcg_shrink;
          reset_momentum();
          --retries_remaining;
          ++num_block_rejects;
          const double next_trust_size = h_scale * block_delta_ref;

          if (print_level >= 2) {
            mfem::out << "PcgBlock block rejected: steps = " << std::setw(3) << trial_steps
                      << ", approx work = " << std::setw(13) << block_predicted << ", achieved work = " << std::setw(13)
                      << block_actual << ", rho = " << std::setw(13) << trust_ratio << ", h_scale = " << std::setw(13)
                      << old_h_scale << " -> " << std::setw(13) << h_scale << ", trust size = " << std::setw(13)
                      << block_trust_size << " -> " << std::setw(13) << next_trust_size << ", reset momentum = 1"
                      << ", retries left = " << retries_remaining << '\n';
          }

          if (retries_remaining <= 0 || h_scale < nonlinear_options.pcg_min_h_scale) {
            block_finished = true;
          } else {
            refreshBlockOperators(X);
          }
        }
      }
    }

    final_iter = it;
    final_norm = norm;
    final_h_scale = h_scale;

    if (print_level == 1) {
      mfem::out << "PcgBlock iteration " << std::setw(3) << final_iter << " : ||r|| = " << std::setw(13) << norm
                << '\n';
    }
    if (!converged && print_level >= 1) {
      mfem::out << "PcgBlock: No convergence!\n";
    }
  }
};

EquationSolver::EquationSolver(NonlinearSolverOptions nonlinear_opts, LinearSolverOptions lin_opts, MPI_Comm comm)
{
  auto [lin_solver, preconditioner] = buildLinearSolverAndPreconditioner(lin_opts, comm);

  lin_solver_ = std::move(lin_solver);
  preconditioner_ = std::move(preconditioner);
  nonlin_solver_ = buildNonlinearSolver(nonlinear_opts, lin_opts, *preconditioner_, comm);
}

EquationSolver::EquationSolver(std::unique_ptr<mfem::NewtonSolver> nonlinear_solver,
                               std::unique_ptr<mfem::Solver> linear_solver,
                               std::unique_ptr<mfem::Solver> preconditioner)
{
  SLIC_ERROR_ROOT_IF(!nonlinear_solver, "Nonlinear solvers must be given to construct an EquationSolver");
  SLIC_ERROR_ROOT_IF(!linear_solver, "Linear solvers must be given to construct an EquationSolver");

  nonlin_solver_ = std::move(nonlinear_solver);
  lin_solver_ = std::move(linear_solver);
  preconditioner_ = std::move(preconditioner);
}

void EquationSolver::setOperator(const mfem::Operator& op)
{
  nonlin_solver_->SetOperator(op);

  // Now that the nonlinear solver knows about the operator, we can set its linear solver
  if (!nonlin_solver_set_solver_called_) {
    nonlin_solver_->SetSolver(linearSolver());
    nonlin_solver_set_solver_called_ = true;
  }
}

void EquationSolver::setMatrixFreeTangentAction(MatrixFreeTangentAction tangent_action)
{
  auto* pcg_block = dynamic_cast<PcgBlockSolver*>(nonlin_solver_.get());
  if (pcg_block) {
    pcg_block->setMatrixFreeTangentAction(std::move(tangent_action));
  }
}

void EquationSolver::setJacobianOperator(JacobianOperatorFactory jacobian_operator)
{
  auto* pcg_block = dynamic_cast<PcgBlockSolver*>(nonlin_solver_.get());
  if (pcg_block) {
    pcg_block->setJacobianOperator(std::move(jacobian_operator));
    return;
  }
  auto* trust_region = dynamic_cast<TrustRegion*>(nonlin_solver_.get());
  if (trust_region) {
    trust_region->setJacobianOperator(std::move(jacobian_operator));
  }
}

void EquationSolver::solve(mfem::Vector& x) const
{
  mfem::Vector zero(x);
  zero = 0.0;
  // KINSOL does not handle non-zero RHS, so we enforce that the RHS
  // of the nonlinear system is zero
  nonlin_solver_->Mult(zero, x);
}

std::optional<PcgBlockDiagnostics> EquationSolver::pcgBlockDiagnostics() const
{
  auto* pcg_block = dynamic_cast<const PcgBlockSolver*>(nonlin_solver_.get());
  if (!pcg_block) {
    return std::nullopt;
  }
  return pcg_block->diagnostics();
}

std::optional<TrustRegionDiagnostics> EquationSolver::trustRegionDiagnostics() const
{
  auto* trust_region = dynamic_cast<const TrustRegion*>(nonlin_solver_.get());
  if (!trust_region) {
    return std::nullopt;
  }
  return trust_region->diagnostics();
}

void SuperLUSolver::Mult(const mfem::Vector& input, mfem::Vector& output) const
{
  SLIC_ERROR_ROOT_IF(!superlu_mat_, "Operator must be set prior to solving with SuperLU");

  // Use the underlying MFEM-based solver and SuperLU matrix type to solve the system
  superlu_solver_.Mult(input, output);
}

std::unique_ptr<mfem::HypreParMatrix> buildMonolithicMatrix(const mfem::BlockOperator& block_operator)
{
  int row_blocks = block_operator.NumRowBlocks();
  int col_blocks = block_operator.NumColBlocks();

  SLIC_ERROR_ROOT_IF(row_blocks != col_blocks, "Attempted to use a direct solver on a non-square block system.");

  mfem::Array2D<const mfem::HypreParMatrix*> hypre_blocks(row_blocks, col_blocks);

  for (int i = 0; i < row_blocks; ++i) {
    for (int j = 0; j < col_blocks; ++j) {
      // checks for presence of empty (null) blocks, which happen fairly common in multirank contact
      if (!block_operator.IsZeroBlock(i, j)) {
        auto* hypre_block = dynamic_cast<const mfem::HypreParMatrix*>(&block_operator.GetBlock(i, j));
        SLIC_ERROR_ROOT_IF(!hypre_block,
                           "Trying to use SuperLU on a block operator that does not contain HypreParMatrix blocks.");

        hypre_blocks(i, j) = hypre_block;
      } else {
        hypre_blocks(i, j) = nullptr;
      }
    }
  }

  // Note that MFEM passes ownership of this matrix to the caller
  return std::unique_ptr<mfem::HypreParMatrix>(mfem::HypreParMatrixFromBlocks(hypre_blocks));
}

void SuperLUSolver::SetOperator(const mfem::Operator& op)
{
  // Check if this is a block operator
  auto* block_operator = dynamic_cast<const mfem::BlockOperator*>(&op);

  // If it is, make a monolithic system from the underlying blocks
  if (block_operator) {
    auto monolithic_mat = buildMonolithicMatrix(*block_operator);

    superlu_mat_ = std::make_unique<mfem::SuperLURowLocMatrix>(*monolithic_mat);
  } else {
    // If this is not a block system, check that the input operator is a HypreParMatrix as expected
    auto* matrix = dynamic_cast<const mfem::HypreParMatrix*>(&op);

    SLIC_ERROR_ROOT_IF(!matrix, "Matrix must be an assembled HypreParMatrix for use with SuperLU");

    superlu_mat_ = std::make_unique<mfem::SuperLURowLocMatrix>(*matrix);
  }

  superlu_solver_.SetOperator(*superlu_mat_);
}

#ifdef MFEM_USE_STRUMPACK

void StrumpackSolver::Mult(const mfem::Vector& input, mfem::Vector& output) const
{
  SLIC_ERROR_ROOT_IF(!strumpack_mat_, "Operator must be set prior to solving with Strumpack");

  // Use the underlying MFEM-based solver and Strumpack matrix type to solve the system
  strumpack_solver_.Mult(input, output);
}

void StrumpackSolver::SetOperator(const mfem::Operator& op)
{
  // Check if this is a block operator
  auto* block_operator = dynamic_cast<const mfem::BlockOperator*>(&op);

  // If it is, make a monolithic system from the underlying blocks
  if (block_operator) {
    auto monolithic_mat = buildMonolithicMatrix(*block_operator);

    strumpack_mat_ = std::make_unique<mfem::STRUMPACKRowLocMatrix>(*monolithic_mat);
  } else {
    // If this is not a block system, check that the input operator is a HypreParMatrix as expected
    auto* matrix = dynamic_cast<const mfem::HypreParMatrix*>(&op);

    SLIC_ERROR_ROOT_IF(!matrix, "Matrix must be an assembled HypreParMatrix for use with Strumpack");

    strumpack_mat_ = std::make_unique<mfem::STRUMPACKRowLocMatrix>(*matrix);
  }
  height = op.Height();
  width = op.Width();
  strumpack_solver_.SetOperator(*strumpack_mat_);
}

#endif

std::unique_ptr<mfem::NewtonSolver> buildNonlinearSolver(NonlinearSolverOptions nonlinear_opts,
                                                         const LinearSolverOptions& linear_opts, mfem::Solver& prec,
                                                         MPI_Comm comm)
{
  std::unique_ptr<mfem::NewtonSolver> nonlinear_solver;

  if (nonlinear_opts.nonlin_solver == NonlinearSolver::Newton) {
    nonlinear_opts.max_line_search_iterations = 0;
    SLIC_ERROR_ROOT_IF(nonlinear_opts.min_iterations != 0, "Newton's method does not support nonzero min_iterations");
    nonlinear_solver = std::make_unique<NewtonSolver>(comm, nonlinear_opts);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::LBFGS) {
    nonlinear_opts.max_line_search_iterations = 0;
    SLIC_ERROR_ROOT_IF(nonlinear_opts.min_iterations != 0, "LBFGS does not support nonzero min_iterations");
    nonlinear_solver = std::make_unique<mfem::LBFGSSolver>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::NewtonLineSearch) {
    nonlinear_solver = std::make_unique<NewtonSolver>(comm, nonlinear_opts);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::TrustRegion) {
    nonlinear_solver = std::make_unique<TrustRegion>(comm, nonlinear_opts, linear_opts, prec);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::PcgBlock) {
    nonlinear_solver = std::make_unique<PcgBlockSolver>(comm, nonlinear_opts, prec);
#ifdef SMITH_USE_PETSC
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::PetscNewton) {
    nonlinear_solver = std::make_unique<mfem_ext::PetscNewtonSolver>(comm, nonlinear_opts);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::PetscNewtonBacktracking) {
    nonlinear_solver = std::make_unique<mfem_ext::PetscNewtonSolver>(comm, nonlinear_opts);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::PetscNewtonCriticalPoint) {
    nonlinear_solver = std::make_unique<mfem_ext::PetscNewtonSolver>(comm, nonlinear_opts);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::PetscTrustRegion) {
    nonlinear_solver = std::make_unique<mfem_ext::PetscNewtonSolver>(comm, nonlinear_opts);
#endif
  }
  // KINSOL
  else {
#ifdef SMITH_USE_SUNDIALS
    nonlinear_opts.max_line_search_iterations = 0;
    SLIC_ERROR_ROOT_IF(nonlinear_opts.min_iterations != 0, "kinsol solvers do not support min_iterations");

    int kinsol_strat = KIN_NONE;

    switch (nonlinear_opts.nonlin_solver) {
      case NonlinearSolver::KINFullStep:
        kinsol_strat = KIN_NONE;
        break;
      case NonlinearSolver::KINBacktrackingLineSearch:
        kinsol_strat = KIN_LINESEARCH;
        break;
      case NonlinearSolver::KINPicard:
        kinsol_strat = KIN_PICARD;
        break;
      default:
        kinsol_strat = KIN_NONE;
        SLIC_ERROR_ROOT("Unknown KINSOL nonlinear solver type given.");
    }
    auto kinsol_solver = std::make_unique<mfem::KINSolver>(comm, kinsol_strat, true);
    nonlinear_solver = std::move(kinsol_solver);
#else
    SLIC_ERROR_ROOT("KINSOL was not enabled when MFEM was built");
#endif
  }

  nonlinear_solver->SetRelTol(nonlinear_opts.relative_tol);
  nonlinear_solver->SetAbsTol(nonlinear_opts.absolute_tol);
  nonlinear_solver->SetMaxIter(nonlinear_opts.max_iterations);
  nonlinear_solver->SetPrintLevel(nonlinear_opts.print_level);

  // Iterative mode indicates we do not zero out the initial guess during the
  // nonlinear solver call. This is required as we apply the essential boundary
  // conditions before the nonlinear solver is applied.
  nonlinear_solver->iterative_mode = true;

  return nonlinear_solver;
}

std::pair<std::unique_ptr<mfem::Solver>, std::unique_ptr<mfem::Solver>> buildLinearSolverAndPreconditioner(
    LinearSolverOptions linear_opts, MPI_Comm comm)
{
  auto preconditioner = buildPreconditioner(linear_opts, comm);

  if (linear_opts.linear_solver == LinearSolver::SuperLU) {
    auto lin_solver = std::make_unique<SuperLUSolver>(linear_opts.print_level, comm);
    return {std::move(lin_solver), std::move(preconditioner)};
  }

#ifdef MFEM_USE_STRUMPACK

  if (linear_opts.linear_solver == LinearSolver::Strumpack) {
    auto lin_solver = std::make_unique<StrumpackSolver>(linear_opts.print_level, comm);
    return {std::move(lin_solver), std::move(preconditioner)};
  }

#endif

  std::unique_ptr<mfem::IterativeSolver> iter_lin_solver;

  switch (linear_opts.linear_solver) {
    case LinearSolver::CG:
      iter_lin_solver = std::make_unique<mfem::CGSolver>(comm);
      break;
    case LinearSolver::GMRES:
      iter_lin_solver = std::make_unique<mfem::GMRESSolver>(comm);
      break;
#ifdef SMITH_USE_PETSC
    case LinearSolver::PetscCG:
      iter_lin_solver = std::make_unique<smith::mfem_ext::PetscKSPSolver>(comm, KSPCG, std::string());
      break;
    case LinearSolver::PetscGMRES:
      iter_lin_solver = std::make_unique<smith::mfem_ext::PetscKSPSolver>(comm, KSPGMRES, std::string());
      break;
#else
    case LinearSolver::PetscCG:
    case LinearSolver::PetscGMRES:
      SLIC_ERROR_ROOT("PETSc linear solver requested for non-PETSc build.");
      exit(1);
      break;
#endif
    default:
      SLIC_ERROR_ROOT("Linear solver type not recognized.");
      exit(1);
  }

  iter_lin_solver->SetRelTol(linear_opts.relative_tol);
  iter_lin_solver->SetAbsTol(linear_opts.absolute_tol);
  iter_lin_solver->SetMaxIter(linear_opts.max_iterations);
  iter_lin_solver->SetPrintLevel(linear_opts.print_level);

  if (preconditioner) {
    iter_lin_solver->SetPreconditioner(*preconditioner);
  }

  return {std::move(iter_lin_solver), std::move(preconditioner)};
}

#ifdef MFEM_USE_AMGX
std::unique_ptr<mfem::AmgXSolver> buildAMGX(const AMGXOptions& options, const MPI_Comm comm)
{
  auto amgx = std::make_unique<mfem::AmgXSolver>();
  conduit::Node options_node;
  options_node["config_version"] = 2;
  auto& solver_options = options_node["solver"];
  solver_options["solver"] = "AMG";
  solver_options["presweeps"] = 1;
  solver_options["postsweeps"] = 2;
  solver_options["interpolator"] = "D2";
  solver_options["max_iters"] = 2;
  solver_options["convergence"] = "ABSOLUTE";
  solver_options["cycle"] = "V";

  if (options.verbose) {
    options_node["solver/obtain_timings"] = 1;
    options_node["solver/monitor_residual"] = 1;
    options_node["solver/print_solve_stats"] = 1;
  }

  // TODO: Use magic_enum here when we can switch to GCC 9+
  // This is an immediately-invoked lambda so that the map
  // can be const without needed to initialize all the values
  // in the constructor
  static const auto solver_names = []() {
    std::unordered_map<AMGXSolver, std::string> names;
    names[AMGXSolver::AMG] = "AMG";
    names[AMGXSolver::PCGF] = "PCGF";
    names[AMGXSolver::CG] = "CG";
    names[AMGXSolver::PCG] = "PCG";
    names[AMGXSolver::PBICGSTAB] = "PBICGSTAB";
    names[AMGXSolver::BICGSTAB] = "BICGSTAB";
    names[AMGXSolver::FGMRES] = "FGMRES";
    names[AMGXSolver::JACOBI_L1] = "JACOBI_L1";
    names[AMGXSolver::GS] = "GS";
    names[AMGXSolver::POLYNOMIAL] = "POLYNOMIAL";
    names[AMGXSolver::KPZ_POLYNOMIAL] = "KPZ_POLYNOMIAL";
    names[AMGXSolver::BLOCK_JACOBI] = "BLOCK_JACOBI";
    names[AMGXSolver::MULTICOLOR_GS] = "MULTICOLOR_GS";
    names[AMGXSolver::MULTICOLOR_DILU] = "MULTICOLOR_DILU";
    return names;
  }();

  options_node["solver/solver"] = solver_names.at(options.solver);
  options_node["solver/smoother"] = solver_names.at(options.smoother);

  // Treat the string as the config (not a filename)
  amgx->ReadParameters(options_node.to_json(), mfem::AmgXSolver::INTERNAL);
  amgx->InitExclusiveGPU(comm);

  return amgx;
}
#endif

std::unique_ptr<mfem::Solver> buildPreconditioner(LinearSolverOptions linear_opts, [[maybe_unused]] MPI_Comm comm)
{
  std::unique_ptr<mfem::Solver> preconditioner_solver;
  auto preconditioner = linear_opts.preconditioner;
  auto preconditioner_print_level = linear_opts.preconditioner_print_level;

  // Handle the preconditioner - currently just BoomerAMG and HypreSmoother are supported
  if (preconditioner == Preconditioner::HypreAMG) {
    auto amg_preconditioner = std::make_unique<mfem::HypreBoomerAMG>();
    amg_preconditioner->SetPrintLevel(preconditioner_print_level);
    preconditioner_solver = std::move(amg_preconditioner);
  } else if (preconditioner == Preconditioner::HypreJacobi) {
    auto jac_preconditioner = std::make_unique<mfem::HypreSmoother>();
    jac_preconditioner->SetType(mfem::HypreSmoother::Type::Jacobi);
    preconditioner_solver = std::move(jac_preconditioner);
  } else if (preconditioner == Preconditioner::HypreL1Jacobi) {
    auto jacl1_preconditioner = std::make_unique<mfem::HypreSmoother>();
    jacl1_preconditioner->SetType(mfem::HypreSmoother::Type::l1Jacobi);
    preconditioner_solver = std::move(jacl1_preconditioner);
  } else if (preconditioner == Preconditioner::HypreGaussSeidel) {
    auto gs_preconditioner = std::make_unique<mfem::HypreSmoother>();
    gs_preconditioner->SetType(mfem::HypreSmoother::Type::GS);
    preconditioner_solver = std::move(gs_preconditioner);
  } else if (preconditioner == Preconditioner::HypreILU) {
    auto ilu_preconditioner = std::make_unique<mfem::HypreILU>();
    ilu_preconditioner->SetLevelOfFill(1);
    ilu_preconditioner->SetPrintLevel(preconditioner_print_level);
    preconditioner_solver = std::move(ilu_preconditioner);
  } else if (preconditioner == Preconditioner::AMGX) {
#ifdef MFEM_USE_AMGX
    preconditioner_solver = buildAMGX(linear_opts.amgx_options, comm);
#else
    SLIC_ERROR_ROOT("AMGX requested in non-GPU build");
#endif
  } else if (preconditioner == Preconditioner::Petsc) {
#ifdef SMITH_USE_PETSC
    preconditioner_solver = mfem_ext::buildPetscPreconditioner(linear_opts.petsc_preconditioner, comm);
#else
    SLIC_ERROR_ROOT("PETSc preconditioner requested in non-PETSc build");
#endif
  } else if (preconditioner == Preconditioner::AMGFContact) {
    auto amgfcontact_preconditioner = std::make_unique<mfem::AMGFSolver>();
    auto amgfcontact_opts = linear_opts.amgfcontact_options;
    amgfcontact_preconditioner->GetAMG().SetPrintLevel(preconditioner_print_level);
    amgfcontact_preconditioner->GetAMG().SetSystemsOptions(amgfcontact_opts.dim_systems_options);
    amgfcontact_preconditioner->GetAMG().SetRelaxType(amgfcontact_opts.relax_type);
    preconditioner_solver = std::move(amgfcontact_preconditioner);
  } else {
    SLIC_ERROR_ROOT_IF(preconditioner != Preconditioner::None, "Unknown preconditioner type requested");
  }

  return preconditioner_solver;
}

void EquationSolver::defineInputFileSchema(axom::inlet::Container& container)
{
  auto& linear_container = container.addStruct("linear", "Linear Equation Solver Parameters");
  linear_container.required().registerVerifier([](const axom::inlet::Container& container_to_verify) {
    // Make sure that the provided options match the desired linear solver type
    const bool is_iterative = (container_to_verify["type"].get<std::string>() == "iterative") &&
                              container_to_verify.contains("iterative_options");
    const bool is_direct =
        (container_to_verify["type"].get<std::string>() == "direct") && container_to_verify.contains("direct_options");
    return is_iterative || is_direct;
  });

  // Enforce the solver type - must be iterative or direct
  linear_container.addString("type", "The type of solver parameters to use (iterative|direct)")
      .required()
      .validValues({"iterative", "direct"});

  auto& iterative_container = linear_container.addStruct("iterative_options", "Iterative solver parameters");
  iterative_container.addDouble("rel_tol", "Relative tolerance for the linear solve.").defaultValue(1.0e-6);
  iterative_container.addDouble("abs_tol", "Absolute tolerance for the linear solve.").defaultValue(1.0e-8);
  iterative_container.addInt("max_iter", "Maximum iterations for the linear solve.").defaultValue(5000);
  iterative_container.addInt("print_level", "Linear print level.").defaultValue(0);
  iterative_container.addString("solver_type", "Solver type (gmres|minres|cg).").defaultValue("gmres");
  iterative_container.addString("prec_type", "Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|ILU|Petsc).")
      .defaultValue("JacobiSmoother");
  iterative_container.addString("petsc_prec_type", "Type of PETSc preconditioner to use.").defaultValue("jacobi");

  auto& direct_container = linear_container.addStruct("direct_options", "Direct solver parameters");
  direct_container.addInt("print_level", "Linear print level.").defaultValue(0);

  // Only needed for nonlinear problems
  auto& nonlinear_container = container.addStruct("nonlinear", "Newton Equation Solver Parameters").required(false);
  nonlinear_container.addDouble("rel_tol", "Relative tolerance for the Newton solve.").defaultValue(1.0e-2);
  nonlinear_container.addDouble("abs_tol", "Absolute tolerance for the Newton solve.").defaultValue(1.0e-4);
  nonlinear_container.addInt("max_iter", "Maximum iterations for the Newton solve.").defaultValue(500);
  nonlinear_container.addInt("print_level", "Nonlinear print level.").defaultValue(0);
  nonlinear_container
      .addString("solver_type", "Solver type (Newton|NewtonLineSearch|TrustRegion|PcgBlock|KINFullStep|KINLineSearch)")
      .defaultValue("Newton");
}

}  // namespace smith

using smith::EquationSolver;
using smith::LinearSolverOptions;
using smith::NonlinearSolverOptions;

smith::LinearSolverOptions FromInlet<smith::LinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  LinearSolverOptions options;
  std::string type = base["type"];

  if (type == "direct") {
    options.linear_solver = smith::LinearSolver::SuperLU;
    options.print_level = base["direct_options/print_level"];
    return options;
  }

  auto config = base["iterative_options"];
  options.relative_tol = config["rel_tol"];
  options.absolute_tol = config["abs_tol"];
  options.max_iterations = config["max_iter"];
  options.print_level = config["print_level"];
  std::string solver_type = config["solver_type"];
  if (solver_type == "gmres") {
    options.linear_solver = smith::LinearSolver::GMRES;
  } else if (solver_type == "cg") {
    options.linear_solver = smith::LinearSolver::CG;
  } else {
    std::string msg = std::format("Unknown Linear solver type given: '{0}'", solver_type);
    SLIC_ERROR_ROOT(msg);
  }
  const std::string prec_type = config["prec_type"];
  if (prec_type == "JacobiSmoother") {
    options.preconditioner = smith::Preconditioner::HypreJacobi;
  } else if (prec_type == "L1JacobiSmoother") {
    options.preconditioner = smith::Preconditioner::HypreL1Jacobi;
  } else if (prec_type == "HypreAMG") {
    options.preconditioner = smith::Preconditioner::HypreAMG;
  } else if (prec_type == "ILU") {
    options.preconditioner = smith::Preconditioner::HypreILU;
#ifdef MFEM_USE_AMGX
  } else if (prec_type == "AMGX") {
    options.preconditioner = smith::Preconditioner::AMGX;
#endif
  } else if (prec_type == "GaussSeidel") {
    options.preconditioner = smith::Preconditioner::HypreGaussSeidel;
#ifdef SMITH_USE_PETSC
  } else if (prec_type == "Petsc") {
    const std::string petsc_prec = config["petsc_prec_type"];
    options.preconditioner = smith::Preconditioner::Petsc;
    options.petsc_preconditioner = smith::mfem_ext::stringToPetscPCType(petsc_prec);
#endif
  } else if (prec_type == "AMGFContact") {
    options.preconditioner = smith::Preconditioner::AMGFContact;
  } else {
    std::string msg = std::format("Unknown preconditioner type given: '{0}'", prec_type);
    SLIC_ERROR_ROOT(msg);
  }

  return options;
}

smith::NonlinearSolverOptions FromInlet<smith::NonlinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  NonlinearSolverOptions options;
  options.relative_tol = base["rel_tol"];
  options.absolute_tol = base["abs_tol"];
  options.max_iterations = base["max_iter"];
  options.print_level = base["print_level"];
  const std::string solver_type = base["solver_type"];
  if (solver_type == "Newton") {
    options.nonlin_solver = smith::NonlinearSolver::Newton;
  } else if (solver_type == "NewtonLineSearch") {
    options.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch;
  } else if (solver_type == "TrustRegion") {
    options.nonlin_solver = smith::NonlinearSolver::TrustRegion;
  } else if (solver_type == "PcgBlock") {
    options.nonlin_solver = smith::NonlinearSolver::PcgBlock;
  } else if (solver_type == "KINFullStep") {
    options.nonlin_solver = smith::NonlinearSolver::KINFullStep;
  } else if (solver_type == "KINLineSearch") {
    options.nonlin_solver = smith::NonlinearSolver::KINBacktrackingLineSearch;
  } else if (solver_type == "KINPicard") {
    options.nonlin_solver = smith::NonlinearSolver::KINPicard;
  } else {
    SLIC_ERROR_ROOT(std::format("Unknown nonlinear solver type given: '{0}'", solver_type));
  }
  return options;
}

smith::EquationSolver FromInlet<smith::EquationSolver>::operator()(const axom::inlet::Container& base)
{
  auto lin = base["linear"].get<LinearSolverOptions>();
  auto nonlin = base["nonlinear"].get<NonlinearSolverOptions>();

  auto [linear_solver, preconditioner] = smith::buildLinearSolverAndPreconditioner(lin, MPI_COMM_WORLD);

  smith::EquationSolver eq_solver(smith::buildNonlinearSolver(nonlin, lin, *preconditioner, MPI_COMM_WORLD),
                                  std::move(linear_solver), std::move(preconditioner));

  return eq_solver;
}
