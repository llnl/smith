// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/steihaug_toint_cg.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/numerics/trust_region_subspace_cache.hpp"

#include <array>
#include <cstdlib>
#include <deque>
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
#include "smith/numerics/deflation.hpp"
#include "smith/numerics/bsr_operator.hpp"
#include "smith/numerics/functional/assembly_timers.hpp"
#include "smith/infrastructure/logger.hpp"

namespace smith {

namespace {

/// Symmetric Lanczos with full reorthogonalization. Computes `n_evecs` approximate leftmost
/// eigenvalues/eigenvectors of the symmetric operator `A` using `n_iter` Krylov iterations
/// (n_iter ≥ n_evecs). `dots` is the MPI-aware dot helper. Start vector is `b_seed` if its
/// size matches; otherwise a deterministic non-zero pattern.
///
/// Returns the approximate eigenpairs sorted ascending by eigenvalue. Approximations to the
/// true leftmost eigenpairs of A converge as `n_iter` grows beyond `n_evecs`; in practice
/// 2×n_evecs Krylov iterations give useful approximations for extreme eigenvalues even on
/// indefinite operators.
void lanczosLeftmostEigvecs(const mfem::Operator& A, int n_iter, int n_evecs, const mfem::Vector& b_seed,
                            std::vector<mfem::Vector>& evecs_out, std::vector<double>& evals_out,
                            const DotManyFunction& dots)
{
  evecs_out.clear();
  evals_out.clear();
  const int N = A.Height();
  n_iter = std::min(std::max(n_iter, n_evecs), N);
  if (n_iter <= 0) return;

  std::vector<mfem::Vector> V;
  V.reserve(static_cast<size_t>(n_iter));
  std::vector<double> alpha(static_cast<size_t>(n_iter), 0.0);
  std::vector<double> beta(static_cast<size_t>(n_iter), 0.0);

  // Seed vector: use b_seed if size matches, else a deterministic non-trivial pattern.
  mfem::Vector v(N);
  if (b_seed.Size() == N) {
    v = b_seed;
  } else {
    for (int i = 0; i < N; ++i) v(i) = std::sin(0.137 * i + 1.7);
  }
  double vv = dots({{&v, &v}})[0];
  if (vv <= 0.0) return;
  v *= 1.0 / std::sqrt(vv);
  V.emplace_back(v);

  mfem::Vector w(N), vprev(N);
  vprev = 0.0;

  for (int j = 0; j < n_iter; ++j) {
    A.Mult(V[static_cast<size_t>(j)], w);
    alpha[static_cast<size_t>(j)] = dots({{&V[static_cast<size_t>(j)], &w}})[0];
    if (j > 0) w.Add(-beta[static_cast<size_t>(j - 1)], V[static_cast<size_t>(j - 1)]);
    w.Add(-alpha[static_cast<size_t>(j)], V[static_cast<size_t>(j)]);
    // Full reorthogonalization (numerical hygiene; small for the n_iter sizes here).
    for (const auto& vi : V) {
      const double c = dots({{&vi, &w}})[0];
      w.Add(-c, vi);
    }
    const double ww = dots({{&w, &w}})[0];
    if (ww <= 1e-30 || j == n_iter - 1) {
      beta[static_cast<size_t>(j)] = std::sqrt(std::max(ww, 0.0));
      break;
    }
    beta[static_cast<size_t>(j)] = std::sqrt(ww);
    w *= 1.0 / beta[static_cast<size_t>(j)];
    V.emplace_back(w);
  }
  const int m = static_cast<int>(V.size());
  if (m == 0) return;

  // Symmetric tridiagonal eigensystem via dense LAPACK (mfem::DenseMatrixEigensystem).
  mfem::DenseMatrix T(m, m);
  T = 0.0;
  for (int i = 0; i < m; ++i) {
    T(i, i) = alpha[static_cast<size_t>(i)];
    if (i + 1 < m) {
      T(i, i + 1) = beta[static_cast<size_t>(i)];
      T(i + 1, i) = beta[static_cast<size_t>(i)];
    }
  }
  mfem::DenseMatrixEigensystem eig(T);
  eig.Eval();
  const mfem::Vector& evals = eig.Eigenvalues();
  const mfem::DenseMatrix& evecs_T = eig.Eigenvectors();

  std::vector<int> order(static_cast<size_t>(m));
  for (int i = 0; i < m; ++i) order[static_cast<size_t>(i)] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return evals(a) < evals(b); });

  const int kk = std::min(n_evecs, m);
  for (int q = 0; q < kk; ++q) {
    const int idx = order[static_cast<size_t>(q)];
    evals_out.push_back(evals(idx));
    mfem::Vector full(N);
    full = 0.0;
    for (int i = 0; i < m; ++i) {
      full.Add(evecs_T(i, idx), V[static_cast<size_t>(i)]);
    }
    evecs_out.emplace_back(std::move(full));
  }
}

size_t rootOnlyPrintLevel(const mfem::NewtonSolver& solver, size_t level)
{
#ifdef MFEM_USE_MPI
  const MPI_Comm comm = solver.GetComm();
  if (level > 0 && comm != MPI_COMM_NULL) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0) {
      return 0;
    }
  }
#endif
  return level;
}

class SolverWithPreconditioner : public mfem::Solver {
 public:
  SolverWithPreconditioner(std::unique_ptr<mfem::Solver> linear_solver, std::unique_ptr<mfem::Solver> preconditioner)
      : linear_solver_(std::move(linear_solver)), preconditioner_(std::move(preconditioner))
  {
    SLIC_ERROR_IF(!linear_solver_, "SolverWithPreconditioner requires a non-null linear solver");
  }

  void SetOperator(const mfem::Operator& op) override
  {
    height = op.Height();
    width = op.Width();
    linear_solver_->SetOperator(op);
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override { linear_solver_->Mult(x, y); }

 private:
  std::unique_ptr<mfem::Solver> linear_solver_;
  std::unique_ptr<mfem::Solver> preconditioner_;
};

bool preconditionerSupportsBlockOperator(Preconditioner preconditioner)
{
  switch (preconditioner) {
    case Preconditioner::None:
    case Preconditioner::BlockDiagonal:
    case Preconditioner::BlockTriangular:
    case Preconditioner::BlockSchur:
      return true;
    default:
      return false;
  }
}

bool linearSolverSupportsBlockOperator(LinearSolver linear_solver)
{
  switch (linear_solver) {
    case LinearSolver::CG:
    case LinearSolver::GMRES:
    case LinearSolver::SuperLU:
#ifdef MFEM_USE_STRUMPACK
    case LinearSolver::Strumpack:
#endif
#ifdef SMITH_USE_PETSC
    case LinearSolver::PetscCG:
    case LinearSolver::PetscGMRES:
#endif
      return true;
    default:
      return false;
  }
}

bool monolithicizeOperatorIfNeeded(const LinearSolverOptions& linear_options, mfem::Operator& assembled_gradient,
                                   mfem::Operator*& gradient_operator)
{
  auto* block_gradient = dynamic_cast<const mfem::BlockOperator*>(&assembled_gradient);
  if (!block_gradient) {
    gradient_operator = &assembled_gradient;
    return false;
  }

  if (!requiresMonolithicOperator(linear_options)) {
    gradient_operator = &assembled_gradient;
    return false;
  }

  gradient_operator = buildMonolithicMatrix(*block_gradient).release();
  SLIC_DEBUG_ROOT(
      axom::fmt::format("Automatically monolithicizing block Jacobian for linear solver {} with "
                        "preconditioner {}",
                        linearName(linear_options.linear_solver), preconditionerName(linear_options.preconditioner)));
  return true;
}

ConvergenceStatus scalarConvergenceStatus(double residual_norm, double initial_norm, double abs_tol, double rel_tol)
{
  ConvergenceStatus status;
  status.block_norms = {residual_norm};
  status.global_norm = residual_norm;
  const double relative_base = initial_norm > 0.0 ? initial_norm : residual_norm;
  status.global_goal = std::max(abs_tol, rel_tol * relative_base);
  status.global_converged = status.global_norm <= status.global_goal;
  status.converged = status.global_converged;
  return status;
}

bool shouldUseSubspaceStep(int subspace_option, TrustRegionResults::Status status, double step_norm, double tr_size,
                           int line_search_iter, bool cg_hit_max_iters, bool cg_model_stagnated)
{
  const bool failed_or_indefinite = status == TrustRegionResults::Status::NonDescentDirection ||
                                    status == TrustRegionResults::Status::NegativeCurvature;
  const bool on_boundary = step_norm > (1.0 - 1.0e-6) * tr_size;
  const bool retrying_on_boundary = on_boundary && line_search_iter > 1;
  const bool poor_inner = cg_hit_max_iters || cg_model_stagnated;
  return ((subspace_option >= 1) && (failed_or_indefinite || retrying_on_boundary || poor_inner)) ||
         ((subspace_option >= 2) && (on_boundary || poor_inner)) || (subspace_option >= 3);
}

enum class SubspaceStepStatus
{
  Unavailable,
  Unchanged,
  Replaced
};

}  // namespace

/// @cond
/// Newton solver with a 2-way line-search.  Reverts to regular Newton if max_line_search_iterations is set to 0.
class NewtonSolver : public mfem::NewtonSolver, public ConvergenceManagedNonlinearSolver {
 protected:
  /// initial solution vector to do line-search off of
  mutable mfem::Vector x0;

  /// nonlinear solver options
  NonlinearSolverOptions nonlinear_options;

  /// linear solver options
  LinearSolverOptions linear_options;

  /// reconstructed smith print level
  mutable size_t print_level = 0;

  /// Tracks if grad was monolithicized and needs deletion
  mutable bool grad_monolithic = false;

  std::shared_ptr<EquationSolverConvergenceManager> convergence_manager_ = nullptr;

 public:
  /// constructor
  NewtonSolver(const NonlinearSolverOptions& nonlinear_opts, const LinearSolverOptions& linear_opts)
      : nonlinear_options(nonlinear_opts), linear_options(linear_opts)
  {
  }

#ifdef MFEM_USE_MPI
  /// parallel constructor
  NewtonSolver(MPI_Comm comm_, const NonlinearSolverOptions& nonlinear_opts, const LinearSolverOptions& linear_opts)
      : mfem::NewtonSolver(comm_), nonlinear_options(nonlinear_opts), linear_options(linear_opts)
  {
  }
#endif

  /// destructor
  virtual ~NewtonSolver()
  {
    if (grad_monolithic) delete grad;
  }

  void setConvergenceManager(std::shared_ptr<EquationSolverConvergenceManager> convergence_manager) override
  {
    convergence_manager_ = std::move(convergence_manager);
  }

  /// Evaluate the residual and convergence status.
  ConvergenceStatus evaluateConvergence(const mfem::Vector& x, mfem::Vector& rOut) const
  {
    SMITH_MARK_FUNCTION;
    ConvergenceStatus status;
    status.global_norm = std::numeric_limits<double>::max();
    status.global_goal = std::numeric_limits<double>::max();
    try {
      oper->Mult(x, rOut);
      if (convergence_manager_) {
        status = convergence_manager_->evaluate(1.0, rOut);
      } else {
        status = scalarConvergenceStatus(Norm(rOut), initial_norm, abs_tol, rel_tol);
      }
    } catch (const std::exception&) {
      status.global_norm = std::numeric_limits<double>::max();
      status.global_goal = std::numeric_limits<double>::max();
    }
    return status;
  }

  /// assemble the jacobian
  void assembleJacobian(const mfem::Vector& x) const
  {
    SMITH_MARK_FUNCTION;
    if (grad_monolithic) {
      delete grad;
      grad = nullptr;
      grad_monolithic = false;
    }
    mfem::Operator& assembled_gradient = oper->GetGradient(x);
    grad_monolithic = monolithicizeOperatorIfNeeded(linear_options, assembled_gradient, grad);
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
  void Mult(const mfem::Vector&, mfem::Vector& x) const override
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

    print_level = static_cast<size_t>(std::max(nonlinear_options.print_level, 0));
    print_level = print_options.iterations ? std::max<size_t>(1, print_level) : print_level;
    print_level = print_options.summary ? std::max<size_t>(2, print_level) : print_level;
    print_level = rootOnlyPrintLevel(*this, print_level);

    using real_t = mfem::real_t;

    ConvergenceStatus status = evaluateConvergence(x, r);
    real_t norm = status.global_norm;
    initial_norm = norm;
    if (norm == 0.0) return;

    if (print_level == 1) {
      mfem::out << "Newton iteration " << std::setw(3) << 0 << " : ||r|| = " << std::setw(13) << norm << "\n";
    }

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

      if (status.converged && it >= nonlinear_options.min_iterations) {
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
      status = evaluateConvergence(x, r);
      norm = status.global_norm;

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
        status = evaluateConvergence(x, r);
        norm = status.global_norm;
      }

      // try the opposite direction and linesearch back from there
      if (max_ls_iters > 0 && ls_iter == max_ls_iters && !is_improved(norm, stepScale)) {
        stepScale = 1.0;
        add(x0, stepScale, c, x);
        status = evaluateConvergence(x, r);
        norm = status.global_norm;

        ls_iter = 0;
        for (; !is_improved(norm, stepScale) && ls_iter < max_ls_iters; ++ls_iter, ++ls_iter_sum) {
          stepScale *= reduction;
          add(x0, stepScale, c, x);
          status = evaluateConvergence(x, r);
          norm = status.global_norm;
        }

        // ok, the opposite direction was also terrible, lets go back, cut in half 1 last time and accept it hoping for
        // the best
        if (ls_iter == max_ls_iters && !is_improved(norm, stepScale)) {
          ++ls_iter_sum;
          stepScale *= reduction;
          add(x0, -stepScale, c, x);
          status = evaluateConvergence(x, r);
          norm = status.global_norm;
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

/// trust region printing utility function
void printTrustRegionInfo(double realWork, double modelObjective, size_t cgIters, double trSize, bool willAccept)
{
  mfem::out << "real work = " << std::setw(13) << realWork << ", model energy = " << std::setw(13) << modelObjective
            << ", cg iter = " << std::setw(7) << cgIters << ", next tr size = " << std::setw(8) << trSize
            << ", accepting = " << willAccept << std::endl;
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
class TrustRegion : public mfem::NewtonSolver, public ConvergenceManagedNonlinearSolver {
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
  /// accepted trust-region steps available for future subspace solves
  mutable std::deque<mfem::Vector> previous_steps;

  /// nonlinear solution options
  NonlinearSolverOptions nonlinear_options;
  /// linear solution options
  LinearSolverOptions linear_options;

  /// handle to the preconditioner used by the trust region, it ignores the linear solver as a SPD preconditioner is
  /// currently required
  Solver& tr_precond;

  /// non-owning view of `tr_precond` when it is a DeflationPreconditioner; nullptr otherwise.
  /// Populated lazily on first SetOperator. Enables tighter coupling: the leftmost coarse
  /// direction is treated as a candidate negative-curvature direction in the linesearch and
  /// (when the SLEPc subspace path is active) is appended to `left_mosts`.
  mutable DeflationPreconditioner* deflation_precond_ = nullptr;
  mutable bool deflation_precond_checked_ = false;

  /// Adaptive deflation-pieces ladder (enabled by deflation_pieces == 0): start at s=1,
  /// probe a doubling every window; accept on a clear wall/outer improvement, otherwise
  /// revert and lock. Decisions use an allreduced window wall so every rank decides
  /// identically. State persists across load steps (the solver object is reused).
  struct AdaptivePiecesState {
    bool enabled = false;
    bool locked = false;
    bool probing = false;          // current window is evaluating a freshly doubled s
    int current_s = 1;
    int previous_s = 1;
    double previous_metric = 0.0;  // accepted-config wall/outer to beat
    size_t window_outers = 0;
    double window_t0 = 0.0;
    static constexpr size_t window = 25;            // outers per decision window
    static constexpr double improve_margin = 0.95;  // accept doubling on a >5% wall/outer win
    static constexpr double attempt_gate = 50.0;    // only experiment if iters/outer >= this
  };
  mutable AdaptivePiecesState adapt_pieces_;
  mutable size_t window_cg_iters_ = 0;
  mutable bool force_precond_refresh_ = false;
  /// Cached coarse leftmost direction and eigenvalue, refreshed each precond rebuild.
  mutable mfem::Vector deflation_leftmost_w_;
  mutable double deflation_leftmost_lambda_ = 0.0;

  /// Lanczos approximate leftmost eigvecs of the assembled Hessian, refreshed each outer
  /// iter (when `trust_num_lanczos > 0`). Pushed into `left_mosts` before solveModelProblem.
  mutable std::vector<mfem::Vector> lanczos_evecs_;
  mutable std::vector<double> lanczos_evals_;
  mutable double lanczos_time_ = 0.0;
  mutable size_t num_lanczos_calls_ = 0;

  /// reconstructed smith print level
  mutable size_t print_level = 0;

  /// optional optimized block sparse row operator
  mutable std::unique_ptr<BSROperator> bsr_operator_;
  /// non-owning view of the current gradient when it is a BSROperator (wrapped or direct-assembled)
  mutable const BSROperator* grad_bsr_view_ = nullptr;

  /// Tracks if grad was monolithicized and needs deletion
  mutable bool grad_monolithic = false;

  // === TIMING BEGIN ===
  mutable double hess_vec_time_ = 0.0;
  mutable double precond_time_ = 0.0;
  mutable double augment_time_ = 0.0;
  mutable double assemble_jacobian_time_ = 0.0;
  mutable double assemble_gradient_time_ = 0.0;
  mutable double bsr_convert_time_ = 0.0;
  mutable double precond_setop_time_ = 0.0;
  mutable double subspace_prepare_time_ = 0.0;
  mutable double subspace_solve_time_ = 0.0;
  mutable double batch_hess_vec_time_ = 0.0;
  mutable size_t num_hess_vecs_ = 0;
  mutable size_t num_preconds_ = 0;
  mutable size_t num_augments_ = 0;
  mutable size_t num_jacobian_assembles_ = 0;
  /// true while `grad` matches the current iterate X; cleared whenever X changes.
  mutable bool grad_is_current_ = false;
  mutable size_t num_precond_setops_ = 0;
  mutable size_t num_subspace_prepares_ = 0;
  mutable size_t num_subspace_solves_ = 0;
  mutable size_t num_batch_hess_vec_calls_ = 0;
  mutable size_t num_batch_hess_vec_actions_ = 0;
  /// In-CG profile accumulators (filled by steihaugTointCG, pointer-passed).
  mutable CGProfile cg_profile_;
  /// Histogram of CG iteration counts per outer TR iteration.
  /// Bins (right-exclusive): [0,10), [10,50), [50,200), [200,1000), [1000,max).
  mutable std::array<size_t, 5> cg_iter_hist_{};
  mutable size_t cg_outer_count_ = 0;
  mutable size_t cg_iter_sum_ = 0;
  mutable size_t cg_iter_max_ = 0;
  /// Per-outer-iter status counts (Interior, NegCurv, OnBoundary, NonDescent).
  mutable std::array<size_t, 4> cg_status_hist_{};
  /// # outer iters where the CG hit `max_cg_iterations` (a subset of Interior).
  mutable size_t cg_hit_max_iters_count_ = 0;
  /// Sum of line-search retries per outer iter.
  mutable size_t line_search_retries_total_ = 0;
  mutable size_t line_search_retries_max_ = 0;
  // === TIMING END ===

  std::shared_ptr<EquationSolverConvergenceManager> convergence_manager_ = nullptr;

  /// Optional exact-energy evaluator. When set, the linesearch acceptance uses
  /// `realObjective = E(x+d) - E(x)` instead of an integrated-work surrogate.
  EquationSolver::EnergyFunction energy_function_;

 public:
  /// Set the exact-energy evaluator. Pass an empty std::function to disable.
  void setEnergyFunction(EquationSolver::EnergyFunction fn) { energy_function_ = std::move(fn); }
  bool hasEnergyFunction() const { return static_cast<bool>(energy_function_); }

#ifdef MFEM_USE_MPI
  /// constructor
  TrustRegion(MPI_Comm comm_, const NonlinearSolverOptions& nonlinear_opts, const LinearSolverOptions& linear_opts,
              Solver& tPrec)
      : mfem::NewtonSolver(comm_), nonlinear_options(nonlinear_opts), linear_options(linear_opts), tr_precond(tPrec)
  {
  }
#endif

  /// destructor
  virtual ~TrustRegion()
  {
    if (grad_monolithic) delete grad;
  }

  void setConvergenceManager(std::shared_ptr<EquationSolverConvergenceManager> convergence_manager) override
  {
    convergence_manager_ = std::move(convergence_manager);
  }

  /// auto-detect BSR block size from the FES vdim; an explicit user setting (>0) wins
  void setBSRBlockSize(int block_size) override
  {
    if (linear_options.bsr_block_size <= 0) linear_options.bsr_block_size = block_size;
  }

  /// compute several vector inner products with a single MPI reduction when possible
  std::vector<double> dot_many(const std::vector<DotPair>& pairs) const
  {
    if (dot_oper) {
      std::vector<double> products(pairs.size(), 0.0);
      for (size_t i = 0; i < pairs.size(); ++i) {
        products[i] = Dot(*pairs[i].first, *pairs[i].second);
      }
      return products;
    }

    std::vector<double> products = smith::dotMany(pairs);

#ifdef MFEM_USE_MPI
    const MPI_Comm dot_comm = GetComm();
    if (dot_comm != MPI_COMM_NULL) {
      std::vector<mfem::real_t> global_products(pairs.size());
      MPI_Allreduce(products.data(), global_products.data(), static_cast<int>(pairs.size()), MFEM_MPI_REAL_T, MPI_SUM,
                    dot_comm);
      products.assign(global_products.begin(), global_products.end());
    }
#endif

    return products;
  }

  /// build reusable subspace data for line-search retries
  bool prepareSubspaceProblemCache([[maybe_unused]] const std::vector<const mfem::Vector*>& ds,
                                   [[maybe_unused]] const std::vector<const mfem::Vector*>& Hds,
                                   [[maybe_unused]] const mfem::Vector& g, [[maybe_unused]] int num_leftmost,
                                   [[maybe_unused]] TrustRegionSubspaceCache& subspace_cache) const
  {
#ifdef MFEM_USE_LAPACK
    SMITH_MARK_FUNCTION;
    std::vector<const mfem::Vector*> directions(ds.begin(), ds.end());
    std::vector<const mfem::Vector*> H_directions(Hds.begin(), Hds.end());
    for (auto& left : left_mosts) directions.emplace_back(left.get());
    for (auto& H_left : H_left_mosts) H_directions.emplace_back(H_left.get());

    mfem::Vector b(g);
    b *= -1;

    ++num_subspace_prepares_;
    double t0 = MPI_Wtime();
    try {
      subspace_cache.prepare(directions, H_directions, b, num_leftmost, GetComm());
    } catch (const std::exception& e) {
      subspace_prepare_time_ += MPI_Wtime() - t0;
      if (print_level >= 1) {
        mfem::out << "subspace preparation failed with " << e.what() << "; using dogleg fallback." << std::endl;
      }
      return false;
    }
    subspace_prepare_time_ += MPI_Wtime() - t0;
    return true;
#else
    return false;
#endif
  }

  /// solve cached exact trust-region subspace problem for current trust-region size
  template <typename HessVecFunc>
  SubspaceStepStatus trySubspaceStep([[maybe_unused]] mfem::Vector& z,
                                     [[maybe_unused]] const HessVecFunc& hess_vec_func,
                                     [[maybe_unused]] const TrustRegionSubspaceCache& subspace_cache,
                                     [[maybe_unused]] const mfem::Vector& g, [[maybe_unused]] double delta) const
  {
#ifdef MFEM_USE_LAPACK
    SMITH_MARK_FUNCTION;
    mfem::Vector sol;
    double energy_change;

    ++num_subspace_solves_;
    double tsol0 = MPI_Wtime();
    try {
      std::tie(sol, std::ignore, std::ignore, energy_change) = subspace_cache.solve(delta);
    } catch (const std::exception& e) {
      subspace_solve_time_ += MPI_Wtime() - tsol0;
      if (print_level >= 1) {
        mfem::out << "subspace solve failed with " << e.what() << "; using dogleg fallback." << std::endl;
      }
      return SubspaceStepStatus::Unavailable;
    }
    subspace_solve_time_ += MPI_Wtime() - tsol0;

    double base_energy = computeEnergy(g, hess_vec_func, z);
    double subspace_energy = computeEnergy(g, hess_vec_func, sol);

    if (print_level >= 2) {
      double leftval = subspace_cache.leftvals.empty() ? 1.0 : subspace_cache.leftvals[0];
      mfem::out << "Energy using subspace solver from: " << base_energy << ", to: " << subspace_energy << " / "
                << energy_change << ".  Min eig: " << leftval << std::endl;
    }

    if (subspace_energy < base_energy) {
      z = sol;
      return SubspaceStepStatus::Replaced;
    }
    return SubspaceStepStatus::Unchanged;
#else
    return SubspaceStepStatus::Unavailable;
#endif
  }

  /// finds tau s.t. (z + tau*(y-z))^2 = trSize^2
  void projectToBoundaryBetweenWithCoefs(mfem::Vector& z, const mfem::Vector& y, double trSize, double zz, double zy,
                                         double yy) const
  {
    double dd = yy - 2 * zy + zz;
    double zd = zy - zz;
    double boundary_gap = std::max(trSize * trSize - zz, 0.0);
    if (boundary_gap == 0.0) return;
    double tau = (std::sqrt(boundary_gap * dd + zd * zd) - zd) / dd;
    z.Add(-tau, z);
    z.Add(tau, y);
  }

  /// take a dogleg step in direction s, solution norm must be within trSize
  void doglegStep(const mfem::Vector& cp, const mfem::Vector& newtonP, double trSize, mfem::Vector& s) const
  {
    SMITH_MARK_FUNCTION;
    const auto dots = dot_many({{&cp, &cp}, {&newtonP, &newtonP}});
    const double cc = dots[0];
    const double nn = dots[1];
    double tt = trSize * trSize;

    s = 0.0;
    if (cc >= tt) {
      add(s, std::sqrt(tt / cc), cp, s);
    } else if (cc > nn) {
      if (print_level >= 2) {
        mfem::out << "cp outside newton, preconditioner likely inaccurate\n";
      }
      add(s, 1.0, cp, s);
    } else if (nn > tt) {  // on the dogleg (we have nn >= cc, and tt >= cc)
      add(s, 1.0, cp, s);
      double cn = Dot(cp, newtonP);
      projectToBoundaryBetweenWithCoefs(s, newtonP, trSize, cc, cn, nn);
    } else {
      s = newtonP;
    }
  }

  /// compute the energy of the linearized system for a given solution vector z
  template <typename HessVecFunc>
  double computeEnergy(const mfem::Vector& r_local, const HessVecFunc& H, const mfem::Vector& z) const
  {
    SMITH_MARK_FUNCTION;
    double rz = Dot(r_local, z);
    mfem::Vector tmp(r_local);
    tmp = 0.0;
    H(z, tmp);
    return rz + 0.5 * Dot(z, tmp);
  }

  /// Minimize quadratic sub-problem given residual vector, the action of the stiffness and a preconditioner
  void solveModelProblem(const mfem::Vector& r0, mfem::Vector& rCurrent, const mfem::Operator& H, const mfem::Solver* P,
                         const TrustRegionSettings& settings, double& trSize, TrustRegionResults& results,
                         double r0_norm_squared) const
  {
    auto dot_many_lambda = [this](const std::vector<DotPair>& pairs) { return dot_many(pairs); };
    try {
      steihaugTointCG(r0, rCurrent, H, P, settings, trSize, results, r0_norm_squared, dot_many_lambda, &cg_profile_);
    } catch (const DeflationIndefiniteCoarseException& e) {
      results.interior_status = TrustRegionResults::Status::NegativeCurvature;
      results.cg_iterations_count = std::max<size_t>(results.cg_iterations_count, 1);
      results.cg_hit_max_iters = false;
      results.cg_model_stagnated = false;
      // Per user spec: z = leftmost direction as-is. The subspace solver picks optimal sign
      // and magnitude. Do NOT append to left_mosts — z itself is fed to the subspace as one
      // of its bases, alongside cauchy, previous_steps, and the existing left_mosts ring.
      results.z = e.direction();
      rCurrent = r0;
      if (print_level >= 2) {
        mfem::out << "Deflation coarse operator is indefinite, lambda_min = " << e.eigenvalue()
                  << "; z := W*v_min, routed to subspace.\n";
      }
    }
  }

  void fallbackToCauchyPoint(TrustRegionResults& results, const char* reason) const
  {
    if (print_level >= 2) {
      mfem::out << reason << "; using cauchy point fallback." << std::endl;
    }
    results.d = results.cauchy_point;
  }

  bool isDescentStep(const mfem::Vector& step, const mfem::Vector& residual) const
  {
    auto dot_many_lambda = [this](const std::vector<DotPair>& pairs) { return dot_many(pairs); };
    return smith::isDescentDirection(step, residual, dot_many_lambda);
  }

  void saveAcceptedStep(const mfem::Vector& step) const
  {
    const int max_previous_steps = nonlinear_options.num_previous_steps;
    if (max_previous_steps <= 0) {
      previous_steps.clear();
      return;
    }

    previous_steps.emplace_back(step);
    while (previous_steps.size() > static_cast<size_t>(max_previous_steps)) {
      previous_steps.pop_front();
    }
  }

  void acceptStep(TrustRegionResults& trResults, const TrustRegionSubspaceCache& subspace_cache,
                  const mfem::Vector& accepted_x, const mfem::Vector& accepted_r,
                  const ConvergenceStatus& predicted_status, mfem::Vector& X, mfem::Vector& r,
                  ConvergenceStatus& status, mfem::real_t& norm) const
  {
    saveAcceptedStep(trResults.d);
    if (!subspace_cache.leftmosts.empty()) {
      left_mosts = subspace_cache.leftmosts;
    }
    grad_is_current_ = false;
    X = accepted_x;
    r = accepted_r;
    status = predicted_status;
    norm = status.global_norm;
  }

  template <typename HessVecFunc>
  void computeHessianActions(const std::vector<const mfem::Vector*>& inputs, const std::vector<mfem::Vector*>& outputs,
                             const HessVecFunc& hess_vec_func) const
  {
    MFEM_VERIFY(inputs.size() == outputs.size(), "Subspace Hessian-vector batch input/output size mismatch");
    ++num_batch_hess_vec_calls_;
    num_batch_hess_vec_actions_ += inputs.size();
    double t0 = MPI_Wtime();
    if (grad_bsr_view_ && grad_bsr_view_->Enabled() && grad == grad_bsr_view_) {
      // one packed halo exchange + one matrix sweep for the whole batch
      grad_bsr_view_->MultBatch(inputs, outputs);
    } else {
      for (size_t i = 0; i < inputs.size(); ++i) {
        hess_vec_func(*inputs[i], *outputs[i]);
      }
    }
    batch_hess_vec_time_ += MPI_Wtime() - t0;
  }

  /// assemble the jacobian
  void assembleJacobian(const mfem::Vector& x) const
  {
    SMITH_MARK_FUNCTION;
    ++num_jacobian_assembles_;
    double t0 = MPI_Wtime();
    if (grad_monolithic) {
      delete grad;
      grad = nullptr;
      grad_monolithic = false;
    }
    mfem::Operator& assembled_gradient = oper->GetGradient(x);
    grad_monolithic = monolithicizeOperatorIfNeeded(linear_options, assembled_gradient, grad);
    double t1 = MPI_Wtime();
    assemble_gradient_time_ += t1 - t0;

    grad_bsr_view_ = dynamic_cast<const BSROperator*>(grad);  // direct-BSR assembly hands us one already
    if (!grad_bsr_view_ && linear_options.use_bsr_spmv) {
      MFEM_VERIFY(linear_options.bsr_block_size > 0,
                  "use_bsr_spmv requires a block size; attach an FES (auto-detect) or set bsr_block_size explicitly");
      if (auto* hypre_grad = dynamic_cast<mfem::HypreParMatrix*>(const_cast<mfem::Operator*>(grad))) {
        bsr_operator_ = std::make_unique<smith::BSROperator>(hypre_grad, linear_options.bsr_block_size);
        grad = bsr_operator_.get();
        grad_bsr_view_ = bsr_operator_.get();
      }
    } else if (!grad_bsr_view_) {
      bsr_operator_.reset();
    }
    bsr_convert_time_ += MPI_Wtime() - t1;
    assemble_jacobian_time_ += MPI_Wtime() - t0;
  }

  /// evaluate the nonlinear residual
  mfem::real_t computeResidual(const mfem::Vector& x_, mfem::Vector& r_) const
  {
    SMITH_MARK_FUNCTION;
    oper->Mult(x_, r_);
    return Norm(r_);
  }

  /// apply the action of the current Jacobian representation to a vector
  ConvergenceStatus evaluateConvergence(const mfem::Vector& x_, mfem::Vector& r_) const
  {
    ConvergenceStatus status;
    status.global_norm = std::numeric_limits<double>::max();
    status.global_goal = std::numeric_limits<double>::max();
    try {
      status.global_norm = computeResidual(x_, r_);
      if (convergence_manager_) {
        status = convergence_manager_->evaluate(1.0, r_);
      } else {
        status = scalarConvergenceStatus(status.global_norm, initial_norm, abs_tol, rel_tol);
      }
    } catch (const std::exception&) {
      status.global_norm = std::numeric_limits<double>::max();
      status.global_goal = std::numeric_limits<double>::max();
    }
    return status;
  }

  /// apply the action of the assembled Jacobian matrix to a vector
  void hessVec(const mfem::Vector& x_, mfem::Vector& v_) const
  {
    SMITH_MARK_FUNCTION;
    ++num_hess_vecs_;
    double t0 = MPI_Wtime();
    grad->Mult(x_, v_);
    hess_vec_time_ += MPI_Wtime() - t0;
  }

  /// apply trust region specific preconditioner
  void precond(const mfem::Vector& x_, mfem::Vector& v_) const
  {
    SMITH_MARK_FUNCTION;
    ++num_preconds_;
    double t0 = MPI_Wtime();
    tr_precond.Mult(x_, v_);
    precond_time_ += MPI_Wtime() - t0;
  };

  /// Lazy dynamic_cast detection of the deflation preconditioner.
  void detectDeflationPrecond() const
  {
    if (deflation_precond_checked_) return;
    deflation_precond_ = dynamic_cast<DeflationPreconditioner*>(&tr_precond);
    deflation_precond_checked_ = true;
  }

  /// Refresh the cached coarse leftmost (W * v_min of W^T A W) after a precond rebuild.
  /// Eigen-decomp of the dense m×m coarse matrix is cheap for m = 12 · num_ranks.
  void refreshDeflationLeftmost() const
  {
    if (!deflation_precond_) return;
    try {
      deflation_leftmost_lambda_ = deflation_precond_->coarseLeftmostEigenvalue();
      deflation_precond_->coarseLeftmostDirection(deflation_leftmost_w_);
    } catch (const std::exception&) {
      deflation_leftmost_w_.SetSize(0);
    }
  }

  /// 2D model-objective minimization in span(d, w): minimize g^T s + 0.5 s^T A s s.t. ||s|| ≤ Δ.
  /// Closed-form optimal α in `s = d + α w` when w^T A w > 0; otherwise step to TR boundary in
  /// the descent direction along w. Replaces `d` only if the resulting model objective strictly
  /// improves on the original `d`. Costs ~2 hess_vecs (Aw and Ad_new).
  template <typename HessVecFunc>
  void augmentDirectionWithDeflationLeftmost(mfem::Vector& d, const mfem::Vector& g, double tr_size,
                                             const HessVecFunc& H) const
  {
    if (!deflation_precond_ || deflation_leftmost_w_.Size() != d.Size()) return;
    const mfem::Vector& w = deflation_leftmost_w_;

    mfem::Vector Ad(d.Size()), Aw(d.Size());
    H(d, Ad);
    H(w, Aw);
    const double q_old = Dot(g, d) + 0.5 * Dot(d, Ad);

    const double wAw = Dot(w, Aw);
    const double dAw = Dot(d, Aw);
    const double gw = Dot(g, w);

    // Unconstrained optimum along α: solves d/dα [g^T(d+αw) + 0.5 (d+αw)^T A (d+αw)] = 0
    double alpha = 0.0;
    if (wAw > 1e-30) {
      alpha = -(dAw + gw) / wAw;
    } else {
      // Negative or zero curvature along w → go to boundary in the descent direction.
      alpha = (dAw + gw > 0.0) ? -1.0 : 1.0;
    }

    // Constrain ||d + α w||^2 ≤ tr_size^2: solve quadratic in α.
    const double dd = Dot(d, d), dw = Dot(d, w), ww = Dot(w, w);
    if (ww > 1e-30) {
      const double radius_sq = tr_size * tr_size;
      const double disc = dw * dw - ww * (dd - radius_sq);
      if (disc < 0.0) return;  // current d already outside TR — leave alone
      const double sq = std::sqrt(disc);
      const double a_hi = (-dw + sq) / ww;
      const double a_lo = (-dw - sq) / ww;
      if (alpha > a_hi) alpha = a_hi;
      if (alpha < a_lo) alpha = a_lo;
    }

    mfem::Vector d_new(d);
    d_new.Add(alpha, w);

    mfem::Vector Adnew(d.Size());
    H(d_new, Adnew);
    const double q_new = Dot(g, d_new) + 0.5 * Dot(d_new, Adnew);
    if (q_new < q_old) d = d_new;
  }

  /// @overload
  void Mult(const mfem::Vector&, mfem::Vector& X) const override
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");
    print_level = static_cast<size_t>(std::max(nonlinear_options.print_level, 0));
    print_level = print_options.iterations ? std::max<size_t>(1, print_level) : print_level;
    print_level = print_options.summary ? std::max<size_t>(2, print_level) : print_level;
    print_level = rootOnlyPrintLevel(*this, print_level);

    using real_t = mfem::real_t;

    ConvergenceStatus status = evaluateConvergence(X, r);
    real_t norm = status.global_norm;
    real_t norm_goal = status.global_goal;
    initial_norm = norm;
    if (norm == 0.0) return;

    if (print_level == 1) {
      mfem::out << "TrustRegion iteration " << std::setw(3) << 0 << " : ||r|| = " << std::setw(13) << norm << "\n";
    }

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
    settings.model_stagnation_tol = linear_options.cg_model_stagnation_tol;
    settings.model_stagnation_window = static_cast<size_t>(std::max(0, linear_options.cg_model_stagnation_window));
    settings.cg_tol = 0.5 * norm_goal;
    settings.t1 = nonlinear_options.tr_decrease_factor;
    settings.t2 = nonlinear_options.tr_increase_factor;
    settings.eta1 = nonlinear_options.tr_eta1;
    settings.eta2 = nonlinear_options.tr_eta2;
    settings.eta3 = nonlinear_options.tr_eta3;
    settings.eta4 = nonlinear_options.tr_eta4;

    // === TIMING BEGIN ===
    hess_vec_time_ = precond_time_ = augment_time_ = 0.0;
    assemble_jacobian_time_ = precond_setop_time_ = 0.0;
    assemble_gradient_time_ = bsr_convert_time_ = 0.0;
    gradient_assemble_timers = {};
    subspace_prepare_time_ = subspace_solve_time_ = batch_hess_vec_time_ = 0.0;
    num_hess_vecs_ = num_preconds_ = num_augments_ = 0;
    num_jacobian_assembles_ = num_precond_setops_ = 0;
    num_subspace_prepares_ = num_subspace_solves_ = 0;
    num_batch_hess_vec_calls_ = num_batch_hess_vec_actions_ = 0;
    cg_profile_ = CGProfile{};
    cg_iter_hist_.fill(0);
    cg_status_hist_.fill(0);
    cg_outer_count_ = cg_iter_sum_ = cg_iter_max_ = 0;
    cg_hit_max_iters_count_ = 0;
    line_search_retries_total_ = line_search_retries_max_ = 0;
    if (auto* defl = dynamic_cast<DeflationPreconditioner*>(&tr_precond)) defl->resetTimers();
    double solve_t0 = MPI_Wtime();
    // === TIMING END ===

    int subspace_option = nonlinear_options.subspace_option;
    int num_leftmost = nonlinear_options.num_leftmost;
    grad_is_current_ = false;
    previous_steps.clear();

    scratch = 1.0;
    double tr_size = nonlinear_options.trust_region_scaling * std::sqrt(Dot(scratch, scratch));
    size_t cumulative_cg_iters_from_last_precond_update = 0;

    // Eisenstat–Walker state. prev_norm starts at the initial residual norm;
    // prev_eta seeds at eta_max so the first outer iter uses the cap directly.
    real_t ew_prev_norm = norm;
    real_t ew_prev_eta = linear_options.cg_ew_eta_max;

    // Adaptive CG cap state: full budget from the configured max, residual progress
    // anchored at the running max norm, and the consecutive radius-shrink streak.
    const size_t cap_max_full = settings.max_cg_iterations;
    double max_norm_seen = norm;
    int consecutive_tr_shrinks = 0;

    // Adaptive pieces: (re)arm per solve; ladder state (current_s/locked) persists across
    // load steps. Enabled by deflation_pieces == 0.
    adapt_pieces_.enabled = (linear_options.deflation_pieces == 0);
    adapt_pieces_.window_outers = 0;
    window_cg_iters_ = 0;
    adapt_pieces_.window_t0 = MPI_Wtime();
    if (adapt_pieces_.probing && deflation_precond_) {
      // A probe interrupted by the load-step boundary is unverified: roll it back (the
      // ladder may retry within the new step). The it==0 refresh rebuilds the basis.
      deflation_precond_->setNumPieces(adapt_pieces_.previous_s);
      adapt_pieces_.current_s = adapt_pieces_.previous_s;
      adapt_pieces_.probing = false;
    }

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

      if (status.converged && it >= nonlinear_options.min_iterations) {
        converged = true;
        break;
      } else if (it >= max_iter) {
        converged = false;
        break;
      }

      // Skip re-assembly when every linesearch retry of the previous outer iteration was
      // rejected: X is unchanged, so the Jacobian (and BSR copy) would be bit-identical.
      if (!grad_is_current_) {
        assembleJacobian(X);
        grad_is_current_ = true;
      }

      if (it == 0 || force_precond_refresh_ ||
          (trResults.cg_iterations_count >= settings.max_cg_iterations ||
           cumulative_cg_iters_from_last_precond_update >= settings.max_cumulative_iteration)) {
        force_precond_refresh_ = false;
        ++num_precond_setops_;
        double tps0 = MPI_Wtime();
        tr_precond.SetOperator(*grad);
        precond_setop_time_ += MPI_Wtime() - tps0;
        cumulative_cg_iters_from_last_precond_update = 0;
        detectDeflationPrecond();
        refreshDeflationLeftmost();
        // Lanczos enrichment: compute approximate leftmost eigvecs of the assembled Hessian
        // directly. Independent of the deflation basis; surfaces the true negative-curvature
        // mode that per-rank polynomial W projections often miss in snap-through problems.
        if (nonlinear_options.trust_num_lanczos > 0) {
          const int n_evecs = nonlinear_options.trust_num_lanczos;
          const int n_iter =
              nonlinear_options.trust_num_lanczos_iters > 0 ? nonlinear_options.trust_num_lanczos_iters : 3 * n_evecs;
          auto dots_lambda = [this](const std::vector<DotPair>& pairs) { return dot_many(pairs); };
          double t_lz = MPI_Wtime();
          lanczosLeftmostEigvecs(*grad, n_iter, n_evecs, r, lanczos_evecs_, lanczos_evals_, dots_lambda);
          lanczos_time_ += MPI_Wtime() - t_lz;
          ++num_lanczos_calls_;
          if (print_level >= 2 && !lanczos_evals_.empty()) {
            mfem::out << "  [lanczos] " << lanczos_evals_.size() << " evals:";
            for (double e : lanczos_evals_) mfem::out << " " << e;
            mfem::out << "\n";
          }
        }
      }

      auto hess_vec_func = [&](const mfem::Vector& x_, mfem::Vector& v_) { hessVec(x_, v_); };

      double cauchyPointNormSquared = tr_size * tr_size;
      trResults.reset();

      {
        hess_vec_func(r, trResults.H_d);
        const double gKg = Dot(r, trResults.H_d);
        const double residual_norm_squared = norm * norm;
        if (gKg > 0) {
          const double alphaCp = -residual_norm_squared / gKg;
          add(trResults.cauchy_point, alphaCp, r, trResults.cauchy_point);
          cauchyPointNormSquared = Dot(trResults.cauchy_point, trResults.cauchy_point);
        } else {
          const double alphaTr = -tr_size / norm;
          add(trResults.cauchy_point, alphaTr, r, trResults.cauchy_point);
          if (print_level >= 2) {
            mfem::out << "Negative curvature un-preconditioned cauchy point direction found."
                      << "\n";
          }
        }
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
        if (linear_options.cg_eisenstat_walker) {
          // Eisenstat–Walker choice 2 with the standard safeguard.
          const double gamma = linear_options.cg_ew_gamma;
          const double alpha = linear_options.cg_ew_alpha;
          const double eta_max = linear_options.cg_ew_eta_max;
          double eta_k = eta_max;
          if (it > 0 && ew_prev_norm > 0.0) {
            eta_k = gamma * std::pow(norm / ew_prev_norm, alpha);
            const double safeguard = gamma * std::pow(ew_prev_eta, alpha);
            if (safeguard > 0.1) {
              eta_k = std::max(eta_k, safeguard);
            }
            eta_k = std::min(eta_k, eta_max);
          }
          // Absolute floor: don't solve past the outer convergence goal.
          settings.cg_tol = std::max(eta_k * norm, 0.5 * norm_goal);
          ew_prev_norm = norm;
          ew_prev_eta = eta_k;
        } else {
          settings.cg_tol = std::max(0.5 * norm_goal, nonlinear_options.cg_forcing_rel * norm);
        }
        max_norm_seen = std::max(max_norm_seen, norm);
        if (nonlinear_options.cg_cap_min > 0) {
          // Adaptive CG budget (see NonlinearSolverOptions::cg_cap_min): geometric ramp from
          // cap_min to the full max over the residual's log-progress toward the goal, cut by
          // gamma per consecutive radius-shrinking step.
          const double cap_min = std::min(static_cast<double>(nonlinear_options.cg_cap_min),
                                          static_cast<double>(cap_max_full));
          double frac = 1.0;
          if (norm > norm_goal && max_norm_seen > norm_goal) {
            frac = std::clamp(std::log(max_norm_seen / norm) / std::log(max_norm_seen / norm_goal), 0.0, 1.0);
          }
          const double cap = cap_min * std::pow(static_cast<double>(cap_max_full) / cap_min, frac) *
                             std::pow(nonlinear_options.cg_cap_gamma, consecutive_tr_shrinks);
          settings.max_cg_iterations =
              static_cast<size_t>(std::clamp(cap, cap_min, static_cast<double>(cap_max_full)));
        }
        // Push Lanczos eigvec approximations of the assembled Hessian's leftmost spectrum
        // into left_mosts as subspace candidates. Applies regardless of W choice.
        for (const auto& v : lanczos_evecs_) {
          if (v.Size() != r.Size()) continue;
          const int k = std::max(1, nonlinear_options.num_leftmost + nonlinear_options.trust_num_lanczos);
          while (left_mosts.size() >= static_cast<size_t>(k)) {
            left_mosts.erase(left_mosts.begin());
          }
          left_mosts.emplace_back(std::make_shared<mfem::Vector>(v));
        }
        // Indefinite-coarse path: push the WtAW lowest mode into left_mosts as a candidate
        // negative-curvature subspace basis vector. Mult itself applies smoother only (PD)
        // so CG runs normally and contributes a Newton-quality direction.
        if (deflation_precond_ && !deflation_precond_->coarseIsSPD() && deflation_leftmost_w_.Size() == r.Size()) {
          const int k = std::max(1, nonlinear_options.num_leftmost);
          while (left_mosts.size() >= static_cast<size_t>(k)) {
            left_mosts.erase(left_mosts.begin());
          }
          left_mosts.emplace_back(std::make_shared<mfem::Vector>(deflation_leftmost_w_));
          if (print_level >= 2) {
            // Diagnostic: verify curvature and descent sign of the leftmost direction.
            const mfem::Vector& w = deflation_leftmost_w_;
            mfem::Vector Aw(w.Size());
            hess_vec_func(w, Aw);
            const double wAw = Dot(w, Aw);
            const double gw = Dot(r, w);
            const double ww = Dot(w, w);
            mfem::out << "  [indef-WtAW] lambda_min=" << deflation_leftmost_lambda_ << " w·Aw=" << wAw << " w·w=" << ww
                      << " curvature=w·Aw/w·w=" << (ww > 0 ? wAw / ww : 0.0) << " g·w=" << gw
                      << " (need <0 for descent)\n";
          }
        }
        solveModelProblem(r, scratch, *grad, &this->tr_precond, settings, tr_size, trResults, norm * norm);
        if (print_level >= 2 && deflation_precond_ && !deflation_precond_->coarseIsSPD()) {
          mfem::out << "  [post-CG] cg_iter=" << trResults.cg_iterations_count
                    << " status=" << static_cast<int>(trResults.interior_status)
                    << " hit_max=" << trResults.cg_hit_max_iters << "\n";
        }
      }
      cumulative_cg_iters_from_last_precond_update += trResults.cg_iterations_count;

      // Adaptive-cap diagnostic (SMITH_TR_CAP_DIAG=1): one line per outer with the state the
      // cap schedule would consume. norm/tr_size are unmodified between the model solve and
      // the line search, so capturing here reflects what CG saw.
      static const bool cap_diag_enabled = [] {
        const char* v = std::getenv("SMITH_TR_CAP_DIAG");
        return v && v[0] != '\0' && v[0] != '0';
      }();
      const double diag_norm_pre = norm;
      const double diag_tr_pre = tr_size;
      const size_t diag_cg_iters = trResults.cg_iterations_count;
      double diag_rho = std::numeric_limits<double>::quiet_NaN();

      // === TIMING BEGIN === — per-outer-iter CG-count histogram + status tally
      {
        const size_t ci = trResults.cg_iterations_count;
        const size_t bin = (ci < 10) ? 0 : (ci < 50) ? 1 : (ci < 200) ? 2 : (ci < 1000) ? 3 : 4;
        ++cg_iter_hist_[bin];
        ++cg_outer_count_;
        cg_iter_sum_ += ci;
        if (ci > cg_iter_max_) cg_iter_max_ = ci;
        switch (trResults.interior_status) {
          case TrustRegionResults::Status::Interior:
            ++cg_status_hist_[0];
            break;
          case TrustRegionResults::Status::NegativeCurvature:
            ++cg_status_hist_[1];
            break;
          case TrustRegionResults::Status::OnBoundary:
            ++cg_status_hist_[2];
            break;
          case TrustRegionResults::Status::NonDescentDirection:
            ++cg_status_hist_[3];
            break;
        }
        if (trResults.cg_hit_max_iters) ++cg_hit_max_iters_count_;
      }
      // === TIMING END ===

      bool have_computed_Hvs = false;
      bool have_prepared_subspace = false;
      TrustRegionSubspaceCache subspace_cache;
      std::vector<mfem::Vector> H_previous_steps;
#ifdef MFEM_USE_LAPACK
      constexpr bool can_use_subspace_solver = true;
#else
      constexpr bool can_use_subspace_solver = false;
#endif

      int lineSearchIter = 0;
      while (lineSearchIter <= nonlinear_options.max_line_search_iterations) {
        ++lineSearchIter;

        doglegStep(trResults.cauchy_point, trResults.z, tr_size, trResults.d);
        const double d_norm = subspace_option >= 1 ? std::sqrt(Dot(trResults.d, trResults.d)) : 0.0;
        const bool use_subspace =
            can_use_subspace_solver &&
            shouldUseSubspaceStep(subspace_option, trResults.interior_status, d_norm, tr_size, lineSearchIter,
                                  trResults.cg_hit_max_iters, trResults.cg_model_stagnated);

        bool subspace_unavailable = false;
        if (use_subspace) {
          if (!have_computed_Hvs) {
            have_computed_Hvs = true;
            std::vector<const mfem::Vector*> subspace_hess_inputs{&trResults.z, &trResults.cauchy_point};
            std::vector<mfem::Vector*> subspace_hess_outputs{&trResults.H_z, &trResults.H_cauchy_point};

            H_previous_steps.resize(previous_steps.size());
            for (size_t i = 0; i < previous_steps.size(); ++i) {
              H_previous_steps[i].SetSize(previous_steps[i].Size());
              subspace_hess_inputs.push_back(&previous_steps[i]);
              subspace_hess_outputs.push_back(&H_previous_steps[i]);
            }

            H_left_mosts.clear();
            for (auto& left : left_mosts) {
              H_left_mosts.emplace_back(std::make_shared<mfem::Vector>(*left));
              subspace_hess_inputs.push_back(left.get());
              subspace_hess_outputs.push_back(H_left_mosts.back().get());
            }

            computeHessianActions(subspace_hess_inputs, subspace_hess_outputs, hess_vec_func);
          }

          if (!have_prepared_subspace) {
            have_prepared_subspace = true;

            std::vector<const mfem::Vector*> ds{&trResults.z, &trResults.cauchy_point};
            std::vector<const mfem::Vector*> H_ds{&trResults.H_z, &trResults.H_cauchy_point};
            for (size_t i = 0; i < previous_steps.size(); ++i) {
              ds.push_back(&previous_steps[i]);
              H_ds.push_back(&H_previous_steps[i]);
            }

            have_prepared_subspace = prepareSubspaceProblemCache(ds, H_ds, r, num_leftmost, subspace_cache);
            subspace_unavailable = !have_prepared_subspace;
          }

          if (have_prepared_subspace) {
            const SubspaceStepStatus subspace_status =
                trySubspaceStep(trResults.d, hess_vec_func, subspace_cache, r, tr_size);
            subspace_unavailable = subspace_status == SubspaceStepStatus::Unavailable;
          }
        }

        if (subspace_unavailable || !isDescentStep(trResults.d, r)) {
          fallbackToCauchyPoint(
              trResults, subspace_unavailable ? "Subspace step unavailable" : "Fallback step is not a descent step");
        }

        static constexpr double roundOffTol = 0.0;  // 1e-14;

        hess_vec_func(trResults.d, trResults.H_d);
        const auto dots = dot_many({{&trResults.d, &trResults.H_d}, {&r, &trResults.d}});
        const double dHd = dots[0];
        const double rd = dots[1];
        double modelObjective = rd + 0.5 * dHd - roundOffTol;

        add(X, trResults.d, x_pred);

        double realObjective = std::numeric_limits<double>::max();
        double normPred = std::numeric_limits<double>::max();
        ConvergenceStatus predicted_status;
        try {
          predicted_status = evaluateConvergence(x_pred, r_pred);
          normPred = predicted_status.global_norm;
          // Real work along the step. Preferred path: exact ΔE via user-supplied energy
          // callback (`setEnergyFunction`). Fallback: integrated `∫₀¹ r·d dτ` via 2/3/5-point
          // quadrature (trapezoid/Simpson/Boole). The exact-energy path guarantees energy
          // descent on accepted steps — no limit cycles possible for potential systems.
          const double rd_a = Dot(r, trResults.d);
          const double rd_b = Dot(r_pred, trResults.d);
          double obj1 = 0.5 * (rd_a + rd_b) - roundOffTol;  // trapezoid: r·d at endpoints
          double real_work = obj1;
          if (energy_function_) {
            const double E0 = energy_function_(X);
            const double E1 = energy_function_(x_pred);
            real_work = E1 - E0;
            // Cauchy-preference guard: the subspace step must beat the Cauchy point in
            // exact energy decrease. Otherwise replace d with the Cauchy step and re-evaluate.
            // This makes limit cycles impossible: the algorithm is at least as good as
            // gradient-descent (which provably converges for bounded-below potential systems).
            mfem::Vector x_c(X.Size());
            add(X, trResults.cauchy_point, x_c);
            const double E_cauchy = energy_function_(x_c);
            const double dE_cauchy = E_cauchy - E0;
            if (dE_cauchy < real_work) {
              if (print_level >= 2) {
                mfem::out << "  [cauchy-preference] subspace ΔE=" << real_work << " > cauchy ΔE=" << dE_cauchy
                          << "; using cauchy.\n";
              }
              trResults.d = trResults.cauchy_point;
              x_pred = x_c;
              // Refresh predicted residual and norms to match the new step.
              predicted_status = evaluateConvergence(x_pred, r_pred);
              normPred = predicted_status.global_norm;
              real_work = dE_cauchy;
              // Recompute model objective at the new d.
              hess_vec_func(trResults.d, trResults.H_d);
              const auto md = dot_many({{&trResults.d, &trResults.H_d}, {&r, &trResults.d}});
              modelObjective = md[1] + 0.5 * md[0] - roundOffTol;
            }
          } else {
            const int qpts = nonlinear_options.trust_work_quadrature_points;
            if (qpts == 3 || qpts == 5) {
              // Simpson: 1 extra eval at midpoint, weights (1,4,1)/6.
              mfem::Vector x_q(X.Size()), r_q(r.Size());
              add(X, 0.5, trResults.d, x_q);
              oper->Mult(x_q, r_q);
              const double rd_mid = Dot(r_q, trResults.d);
              real_work = (1.0 / 6.0) * (rd_a + 4.0 * rd_mid + rd_b) - roundOffTol;
              if (qpts == 5) {
                // Boole's 5-point closed Newton-Cotes: nodes 0, 1/4, 1/2, 3/4, 1;
                // weights (7,32,12,32,7)/90. Need r at 1/4 and 3/4 (already have 0, 1/2, 1).
                add(X, 0.25, trResults.d, x_q);
                oper->Mult(x_q, r_q);
                const double rd_q1 = Dot(r_q, trResults.d);
                add(X, 0.75, trResults.d, x_q);
                oper->Mult(x_q, r_q);
                const double rd_q3 = Dot(r_q, trResults.d);
                real_work = (1.0 / 90.0) * (7.0 * rd_a + 32.0 * rd_q1 + 12.0 * rd_mid + 32.0 * rd_q3 + 7.0 * rd_b) -
                            roundOffTol;
              }
              if (print_level >= 2 && std::abs(real_work - obj1) > 0.5 * std::max(std::abs(obj1), 1e-12)) {
                mfem::out << "  [work quadrature: trapezoid=" << obj1 << "  qpt-" << qpts << "=" << real_work << "]\n";
              }
            }
          }
          realObjective = real_work;
          if (predicted_status.converged) {
            acceptStep(trResults, subspace_cache, x_pred, r_pred, predicted_status, X, r, status, norm);
            if (print_level >= 2) {
              printTrustRegionInfo(realObjective, modelObjective, trResults.cg_iterations_count, tr_size, true);
              trResults.cg_iterations_count = 0;
            }
            break;
          }
        } catch (const std::exception&) {
          realObjective = std::numeric_limits<double>::max();
          normPred = std::numeric_limits<double>::max();
        }

        // accept immediately if converged — no need to check model quality (rho)
        if (normPred <= norm_goal) {
          acceptStep(trResults, subspace_cache, x_pred, r_pred, predicted_status, X, r, status, norm);
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
        diag_rho = rho;
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
        // Residual-norm safeguard against the trapezoid/Simpson blind spot. Skipped when the
        // exact-energy callback is in use, because ΔE is then sign-accurate by construction
        // and the safeguard would over-reject legitimate energy-descent steps that grow ‖r‖.
        const double residual_growth_cap = nonlinear_options.residual_growth_cap;
        const bool residual_safe = energy_function_ || (normPred <= residual_growth_cap * norm);
        const bool willAccept = rho >= settings.eta1 && rho <= settings.eta4 && residual_safe;
        if (!residual_safe) tr_size *= settings.t1;

        if (print_level >= 2) {
          printTrustRegionInfo(realObjective, modelObjective, trResults.cg_iterations_count, tr_size, willAccept);
          trResults.cg_iterations_count =
              0;  // zero this output so it doesn't look like the linesearch is doing cg iterations
        }

        if (willAccept) {
          acceptStep(trResults, subspace_cache, x_pred, r_pred, predicted_status, X, r, status, norm);
          break;
        }
      }
      // === TIMING BEGIN ===
      {
        const size_t ls = static_cast<size_t>(lineSearchIter);
        line_search_retries_total_ += ls;
        if (ls > line_search_retries_max_) line_search_retries_max_ = ls;
      }
      // === TIMING END ===

      if (tr_size < diag_tr_pre) {
        ++consecutive_tr_shrinks;
      } else {
        consecutive_tr_shrinks = 0;
      }

      // Adaptive pieces ladder: every `window` outers, decide using the allreduced window
      // wall (identical on every rank — decisions must not diverge) and the replicated
      // iteration count. Baseline window: attempt a doubling only if the solve looks
      // iteration-bound and the suggested cap allows. Probe window: keep the doubling on a
      // clear improvement, otherwise revert and lock.
      if (adapt_pieces_.enabled && !adapt_pieces_.locked && deflation_precond_) {
        ++adapt_pieces_.window_outers;
        window_cg_iters_ += diag_cg_iters;
        if (adapt_pieces_.window_outers >= AdaptivePiecesState::window) {
          double window_wall = MPI_Wtime() - adapt_pieces_.window_t0;
          MPI_Comm comm = GetComm() != MPI_COMM_NULL ? GetComm() : MPI_COMM_WORLD;
          MPI_Allreduce(MPI_IN_PLACE, &window_wall, 1, MPI_DOUBLE, MPI_MAX, comm);
          const double metric = window_wall / static_cast<double>(adapt_pieces_.window_outers);
          const double iters_per_outer =
              static_cast<double>(window_cg_iters_) / static_cast<double>(adapt_pieces_.window_outers);
          auto announce = [&](const char* what, int s_from, int s_to) {
            if (print_level >= 1) {
              mfem::out << "[adaptive-pieces] " << what << ": s " << s_from << " -> " << s_to
                        << " (wall/outer=" << metric << " s, iters/outer=" << iters_per_outer << ")\n";
            }
          };
          if (adapt_pieces_.probing) {
            if (metric < AdaptivePiecesState::improve_margin * adapt_pieces_.previous_metric) {
              announce("keep", adapt_pieces_.previous_s, adapt_pieces_.current_s);
              adapt_pieces_.previous_s = adapt_pieces_.current_s;
              adapt_pieces_.previous_metric = metric;
              adapt_pieces_.probing = false;  // next window may attempt a further doubling
            } else {
              announce("revert+lock", adapt_pieces_.current_s, adapt_pieces_.previous_s);
              deflation_precond_->setNumPieces(adapt_pieces_.previous_s);
              adapt_pieces_.current_s = adapt_pieces_.previous_s;
              adapt_pieces_.probing = false;
              adapt_pieces_.locked = true;
              force_precond_refresh_ = true;
            }
          } else if (iters_per_outer >= AdaptivePiecesState::attempt_gate &&
                     2 * adapt_pieces_.current_s <= deflation_precond_->suggestedMaxPieces()) {
            announce("probe", adapt_pieces_.current_s, 2 * adapt_pieces_.current_s);
            adapt_pieces_.previous_metric = metric;
            adapt_pieces_.previous_s = adapt_pieces_.current_s;
            adapt_pieces_.current_s *= 2;
            deflation_precond_->setNumPieces(adapt_pieces_.current_s);
            adapt_pieces_.probing = true;
            force_precond_refresh_ = true;
          } else {
            adapt_pieces_.previous_metric = metric;  // keep the baseline fresh
          }
          adapt_pieces_.window_outers = 0;
          window_cg_iters_ = 0;
          adapt_pieces_.window_t0 = MPI_Wtime();
        }
      }

      if (cap_diag_enabled) {
        // The line-search loop only exits via accept-break or retry exhaustion.
        const bool diag_accepted = lineSearchIter <= nonlinear_options.max_line_search_iterations;
        mfem::out << "[capdiag] it=" << it << " norm=" << diag_norm_pre << " norm_goal=" << norm_goal
                  << " tr_pre=" << diag_tr_pre << " tr_post=" << tr_size << " cg=" << diag_cg_iters
                  << " hit_max=" << trResults.cg_hit_max_iters << " stag=" << trResults.cg_model_stagnated
                  << " status=" << static_cast<int>(trResults.interior_status) << " rho=" << diag_rho
                  << " accepted=" << diag_accepted << " ls=" << lineSearchIter << "\n";
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

    // === TIMING BEGIN ===
    double solve_total = MPI_Wtime() - solve_t0;
    MPI_Comm comm = MPI_COMM_WORLD;
#ifdef MFEM_USE_MPI
    if (GetComm() != MPI_COMM_NULL) comm = GetComm();
#endif
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    auto rmax = [comm](double v) {
      double out = 0.0;
      MPI_Reduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      return out;
    };
    double t_hv = rmax(hess_vec_time_);
    double t_pc = rmax(precond_time_);
    double t_ag = rmax(augment_time_);
    double t_aj = rmax(assemble_jacobian_time_);
    double t_ag_grad = rmax(assemble_gradient_time_);
    double t_bsr = rmax(bsr_convert_time_);
    double t_asm_kernels = rmax(gradient_assemble_timers.kernels);
    double t_asm_scatter = rmax(gradient_assemble_timers.scatter);
    double t_asm_rap = rmax(gradient_assemble_timers.rap);
    double t_ps = rmax(precond_setop_time_);
    double t_sp = rmax(subspace_prepare_time_);
    double t_ss = rmax(subspace_solve_time_);
    double t_bh = rmax(batch_hess_vec_time_);
    double t_total = rmax(solve_total);
    double t_cg_H = rmax(cg_profile_.H_mult_time);
    double t_cg_P = rmax(cg_profile_.P_mult_time);
    double t_cg_dots = rmax(cg_profile_.dots_time);
    if (rank == 0) {
      mfem::out << "\n========= TrustRegion solve timing (max across ranks) =========\n"
                << "  total solve            : " << t_total << " s\n"
                << "  assembleJacobian       : " << t_aj << " s  (" << num_jacobian_assembles_ << " calls)\n"
                << "    GetGradient          : " << t_ag_grad << " s\n"
                << "      element kernels    : " << t_asm_kernels << " s\n"
                << "      scatter (SearchRow): " << t_asm_scatter << " s\n"
                << "      hypre + RAP        : " << t_asm_rap << " s\n"
                << "    CSR->BSR convert     : " << t_bsr << " s\n"
                << "  precond SetOperator    : " << t_ps << " s  (" << num_precond_setops_ << " calls)\n"
                << "  hess_vec total         : " << t_hv << " s  (" << num_hess_vecs_ << " calls)\n"
                << "  precond Mult total     : " << t_pc << " s  (" << num_preconds_ << " calls)\n"
                << "  batched hess_vec       : " << t_bh << " s  (" << num_batch_hess_vec_calls_ << " calls; "
                << num_batch_hess_vec_actions_ << " H-actions)\n"
                << "  subspace prepare       : " << t_sp << " s  (" << num_subspace_prepares_ << " calls)\n"
                << "  subspace solve         : " << t_ss << " s  (" << num_subspace_solves_ << " calls)\n"
                << "  augment-direction      : " << t_ag << " s  (" << num_augments_ << " calls)\n"
                << "  --- in-CG breakdown ---\n"
                << "  in-CG H.Mult           : " << t_cg_H << " s  (" << cg_profile_.H_mult_count << " calls)\n"
                << "  in-CG P.Mult           : " << t_cg_P << " s  (" << cg_profile_.P_mult_count << " calls)\n"
                << "  in-CG dot_many (+Allred): " << t_cg_dots << " s  (" << cg_profile_.dot_call_count << " calls)\n"
                << "  --- CG iter histogram (per outer; " << cg_outer_count_ << " outers) ---\n"
                << "  [0,10)   : " << cg_iter_hist_[0] << "\n"
                << "  [10,50)  : " << cg_iter_hist_[1] << "\n"
                << "  [50,200) : " << cg_iter_hist_[2] << "\n"
                << "  [200,1k) : " << cg_iter_hist_[3] << "\n"
                << "  [1k,+)   : " << cg_iter_hist_[4] << "\n"
                << "  mean = " << (cg_outer_count_ ? cg_iter_sum_ / cg_outer_count_ : 0) << "  max = " << cg_iter_max_
                << "  total = " << cg_iter_sum_ << "\n"
                << "  --- CG exit status ---\n"
                << "  Interior   : " << cg_status_hist_[0] << "\n"
                << "  NegCurv    : " << cg_status_hist_[1] << "\n"
                << "  OnBoundary : " << cg_status_hist_[2] << "\n"
                << "  NonDescent : " << cg_status_hist_[3] << "\n"
                << "  cg_hit_max : " << cg_hit_max_iters_count_ << " (subset of Interior; routed to subspace)\n"
                << "  --- line-search retries ---\n"
                << "  total = " << line_search_retries_total_
                << "  mean = " << (cg_outer_count_ ? line_search_retries_total_ / cg_outer_count_ : 0)
                << "  max = " << line_search_retries_max_ << "\n"
                << "===============================================================\n\n";
    }
    if (auto* defl = dynamic_cast<DeflationPreconditioner*>(&tr_precond)) defl->printTimingSummary(comm);
    // === TIMING END ===
  }
};
/// @endcond

EquationSolver::EquationSolver(NonlinearSolverOptions nonlinear_opts, LinearSolverOptions lin_opts, MPI_Comm comm)
{
  auto [lin_solver, preconditioner] = buildLinearSolverAndPreconditioner(lin_opts, comm);

  lin_solver_ = std::move(lin_solver);
  preconditioner_ = std::move(preconditioner);
  nonlin_solver_ = buildNonlinearSolver(nonlinear_opts, lin_opts, *preconditioner_, comm);
  convergence_manager_ = std::make_shared<EquationSolverConvergenceManager>(comm, nonlinear_opts.absolute_tol,
                                                                            nonlinear_opts.relative_tol);
  attachConvergenceManager();
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

void EquationSolver::attachDeflationFES(mfem::ParFiniteElementSpace& fes)
{
  if (auto* managed = dynamic_cast<ConvergenceManagedNonlinearSolver*>(nonlin_solver_.get())) {
    managed->setBSRBlockSize(fes.GetVDim());
  }
  if (!preconditioner_) return;
  auto* defl = dynamic_cast<DeflationPreconditioner*>(preconditioner_.get());
  if (defl && !defl->hasFES()) defl->attachFES(fes);
}

void EquationSolver::attachConvergenceManager() const
{
  if (!convergence_manager_ || !nonlin_solver_) {
    return;
  }

  if (auto* managed_solver = dynamic_cast<ConvergenceManagedNonlinearSolver*>(nonlin_solver_.get())) {
    managed_solver->setConvergenceManager(convergence_manager_);
  }
}

void EquationSolver::initializeConvergenceManager(double abs_tol, double rel_tol, MPI_Comm comm) const
{
  if (!convergence_manager_) {
    convergence_manager_ = std::make_shared<EquationSolverConvergenceManager>(comm, abs_tol, rel_tol);
  } else {
    convergence_manager_->setTolerances(abs_tol, rel_tol);
  }
  attachConvergenceManager();
}

void EquationSolver::setOperator(const mfem::Operator& op)
{
  attachConvergenceManager();
  nonlin_solver_->SetOperator(op);

  // Now that the nonlinear solver knows about the operator, we can set its linear solver
  if (!nonlin_solver_set_solver_called_) {
    nonlin_solver_->SetSolver(linearSolver());
    nonlin_solver_set_solver_called_ = true;
  }
}

void EquationSolver::setConvergenceTolerances(double abs_tol, double rel_tol, MPI_Comm comm) const
{
  initializeConvergenceManager(abs_tol, rel_tol, comm);
}

void EquationSolver::resetConvergenceState() const
{
  if (convergence_manager_) {
    convergence_manager_->reset();
  }
}

void EquationSolver::setEnergyFunction(EnergyFunction fn)
{
  // TrustRegion lives in the anonymous namespace of this TU; cast in-file.
  if (auto* tr = dynamic_cast<TrustRegion*>(nonlin_solver_.get())) {
    tr->setEnergyFunction(std::move(fn));
  }
}

void EquationSolver::solve(mfem::Vector& x) const
{
  resetConvergenceState();
  mfem::Vector zero(x);
  zero = 0.0;
  // KINSOL does not handle non-zero RHS, so we enforce that the RHS
  // of the nonlinear system is zero
  nonlin_solver_->Mult(zero, x);
}

void SuperLUSolver::Mult(const mfem::Vector& input, mfem::Vector& output) const
{
  SLIC_ERROR_ROOT_IF(!superlu_mat_, "Operator must be set prior to solving with SuperLU");

  // Use the underlying MFEM-based solver and SuperLU matrix type to solve the system
  superlu_solver_.Mult(input, output);
}

/**
 * @brief Build a monolithic HypreParMatrix from a BlockOperator.
 *
 * PERFORMANCE NOTE: This function creates a NEW monolithic matrix by copying data from
 * the block structure. This incurs a performance overhead:
 * - Memory: Allocates new matrix storage
 * - Time: Copies all block data into monolithic format
 *
 * This is necessary when using direct solvers (SuperLU, Strumpack) that require
 * monolithic matrices. For iterative solvers, the BlockOperator can be used directly
 * without this copy overhead.
 *
 * @param block_operator The block operator to convert.
 * @return Unique pointer to the new monolithic HypreParMatrix.
 */
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

  // Note that MFEM passes ownership of this matrix to the caller.
  // MFEM creates a new monolithic matrix (not a view), so this is a COPY operation.
  return std::unique_ptr<mfem::HypreParMatrix>(mfem::HypreParMatrixFromBlocks(hypre_blocks));
}

void SuperLUSolver::SetOperator(const mfem::Operator& op)
{
  // Check if this is a block operator
  auto* block_operator = dynamic_cast<const mfem::BlockOperator*>(&op);

  // If it is, make a monolithic system from the underlying blocks
  if (block_operator) {
    monolithic_mat_ = buildMonolithicMatrix(*block_operator);

    superlu_mat_ = std::make_unique<mfem::SuperLURowLocMatrix>(*monolithic_mat_);
  } else {
    // If this is not a block system, check that the input operator is a HypreParMatrix as expected
    auto* matrix = dynamic_cast<const mfem::HypreParMatrix*>(&op);

    SLIC_ERROR_ROOT_IF(!matrix, "Matrix must be an assembled HypreParMatrix for use with SuperLU");

    superlu_mat_ = std::make_unique<mfem::SuperLURowLocMatrix>(*matrix);
  }
  height = op.Height();
  width = op.Width();
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
    monolithic_mat_ = buildMonolithicMatrix(*block_operator);

    strumpack_mat_ = std::make_unique<mfem::STRUMPACKRowLocMatrix>(*monolithic_mat_);
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
    nonlinear_solver = std::make_unique<NewtonSolver>(comm, nonlinear_opts, linear_opts);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::LBFGS) {
    nonlinear_opts.max_line_search_iterations = 0;
    SLIC_ERROR_ROOT_IF(nonlinear_opts.min_iterations != 0, "LBFGS does not support nonzero min_iterations");
    nonlinear_solver = std::make_unique<mfem::LBFGSSolver>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::NewtonLineSearch) {
    nonlinear_solver = std::make_unique<NewtonSolver>(comm, nonlinear_opts, linear_opts);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::TrustRegion) {
    nonlinear_solver = std::make_unique<TrustRegion>(comm, nonlinear_opts, linear_opts, prec);
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

bool requiresMonolithicOperator(const LinearSolverOptions& linear_opts)
{
  return !linearSolverSupportsBlockOperator(linear_opts.linear_solver) ||
         !preconditionerSupportsBlockOperator(linear_opts.preconditioner);
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
  } else if (preconditioner == Preconditioner::Deflation) {
    // Allow null deflation_fes here — framework callers (e.g. SolidMechanics) cannot
    // provide the FES at solver-construction time and bind it post-hoc via attachFES.
    std::unique_ptr<DeflationPreconditioner> defl;
    if (linear_opts.deflation_fes) {
      defl = std::make_unique<DeflationPreconditioner>(*linear_opts.deflation_fes);
    } else {
      defl = std::make_unique<DeflationPreconditioner>();
    }
    defl->setDeflationOrder(linear_opts.deflation_order);
    defl->setNumPieces(linear_opts.deflation_pieces);
    defl->setCoarseMode(linear_opts.deflation_coarse_mode);
    if (linear_opts.deflation_smoother == "block") {
      defl->setSmootherVariant(DeflationSmoother::BlockJacobi);
    } else if (linear_opts.deflation_smoother == "jacobi") {
      defl->setSmootherVariant(DeflationSmoother::PointJacobi);
    } else if (linear_opts.deflation_smoother == "hypre") {
      defl->setSmootherVariant(DeflationSmoother::Hypre);
    } else {
      SLIC_ERROR_ROOT("Unknown deflation_smoother value '" + linear_opts.deflation_smoother +
                      "' (expected hypre|jacobi|block)");
    }
    preconditioner_solver = std::move(defl);
  } else if (preconditioner == Preconditioner::AMGFContact) {
    auto amgfcontact_preconditioner = std::make_unique<mfem::AMGFSolver>();
    auto amgfcontact_opts = linear_opts.amgfcontact_options;
    amgfcontact_preconditioner->GetAMG().SetPrintLevel(preconditioner_print_level);
    amgfcontact_preconditioner->GetAMG().SetSystemsOptions(amgfcontact_opts.dim_systems_options);
    amgfcontact_preconditioner->GetAMG().SetRelaxType(amgfcontact_opts.relax_type);
    preconditioner_solver = std::move(amgfcontact_preconditioner);
  } else if (preconditioner == Preconditioner::BlockDiagonal || preconditioner == Preconditioner::BlockTriangular ||
             preconditioner == Preconditioner::BlockSchur) {
    std::vector<std::unique_ptr<mfem::Solver>> inner_solvers;
    for (const auto& opt : linear_opts.sub_block_linear_solver_options) {
      auto [lin, prec] = buildLinearSolverAndPreconditioner(opt, comm);
      inner_solvers.push_back(std::make_unique<SolverWithPreconditioner>(std::move(lin), std::move(prec)));
    }

    if (preconditioner == Preconditioner::BlockDiagonal) {
      preconditioner_solver = std::make_unique<BlockDiagonalPreconditioner>(std::move(inner_solvers));
    } else if (preconditioner == Preconditioner::BlockTriangular) {
      preconditioner_solver =
          std::make_unique<BlockTriangularPreconditioner>(std::move(inner_solvers), linear_opts.block_triangular_type);
    } else if (preconditioner == Preconditioner::BlockSchur) {
      preconditioner_solver = std::make_unique<BlockSchurPreconditioner>(
          std::move(inner_solvers), linear_opts.block_schur_type, linear_opts.schur_approx_type);
    }
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
      .addString("solver_type", "Solver type (Newton|NewtonLineSearch|TrustRegion|KINFullStep|KINLineSearch)")
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
