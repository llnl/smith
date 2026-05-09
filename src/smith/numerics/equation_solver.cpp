// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/steihaug_toint_cg.hpp"

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
#include "smith/infrastructure/logger.hpp"

namespace smith {

namespace {

#ifdef MFEM_USE_MPI
size_t rootOnlyPrintLevel(size_t level, MPI_Comm comm)
{
  if (level > 0 && comm != MPI_COMM_NULL) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0) {
      return 0;
    }
  }
  return level;
}
#else
size_t rootOnlyPrintLevel(size_t level) { return level; }
#endif

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
    print_level = rootOnlyPrintLevel(print_level
#ifdef MFEM_USE_MPI
                                     ,
                                     GetComm()
#endif
    );

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
class TrustRegion : public mfem::NewtonSolver, public SteihaugTointDelegate {
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
  mutable std::deque<std::shared_ptr<mfem::Vector>> accepted_step_history;
  /// initial state for this nonlinear solve, used as an optional history direction
  mutable mfem::Vector solve_start_x;
  mutable mfem::Vector min_residual_x;
  mutable double min_residual_norm = -1.0;
  
  /// Workspace vector for exact subspace solver to avoid small allocations
  mutable mfem::Vector exact_solver_workspace;

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
#ifdef MFEM_USE_MPI
  /// constructor
  TrustRegion(MPI_Comm comm_, const NonlinearSolverOptions& nonlinear_opts, const LinearSolverOptions& linear_opts,
              Solver& tPrec)
      : mfem::NewtonSolver(comm_), nonlinear_options(nonlinear_opts), linear_options(linear_opts), tr_precond(tPrec)
  {
  }
#endif

  template <typename... Args>
  std::array<double, sizeof...(Args) / 2> dot_many(const Args&... args) const
  {
    static_assert(sizeof...(Args) % 2 == 0, "dot_many requires an even number of arguments");
    constexpr size_t num_pairs = sizeof...(Args) / 2;
    std::array<double, num_pairs> products;
    products.fill(0.0);

    if (dot_oper) {
      auto tuple_args = std::tie(args...);
      auto do_dots = [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((products[I] = Dot(std::get<2 * I>(tuple_args), std::get<2 * I + 1>(tuple_args))), ...);
      };
      do_dots(std::make_index_sequence<num_pairs>{});
      return products;
    }

    auto tuple_args = std::tie(args...);
    std::array<int, num_pairs> sizes;
    std::array<const double*, num_pairs> ptr_a;
    std::array<const double*, num_pairs> ptr_b;
    
    auto populate_arrays = [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((
        sizes[I] = std::get<2 * I>(tuple_args).Size(),
        [&](){ MFEM_ASSERT(sizes[I] == std::get<2 * I + 1>(tuple_args).Size(), "Incompatible vector sizes."); }(),
        ptr_a[I] = std::get<2 * I>(tuple_args).GetData(),
        ptr_b[I] = std::get<2 * I + 1>(tuple_args).GetData()
      ), ...);
    };
    populate_arrays(std::make_index_sequence<num_pairs>{});

    bool all_same_size = true;
    for (size_t i = 1; i < num_pairs; ++i) {
      if (sizes[i] != sizes[0]) {
        all_same_size = false;
        break;
      }
    }

    if (all_same_size && num_pairs > 0) {
      for (int j = 0; j < sizes[0]; ++j) {
        for (size_t i = 0; i < num_pairs; ++i) {
          products[i] += ptr_a[i][j] * ptr_b[i][j];
        }
      }
    } else {
      for (size_t i = 0; i < num_pairs; ++i) {
        for (int j = 0; j < sizes[i]; ++j) {
          products[i] += ptr_a[i][j] * ptr_b[i][j];
        }
      }
    }

#ifdef MFEM_USE_MPI
    const MPI_Comm dot_comm = GetComm();
    if (dot_comm != MPI_COMM_NULL) {
      std::array<mfem::real_t, num_pairs> global_products;
      MPI_Allreduce(products.data(), global_products.data(), num_pairs, MFEM_MPI_REAL_T, MPI_SUM, dot_comm);
      for (size_t i = 0; i < num_pairs; ++i) {
        products[i] = global_products[i];
      }
    }
#endif

    return products;
  }

  template <typename HessVecFunc>
  void batchedSubspaceHessVec(HessVecFunc hess_vec_func, const std::vector<const mfem::Vector*>& inputs,
                              const std::vector<mfem::Vector*>& outputs) const
  {
    MFEM_VERIFY(inputs.size() == outputs.size(), "Subspace Hessian-vector batch input/output size mismatch");
    if (inputs.empty()) {
      return;
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
      hess_vec_func(*inputs[i], *outputs[i]);
    }
  }

  void pushAcceptedStepHistory(const mfem::Vector& step) const
  {
    if (nonlinear_options.trust_num_past_steps <= 0) {
      accepted_step_history.clear();
      return;
    }

    accepted_step_history.push_front(std::make_shared<mfem::Vector>(step));
    const size_t max_size = static_cast<size_t>(nonlinear_options.trust_num_past_steps) + 1;
    while (accepted_step_history.size() > max_size) {
      accepted_step_history.pop_back();
    }
  }

  std::array<double, 4> dot_many_4(const mfem::Vector& a0, const mfem::Vector& b0,
                                   const mfem::Vector& a1, const mfem::Vector& b1,
                                   const mfem::Vector& a2, const mfem::Vector& b2,
                                   const mfem::Vector& a3, const mfem::Vector& b3) const override
  {
    return dot_many(a0, b0, a1, b1, a2, b2, a3, b3);
  }

  std::array<double, 2> dot_many_2(const mfem::Vector& a0, const mfem::Vector& b0,
                                   const mfem::Vector& a1, const mfem::Vector& b1) const override
  {
    return dot_many(a0, b0, a1, b1);
  }

  void projectToBoundaryWithCoefs(mfem::Vector& z, const mfem::Vector& d, double delta, double zz, double zd,
                                  double dd) const override
  {
    double deltadelta_m_zz = delta * delta - zz;
    if (deltadelta_m_zz == 0) return;
    double tau = (std::sqrt(deltadelta_m_zz * dd + zd * zd) - zd) / dd;
    z.Add(tau, d);
  }

  /// solve the exact trust-region subspace problem with directions ds, and the leftmosts
  template <typename HessVecFunc>
  void solveTheSubspaceProblem([[maybe_unused]] mfem::Vector& z, [[maybe_unused]] const HessVecFunc& hess_vec_func,
                               [[maybe_unused]] const std::vector<const mfem::Vector*> ds,
                               [[maybe_unused]] const std::vector<const mfem::Vector*> Hds,
                               [[maybe_unused]] const mfem::Vector& g, [[maybe_unused]] double delta,
                               [[maybe_unused]] int num_leftmost,
                               [[maybe_unused]] std::vector<std::shared_ptr<mfem::Vector>>& candidate_left_mosts) const
  {
    SMITH_MARK_FUNCTION;
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
      if (exact_solver_workspace.Size() < 2000) {
        exact_solver_workspace.SetSize(2000);
      }
      std::tie(sol, leftvecs, leftvals, energy_change) =
          solveSubspaceProblem(directions, H_directions, b, delta, num_leftmost, exact_solver_workspace);
    } catch (const std::exception& e) {
      if (print_level >= 1) {
        mfem::out << "subspace solve failed with " << e.what() << std::endl;
      }
      return;
    }

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
  }

  /// finds tau s.t. (z + tau*(y-z))^2 = trSize^2
  void projectToBoundaryBetweenWithCoefs(mfem::Vector& z, const mfem::Vector& y, double trSize, double zz, double zy,
                                         double yy) const
  {
    double dd = yy - 2 * zy + zz;
    double zd = zy - zz;
    double tau = (std::sqrt((trSize * trSize - zz) * dd + zd * zd) - zd) / dd;
    z.Add(-tau, z);
    z.Add(tau, y);
  }

  /// take a dogleg step in direction s, solution norm must be within trSize
  void doglegStep(const mfem::Vector& cp, const mfem::Vector& newtonP, double trSize, mfem::Vector& s) const
  {
    SMITH_MARK_FUNCTION;
    auto [cc, nn] = dot_many(cp, cp, newtonP, newtonP);
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
  void solveModelProblem(const mfem::Vector& r0, mfem::Vector& rCurrent, const mfem::Operator& H,
                         const mfem::Solver* P, const TrustRegionSettings& settings, double& trSize,
                         TrustRegionResults& results, double r0_norm_squared) const
  {
    steihaugTointCG(r0, rCurrent, H, P, settings, trSize, results, r0_norm_squared, *this);
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
    grad = &oper->GetGradient(x);
    if (nonlinear_options.force_monolithic) {
      auto* grad_blocked = dynamic_cast<mfem::BlockOperator*>(grad);
      if (grad_blocked) grad = buildMonolithicMatrix(*grad_blocked).release();
    }
  }

  /// evaluate the nonlinear residual
  mfem::real_t computeResidual(const mfem::Vector& x_, mfem::Vector& r_) const
  {
    SMITH_MARK_FUNCTION;
    oper->Mult(x_, r_);
    return Norm(r_);
  }

  /// apply the action of the current Jacobian representation to a vector
  void hessVec(const mfem::Vector& x_, mfem::Vector& v_) const
  {
    SMITH_MARK_FUNCTION;
    grad->Mult(x_, v_);
  }

  /// apply trust region specific preconditioner
  void precond(const mfem::Vector& x_, mfem::Vector& v_) const
  {
    SMITH_MARK_FUNCTION;
    tr_precond.Mult(x_, v_);
  };

  /// @overload
  void Mult(const mfem::Vector&, mfem::Vector& X) const override
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");
    print_level = static_cast<size_t>(std::max(nonlinear_options.print_level, 0));
    print_level = print_options.iterations ? std::max<size_t>(1, print_level) : print_level;
    print_level = print_options.summary ? std::max<size_t>(2, print_level) : print_level;
    print_level = rootOnlyPrintLevel(print_level
#ifdef MFEM_USE_MPI
                                     ,
                                     GetComm()
#endif
    );

    using real_t = mfem::real_t;

    solve_start_x.SetSize(X.Size());
    solve_start_x = X;
    min_residual_x.SetSize(X.Size());
    min_residual_x = X;
    previous_H_left_mosts.clear();

    real_t norm, norm_goal = 0.0;
    norm = initial_norm = computeResidual(X, r);
    min_residual_norm = initial_norm;
    if (norm == 0.0) return;

    norm_goal = std::max(rel_tol * initial_norm, abs_tol);

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
    settings.cg_tol = 0.5 * norm_goal;

    int subspace_option = nonlinear_options.subspace_option;
    int num_leftmost = nonlinear_options.num_leftmost;

    scratch = 1.0;
    double tr_size = nonlinear_options.trust_region_scaling * std::sqrt(Dot(scratch, scratch));
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

      assembleJacobian(X);

      if (it == 0 || (trResults.cg_iterations_count >= settings.max_cg_iterations ||
                      cumulative_cg_iters_from_last_precond_update >= settings.max_cumulative_iteration)) {
        tr_precond.SetOperator(*grad);
        cumulative_cg_iters_from_last_precond_update = 0;
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
        settings.cg_tol = std::max(0.5 * norm_goal, 5e-5 * norm);
        solveModelProblem(r, scratch, *this->oper, &this->tr_precond, settings, tr_size, trResults,
                                     norm * norm);
      }
      cumulative_cg_iters_from_last_precond_update += trResults.cg_iterations_count;

      bool have_computed_Hvs = false;
      bool have_computed_H_left_mosts = false;
      std::vector<std::shared_ptr<mfem::Vector>> candidate_left_mosts;

      int lineSearchIter = 0;
      while (lineSearchIter <= nonlinear_options.max_line_search_iterations) {
        ++lineSearchIter;

        doglegStep(trResults.cauchy_point, trResults.z, tr_size, trResults.d);
        const bool check_subspace_boundary = subspace_option >= 1;
        const double d_norm = check_subspace_boundary ? std::sqrt(Dot(trResults.d, trResults.d)) : 0.0;
        bool use_with_option1 =
            (subspace_option >= 1) && (trResults.interior_status == TrustRegionResults::Status::NonDescentDirection ||
                                       trResults.interior_status == TrustRegionResults::Status::NegativeCurvature ||
                                       ((d_norm > (1.0 - 1.0e-6) * tr_size) && lineSearchIter > 1));
        bool use_with_option2 = (subspace_option >= 2) && (d_norm > (1.0 - 1.0e-6) * tr_size);
        bool use_with_option3 = (subspace_option >= 3);

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
            previous_H_left_mosts = H_left_mosts;
            H_left_mosts.clear();
            std::vector<const mfem::Vector*> leftmost_inputs;
            std::vector<mfem::Vector*> leftmost_outputs;
            for (auto& left : left_mosts) {
              H_left_mosts.emplace_back(std::make_shared<mfem::Vector>(*left));
              leftmost_inputs.push_back(left.get());
              leftmost_outputs.push_back(H_left_mosts.back().get());
            }
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
              batchedSubspaceHessVec(hess_vec_func, min_res_inputs, min_res_outputs);
              ds.push_back(&min_residual_direction);
              H_ds.push_back(&H_min_residual_direction);
            }
          }
          solveTheSubspaceProblem(trResults.d, hess_vec_func, ds, H_ds, r, tr_size, num_leftmost, candidate_left_mosts);
        }

        static constexpr double roundOffTol = 0.0;  // 1e-14;

        hess_vec_func(trResults.d, trResults.H_d);
        const auto [dHd, rd] = dot_many(trResults.d, trResults.H_d, r, trResults.d);
        double modelObjective = rd + 0.5 * dHd - roundOffTol;

        add(X, trResults.d, x_pred);

        double realObjective = std::numeric_limits<double>::max();
        double normPred = std::numeric_limits<double>::max();
        try {
          normPred = computeResidual(x_pred, r_pred);
          if (normPred < min_residual_norm) {
            min_residual_norm = normPred;
            min_residual_x = x_pred;
          }
          double obj1 = 0.5 * (rd + Dot(r_pred, trResults.d)) - roundOffTol;
          realObjective = obj1;
        } catch (const std::exception&) {
          realObjective = std::numeric_limits<double>::max();
          normPred = std::numeric_limits<double>::max();
        }

        if (normPred <= norm_goal) {
          trResults.d_old = trResults.d;
          trResults.H_d_old_at_accept = trResults.H_d;
          trResults.has_d_old = true;
          pushAcceptedStepHistory(trResults.d);
          if (!candidate_left_mosts.empty()) {
            left_mosts = std::move(candidate_left_mosts);
          }
          X = x_pred;
          r = r_pred;
          norm = normPred;
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
        const bool willAccept = rho >= settings.eta1 && rho <= settings.eta4;

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
          X = x_pred;
          r = r_pred;
          norm = normPred;
          break;
        }
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

void EquationSolver::solve(mfem::Vector& x) const
{
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
