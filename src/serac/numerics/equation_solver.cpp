// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/equation_solver.hpp"

#include <iomanip>
#include <sstream>
#include <ios>
#include <iostream>

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/profiling.hpp"

namespace serac {

class SolveException : public std::exception {
public:
    SolveException(const std::string& message) : msg(message) {}

    const char* what() const noexcept override {
        return msg.c_str();
    }

private:
    std::string msg;
};

/// Newton solver with a 2-way line-search.  Reverts to regular Newton if max_line_search_iterations is set to 0.
class NewtonSolver : public mfem::NewtonSolver {
protected:
  /// initial solution vector to do line-search off of
  mutable mfem::Vector x0;
  /// nonlinear solver options
  NonlinearSolverOptions nonlinear_options;

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
    SERAC_MARK_FUNCTION;
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
    SERAC_MARK_FUNCTION;
    grad = &oper->GetGradient(x);
    if (nonlinear_options.force_monolithic) {
      auto* grad_blocked = dynamic_cast<mfem::BlockOperator*>(grad);
      if (grad_blocked) grad = buildMonolithicMatrix(*grad_blocked).release();
    }
  }

  /// set the preconditioner for the linear solver
  void setPreconditioner() const
  {
    SERAC_MARK_FUNCTION;
    prec->SetOperator(*grad);
  }

  /// solve the linear system
  void solveLinearSystem(const mfem::Vector& r_, mfem::Vector& c_) const
  {
    SERAC_MARK_FUNCTION;
    prec->Mult(r_, c_);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
  }

  /// @overload
  void Mult(const mfem::Vector&, mfem::Vector& x) const
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

    using real_t = mfem::real_t;

    real_t norm, norm_goal;
    norm = initial_norm = evaluateNorm(x, r);

    if (print_options.first_and_last && !print_options.iterations) {
      mfem::out << "Newton iteration " << std::setw(3) << 0 << " : ||r|| = " << std::setw(13) << norm << "...\n";
    }

    norm_goal            = std::max(rel_tol * initial_norm, abs_tol);
    prec->iterative_mode = false;

    int it = 0;
    for (; true; it++) {
      MFEM_ASSERT(mfem::IsFinite(norm), "norm = " << norm);
      if (print_options.iterations) {
        mfem::out << "Newton iteration " << std::setw(3) << it << " : ||r|| = " << std::setw(13) << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << std::setw(13) << (initial_norm != 0.0 ? norm / initial_norm : norm);
        }
        mfem::out << '\n';
      }

      if (norm != norm) {
        if (print_options.first_and_last) {
          mfem::out << "Initial residual for Newton iteration is undefined/nan." << std::endl;
          mfem::out << "Newton: No convergence!\n";
        }
        throw SolveException("Initial residual for Newton iteration is undefined/nan.");
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

      const int               max_ls_iters = nonlinear_options.max_line_search_iterations;
      static constexpr real_t reduction    = 0.5;

      const double sufficientDecreaseParam = 0.0;  // 1e-15;
      const double cMagnitudeInR           = sufficientDecreaseParam != 0.0 ? std::abs(Dot(c, r)) / norm_nm1 : 0.0;

      auto is_improved = [=](real_t currentNorm, real_t c_scale) {
        return currentNorm < norm_nm1 - sufficientDecreaseParam * c_scale * cMagnitudeInR;
      };

      // back-track linesearch
      int ls_iter     = 0;
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
        if (print_options.iterations) {
          mfem::out << "Number of line search steps taken = " << ls_iter_sum << std::endl;
        }
        if (print_options.warnings && (ls_iter_sum == 2 * max_ls_iters + 1)) {
          mfem::out << "The maximum number of line search cut back have occurred, the resulting residual may not have "
                       "decreased. "
                    << std::endl;
        }
      }
    }

    final_iter = it;
    final_norm = norm;

    if (print_options.summary || (!converged && print_options.warnings) || print_options.first_and_last) {
      mfem::out << "Newton: Number of iterations: " << final_iter << '\n' << "   ||r|| = " << final_norm << '\n';
    }

    if (!converged && (print_options.summary || print_options.warnings)) {
      mfem::out << "Newton: No convergence!\n";
    }

    if (!converged) {
      throw SolveException("Newton algorithm failed to converge.");
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
};

/// Internal structure for storing trust region stateful data
struct TrustRegionResults {
  /// Constructor takes the size of the solution vector
  TrustRegionResults(int size)
  {
    z.SetSize(size);
    d.SetSize(size);
    Pr.SetSize(size);
    Hd.SetSize(size);
    cauchy_point.SetSize(size);
  }

  /// resets trust region results for a new outer iteration
  void reset()
  {
    z            = 0.0;
    cauchy_point = 0.0;
  }

  /// enumerates the possible final status of the trust region steps
  enum class Status
  {
    Interior,
    NegativeCurvature,
    OnBoundary,
  };

  /// step direction
  mfem::Vector z;
  /// incrementalCG direction
  mfem::Vector d;
  /// preconditioned residual
  mfem::Vector Pr;
  /// action of hessian on direction d
  mfem::Vector Hd;
  /// cauchy point
  mfem::Vector cauchy_point;
  /// specifies if step is interior, exterior, negative curvature, etc.
  Status interior_status = Status::Interior;
  /// iteration counter
  size_t cg_iterations_count = 0;
};

/// trust region printing utility function
void printTrustRegionInfo(double realObjective, double modelObjective, size_t cgIters, double trSize, bool willAccept)
{
  mfem::out << "real energy = " << std::setw(13) << realObjective << ", model energy = " << std::setw(13)
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
  /// predicted solution
  mutable mfem::Vector x_mid;
  /// predicted residual
  mutable mfem::Vector r_pred;
  /// mid residual
  mutable mfem::Vector r_mid;
  /// extra vector of scratch space for doing temporary calculations
  mutable mfem::Vector scratch;

  /// nonlinear solution options
  NonlinearSolverOptions nonlinear_options;
  /// linear solution options
  LinearSolverOptions linear_options;
  /// handle to the preconditioner used by the trust region, it ignores the linear solver as a SPD preconditioner is
  /// currently required
  Solver& tr_precond;

public:
#ifdef MFEM_USE_MPI
  /// constructor
  TrustRegion(MPI_Comm comm_, const NonlinearSolverOptions& nonlinear_opts, const LinearSolverOptions& linear_opts,
              Solver& tPrec)
      : mfem::NewtonSolver(comm_), nonlinear_options(nonlinear_opts), linear_options(linear_opts), tr_precond(tPrec)
  {
  }
#endif

  /// finds tau s.t. (z + tau*d)^2 = trSize^2
  void projectToBoundaryWithCoefs(mfem::Vector& z, const mfem::Vector& d, double trSize, double zz, double zd,
                                  double dd) const
  {
    double tau = (std::sqrt((trSize * trSize - zz) * dd + zd * zd) - zd) / dd;
    z.Add(tau, d);
  }

  /// finds tau s.t. (z + tau*(y-z))^2 = trSize^2
  void projectToBoundaryBetweenWithCoefs(mfem::Vector& z, const mfem::Vector& y, double trSize, double zz, double zy,
                                         double yy) const
  {
    double dd  = yy - 2 * zy + zz;
    double zd  = zy - zz;
    double tau = (std::sqrt((trSize * trSize - zz) * dd + zd * zd) - zd) / dd;
    z.Add(-tau, z);
    z.Add(tau, y);
  }

  /// take a dogleg step in direction s, solution norm must be within trSize
  void doglegStep(const mfem::Vector& cp, const mfem::Vector& newtonP, double trSize, mfem::Vector& s) const
  {
    SERAC_MARK_FUNCTION;
    // MRT, could optimize some of these eventually, compute on the outside and save
    double cc = Dot(cp, cp);
    double nn = Dot(newtonP, newtonP);
    double tt = trSize * trSize;

    s = 0.0;
    if (cc >= tt) {
      add(s, std::sqrt(tt / cc), cp, s);
    } else if (cc > nn) {
      if (print_options.warnings) {
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

  /// Minimize quadratic sub-problem given residual vector, the action of the stiffness and a preconditioner
  template <typename HessVecFunc, typename PrecondFunc>
  void solveTrustRegionModelProblem(const mfem::Vector& r0, mfem::Vector& rCurrent, HessVecFunc hess_vec_func,
                                    PrecondFunc precond, const TrustRegionSettings& settings, double& trSize,
                                    TrustRegionResults& results) const
  {
    SERAC_MARK_FUNCTION;
    // minimize r@z + 0.5*z@J@z
    results.interior_status     = TrustRegionResults::Status::Interior;
    results.cg_iterations_count = 0;

    auto& z      = results.z;
    auto& cgIter = results.cg_iterations_count;
    auto& d      = results.d;
    auto& Pr     = results.Pr;
    auto& Hd     = results.Hd;

    const double cg_tol_squared = settings.cg_tol * settings.cg_tol;

    if (Dot(r0, r0) <= cg_tol_squared && cgIter >= settings.min_cg_iterations) {
      return;
    }

    rCurrent = r0;
    precond(rCurrent, Pr);
    d = 0.0;
    add(d, -1.0, Pr, d);  // d = -Pr

    z          = 0.0;
    double zz  = 0.;
    double rPr = Dot(rCurrent, Pr);
    double zd  = 0.0;
    double dd  = Dot(d, d);

    for (cgIter = 1; cgIter <= settings.max_cg_iterations; ++cgIter) {
      hess_vec_func(d, Hd);
      const double curvature = Dot(d, Hd);
      const double alphaCg   = curvature != 0.0 ? rPr / curvature : 0.0;

      auto& zPred = Hd;  // re-use Hd, this is where bugs come from
      add(z, alphaCg, d, zPred);
      double zzNp1 = Dot(zPred, zPred);  // can optimize this eventually

      if (curvature <= 0) {
        // mfem::out << "negative curvature found.\n";
        projectToBoundaryWithCoefs(z, d, trSize, zz, zd, dd);
        results.interior_status = TrustRegionResults::Status::NegativeCurvature;
        return;
      } else if (zzNp1 > (trSize * trSize)) {
        // mfem::out << "step outside trust region.\n";
        projectToBoundaryWithCoefs(z, d, trSize, zz, zd, dd);
        results.interior_status = TrustRegionResults::Status::OnBoundary;
        return;
      }

      z = zPred;

      hess_vec_func(d, Hd);
      add(rCurrent, alphaCg, Hd, rCurrent);

      precond(rCurrent, Pr);
      double rPrNp1 = Dot(rCurrent, Pr);

      if (Dot(rCurrent, rCurrent) <= cg_tol_squared && cgIter >= settings.min_cg_iterations) {
        return;
      }

      double beta = rPrNp1 / rPr;
      rPr         = rPrNp1;
      add(-1.0, Pr, beta, d, d);

      zz = zzNp1;
      zd = Dot(z, d);
      dd = Dot(d, d);
    }
  }

  /// assemble the jacobian
  void assembleJacobian(const mfem::Vector& x) const
  {
    SERAC_MARK_FUNCTION;
    grad = &oper->GetGradient(x);
    if (nonlinear_options.force_monolithic) {
      auto* grad_blocked = dynamic_cast<mfem::BlockOperator*>(grad);
      if (grad_blocked) grad = buildMonolithicMatrix(*grad_blocked).release();
    }
  }

  /// evaluate the nonlinear residual
  mfem::real_t computeResidual(const mfem::Vector& x_, mfem::Vector& r_) const
  {
    SERAC_MARK_FUNCTION;
    oper->Mult(x_, r_);
    return Norm(r_);
  }

  /// apply the action of the assembled Jacobian matrix to a vector
  void hessVec(const mfem::Vector& x_, mfem::Vector& v_) const
  {
    SERAC_MARK_FUNCTION;
    grad->Mult(x_, v_);
  }

  /// apply trust region specific preconditioner
  void precond(const mfem::Vector& x_, mfem::Vector& v_) const
  {
    SERAC_MARK_FUNCTION;
    tr_precond.Mult(x_, v_);
  };

  /// @overload
  void Mult(const mfem::Vector&, mfem::Vector& X) const
  {
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

    using real_t = mfem::real_t;

    real_t norm, norm_goal;
    norm = initial_norm = computeResidual(X, r);
    norm_goal           = std::max(rel_tol * initial_norm, abs_tol);
    if (print_options.first_and_last && !print_options.iterations) {
      mfem::out << "Newton iteration " << std::setw(3) << 0 << " : ||r|| = " << std::setw(13) << norm << "...\n";
    }
    prec->iterative_mode      = false;
    tr_precond.iterative_mode = false;

    // local arrays
    x_pred.SetSize(X.Size());
    x_pred = 0.0;
    x_mid.SetSize(X.Size());
    x_mid = 0.0;
    r_pred.SetSize(X.Size());
    r_pred = 0.0;
    r_mid.SetSize(X.Size());
    r_mid = 0.0;
    scratch.SetSize(X.Size());
    scratch = 0.0;

    TrustRegionResults  trResults(X.Size());
    TrustRegionSettings settings;
    settings.min_cg_iterations = static_cast<size_t>(nonlinear_options.min_iterations);
    settings.max_cg_iterations = static_cast<size_t>(linear_options.max_iterations);
    settings.cg_tol            = 0.5 * norm_goal;

    scratch        = 1.0;
    double tr_size = nonlinear_options.trust_region_scaling * std::sqrt(Dot(scratch, scratch));
    size_t cumulative_cg_iters_from_last_precond_update = 0;

    auto& d  = trResults.d;   // reuse, maybe dangerous!
    auto& Hd = trResults.Hd;  // reuse, maybe dangerous!

    int it = 0;
    for (; true; it++) {
      MFEM_ASSERT(mfem::IsFinite(norm), "norm = " << norm);
      if (print_options.iterations) {
        mfem::out << "Newton iteration " << std::setw(3) << it << " : ||r|| = " << std::setw(13) << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << std::setw(13) << (initial_norm != 0.0 ? norm / initial_norm : norm);
          mfem::out << ", x_incr = " << std::setw(13) << d.Norml2();
        } else {
          mfem::out << ", norm goal = " << std::setw(13) << norm_goal;
        }
        mfem::out << " :: ";
      }

      if (norm != norm) {
        if (print_options.first_and_last) {
          mfem::out << "Initial residual for trust-region iteration is undefined/nan." << std::endl;
          mfem::out << "Newton: No convergence!\n";
        }
        throw SolveException("Initial residual for trust-region iteration is undefined/nan.");
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
        if (print_options.iterations) {
          // currently it will always be updated
        }
      }

      auto hess_vec_func = [&](const mfem::Vector& x_, mfem::Vector& v_) { hessVec(x_, v_); };
      auto precond_func  = [&](const mfem::Vector& x_, mfem::Vector& v_) { precond(x_, v_); };

      double cauchyPointNormSquared = tr_size * tr_size;
      trResults.reset();

      hess_vec_func(r, trResults.Hd);
      const double gKg = Dot(r, trResults.Hd);
      if (gKg > 0) {
        const double alphaCp = -Dot(r, r) / gKg;
        add(trResults.cauchy_point, alphaCp, r, trResults.cauchy_point);
        cauchyPointNormSquared = Dot(trResults.cauchy_point, trResults.cauchy_point);
      } else {
        const double alphaTr = -tr_size / std::sqrt(Dot(r, r));
        add(trResults.cauchy_point, alphaTr, r, trResults.cauchy_point);
        if (print_options.iterations) {
          mfem::out << "Negative curvature un-preconditioned cauchy point direction found."
                    << "\n";
        }
      }

      if (cauchyPointNormSquared >= tr_size * tr_size) {
        if (print_options.iterations) {
          mfem::out << "Un-preconditioned gradient cauchy point outside trust region, step size = "
                    << std::sqrt(cauchyPointNormSquared) << "\n";
        }
        trResults.cauchy_point *= (tr_size / std::sqrt(cauchyPointNormSquared));
        trResults.z                   = trResults.cauchy_point;
        trResults.cg_iterations_count = 1;
        trResults.interior_status     = TrustRegionResults::Status::OnBoundary;
      } else {
        settings.cg_tol = std::max(0.5 * norm_goal, 5e-5 * norm);
        solveTrustRegionModelProblem(r, scratch, hess_vec_func, precond_func, settings, tr_size, trResults);
      }
      cumulative_cg_iters_from_last_precond_update += trResults.cg_iterations_count;

      bool happyAboutTrSize = false;
      int  lineSearchIter   = 0;
      while (!happyAboutTrSize && lineSearchIter <= nonlinear_options.max_line_search_iterations) {
        ++lineSearchIter;

        doglegStep(trResults.cauchy_point, trResults.z, tr_size, d);

        static constexpr double roundOffTol = 1e-15;

        hess_vec_func(d, Hd);
        double dHd            = Dot(d, Hd);
        double modelObjective = Dot(r, d) + 0.5 * dHd - roundOffTol;

        add(X, d, x_pred);

        double realObjective = std::numeric_limits<double>::max();
        double normPred      = std::numeric_limits<double>::max();
        try {
          normPred    = computeResidual(x_pred, r_pred);
          double obj1 = 0.5 * (Dot(r, d) + Dot(r_pred, d)) - roundOffTol;
          realObjective = obj1;
        } catch (const std::exception&) {
          realObjective = std::numeric_limits<double>::max();
          normPred      = std::numeric_limits<double>::max();
        }

        if (normPred <= norm_goal) {
          X                = x_pred;
          r                = r_pred;
          norm             = normPred;
          happyAboutTrSize = true;
          if (print_options.iterations) {
            printTrustRegionInfo(realObjective, modelObjective, trResults.cg_iterations_count, tr_size, true);
            // zero this output so it doesn't look like the linesearch is doing cg iterations
            trResults.cg_iterations_count = 0;
          }
          break;
        }

        double modelImprove = -modelObjective;
        double realImprove  = -realObjective;

        double rho = realImprove / modelImprove;
        if (modelObjective > 0) {
          if (print_options.iterations || print_options.warnings) {
            mfem::out << "Found a positive model objective increase.  Debug if you see this.\n";
          }
          rho = 0.0; //realImprove / -modelImprove;
        }

        if (!(rho >= settings.eta2)) {  // write it this way to handle NaNs
          tr_size *= settings.t1;
        } else if ((rho > settings.eta3) && (trResults.interior_status == TrustRegionResults::Status::OnBoundary)) {
          tr_size *= settings.t2;
        }

        // eventually extend to handle this case to handle occasional roundoff issues
        // modelRes = g + Jd
        // modelResNorm = np.linalg.norm(modelRes)
        // realResNorm = np.linalg.norm(gy)
        bool willAccept = rho >= settings.eta1;  // or (rho >= -0 and realResNorm <= gNorm)

        if (print_options.iterations) {
          printTrustRegionInfo(realObjective, modelObjective, trResults.cg_iterations_count, tr_size, willAccept);
          trResults.cg_iterations_count =
              0;  // zero this output so it doesn't look like the linesearch is doing cg iterations
        }

        if (willAccept) {
          X                = x_pred;
          r                = r_pred;
          norm             = normPred;
          happyAboutTrSize = true;
          break;
        }
      }
    }

    final_iter = it;
    final_norm = norm;

    if (!converged) {
      throw SolveException("Trust-region algorithm failed to converge.");
    }

    if (print_options.summary || (!converged && print_options.warnings) || print_options.first_and_last) {
      mfem::out << "Newton: Number of iterations: " << final_iter << '\n' << "   ||r|| = " << final_norm << '\n';
    }

    if (!converged && (print_options.summary || print_options.warnings)) {
      mfem::out << "Newton: No convergence!\n";
    }
  }
};

EquationSolver::EquationSolver(NonlinearSolverOptions nonlinear_opts, LinearSolverOptions lin_opts, MPI_Comm comm)
{
  auto [lin_solver, preconditioner] = buildLinearSolverAndPreconditioner(lin_opts, comm);

  lin_solver_     = std::move(lin_solver);
  preconditioner_ = std::move(preconditioner);
  nonlin_solver_  = buildNonlinearSolver(nonlinear_opts, lin_opts, *preconditioner_, comm);
}

EquationSolver::EquationSolver(std::unique_ptr<mfem::NewtonSolver> nonlinear_solver,
                               std::unique_ptr<mfem::Solver>       linear_solver,
                               std::unique_ptr<mfem::Solver>       preconditioner)
{
  SLIC_ERROR_ROOT_IF(!nonlinear_solver, "Nonlinear solvers must be given to construct an EquationSolver");
  SLIC_ERROR_ROOT_IF(!linear_solver, "Linear solvers must be given to construct an EquationSolver");

  nonlin_solver_  = std::move(nonlinear_solver);
  lin_solver_     = std::move(linear_solver);
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

  strumpack_solver_.SetOperator(*strumpack_mat_);
}

#endif

std::unique_ptr<mfem::NewtonSolver> buildNonlinearSolver(const NonlinearSolverOptions& nonlinear_opts,
                                                         const LinearSolverOptions& linear_opts, mfem::Solver& prec,
                                                         MPI_Comm comm)
{
  std::unique_ptr<mfem::NewtonSolver> nonlinear_solver;

  if (nonlinear_opts.nonlin_solver == NonlinearSolver::Newton) {
    SLIC_ERROR_ROOT_IF(nonlinear_opts.min_iterations != 0 || nonlinear_opts.max_line_search_iterations != 0,
                       "Newton's method does not support nonzero min_iterations or max_line_search_iterations");
    nonlinear_solver = std::make_unique<NewtonSolver>(comm, nonlinear_opts);
    // nonlinear_solver = std::make_unique<mfem::NewtonSolver>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::LBFGS) {
    SLIC_ERROR_ROOT_IF(nonlinear_opts.min_iterations != 0 || nonlinear_opts.max_line_search_iterations != 0,
                       "LBFGS does not support nonzero min_iterations or max_line_search_iterations");
    nonlinear_solver = std::make_unique<mfem::LBFGSSolver>(comm);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::NewtonLineSearch) {
    nonlinear_solver = std::make_unique<NewtonSolver>(comm, nonlinear_opts);
  } else if (nonlinear_opts.nonlin_solver == NonlinearSolver::TrustRegion) {
    nonlinear_solver = std::make_unique<TrustRegion>(comm, nonlinear_opts, linear_opts, prec);
#ifdef SERAC_USE_PETSC
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
#ifdef SERAC_USE_SUNDIALS

    SLIC_ERROR_ROOT_IF(nonlinear_opts.min_iterations != 0 || nonlinear_opts.max_line_search_iterations != 0,
                       "kinsol solvers do not support min_iterations or max_line_search_iterations");

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
    nonlinear_solver   = std::move(kinsol_solver);
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
#ifdef SERAC_USE_PETSC
    case LinearSolver::PetscCG:
      iter_lin_solver = std::make_unique<serac::mfem_ext::PetscKSPSolver>(comm, KSPCG, std::string());
      break;
    case LinearSolver::PetscGMRES:
      iter_lin_solver = std::make_unique<serac::mfem_ext::PetscKSPSolver>(comm, KSPGMRES, std::string());
      break;
#else
    case LinearSolver::PetscCG:
    case LinearSolver::PetscGMRES:
      SLIC_ERROR_ROOT("PETSc linear solver requested for non-PETSc build.");
      exitGracefully(true);
      break;
#endif
    default:
      SLIC_ERROR_ROOT("Linear solver type not recognized.");
      exitGracefully(true);
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
  auto          amgx = std::make_unique<mfem::AmgXSolver>();
  conduit::Node options_node;
  options_node["config_version"] = 2;
  auto& solver_options           = options_node["solver"];
  solver_options["solver"]       = "AMG";
  solver_options["presweeps"]    = 1;
  solver_options["postsweeps"]   = 2;
  solver_options["interpolator"] = "D2";
  solver_options["max_iters"]    = 2;
  solver_options["convergence"]  = "ABSOLUTE";
  solver_options["cycle"]        = "V";

  if (options.verbose) {
    options_node["solver/obtain_timings"]    = 1;
    options_node["solver/monitor_residual"]  = 1;
    options_node["solver/print_solve_stats"] = 1;
  }

  // TODO: Use magic_enum here when we can switch to GCC 9+
  // This is an immediately-invoked lambda so that the map
  // can be const without needed to initialize all the values
  // in the constructor
  static const auto solver_names = []() {
    std::unordered_map<AMGXSolver, std::string> names;
    names[AMGXSolver::AMG]             = "AMG";
    names[AMGXSolver::PCGF]            = "PCGF";
    names[AMGXSolver::CG]              = "CG";
    names[AMGXSolver::PCG]             = "PCG";
    names[AMGXSolver::PBICGSTAB]       = "PBICGSTAB";
    names[AMGXSolver::BICGSTAB]        = "BICGSTAB";
    names[AMGXSolver::FGMRES]          = "FGMRES";
    names[AMGXSolver::JACOBI_L1]       = "JACOBI_L1";
    names[AMGXSolver::GS]              = "GS";
    names[AMGXSolver::POLYNOMIAL]      = "POLYNOMIAL";
    names[AMGXSolver::KPZ_POLYNOMIAL]  = "KPZ_POLYNOMIAL";
    names[AMGXSolver::BLOCK_JACOBI]    = "BLOCK_JACOBI";
    names[AMGXSolver::MULTICOLOR_GS]   = "MULTICOLOR_GS";
    names[AMGXSolver::MULTICOLOR_DILU] = "MULTICOLOR_DILU";
    return names;
  }();

  options_node["solver/solver"]   = solver_names.at(options.solver);
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
  auto                          preconditioner = linear_opts.preconditioner;
  auto                          print_level    = linear_opts.print_level;

  // Handle the preconditioner - currently just BoomerAMG and HypreSmoother are supported
  if (preconditioner == Preconditioner::HypreAMG) {
    auto amg_preconditioner = std::make_unique<mfem::HypreBoomerAMG>();
    amg_preconditioner->SetPrintLevel(print_level);
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
    ilu_preconditioner->SetPrintLevel(print_level);
    preconditioner_solver = std::move(ilu_preconditioner);
  } else if (preconditioner == Preconditioner::AMGX) {
#ifdef MFEM_USE_AMGX
    preconditioner_solver = buildAMGX(linear_opts.amgx_options, comm);
#else
    SLIC_ERROR_ROOT("AMGX requested in non-GPU build");
#endif
  } else if (preconditioner == Preconditioner::Petsc) {
#ifdef SERAC_USE_PETSC
    preconditioner_solver = mfem_ext::buildPetscPreconditioner(linear_opts.petsc_preconditioner, comm);
#else
    SLIC_ERROR_ROOT("PETSc preconditioner requested in non-PETSc build");
#endif
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
  nonlinear_container.addString("solver_type", "Solver type (Newton|KINFullStep|KINLineSearch)").defaultValue("Newton");
}

}  // namespace serac

using serac::EquationSolver;
using serac::LinearSolverOptions;
using serac::NonlinearSolverOptions;

serac::LinearSolverOptions FromInlet<serac::LinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  LinearSolverOptions options;
  std::string         type = base["type"];

  if (type == "direct") {
    options.linear_solver = serac::LinearSolver::SuperLU;
    options.print_level   = base["direct_options/print_level"];
    return options;
  }

  auto config             = base["iterative_options"];
  options.relative_tol    = config["rel_tol"];
  options.absolute_tol    = config["abs_tol"];
  options.max_iterations  = config["max_iter"];
  options.print_level     = config["print_level"];
  std::string solver_type = config["solver_type"];
  if (solver_type == "gmres") {
    options.linear_solver = serac::LinearSolver::GMRES;
  } else if (solver_type == "cg") {
    options.linear_solver = serac::LinearSolver::CG;
  } else {
    std::string msg = axom::fmt::format("Unknown Linear solver type given: '{0}'", solver_type);
    SLIC_ERROR_ROOT(msg);
  }
  const std::string prec_type = config["prec_type"];
  if (prec_type == "JacobiSmoother") {
    options.preconditioner = serac::Preconditioner::HypreJacobi;
  } else if (prec_type == "L1JacobiSmoother") {
    options.preconditioner = serac::Preconditioner::HypreL1Jacobi;
  } else if (prec_type == "HypreAMG") {
    options.preconditioner = serac::Preconditioner::HypreAMG;
  } else if (prec_type == "ILU") {
    options.preconditioner = serac::Preconditioner::HypreILU;
#ifdef MFEM_USE_AMGX
  } else if (prec_type == "AMGX") {
    options.preconditioner = serac::Preconditioner::AMGX;
#endif
  } else if (prec_type == "GaussSeidel") {
    options.preconditioner = serac::Preconditioner::HypreGaussSeidel;
#ifdef SERAC_USE_PETSC
  } else if (prec_type == "Petsc") {
    const std::string petsc_prec = config["petsc_prec_type"];
    options.preconditioner       = serac::Preconditioner::Petsc;
    options.petsc_preconditioner = serac::mfem_ext::stringToPetscPCType(petsc_prec);
#endif
  } else {
    std::string msg = axom::fmt::format("Unknown preconditioner type given: '{0}'", prec_type);
    SLIC_ERROR_ROOT(msg);
  }

  return options;
}

serac::NonlinearSolverOptions FromInlet<serac::NonlinearSolverOptions>::operator()(const axom::inlet::Container& base)
{
  NonlinearSolverOptions options;
  options.relative_tol          = base["rel_tol"];
  options.absolute_tol          = base["abs_tol"];
  options.max_iterations        = base["max_iter"];
  options.print_level           = base["print_level"];
  const std::string solver_type = base["solver_type"];
  if (solver_type == "Newton") {
    options.nonlin_solver = serac::NonlinearSolver::Newton;
  } else if (solver_type == "KINFullStep") {
    options.nonlin_solver = serac::NonlinearSolver::KINFullStep;
  } else if (solver_type == "KINLineSearch") {
    options.nonlin_solver = serac::NonlinearSolver::KINBacktrackingLineSearch;
  } else if (solver_type == "KINPicard") {
    options.nonlin_solver = serac::NonlinearSolver::KINPicard;
  } else {
    SLIC_ERROR_ROOT(axom::fmt::format("Unknown nonlinear solver type given: '{0}'", solver_type));
  }
  return options;
}

serac::EquationSolver FromInlet<serac::EquationSolver>::operator()(const axom::inlet::Container& base)
{
  auto lin    = base["linear"].get<LinearSolverOptions>();
  auto nonlin = base["nonlinear"].get<NonlinearSolverOptions>();

  auto [linear_solver, preconditioner] = serac::buildLinearSolverAndPreconditioner(lin, MPI_COMM_WORLD);

  serac::EquationSolver eq_solver(serac::buildNonlinearSolver(nonlin, lin, *preconditioner, MPI_COMM_WORLD),
                                  std::move(linear_solver), std::move(preconditioner));

  return eq_solver;
}
