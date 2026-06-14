// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solver_config.hpp
 *
 * @brief This file contains enumerations and record types for physics solver configuration
 */

#pragma once

#include <variant>
#include <vector>

#include "mfem.hpp"
#include "smith/infrastructure/format.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/numerics/deflation.hpp"

namespace smith {

/**
 * @brief Timestep method of a solver
 */
enum class TimestepMethod
{
  QuasiStatic, /**< Quasistatic */

  // options for first order ODEs
  BackwardEuler,    /**< FirstOrderODE option */
  SDIRK33,          /**< FirstOrderODE option */
  ForwardEuler,     /**< FirstOrderODE option */
  RK2,              /**< FirstOrderODE option */
  RK3SSP,           /**< FirstOrderODE option */
  RK4,              /**< FirstOrderODE option */
  GeneralizedAlpha, /**< FirstOrderODE option */
  ImplicitMidpoint, /**< FirstOrderODE option */
  SDIRK23,          /**< FirstOrderODE option */
  SDIRK34,          /**< FirstOrderODE option */

  // options for second order ODEs
  //
  // note: we don't have a way to communicate
  //       parameters to the TimestepMethod,
  //       right now, so Newmark implies
  //       (beta = 0.25, gamma = 0.5)
  Newmark,             /**< SecondOrderODE option */
  HHTAlpha,            /**< SecondOrderODE option */
  WBZAlpha,            /**< SecondOrderODE option */
  AverageAcceleration, /**< SecondOrderODE option */
  LinearAcceleration,  /**< SecondOrderODE option */
  CentralDifference,   /**< SecondOrderODE option */
  FoxGoodwin           /**< SecondOrderODE option */
};

/**
 * @brief this enum describes which way to enforce the time-varying constraint u(t) == U(t)
 */

enum class DirichletEnforcementMethod
{
  /**
   * Satisfies u(t+dt) == U(t+dt)
   *
   * This method imposes additional stability criteria
   * for the case of second order differential equations
   */
  DirectControl,

  /**
   * (default value)
   * Satisfies dudt(t+dt) == dUdt(t+dt)
   *
   * This method does not impose any additional stability criteria
   * for the case of second order differential equations.
   */
  RateControl,

  /**
   * satisfies u(t+dt) == U(t+dt),
   *           dudt(t+dt) == dUdt(t+dt),
   * (and      d2udt2(t+dt) == d2Udt2(t+dt), for a second order ODE)
   *
   * Empirically, this method tends to be the most accurate
   * for small timesteps (by a constant factor),  but is more
   * expensive to evaluate
   */
  FullControl
};

/// A timestep and boundary condition enforcement method for a dynamic solver
struct TimesteppingOptions {
  /// The timestepping method to be applied
  TimestepMethod timestepper = TimestepMethod::QuasiStatic;

  /// The essential boundary enforcement method to use
  DirichletEnforcementMethod enforcement_method = DirichletEnforcementMethod::RateControl;
};

// _linear_solvers_start
/// Linear solution method indicator
enum class LinearSolver
{
  CG,        /**< Conjugate gradient */
  GMRES,     /**< Generalized minimal residual method */
  SuperLU,   /**< SuperLU MPI-enabled direct nodal solver */
  Strumpack, /**< Strumpack MPI-enabled direct frontal solver*/
  PetscCG,   /**< PETSc MPI-enabled conjugate gradient solver */
  PetscGMRES /**< PETSc MPI-enabled generalize minimal residual solver */
};
// _linear_solvers_end

/// Convert linear solver enums to their string names
inline std::string linearName(const LinearSolver& s)
{
  switch (s) {
    case LinearSolver::CG:
      return "CG";
    case LinearSolver::GMRES:
      return "GMRES";
    case LinearSolver::SuperLU:
      return "SuperLU";
    case LinearSolver::Strumpack:
      return "Strumpack";
    case LinearSolver::PetscCG:
      return "PetscCG";
    case LinearSolver::PetscGMRES:
      return "PetscGMRES";
  }
  // This cannot happen, but GCC doesn't know that
  return "UNKNOWN";
}

/// output linear solver string representation to a stream
inline std::ostream& operator<<(std::ostream& os, LinearSolver s) { return os << linearName(s); }

/// string->value matching for optionally entering options as string in command line
inline std::map<std::string, LinearSolver> linearSolverMap = {
    {"CG", LinearSolver::CG},           {"GMRES", LinearSolver::GMRES},
    {"SuperLU", LinearSolver::SuperLU}, {"Strumpack", LinearSolver::Strumpack},
    {"PetscCG", LinearSolver::PetscCG}, {"PetscGMRES", LinearSolver::PetscGMRES},
};

// Add a custom list of strings? conduit node?
// Arbitrary string (e.g. json) to define parameters?

// _nonlinear_solvers_start
/// Nonlinear solver method indicator
enum class NonlinearSolver
{
  Newton,                    /**< MFEM-native Newton-Raphson */
  LBFGS,                     /**< MFEM-native Limited memory BFGS */
  NewtonLineSearch,          /**< Custom solver using preconditioned earch direction with backtracking line search */
  TrustRegion,               /**< Custom solver using a trust region solver */
  KINFullStep,               /**< KINSOL Full Newton (Sundials must be enabled) */
  KINBacktrackingLineSearch, /**< KINSOL Newton with Backtracking Line Search (Sundials must be enabled) */
  KINPicard,                 /**< KINSOL Picard (Sundials must be enabled) */
  PetscNewton,               /**< PETSc Full Newton */
  PetscNewtonBacktracking,   /**< PETSc Newton with backtracking line search */
  PetscNewtonCriticalPoint,  /**< PETSc Newton with critical point line search */
  PetscTrustRegion           /**< PETSc trust region solver */
};
// _nonlinear_solvers_end

/// Convert nonlinear linear solver enums to their string names
inline std::string nonlinearName(const NonlinearSolver& s)
{
  switch (s) {
    case NonlinearSolver::Newton:
      return "Newton";
    case NonlinearSolver::LBFGS:
      return "LBFGS";
    case NonlinearSolver::NewtonLineSearch:
      return "NewtonLineSearch";
    case NonlinearSolver::TrustRegion:
      return "TrustRegion";
    case NonlinearSolver::KINFullStep:
      return "KINFullStep";
    case NonlinearSolver::KINBacktrackingLineSearch:
      return "KINBacktrackingLineSearch";
    case NonlinearSolver::KINPicard:
      return "KINPicard";
    case NonlinearSolver::PetscNewton:
      return "PetscNewton";
    case NonlinearSolver::PetscNewtonBacktracking:
      return "PetscNewtonBacktracking";
    case NonlinearSolver::PetscNewtonCriticalPoint:
      return "PetscNewtonCriticalPoint";
    case NonlinearSolver::PetscTrustRegion:
      return "PetscTrustRegion";
  }
  // This cannot happen, but GCC doesn't know that
  return "UNKNOWN";
}

/// output nonlinear solver string representation to a stream
inline std::ostream& operator<<(std::ostream& os, NonlinearSolver s) { return os << nonlinearName(s); }

/// string->value matching for optionally entering options as string in command line
inline std::map<std::string, NonlinearSolver> nonlinearSolverMap = {
    {"Newton", NonlinearSolver::Newton},
    {"LBFGS", NonlinearSolver::LBFGS},
    {"NewtonLineSearch", NonlinearSolver::NewtonLineSearch},
    {"TrustRegion", NonlinearSolver::TrustRegion},
    {"KINFullStep", NonlinearSolver::KINFullStep},
    {"KINBacktrackingLineSearch", NonlinearSolver::KINBacktrackingLineSearch},
    {"KINPicard", NonlinearSolver::KINPicard},
    {"PetscNewton", NonlinearSolver::PetscNewton},
    {"PetscNewtonBacktracking", NonlinearSolver::PetscNewtonBacktracking},
    {"PetscNewtonCriticalPoint", NonlinearSolver::PetscNewtonCriticalPoint},
    {"PetscTrustRegion", NonlinearSolver::PetscTrustRegion},
};

/**
 * @brief Solver types supported by AMGX
 */
enum class AMGXSolver
{
  AMG,            /**< GPU Algebraic Multigrid */
  PCGF,           /**< GPU PCGF */
  CG,             /**< GPU CG */
  PCG,            /**< GPU PCG */
  PBICGSTAB,      /**< GPU PBICGSTAB */
  BICGSTAB,       /**< GPU BICGSTAB */
  FGMRES,         /**< GPU FGMRES */
  JACOBI_L1,      /**< GPU JACOBI_L1 */
  GS,             /**< GPU GS */
  POLYNOMIAL,     /**< GPU POLYNOMIAL */
  KPZ_POLYNOMIAL, /**< GPU KPZ_POLYNOMIAL */
  BLOCK_JACOBI,   /**< GPU BLOCK_JACOBI */
  MULTICOLOR_GS,  /**< GPU MULTICOLOR_GS */
  MULTICOLOR_DILU /**< GPU MULTICOLOR_DILU */
};

/**
 * @brief Stores the information required to configure a NVIDIA AMGX preconditioner
 */
struct AMGXOptions {
  /**
   * @brief The solver algorithm
   */
  AMGXSolver solver = AMGXSolver::AMG;
  /**
   * @brief The smoother algorithm
   */
  AMGXSolver smoother = AMGXSolver::JACOBI_L1;
  /**
   * @brief Whether to display statistics from AMGX
   */
  bool verbose = false;
};

/**
 * @brief Stores the configuration information for an AMGFContact preconditioner
 */
struct AMGFContactOptions {
  /**
   * @brief The amg relaxation type
   */
  int relax_type = 88;  // l1-hybrid symmetric Gauss-Seidel smoother
  /**
   * @brief amg DimSystemsOptions
   */
  int dim_systems_options =
      3;  // geometric dimension of problem, used to set more robust options for systems such as elasticity
};

/**
 * @brief Preconditioner types supported by PETSc
 */
enum class PetscPCType
{
  JACOBI,        /**< Jacobi with diagonal scaling */
  JACOBI_L1,     /**< Jacobi with row-wise L1 norm scaling */
  JACOBI_ROWSUM, /**< Jacobi with row sum (no absolute value) scaling */
  JACOBI_ROWMAX, /**< Jacobi with L-infinity norm scaling */
  PBJACOBI,      /**< Point-block Jacobi with LU factorization on sub-blocks */
  BJACOBI,       /**< Block Jacobi with LU factorization on sub-blocks, set number of blocks with -pc_bjacobi_blocks */
  LU,            /**< Direct solver based on LU factorization */
  ILU,           /**< Incomplete LU factorization */
  CHOLESKY,      /**< Cholesky factorization */
  SVD,           /**< LAPACK xGESVD SVD decomposition, fully redundant (SLOW for MPI) */
  ASM,  /**< Additive Schwarz method, each block is solved with its own KSP object, blocks cannot be shared between MPI
           processes. Set total number of blocks with -pc_asm_blocks N */
  GASM, /**< Additive Schwarz method, each block is solved with its own KSP object, blocks can be shared between MPI
           processes. Set total number of blocks with -pc_gasm_total_subdomains N */
  GAMG, /**< PETSc built-in AMG preconditioner */
  HMG,  /**< Hierarchical AMG for multi-component PDE problems */
  NONE, /**< No preconditioner, or type set via -pc_type CLI flag */
};

/// Convert Petsc preconditioner enums to their string names
inline std::string petscPCName(const PetscPCType& s)
{
  switch (s) {
    case PetscPCType::JACOBI:
      return "JACOBI";
    case PetscPCType::JACOBI_L1:
      return "JACOBI_L1";
    case PetscPCType::JACOBI_ROWSUM:
      return "JACOBI_ROWSUM";
    case PetscPCType::JACOBI_ROWMAX:
      return "JACOBI_ROWMAX";
    case PetscPCType::PBJACOBI:
      return "PBJACOBI";
    case PetscPCType::BJACOBI:
      return "BJACOBI";
    case PetscPCType::LU:
      return "LU";
    case PetscPCType::ILU:
      return "ILU";
    case PetscPCType::CHOLESKY:
      return "CHOLESKY";
    case PetscPCType::SVD:
      return "SVD";
    case PetscPCType::ASM:
      return "ASM";
    case PetscPCType::GASM:
      return "GASM";
    case PetscPCType::GAMG:
      return "GAMG";
    case PetscPCType::HMG:
      return "HMG";
    case PetscPCType::NONE:
      return "NONE";
  }
  // This cannot happen, but GCC doesn't know that
  return "UNKNOWN";
}

/// output PETSc preconditioner string representation to a stream
inline std::ostream& operator<<(std::ostream& os, PetscPCType s) { return os << petscPCName(s); }

// _preconditioners_start
/// The type of preconditioner to be used
enum class Preconditioner
{
  HypreJacobi,      /**< Hypre-based Jacobi */
  HypreL1Jacobi,    /**< Hypre-based L1-scaled Jacobi */
  HypreGaussSeidel, /**< Hypre-based Gauss-Seidel */
  HypreAMG,         /**< Hypre's BoomerAMG algebraic multi-grid */
  HypreILU,         /**< Hypre's Incomplete LU */
  AMGX,             /**< NVIDIA's AMGX GPU-enabled algebraic multi-grid, GPU builds only */
  Petsc,            /**< PETSc preconditioner,  */
  AMGFContact,      /**< MFEM-based AMG with filtering (AMGF), contact problems only */
  Deflation,        /**< Custom two-level deflation (per-rank affine basis + Jacobi) */
  BlockDiagonal,    /**< Block diagonal preconditioner */
  BlockTriangular,  /**< Block triangular preconditioner */
  BlockSchur,       /**< Block Schur preconditioner */
  None              /**< No preconditioner used */
};
// _preconditioners_end

/// Convert preconditioner enums to their string names
inline std::string preconditionerName(Preconditioner p)
{
  switch (p) {
    case Preconditioner::HypreJacobi:
      return "HypreJacobi";
    case Preconditioner::HypreL1Jacobi:
      return "HypreL1Jacobi";
    case Preconditioner::HypreGaussSeidel:
      return "HypreGaussSeidel";
    case Preconditioner::HypreAMG:
      return "HypreAMG";
    case Preconditioner::HypreILU:
      return "HypreILU";
    case Preconditioner::AMGX:
      return "AMGX";
    case Preconditioner::Petsc:
      return "Petsc";
    case Preconditioner::AMGFContact:
      return "AMGFContact";
    case Preconditioner::Deflation:
      return "Deflation";
    case Preconditioner::BlockDiagonal:
      return "BlockDiagonal";
    case Preconditioner::BlockTriangular:
      return "BlockTriangular";
    case Preconditioner::BlockSchur:
      return "BlockSchur";
    case Preconditioner::None:
      return "None";
  }
  // This cannot happen, but GCC doesn't know that
  return "UNKNOWN";
}

/// output preconditioner string representation to a stream
inline std::ostream& operator<<(std::ostream& os, Preconditioner p) { return os << preconditionerName(p); }

/// string->value matching for optionally entering options as string in command line
inline std::map<std::string, Preconditioner> preconditionerMap = {
    {"HypreJacobi", Preconditioner::HypreJacobi},
    {"HypreL1Jacobi", Preconditioner::HypreL1Jacobi},
    {"HypreGaussSeidel", Preconditioner::HypreGaussSeidel},
    {"HypreAMG", Preconditioner::HypreAMG},
    {"HypreILU", Preconditioner::HypreILU},
    {"AMGX", Preconditioner::AMGX},
    {"Petsc", Preconditioner::Petsc},
    {"AMGFContact", Preconditioner::AMGFContact},
    {"Deflation", Preconditioner::Deflation},
    {"BlockDiagonal", Preconditioner::BlockDiagonal},
    {"BlockTriangular", Preconditioner::BlockTriangular},
    {"BlockSchur", Preconditioner::BlockSchur},
    {"None", Preconditioner::None},
};

// _linear_options_start
/// Parameters for an iterative linear solution scheme
struct LinearSolverOptions {
  /// Linear solver selection
  LinearSolver linear_solver = LinearSolver::GMRES;

  /// PreconditionerOptions selection
  Preconditioner preconditioner = Preconditioner::HypreJacobi;

  /// AMGX Options, used for Preconditioner::AMGX
  AMGXOptions amgx_options = AMGXOptions{};

  /// AMGFContact Options, used for Preconditioner::AMGFContact
  AMGFContactOptions amgfcontact_options = AMGFContactOptions{};

  /// ParFiniteElementSpace handle, REQUIRED for Preconditioner::Deflation; ignored otherwise.
  /// Non-owning. Lifetime must outlive the preconditioner.
  mfem::ParFiniteElementSpace* deflation_fes = nullptr;

  /// Polynomial order of the per-rank deflation basis (Affine = const+linear,
  /// Quadratic = const+linear+quadratic). Ignored unless using
  /// Preconditioner::Deflation.
  DeflationOrder deflation_order = DeflationOrder::Affine;

  /// Coarse-correction mode for the deflation preconditioner. Ignored unless
  /// using Preconditioner::Deflation.
  CoarseMode deflation_coarse_mode = CoarseMode::Additive;

  /// Sub-rank pieces for the deflation basis: each rank's elements are split into this
  /// many geometric pieces (recursive coordinate bisection), each with an independent
  /// piecewise-affine basis. Enriches the coarse space at low processor counts
  /// (m = pieces * ranks * mpr). 1 = the classic per-rank basis.
  int deflation_pieces = 1;

  /// PETSc preconditioner type
  PetscPCType petsc_preconditioner = PetscPCType::JACOBI;

  /// Relative tolerance
  double relative_tol = 1.0e-8;

  /// Absolute tolerance
  double absolute_tol = 1.0e-12;

  /// Maximum number of iterations
  int max_iterations = 300;

  /// Relative quadratic-model-decrease stagnation tolerance for Steihaug-CG (0 disables).
  /// Inner CG exits when per-iter model decrement / |model| stays below this for
  /// `cg_model_stagnation_window` consecutive iters. Adaptive alternative to a
  /// hand-picked `max_iterations` cap. Typical: 1e-3.
  double cg_model_energy_stagnation_reltol = 0.0;

  /// Consecutive stagnant iters required to trigger model-stagnation exit (0 disables).
  /// Typical: 5.
  int cg_model_stagnation_window = 0;

  /// Enable Eisenstat–Walker choice-2 adaptive forcing term:
  /// `eta_k = gamma * (||F_k||/||F_{k-1}||)^alpha`, capped at eta_max, with the
  /// standard safeguard `eta_k = max(eta_k, gamma * eta_{k-1}^alpha)` when the
  /// previous term wasn't already small. Replaces the fixed `cg_tol`. When
  /// disabled (default), the legacy `max(0.5*norm_goal, 5e-5*norm)` rule is used.
  bool cg_eisenstat_walker = false;

  /// EW gamma (in (0, 1]). Standard value 0.9.
  double cg_ew_gamma = 0.9;

  /// EW alpha (exponent). Standard value (1+sqrt(5))/2 ≈ 1.618.
  double cg_ew_alpha = 1.618033988749895;

  /// EW upper bound on the forcing term. Standard value 0.5.
  double cg_ew_eta_max = 0.5;

  /// Debugging print level for the linear solver
  int print_level = 0;

  /// Debugging print level for the preconditioner
  int preconditioner_print_level = 0;

  /// Toggle to enable conversion to Block Sparse Row (BSR) format for local SpMV
  bool use_bsr_spmv = false;

  /// BSR block size. 0 (default) = auto-detect from the attached FES vdim; set >0 to override.
  int bsr_block_size = 0;

  /// Deflation first-level smoother: "hypre" (scalar Hypre Jacobi, default), "jacobi"
  /// (point Jacobi with the diagonal read from the BSR blocks — same answer as hypre,
  /// but works without a hypre matrix), or "block" (vdim x vdim block-Jacobi).
  std::string deflation_smoother = "hypre";

  /// Subblock linear solver options for block preconditioners
  std::vector<LinearSolverOptions> sub_block_linear_solver_options = {};

  /// Block Triangular Preconditioner factorization type
  BlockTriangularType block_triangular_type = BlockTriangularType::Lower;

  /// Block Schur preconditioner factorization type
  BlockSchurType block_schur_type = BlockSchurType::Full;

  /// Schur approximation type
  SchurApproxType schur_approx_type = SchurApproxType::DiagInv;
};
// _linear_options_end

/// Enumerated options for when to use trust-region subspace solver
enum SubSpaceOptions
{
  NEVER,
  WHEN_INDEFINITE,
  WHEN_INDEFINITE_OR_BOUNDARY,
  ALWAYS
};

// _nonlinear_options_start
/// Nonlinear solution scheme parameters
struct NonlinearSolverOptions {
  /// Nonlinear solver selection
  NonlinearSolver nonlin_solver = NonlinearSolver::NewtonLineSearch;

  /// Relative tolerance
  double relative_tol = 1.0e-8;

  /// Absolute tolerance
  double absolute_tol = 1.0e-12;

  /// Minimum number of iterations
  int min_iterations = 0;

  /// Maximum number of iterations
  int max_iterations = 20;

  /// Maximum line search cutbacks
  int max_line_search_iterations = 8;

  /// Debug print level
  int print_level = 0;

  /// Scaling for the initial trust region size
  double trust_region_scaling = 0.1;

  /// Option for how when the subspace solver should be utilized within trust-region solver
  SubSpaceOptions subspace_option = SubSpaceOptions::NEVER;

  /// Number of extra leftmost eigenvector to be stored between solves
  int num_leftmost = 1;

  /// Number of previous accepted steps to include in trust-region subspace solves
  int num_previous_steps = 1;

  /// Quadrature points used to estimate the real work ∫₀¹ r(x+τd)·d dτ along an
  /// accepted trust-region step. Default 2 = trapezoid (endpoints only, no extra
  /// residual evals). 3 = Simpson (one extra eval at midpoint). 5 = Boole
  /// (three extra evals at τ = 1/4, 1/2, 3/4). Higher-order rules cost extra
  /// residual evaluations per outer iter but give a much more accurate ΔE
  /// estimate for strongly nonlinear functions near indefinite Hessians.
  int trust_work_quadrature_points = 2;

  /// Number of leftmost eigvecs of the assembled Hessian to compute via symmetric Lanczos at
  /// each TR SetOperator, and to push into `left_mosts` as subspace candidates. 0 disables.
  /// Independent of the deflation basis — applies to Jacobi/Affine/Quadratic alike. Cost is
  /// roughly `trust_num_lanczos_iters` matvecs per outer iter; typical 5-10 for good
  /// approximation of the dominant negative eigenmode in buckling problems.
  int trust_num_lanczos = 0;

  /// Krylov dimension for the Lanczos pass. Should be ≥ `trust_num_lanczos`; 2-3× gives
  /// better convergence to extreme eigenvalues. Defaults to 0 (use 3× `trust_num_lanczos`
  /// when active).
  int trust_num_lanczos_iters = 0;

  /// Relative CG forcing term: each TrustRegion model solve uses
  /// cg_tol = max(0.5 * outer_goal, cg_forcing_rel * ||r||).
  double cg_forcing_rel = 5.0e-5;

  /// Adaptive CG iteration cap floor (0 disables). When > 0, each outer's CG budget is
  ///   cap = clamp(cg_cap_min, max_cg, cg_cap_min * (max_cg/cg_cap_min)^frac * gamma^m)
  /// where frac in [0,1] is the nonlinear residual's log-progress from the running max
  /// toward the convergence goal, and m counts consecutive radius-shrinking outer steps.
  /// Tight budget far from tolerance / after poor steps; full max_cg accuracy only when
  /// the Newton model is trustworthy. Truncated solves remain globally safe (Steihaug)
  /// and are routed to the subspace step.
  int cg_cap_min = 0;

  /// Budget decay per consecutive radius-shrinking step for the adaptive CG cap.
  double cg_cap_gamma = 0.7;

  /// TrustRegion acceptance safeguard (only without an exact-energy callback): reject a
  /// candidate step whose predicted residual norm exceeds this multiple of the current one.
  double residual_growth_cap = 3.0;

  /// Trust-region radius decrease factor on rejected / low-quality steps.
  double tr_decrease_factor = 0.25;

  /// Trust-region radius increase factor on high-quality boundary steps.
  double tr_increase_factor = 1.75;

  /// Worst-case work-ratio rho for accepting a step (eta1).
  double tr_eta1 = 1.0e-9;

  /// rho below which the radius shrinks (eta2).
  double tr_eta2 = 0.1;

  /// rho above which a boundary step grows the radius (eta3).
  double tr_eta3 = 0.6;

  /// rho ceiling: steps with rho above this are distrusted and rejected (eta4).
  double tr_eta4 = 4.2;
};
// _nonlinear_options_end

}  // namespace smith

// std::format support for Smith solver enums
namespace std {
template <>
/// @brief Formats `smith::NonlinearSolver` values with stream output.
struct formatter<smith::NonlinearSolver> : smith::format::OstreamFormatter {};

template <>
/// @brief Formats `smith::LinearSolver` values with stream output.
struct formatter<smith::LinearSolver> : smith::format::OstreamFormatter {};

template <>
/// @brief Formats `smith::Preconditioner` values with stream output.
struct formatter<smith::Preconditioner> : smith::format::OstreamFormatter {};

template <>
/// @brief Formats `smith::PetscPCType` values with stream output.
struct formatter<smith::PetscPCType> : smith::format::OstreamFormatter {};
}  // namespace std
