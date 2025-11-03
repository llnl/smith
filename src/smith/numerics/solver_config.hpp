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

#include "mfem.hpp"
#include "axom/fmt.hpp"

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

  /// PETSc preconditioner type
  PetscPCType petsc_preconditioner = PetscPCType::JACOBI;

  /// Relative tolerance
  double relative_tol = 1.0e-8;

  /// Absolute tolerance
  double absolute_tol = 1.0e-12;

  /// Maximum number of iterations
  int max_iterations = 300;

  /// Debugging print level for the linear solver
  int print_level = 0;

  /// Debugging print level for the preconditioner
  int preconditioner_print_level = 0;
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
  int max_line_search_iterations = 0;

  /// Debug print level
  int print_level = 0;

  /// Scaling for the initial trust region size
  double trust_region_scaling = 0.1;

  /// Option for how when the subspace solver should be utilized within trust-region solver
  SubSpaceOptions subspace_option = SubSpaceOptions::NEVER;

  /// Number of extra leftmost eigenvector to be stored between solves
  int num_leftmost = 1;

  /// Should the gradient be converted to a monolithic matrix
  bool force_monolithic = false;
};
// _nonlinear_options_end

}  // namespace smith

// fmt support for smith::NonlinearSolver
namespace axom::fmt {
template <>
struct formatter<smith::NonlinearSolver> : ostream_formatter {};

// fmt support for smith::LinearSolver
template <>
struct formatter<smith::LinearSolver> : ostream_formatter {};

// fmt support for smith::Preconditioner
template <>
struct formatter<smith::Preconditioner> : ostream_formatter {};

// fmt support for smith::PetscPCType
template <>
struct formatter<smith::PetscPCType> : ostream_formatter {};
}  // namespace axom::fmt
