// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <memory>
#include <string>

#include "mpi.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/numerics/solver_config.hpp"

#ifdef SMITH_USE_PETSC

#include "petsc.h"
#include "petscpc.h"

namespace smith::mfem_ext {
class PetscKSPSolver;

/**
 * @brief Wrapper around mfem::Preconditioner to allow for better interoperability with HYPRE-based mfem solvers
 */
class PetscPCSolver : public mfem::PetscPreconditioner {
 protected:
  /// @brief Flag used to prevent extra checks for conversion to MATAIJ
  mutable bool checked_for_convert_ = false;
  /// @brief Matrix passed to SetOperator() converted to MATAIJ format, if such a conversion was needed.  Mutable, as
  /// this may be set by a PETSc callback within Mult(), which is marked as const.
  mutable std::unique_ptr<mfem::PetscParMatrix> converted_matrix_;

  mutable std::unique_ptr<PetscPreconditioner> fieldsplit_pc_;

  /**
   * @brief Pre-solve callback used to convert the matrix type to MATAIJ, which is needed for most PETSc preconditioners
   *
   * @param pc PETSc preconditioner object.
   * @param ksp PETSc linear solver object on which this preconditioner is acting.
   *
   * @return Error code, or 0 on success.
   *
   * @note Since this function doesn't accept a context object, we have to set @a this as the application context on the
   * wrapped PC.
   */
  friend PetscErrorCode convertPCPreSolve(PC pc, KSP);
  friend PetscErrorCode convertKSPPreSolve(KSP, Vec, Vec, void*);
  friend class PetscKSPSolver;

 public:
  /**
   * @brief Construct a PETSc-based preconditioner of a particular PCType. The operator must be set with
   * SetOperator().
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param pc_type The PETSc PCType of the preconditioner
   * @param prefix The command-line prefix for all arguments passed to the preconditioner
   *
   * @note Additional arguments for the preconditioner can be set via the command line, e.g. `-pc_jacobi_type
   * rowsum`. Use `-pc_type TYPE -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscPCSolver(MPI_Comm comm, PCType pc_type = PCJACOBI, const std::string& prefix = std::string());

  /**
   * @brief Construct a PETSc-based preconditioner of a particular PCType to precondition the provided matrix.
   * @param A The mfem::PetscParMatrix operator to precondition for
   * @param pc_type The PETSc PCType of the preconditioner
   * @param prefix The command-line prefix for all arguments passed to the preconditioner
   *
   * @note Additional arguments for the preconditioner can be set via the command line, e.g. `-pc_jacobi_type rowsum`.
   * Use `-pc_type TYPE -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscPCSolver(mfem::PetscParMatrix& A, PCType pc_type = PCJACOBI, const std::string& prefix = std::string());

  /**
   * @brief Construct a PETSc-based preconditioner of a particular PCType to precondition a general operator.
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param op The general mfem::Operator to precondition. This will be converted to the sparse MATAIJ format.
   * @param pc_type The PETSc PCType of the preconditioner
   * @param prefix The command-line prefix for all arguments passed to the preconditioner
   *
   * @note Additional arguments for the preconditioner can be set via the command line, e.g. `-pc_jacobi_type rowsum`.
   * Use `-pc_type TYPE -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscPCSolver(MPI_Comm comm, mfem::Operator& op, PCType pc_type = PCJACOBI,
                const std::string& prefix = std::string());

  virtual void SetOperator(const Operator& op);
  virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const;
  virtual void MultTranspose(const mfem::Vector& b, mfem::Vector& x) const;
};

/**
 * @brief A PETSC-based preconditioner which requires information about the underlying finite element space
 */
class PetscPreconditionerSpaceDependent : public PetscPCSolver {
 protected:
  /// @brief Finite element space defining the operator
  mfem::ParFiniteElementSpace* fespace_ = nullptr;

 public:
  /**
   * @brief Construct a PetscPreconditionerSpaceDependent without an explicit operator (must be set later with
   * SetOperator())
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param pc_type The PETSc PCType of the preconditioner
   * @param prefix Prefix for all PETSc command line options options
   *
   * @note Additional arguments for the preconditioner can be set via the command line, e.g. `-pc_jacobi_type rowsum`.
   * Use `-pc_type TYPE -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscPreconditionerSpaceDependent(MPI_Comm& comm, PCType pc_type = PCHMG, const std::string& prefix = std::string())
      : PetscPCSolver(comm, pc_type, prefix)
  {
  }

  /**
   * @brief Construct a PetscPreconditionerSpaceDependent to approximate A^{-1}
   * @param A The mfem::PetscParMatrix for which the preconditioner is constructed
   * @param pc_type The PETSc PCType of the preconditioner
   * @param prefix Prefix for all PETSc command line options options
   *
   * @note Additional arguments for the preconditioner can be set via the command line, e.g. `-pc_jacobi_type rowsum`.
   * Use `-pc_type TYPE -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscPreconditionerSpaceDependent(mfem::PetscParMatrix& A, PCType pc_type = PCHMG,
                                    const std::string& prefix = std::string())
      : PetscPCSolver(A, pc_type, prefix)
  {
  }

  /**
   * @brief Construct a PetscPreconditionerSpaceDependent from a generic operator
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param op mfem::Operator used to construct the preconditioner
   * @param pc_type The PETSc PCType of the preconditioner
   * @param prefix Prefix for all PETSc command line options options
   *
   * @note Additional arguments for the preconditioner can be set via the command line, e.g. `-pc_jacobi_type rowsum`.
   * Use `-pc_type TYPE -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscPreconditionerSpaceDependent(MPI_Comm comm, Operator& op, PCType pc_type = PCHMG,
                                    const std::string& prefix = std::string())
      : PetscPCSolver(comm, op, pc_type, prefix)
  {
  }

  /**
   * @brief Set the underlying matrix operator to use in the solution algorithm
   * @param[in] op The matrix operator for which to approximate the inverse
   */
  virtual void SetOperator(const Operator& op) override;

  /**
   * @brief Set the finite element space used to create matrices and vectors
   * @param fespace The (usually displacement) finite element space
   */
  virtual void SetFESpace(mfem::ParFiniteElementSpace* fespace) { fespace_ = fespace; }
};

/**
 * @brief Wrapper for applying the PETSc algebraic multigrid preconditioner PCGAMG
 */
class PetscGAMGSolver : public PetscPreconditionerSpaceDependent {
 public:
  /**
   * @brief Construct a PetscGAMGSolver without an explicit operator (must be set later with SetOperator())
   *
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param prefix Prefix for all PETSc command line options options
   *
   * @note Additional arguments for the preconditioner can be set via the command line, e.g. `-pc_jacobi_type rowsum`.
   * Use `-pc_type TYPE -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscGAMGSolver(MPI_Comm& comm, const std::string& prefix = std::string());

  /**
   * @brief Construct a PetscGAMGSolver to approximate A^{-1}
   * @param A The mfem::PetscParMatrix for which the preconditioner is constructed
   * @param prefix Prefix for all PETSc command line options options
   *
   * @note Additional arguments for the preconditioner can be set via the command line, e.g. `-pc_jacobi_type rowsum`.
   * Use `-pc_type TYPE -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscGAMGSolver(mfem::PetscParMatrix& A, const std::string& prefix = std::string());

  /**
   * @brief Construct a PetscGAMGSolver from a generic operator
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param op mfem::Operator used to construct the preconditioner
   * @param prefix Prefix for all PETSc command line options options
   *
   * @note Additional arguments for the preconditioner can be set via the command line, e.g. `-pc_jacobi_type rowsum`.
   * Use `-pc_type TYPE -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscGAMGSolver(MPI_Comm comm, Operator& op, const std::string& prefix = std::string());

  /**
   * @brief Set the underlying matrix operator to use in the solution algorithm
   * @param[in] op The matrix operator for which to approximate the inverse
   */
  virtual void SetOperator(const Operator& op) override;

  /**
   * @brief Set up the near null space for the operator using the finite element space provided to SetFESpace
   */
  void SetupNearNullSpace();
};

/**
 * @brief Build a PETSc preconditioner
 *
 * @param pc_type Type of PETSc preconditioner to construct
 * @param comm The communicator for the underlying operator and HypreParVectors
 * @return The constructed PETSc preconditioner
 */
std::unique_ptr<PetscPCSolver> buildPetscPreconditioner(PetscPCType pc_type, const MPI_Comm comm);

/**
 * @brief Convert a string to the corresponding PetscPCType
 *
 * @param type_str String to convert
 * @return The converted PetscPCType
 */
PetscPCType stringToPetscPCType(const std::string& type_str);

/**
 * @brief Wrapper around mfem::PetscLinearSolver supporting the mfem::IterativeSolver interface
 */
class PetscKSPSolver : virtual public mfem::IterativeSolver, public mfem::PetscLinearSolver {
 private:
  /// @brief Flag determining whether the mfem::Operator is wrapped or converted
  bool wrap_ = false;
  /// @brief Flag indicating whether convertKSPPreSolve has been called
  bool checked_for_convert_ = false;
  /// @brief mfem::HypreParMatrix wrapping an existing MATHYPRE. Mutable, as this may be set by a PETSc callback within
  /// Mult(), which is marked as const.
  mutable std::unique_ptr<mfem::HypreParMatrix> wrapped_matrix_;
  /// @brief Cached converted operator
  mfem::PetscParMatrix* pA_ = nullptr;
  /// @brief Flag which is true if SetOperator is never called and the preconditioner is not a PETSc preconditioner.
  /// Mutable, as this may be set by a PETSc callback within Mult(), which is marked as const.
  mutable bool needs_hypre_wrapping_ = false;
  // mutable PC   fieldsplit_preconditioner_ = nullptr;
  /**
   * @brief Pre-solve callback passed to KSPSetPresolve
   *
   * @param ksp PETSc linear solver object
   * @param rhs Right-hand side vector for solve
   * @param x Solution/initial vector
   * @param ctx Context, set as @a this pointer
   *
   * @return Error code or 0 on success
   */

  friend PetscErrorCode convertKSPPreSolve(KSP ksp, [[maybe_unused]] Vec rhs, [[maybe_unused]] Vec x, void* ctx);

 protected:
  /**
   * @brief Set the tolerances on the underlying PETSc nonlinear operator
   */
  virtual void SetTolerances();

 public:
  /**
   * @brief Construct a wrapper for using a PETSc linear solver
   *
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param ksp_type Type of PETSc linear solver to construct, e.g. KSPCG or KSPGMRES
   * @param prefix Prefix for all PETSc command line options options
   * @param wrap Flag determining whether the mfem::Operator is wrapped or converted, defaults to false
   * @param iter_mode Flag controlling whether the second argument to Mult() is treated as an initial guess
   *
   * @note If @a wrap is true, then the MatMult ops of HypreParMatrix are wrapped.
   *       No preconditioner can be automatically constructed from PETSc.
   *       If @a wrap is false, the HypreParMatrix is converted into a the AIJ
   *       PETSc format, which is suitable for most preconditioning methods.
   * @note Additional arguments for the linear solver can be set via the command line, e.g. `-ksp_rtol 1e-8`.
   * Use `-ksp_type [ksp_type] -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscKSPSolver(MPI_Comm comm, KSPType ksp_type = KSPCG, const std::string& prefix = std::string(), bool wrap = false,
                 bool iter_mode = false);

  /**
   * @brief Constructs a solver using a HypreParMatrix.
   *
   * @param A The mfem::PetscParMatrix defining the linear system
   * @param ksp_type Type of PETSc linear solver to construct, e.g. KSPCG or KSPGMRES
   * @param prefix Prefix for all PETSc command line options options
   * @param iter_mode Flag controlling whether the second argument to Mult() is treated as an initial guess
   *
   * @note Additional arguments for the linear solver can be set via the command line, e.g. `-ksp_rtol 1e-8`.
   * Use `-ksp_type [ksp_type] -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscKSPSolver(const mfem::PetscParMatrix& A, KSPType ksp_type = KSPCG, const std::string& prefix = std::string(),
                 bool iter_mode = false);
  /**
   * @brief Constructs a solver using a HypreParMatrix.
   *
   * @param A The mfem::HypreParMatrix to use to construct the iterative linear solver
   * @param ksp_type Type of PETSc linear solver to construct, e.g. KSPCG or KSPGMRES
   * @param prefix Prefix for all PETSc command line options options
   * @param wrap Flag determining whether the mfem::Operator is wrapped or converted, defaults to false
   * @param iter_mode Flag controlling whether the second argument to Mult() is treated as an initial guess
   *
   * @note If @a wrap is true, then the MatMult ops of HypreParMatrix are wrapped.
   *       No preconditioner can be automatically constructed from PETSc.
   *       If @a wrap is false, the HypreParMatrix is converted into a the AIJ
   *       PETSc format, which is suitable for most preconditioning methods.
   * @note Additional arguments for the linear solver can be set via the command line, e.g. `-ksp_rtol 1e-8`.
   * Use `-ksp_type [ksp_type] -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscKSPSolver(const mfem::HypreParMatrix& A, KSPType ksp_type = KSPCG, const std::string& prefix = std::string(),
                 bool wrap = false, bool iter_mode = false);

  /**
   * @brief Get the MPI communicator
   * @return The MPI communicator used by the vectors and matrices in the solve
   */
  virtual MPI_Comm GetComm() const { return IterativeSolver::GetComm(); }

  /**
   * @brief Solve x = Op^{-1} b
   *
   * @param[in] b Right-hand side vector
   * @param[in,out] x Output solution vector (and initial guess if iterative mode is enabled)
   */
  virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

  /**
   * @brief Solve x = Op^{-T} b
   *
   * @param[in] b Right-hand side vector
   * @param[in,out] x Output solution vector (and initial guess if iterative mode is enabled)
   */
  virtual void MultTranspose(const mfem::Vector& b, mfem::Vector& x) const override;

  /**
   * @brief Set the underlying matrix operator to use in the solution algorithm
   *
   * @param op The matrix operator, either wrapped as a MATSHELL or converted to a PETSc MatType (often MATAIJ or
   * MATHYPRE)
   */
  virtual void SetOperator(const mfem::Operator& op) override;

  /**
   * @brief Set the preconditioner for the linear solver
   *
   * @param pc The preconditioner solver
   *
   * @note For best results, @a pc should be a PetscPCSolver. HYPRE-based mfem preconditioners also avoid convertion
   * costs, but do require re-wrapping the pointer to the interal HYPRE matrix upon each call to SetOperator().
   */
  virtual void SetPreconditioner(mfem::Solver& pc) override;

  /**
   * @brief Get the preconditioner, if set
   *
   * @return Pointer to preconditioner set with SetPreconditioner(), or nullptr if not set
   */
  virtual mfem::Solver* GetPreconditioner() { return prec; }

  void SetMaxIter(int max_its) { mfem::IterativeSolver::SetMaxIter(max_its); }
  void SetRelTol(mfem::real_t rtol) { mfem::IterativeSolver::SetRelTol(rtol); }
  void SetAbsTol(mfem::real_t atol) { mfem::IterativeSolver::SetAbsTol(atol); }
  int GetConverged() { return mfem::PetscLinearSolver::GetConverged(); }
  mfem::real_t GetFinalNorm() { return mfem::PetscLinearSolver::GetFinalNorm(); }
  int GetNumIterations() { return mfem::PetscLinearSolver::GetNumIterations(); }
  void SetPrintLevel(int print_lev) override { mfem::PetscLinearSolver::SetPrintLevel(print_lev); }
};

/**
 * @brief Wrapper for PETSc based nonlinear solvers
 */
class PetscNewtonSolver : public mfem::NewtonSolver, public mfem::PetscNonlinearSolver {
 protected:
  /// @brief Convergence tolerance for norm of solution update
  mfem::real_t step_tol_ = PETSC_DEFAULT;
  /// @brief Type of PETSc nonlinear solver
  SNESType snes_type_;
  /// @brief Linesearch type to use for PETSc nonlinear solver
  SNESLineSearchType linesearch_type_;
  /// @brief Nonlinear solver options
  NonlinearSolverOptions nonlinear_options_;

  static SNESType SNESTypeFromOptions(NonlinearSolverOptions nonlinear_opts)
  {
    switch (nonlinear_opts.nonlin_solver) {
      case NonlinearSolver::PetscNewton:
      case NonlinearSolver::PetscNewtonBacktracking:
      case NonlinearSolver::PetscNewtonCriticalPoint:
        return SNESNEWTONLS;
      case NonlinearSolver::PetscTrustRegion:
        return SNESNEWTONTR;
      default:
        return SNESNEWTONLS;
    }
  }

  static SNESLineSearchType SNESLineSearchTypeFromOptions(NonlinearSolverOptions nonlinear_opts)
  {
    switch (nonlinear_opts.nonlin_solver) {
      case NonlinearSolver::PetscNewton:
        return SNESLINESEARCHBASIC;
      case NonlinearSolver::PetscNewtonBacktracking:
        return SNESLINESEARCHBT;
      case NonlinearSolver::PetscNewtonCriticalPoint:
        return SNESLINESEARCHCP;
      case NonlinearSolver::PetscTrustRegion:
        return SNESLINESEARCHBASIC;
      default:
        return SNESLINESEARCHBASIC;
    }
  }

  /**
   * @brief Set the tolerances on the underlying PETSc nonlinear operator
   */
  virtual void SetTolerances();

  /**
   * @brief Wrap a non-PETSc linear solver in a PCSHELL
   * @param solver The solver to wrap
   */
  void SetNonPetscSolver(mfem::Solver& solver);

  /**
   * @brief Construct a PETSc nonlinear solver
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param snes_type The type of PETSc nonlinear solver to use, e.g. `SNESNEWTONLS`, `SNESNEWTONTR`, `SNESARCLENGTH`
   * @param linesearch_type The type of PETSc nonlinear solver to use, e.g. `SNESLINESEARCHBASIC` (full step),
   * `SNESLINESEARCHBT` (backtracking), `SNESLINESEARCHCP` (critical point)
   * @param prefix Prefix for all PETSc command line options options
   *
   * @note Additional arguments for the linear solver can be set via the command line, e.g. `-snes_rtol 1e-8`.
   * Use `-snes_type [snes_type] -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscNewtonSolver(MPI_Comm comm, SNESType snes_type = SNESNEWTONLS,
                    SNESLineSearchType linesearch_type = SNESLINESEARCHBASIC,
                    const std::string& prefix = std::string());

 public:
  /**
   * @brief Public constructor for PETSc nonlinear solvers from a NonlinearSolverOptions object
   * @param comm The MPI communicator used by the vectors and matrices in the solve
   * @param nonlinear_opts Options structure describing the solver type and tolerances
   * @param prefix Prefix for all PETSc command line options options
   *
   * @note Additional arguments for the linear solver can be set via the command line, e.g. `-snes_rtol 1e-8`.
   * Use `-snes_type [snes_type] -help` for a complete list of options.
   * @note If @a prefix is provided, the options should be set as `-[prefix]_[option] [value]`.
   */
  PetscNewtonSolver(MPI_Comm comm, NonlinearSolverOptions nonlinear_opts, const std::string& prefix = std::string());

  /**
   * @brief Set the linear solver for the algorithn
   * @param solver The linear solver to use
   */
  void SetSolver(mfem::Solver& solver) override;

  /**
   * @brief Set the type of PETSc nonlinear solver
   * @param snes_type The type of PETSc nonlinear solver to use
   *
   * @note For `SNESNEWTONLS` (Newton with line search), it is recommended to set the linesearch type with
   * SetLineSearchType()
   */
  void SetSNESType(SNESType snes_type);

  /**
   * @brief Set the linesearch type for the PETSc nonlinear solver
   * @param linesearch_type The type of linesearch to use
   */
  void SetLineSearchType(SNESLineSearchType linesearch_type);

  /**
   * @brief Solve F(x) = b
   * @param[in] b Right-hand side vector
   * @param[in,out] x Output solution vector and initial guess if iterative mode is enabled
   */
  void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

  /**
   * @brief Set nonlinear operator
   * @param[in] op Nonlinear operator, must implement the GetGradient() method
   */
  virtual void SetOperator(const mfem::Operator& op) override;

  /**
   * @brief Get the MPI communicator
   * @return The MPI communicator used by the vectors and matrices in the solve
   */
  MPI_Comm GetComm() const { return NewtonSolver::GetComm(); }

  void SetMaxIter(int max_its) { mfem::NewtonSolver::SetMaxIter(max_its); }
  void SetRelTol(mfem::real_t rtol) { mfem::NewtonSolver::SetRelTol(rtol); }
  void SetAbsTol(mfem::real_t atol) { mfem::NewtonSolver::SetAbsTol(atol); }
  int GetConverged() { return mfem::PetscNonlinearSolver::GetConverged(); }
  mfem::real_t GetFinalNorm() { return mfem::PetscNonlinearSolver::GetFinalNorm(); }
  int GetNumIterations() { return mfem::PetscNonlinearSolver::GetNumIterations(); }
  void SetPrintLevel(int print_lev) override { mfem::PetscNonlinearSolver::SetPrintLevel(print_lev); }
};

}  // namespace smith::mfem_ext

#endif
