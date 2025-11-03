// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/petsc_solvers.hpp"

#include <unordered_map>

#include "smith/infrastructure/logger.hpp"

#ifdef SMITH_USE_PETSC

#include "petsc/private/matimpl.h"
#include "petsc/private/linesearchimpl.h"
#include "petscmathypre.h"

namespace smith::mfem_ext {

// Static functions needed to create a shell PC

/// @brief Context to store wrapped solver object
typedef struct {
  /// @brief Wrapped solver object
  mfem::Solver* solver;

  /// @brief Flag indicating whether @a solver is owned by the preconditioner, generally false.
  bool owns_op;
} SolverWrapperCtx;

/**
 * @brief Callback for printing information about the preconditioner object to a viewer
 *
 * @param[in] pc Shell preconditioner object to print
 * @param[out] viewer Viewer to output information to
 *
 * @return Error code, or 0 on success.
 */
static PetscErrorCode solverWrapperView(PC pc, PetscViewer viewer)
{
  SolverWrapperCtx* ctx;

  PetscFunctionBeginUser;
  auto* void_ctx = static_cast<void*>(&ctx);
  PetscCall(PCShellGetContext(pc, &void_ctx));
  ctx = static_cast<SolverWrapperCtx*>(void_ctx);
  if (ctx->solver) {
    mfem::PetscPreconditioner* ppc = dynamic_cast<mfem::PetscPreconditioner*>(ctx->solver);
    if (ppc) {
      PetscCall(PCView(*ppc, viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Callback for applying the Mult() function of the wrapped solver
 *
 * @param[in,out] pc Shell preconditioner object containing the solver
 * @param[in] x The input (right-hand-side) vector
 * @param[out] y The output (solution) vector
 *
 * @return Error code, or 0 on success.
 */
static PetscErrorCode solverWrapperMult(PC pc, Vec x, Vec y)
{
  Mat A;
  PetscBool is_hypre;
  SolverWrapperCtx* ctx;

  PetscFunctionBeginUser;
  auto* void_ctx = static_cast<void*>(&ctx);
  PetscCall(PCShellGetContext(pc, &void_ctx));
  ctx = static_cast<SolverWrapperCtx*>(void_ctx);
  mfem::PetscParVector xx(x, true);
  mfem::PetscParVector yy(y, true);
  // Get the operator from the nonlinear solver and wrap as mfem::PetscParMatrix
  PetscCall(PCGetOperators(pc, nullptr, &A));
  PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), MATHYPRE, &is_hypre));
  std::unique_ptr<mfem::Operator> mat;
  // If the MatType is MATHYPRE, we should wrap as a HypreParMatrix for non-PETSc solvers
  if (is_hypre) {
    hypre_ParCSRMatrix* hypre_mat;
    PetscCall(MatHYPREGetParCSR(A, &hypre_mat));
    mat = std::make_unique<mfem::HypreParMatrix>(hypre_mat, false);
  } else {
    mat = std::make_unique<mfem::PetscParMatrix>(A, true);
  }
  ctx->solver->SetOperator(*mat);
  if (ctx->solver) {
    ctx->solver->Mult(xx, yy);
    yy.UpdateVecFromFlags();
  } else  // operator is not present, copy x
  {
    yy = xx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Callback for applying the MultTranspose() function of the wrapped solver
 *
 * @param[in,out] pc Shell preconditioner object containing the solver
 * @param[in] x The input (right-hand-side) vector
 * @param[out] y The output (solution) vector
 *
 * @return Error code, or 0 on success.
 */
static PetscErrorCode solverWrapperMultTranspose(PC pc, Vec x, Vec y)
{
  Mat A;
  PetscBool is_hypre;
  SolverWrapperCtx* ctx;

  PetscFunctionBeginUser;
  auto* void_ctx = static_cast<void*>(&ctx);
  PetscCall(PCShellGetContext(pc, &void_ctx));
  ctx = static_cast<SolverWrapperCtx*>(void_ctx);
  mfem::PetscParVector xx(x, true);
  mfem::PetscParVector yy(y, true);
  // Get the operator from the nonlinear solver and wrap as mfem::PetscParMatrix
  PetscCall(PCGetOperators(pc, nullptr, &A));
  PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), MATHYPRE, &is_hypre));
  std::unique_ptr<mfem::Operator> mat;
  // If the MatType is MATHYPRE, we should wrap as a HypreParMatrix for non-PETSc solvers
  if (is_hypre) {
    hypre_ParCSRMatrix* hypre_mat;
    PetscCall(MatHYPREGetParCSR(A, &hypre_mat));
    mat = std::make_unique<mfem::HypreParMatrix>(hypre_mat, false);
  } else {
    mat = std::make_unique<mfem::PetscParMatrix>(A, true);
  }
  ctx->solver->SetOperator(*mat);
  if (ctx->solver) {
    ctx->solver->MultTranspose(xx, yy);
    yy.UpdateVecFromFlags();
  } else  // operator is not present, copy x
  {
    yy = xx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Callback to destroy the SolverWrapperCtx object
 *
 * @param[in,out] pc Shell preconditioner object
 *
 * @return Error code, or 0 on success.
 */
static PetscErrorCode solverWrapperDestroy(PC pc)
{
  PetscFunctionBeginUser;
  void* void_ctx;
  PetscCall(PCShellGetContext(pc, &void_ctx));
  SolverWrapperCtx* ctx = static_cast<SolverWrapperCtx*>(void_ctx);
  if (ctx->owns_op) {
    delete ctx->solver;
  }
  delete ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Helper function to create a shell preconditioner wrapping a generic mfem::Solver
 *
 * @param[in,out] pc Preconditioner object to set up as shell
 * @param[in] solver Solver to wrap in the shell preconditioner
 * @param[in] owns_op Flag specifying whether the solver object should be destroyed with the preconditioner
 *
 * @return Error code, or 0 on success.
 */
static PetscErrorCode wrapSolverInShellPC(PC pc, mfem::Solver& solver, bool owns_op = false)
{
  PetscFunctionBeginUser;
  SolverWrapperCtx* ctx = new SolverWrapperCtx{&solver, owns_op};

  // In case the PC was already of type SHELL, this will destroy any
  // previous user-defined data structure
  // We cannot call PCReset as it will wipe out any operator already set
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetName(pc, "MFEM Solver"));
  PetscCall(PCShellSetContext(pc, static_cast<void*>(ctx)));
  PetscCall(PCShellSetApply(pc, solverWrapperMult));
  PetscCall(PCShellSetApplyTranspose(pc, solverWrapperMultTranspose));
  PetscCall(PCShellSetView(pc, solverWrapperView));
  PetscCall(PCShellSetDestroy(pc, solverWrapperDestroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscPCSolver methods

PetscErrorCode convertPCPreSolve(PC pc, [[maybe_unused]] KSP ksp)
{
  PetscPCSolver* solver;
  Mat A;
  void* void_solver;

  PetscFunctionBeginUser;
  PetscCall(PCGetApplicationContext(pc, &void_solver));
  solver = static_cast<PetscPCSolver*>(void_solver);
  // If this function is called, we have a PETSc preconditioner
  // That means we have to ensure the matrix is MATAIJ
  if (!solver->checked_for_convert_ || solver->converted_matrix_) {
    PetscCall(PCGetOperators(pc, nullptr, &A));
    char* found;
    MatType mat_type;
    PetscCall(MatGetType(A, &mat_type));
    PetscCall(PetscStrstr(mat_type, "aij", &found));
    if (found) {
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    SLIC_DEBUG_ROOT("convertPCPreSolve(...) - Converting operators to MATAIJ format.");
    mfem::PetscParMatrix temp_mat(A, true);
    solver->converted_matrix_ = std::make_unique<mfem::PetscParMatrix>(temp_mat, mfem::Operator::PETSC_MATAIJ);
    PetscCall(PCSetOperators(pc, *solver->converted_matrix_, *solver->converted_matrix_));
  }
  solver->checked_for_convert_ = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscPCSolver::PetscPCSolver(MPI_Comm comm_, PCType pc_type, const std::string& prefix)
    : PetscPreconditioner(comm_, prefix)
{
  PetscCallAbort(GetComm(), PCSetType(*this, pc_type));
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, convertPCPreSolve));
  clcustom = false;
  Customize();
}

PetscPCSolver::PetscPCSolver(mfem::PetscParMatrix& A, PCType pc_type, const std::string& prefix)
    : PetscPreconditioner(A, prefix)
{
  PetscCallAbort(GetComm(), PCSetType(*this, pc_type));
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, convertPCPreSolve));
  clcustom = false;
  Customize();
}

PetscPCSolver::PetscPCSolver(MPI_Comm comm_, Operator& op, PCType pc_type, const std::string& prefix)
    : PetscPreconditioner(comm_, op, prefix)
{
  PetscCallAbort(GetComm(), PCSetType(*this, pc_type));
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, convertPCPreSolve));
  clcustom = false;
  Customize();
}

void PetscPCSolver::SetOperator(const mfem::Operator& op)
{
  PetscBool is_nest;
  bool delete_pA = false;

  auto* pA = const_cast<mfem::PetscParMatrix*>(dynamic_cast<const mfem::PetscParMatrix*>(&op));
  if (!pA) {
    pA = new mfem::PetscParMatrix(GetComm(), &op, PETSC_MATAIJ);
    delete_pA = true;
  }
  PetscCallAbort(GetComm(), PetscObjectTypeCompare(*pA, MATNEST, &is_nest));
  if (!is_nest) {
    SLIC_DEBUG_ROOT("Not a MATNEST, setting operator directly");
    smith::logger::flush();
    fieldsplit_pc_.reset();
    mfem::PetscPreconditioner::SetOperator(*pA);
    if (delete_pA) {
      delete pA;
    }
    return;
  }
  SLIC_DEBUG_ROOT("MATNEST detected, setting up fieldsplit");

  if (fieldsplit_pc_) {
    SLIC_DEBUG_ROOT("Fieldsplit exists, setting operator");
    fieldsplit_pc_->SetOperator(*pA);
  } else {
    SLIC_DEBUG_ROOT("Creating fieldsplit");
    fieldsplit_pc_ = std::make_unique<mfem::PetscFieldSplitSolver>(GetComm(), *pA, "nest");
    PetscCallAbort(GetComm(), PCFieldSplitSetType(*fieldsplit_pc_, PC_COMPOSITE_ADDITIVE));
    PetscCallAbort(GetComm(), PCSetFromOptions(*fieldsplit_pc_));
    PetscCallAbort(GetComm(), PCSetUp(*fieldsplit_pc_));
    KSP* sub_ksps;
    PetscInt n = 1;
    PetscCallAbort(GetComm(), PCFieldSplitGetSubKSP(*fieldsplit_pc_, &n, &sub_ksps));
    PetscCallAbort(GetComm(), KSPSetPC(sub_ksps[0], *this));
    PC sub_pc_1;
    PetscCallAbort(GetComm(), KSPGetPC(sub_ksps[1], &sub_pc_1));
    PetscCallAbort(GetComm(), PCSetType(sub_pc_1, PCNONE));
    PetscCallAbort(GetComm(), KSPSetFromOptions(sub_ksps[1]));
    PetscCallAbort(GetComm(), PetscFree(sub_ksps));
  }
  Mat A11;
  PetscCallAbort(GetComm(), MatNestGetSubMat(*pA, 0, 0, &A11));
  mfem::PetscParMatrix pA11(A11, true);
  mfem::PetscPreconditioner::SetOperator(pA11);
  Mat A22;
  Vec zero;
  PetscCallAbort(GetComm(), MatNestGetSubMat(*pA, 1, 1, &A22));
  // Make sure all diagonal elements are set
  PetscCallAbort(GetComm(), MatSetOption(A22, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCallAbort(GetComm(), MatCreateVecs(A22, &zero, nullptr));
  PetscCallAbort(GetComm(), VecSet(zero, 0));
  PetscCallAbort(GetComm(), MatDiagonalSet(A22, zero, ADD_VALUES));
  if (delete_pA) {
    delete pA;
  }
  smith::logger::flush();
}

void PetscPCSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const
{
  if (fieldsplit_pc_) {
    fieldsplit_pc_->Mult(b, x);
  } else {
    mfem::PetscPreconditioner::Mult(b, x);
  }
}

void PetscPCSolver::MultTranspose(const mfem::Vector& b, mfem::Vector& x) const
{
  if (fieldsplit_pc_) {
    fieldsplit_pc_->MultTranspose(b, x);
  } else {
    mfem::PetscPreconditioner::MultTranspose(b, x);
  }
}

// PetscPreconditionerSpaceDependent methods

void PetscPreconditionerSpaceDependent::SetOperator(const Operator& op)
{
  // Update parent class
  PetscPCSolver::SetOperator(op);
  SLIC_WARNING_ROOT_IF(
      !fespace_,
      "Finite element space not set with SetFESpace() method, expect performance and/or convergence issues.");
  if (fespace_) {
    Mat pA, ppA;
    PetscCallAbort(GetComm(), PCGetOperators(*this, nullptr, &ppA));
    int vdim = fespace_->GetVDim();

    // Ideally, the block size should be set at matrix creation
    // but the MFEM assembly does not allow us to do so
    PetscCallAbort(GetComm(), MatSetBlockSize(ppA, vdim));
    PetscCallAbort(GetComm(), PCGetOperators(*this, &pA, nullptr));
    if (ppA != pA) {
      PetscCallAbort(GetComm(), MatSetBlockSize(pA, vdim));
    }
  }
}

// PetscGAMGSolver methods

static PetscErrorCode gamg_pre_solve(PC pc, KSP ksp)
{
  PetscGAMGSolver* solver;
  void* void_solver;

  PetscFunctionBeginUser;
  PetscCall(convertPCPreSolve(pc, ksp));
  PetscCall(PCGetApplicationContext(pc, &void_solver));
  solver = static_cast<PetscGAMGSolver*>(void_solver);
  solver->SetupNearNullSpace();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscGAMGSolver::PetscGAMGSolver(MPI_Comm& comm_, const std::string& prefix)
    : PetscPreconditionerSpaceDependent(comm_, PCGAMG, prefix)
{
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, gamg_pre_solve));
  Customize();
}

PetscGAMGSolver::PetscGAMGSolver(mfem::PetscParMatrix& A, const std::string& prefix)
    : PetscPreconditionerSpaceDependent(A, PCGAMG, prefix)
{
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, gamg_pre_solve));
}

PetscGAMGSolver::PetscGAMGSolver(MPI_Comm comm_, Operator& op, const std::string& prefix)
    : PetscPreconditionerSpaceDependent(comm_, op, PCGAMG, prefix)
{
  PetscCallAbort(GetComm(), PCSetApplicationContext(*this, this));
  PetscCallAbort(GetComm(), PCSetPreSolve(*this, gamg_pre_solve));
}

static void func_coords(const mfem::Vector& x, mfem::Vector& y) { y = x; }

void PetscGAMGSolver::SetupNearNullSpace()
{
  Mat pA;
  PetscCallAbort(GetComm(), PCGetOperators(*this, nullptr, &pA));
  MatNullSpace nnsp;
  PetscCallAbort(GetComm(), MatGetNearNullSpace(pA, &nnsp));
  if (!fespace_ || nnsp) {
    return;
  }

  // get PETSc object
  PC pc = *this;

  PetscBool is_op_set;
  PetscCallAbort(GetComm(), PCGetOperatorsSet(pc, nullptr, &is_op_set));
  if (!is_op_set) {
    return;
  }

  // coordinates
  const mfem::FiniteElementCollection* fec = fespace_->FEColl();
  bool h1space = dynamic_cast<const mfem::H1_FECollection*>(fec);
  if (h1space) {
    PetscInt sdim = fespace_->GetParMesh()->SpaceDimension();
    int vdim = fespace_->GetVDim();

    SLIC_DEBUG_ROOT("PetscGAMGSolver::SetupNearNullSpace(...) - Setting up near null space");
    mfem::ParFiniteElementSpace* fespace_coords = fespace_;

    if (vdim != sdim || fespace_->GetOrdering() != mfem::Ordering::byVDIM) {
      SLIC_WARNING_ROOT(
          "PetscGAMGSolver::SetupNearNullSpace(...) - Wrong displacement finite element space ordering - should be "
          "byVDIM");
      fespace_coords = new mfem::ParFiniteElementSpace(fespace_->GetParMesh(), fec, vdim, mfem::Ordering::byVDIM);
    }
    mfem::VectorFunctionCoefficient coeff_coords(sdim, func_coords);
    mfem::ParGridFunction gf_coords(fespace_coords);
    gf_coords.ProjectCoefficient(coeff_coords);
    mfem::HypreParVector* hvec_coords = gf_coords.ParallelProject();
    auto data_coords = const_cast<PetscScalar*>(mfem::Read(hvec_coords->GetMemory(), hvec_coords->Size(), false));

    Vec pvec_coords;
    PetscCallAbort(GetComm(), VecCreateMPIWithArray(GetComm(), sdim, hvec_coords->Size(),
                                                    fespace_coords->GlobalTrueVSize(), data_coords, &pvec_coords));
    PetscCallAbort(GetComm(), MatNullSpaceCreateRigidBody(pvec_coords, &nnsp));
    PetscCallAbort(GetComm(), MatSetNearNullSpace(pA, nnsp));
    PetscCallAbort(GetComm(), MatNullSpaceDestroy(&nnsp));
    PetscCallAbort(GetComm(), VecDestroy(&pvec_coords));
    if (fespace_coords != fespace_) {
      delete fespace_coords;
    }
    delete hvec_coords;
  }
  PetscCallAbort(GetComm(), MatGetNearNullSpace(pA, &nnsp));
  SLIC_WARNING_ROOT_IF(!nnsp, "Global near null space was not set successfully, expect slow (or no) convergence.");
  SLIC_DEBUG_ROOT_IF(nnsp, "PetscGAMGSolver::SetupNearNullSpace(...) - Near null space set successfully.");
}

void PetscGAMGSolver::SetOperator(const Operator& op)
{
  // Update parent class
  PetscPreconditionerSpaceDependent::SetOperator(op);
  // Set rigid body near null space
  if (fespace_) {
    SetupNearNullSpace();
  }
}

// Helper functions

std::unique_ptr<PetscPCSolver> buildPetscPreconditioner(PetscPCType pc_type, MPI_Comm comm)
{
  std::unique_ptr<PetscPCSolver> preconditioner;
  switch (pc_type) {
    case PetscPCType::JACOBI:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCJACOBI);
      PetscCallAbort(comm, PCJacobiSetType(*preconditioner, PC_JACOBI_DIAGONAL));
      break;
    case PetscPCType::JACOBI_L1:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCJACOBI);
      PetscCallAbort(comm, PCJacobiSetType(*preconditioner, PC_JACOBI_ROWL1));
      break;
    case PetscPCType::JACOBI_ROWMAX:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCJACOBI);
      PetscCallAbort(comm, PCJacobiSetType(*preconditioner, PC_JACOBI_ROWMAX));
      break;
    case PetscPCType::JACOBI_ROWSUM:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCJACOBI);
      PetscCallAbort(comm, PCJacobiSetType(*preconditioner, PC_JACOBI_ROWSUM));
      break;
    case PetscPCType::PBJACOBI:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCPBJACOBI);
      break;
    case PetscPCType::BJACOBI:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCBJACOBI);
      break;
    case PetscPCType::LU:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCLU);
      // Automatically shift the LU factorization to ensure positive definiteness
      PetscCallAbort(comm, PCFactorSetShiftType(*preconditioner, MAT_SHIFT_POSITIVE_DEFINITE));
      break;
    case PetscPCType::ILU:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCILU);
      // Automatically shift the ILU factorization to ensure positive definiteness
      PetscCallAbort(comm, PCFactorSetShiftType(*preconditioner, MAT_SHIFT_POSITIVE_DEFINITE));
      break;
    case PetscPCType::CHOLESKY:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCCHOLESKY);
      // Automatically shift the ILU factorization to ensure positive definiteness
      PetscCallAbort(comm, PCFactorSetShiftType(*preconditioner, MAT_SHIFT_POSITIVE_DEFINITE));
      break;
    case PetscPCType::SVD:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCSVD);
      break;
    case PetscPCType::ASM:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCASM);
      break;
    case PetscPCType::GASM:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCGASM);
      break;
    case PetscPCType::HMG: {
      preconditioner = std::make_unique<PetscPreconditionerSpaceDependent>(comm, PCHMG);
      // Coarsen using component-based subspaces
      PetscCallAbort(comm, PCHMGSetUseSubspaceCoarsening(*preconditioner, PETSC_TRUE));
      // Reuse interpolation matrices to speed up computation
      PetscCallAbort(comm, PCHMGSetReuseInterpolation(*preconditioner, PETSC_TRUE));
      // Use GAMG for the inner preconditioner (faster)
      PetscCallAbort(comm, PCHMGSetInnerPCType(*preconditioner, PCGAMG));
      PetscCallAbort(comm, PCSetFromOptions(*preconditioner));
      break;
    }
    case PetscPCType::GAMG: {
      // Special type, as we need to attach near null space
      preconditioner = std::make_unique<PetscGAMGSolver>(comm);
      // Automatically shift the LU factorization to ensure positive definiteness
      PetscOptionsInsertString(nullptr, "-mg_coarse_sub_pc_factor_shift_type positive_definite");
      PetscCallAbort(comm, PCSetFromOptions(*preconditioner));
      break;
    }
    case PetscPCType::NONE:
      preconditioner = std::make_unique<PetscPCSolver>(comm, PCNONE);
      break;
  }
  // Allow further customization via command-line arguments (PETSc has many)
  preconditioner->Customize();
  return preconditioner;
}

PetscPCType stringToPetscPCType(const std::string& type_str)
{
  std::unordered_map<std::string, PetscPCType> types{
      {"jacobi", PetscPCType::JACOBI},
      {"jacobi_l1", PetscPCType::JACOBI_L1},
      {"jacobi_rowsum", PetscPCType::JACOBI_ROWSUM},
      {"jacobi_rowmax", PetscPCType::JACOBI_ROWMAX},
      {"pbjacobi", PetscPCType::PBJACOBI},
      {"bjacobi", PetscPCType::BJACOBI},
      {"lu", PetscPCType::LU},
      {"ilu", PetscPCType::ILU},
      {"cholesky", PetscPCType::CHOLESKY},
      {"svd", PetscPCType::SVD},
      {"asm", PetscPCType::ASM},
      {"gasm", PetscPCType::GASM},
      {"gamg", PetscPCType::GAMG},
      {"hmg", PetscPCType::HMG},
      {"none", PetscPCType::NONE},
  };
  return types.at(type_str);
}

// PetscKSPSolver methods

PetscErrorCode convertKSPPreSolve(KSP ksp, [[maybe_unused]] Vec rhs, [[maybe_unused]] Vec x, void* ctx)
{
  PetscKSPSolver* solver;
  Mat A;
  PetscBool is_nest;

  PetscFunctionBeginUser;
  solver = static_cast<PetscKSPSolver*>(ctx);
  auto* prec = solver->prec;
  PetscPCSolver* petsc_pc = dynamic_cast<PetscPCSolver*>(prec);
  PetscCall(KSPGetOperators(ksp, nullptr, &A));
  if (petsc_pc) {
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), MATNEST, &is_nest));
    PC pc_orig;
    PetscBool is_fieldsplit;
    PetscCall(KSPGetPC(ksp, &pc_orig));
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(pc_orig), PCFIELDSPLIT, &is_fieldsplit));
    if (is_nest && !is_fieldsplit) {
      SLIC_ERROR_ROOT_IF(!petsc_pc, "MATNEST only supported for PETSc preconditioners");
      SLIC_DEBUG_ROOT_IF(is_nest, "convertKSPPreSolve(...) - Using MATNEST, must set up fieldsplit preconditioner.");
      smith::logger::flush();
      mfem::PetscParMatrix pA(A, true);
      petsc_pc->SetOperator(pA);
      SLIC_ERROR_ROOT_IF(!petsc_pc->fieldsplit_pc_, "Fieldsplit was not created successfully.");
      PetscCall(KSPSetPC(ksp, *petsc_pc->fieldsplit_pc_));
      PetscFunctionReturn(PETSC_SUCCESS);
    } else if (is_fieldsplit && !is_nest) {
      PetscCall(KSPSetPC(ksp, *petsc_pc));
    }
  }
  if (!solver->checked_for_convert_ || solver->needs_hypre_wrapping_) {
    PetscBool is_hypre;
    PetscCall(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A), MATHYPRE, &is_hypre));
    SLIC_DEBUG_ROOT_IF(
        is_hypre && petsc_pc,
        "convertKSPPreSolve(...) - MATHYPRE is not supported for most PETSc preconditioners, converting to MATAIJ.");
    if (!is_hypre || petsc_pc) {
      solver->checked_for_convert_ = true;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    hypre_ParCSRMatrix *hypre_csr = nullptr, *old_hypre_csr = nullptr;
    PetscCall(MatHYPREGetParCSR(A, &hypre_csr));
    if (solver->wrapped_matrix_) {
      old_hypre_csr = *solver->wrapped_matrix_;
    }
    if (old_hypre_csr != hypre_csr || !solver->wrapped_matrix_) {
      SLIC_DEBUG_ROOT("convertKSPPreSolve(...) - Rebuilding HypreParMatrix wrapper");
      solver->wrapped_matrix_ = std::make_unique<mfem::HypreParMatrix>(hypre_csr, false);
    }
    SLIC_DEBUG_ROOT("convertKSPPreSolve(...) - Setting operator for preconditioner");
    if (prec) {
      prec->SetOperator(*solver->wrapped_matrix_);
    }
    solver->needs_hypre_wrapping_ = true;
  }
  solver->checked_for_convert_ = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscKSPSolver::PetscKSPSolver(MPI_Comm comm_, KSPType ksp_type, const std::string& prefix, bool wrap_op,
                               bool iter_mode)
    : mfem::IterativeSolver(comm_), mfem::PetscLinearSolver(comm_, prefix, wrap_op, iter_mode)
{
  abs_tol = PETSC_DEFAULT;
  rel_tol = PETSC_DEFAULT;
  max_iter = PETSC_DEFAULT;
  PetscCallAbort(GetComm(), KSPConvergedDefaultSetConvergedMaxits(*this, PETSC_TRUE));
  PetscCallAbort(GetComm(), KSPSetType(*this, ksp_type));
  PetscCallAbort(GetComm(), KSPSetPreSolve(*this, convertKSPPreSolve, this));
  clcustom = false;
  Customize();
}

PetscKSPSolver::PetscKSPSolver(const mfem::PetscParMatrix& A, KSPType ksp_type, const std::string& prefix,
                               bool iter_mode)
    : mfem::IterativeSolver(A.GetComm()), mfem::PetscLinearSolver(A, prefix, iter_mode), wrap_(false)
{
  abs_tol = PETSC_DEFAULT;
  rel_tol = PETSC_DEFAULT;
  max_iter = PETSC_DEFAULT;
  PetscCallAbort(GetComm(), KSPConvergedDefaultSetConvergedMaxits(*this, PETSC_TRUE));
  PetscCallAbort(GetComm(), KSPSetType(*this, ksp_type));
  PetscCallAbort(GetComm(), KSPSetPreSolve(*this, convertKSPPreSolve, this));
  clcustom = false;
  Customize();
}

PetscKSPSolver::PetscKSPSolver(const mfem::HypreParMatrix& A, KSPType ksp_type, const std::string& prefix, bool wrap_op,
                               bool iter_mode)
    : mfem::IterativeSolver(A.GetComm()), mfem::PetscLinearSolver(A, wrap_op, prefix, iter_mode), wrap_(wrap_op)
{
  abs_tol = PETSC_DEFAULT;
  rel_tol = PETSC_DEFAULT;
  max_iter = PETSC_DEFAULT;
  PetscCallAbort(GetComm(), KSPConvergedDefaultSetConvergedMaxits(*this, PETSC_TRUE));
  PetscCallAbort(GetComm(), KSPSetType(*this, ksp_type));
  PetscCallAbort(GetComm(), KSPSetPreSolve(*this, convertKSPPreSolve, this));
  clcustom = false;
  Customize();
}

void PetscKSPSolver::SetTolerances()
{
  PetscCallAbort(GetComm(), KSPSetTolerances(*this, rel_tol, abs_tol, PETSC_DEFAULT, max_iter));
}

void PetscKSPSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const { mfem::PetscLinearSolver::Mult(b, x); }

void PetscKSPSolver::MultTranspose(const mfem::Vector& b, mfem::Vector& x) const
{
  mfem::PetscLinearSolver::MultTranspose(b, x);
}

void PetscKSPSolver::SetOperator(const mfem::Operator& op)
{
  const mfem::HypreParMatrix* hA = dynamic_cast<const mfem::HypreParMatrix*>(&op);
  mfem::PetscParMatrix* pA = const_cast<mfem::PetscParMatrix*>(dynamic_cast<const mfem::PetscParMatrix*>(&op));
  const mfem::Operator* oA = dynamic_cast<const mfem::Operator*>(&op);

  // set tolerances from user
  SetTolerances();

  // Check if preconditioner can use HYPRE matrices
  PetscPCSolver* petsc_pc = dynamic_cast<PetscPCSolver*>(prec);

  // delete existing matrix, if created
  if (pA_) {
    delete pA_;
  }
  pA_ = nullptr;
  // update base classes: Operator, Solver, PetscLinearSolver
  if (!pA) {
    if (hA) {
      // Create MATSHELL object or convert into a format suitable to construct preconditioners
      if (PETSC_HAVE_HYPRE && !petsc_pc) {
        SLIC_DEBUG_ROOT("PetscKSPSolver::SetOperator(...) - Wrapping existing HYPRE matrix");
        pA = new mfem::PetscParMatrix(hA, wrap_ ? PETSC_MATSHELL : PETSC_MATHYPRE);
      } else {
        SLIC_DEBUG_ROOT("PetscKSPSolver::SetOperator(...) - Converting operator from HYPRE to MATAIJ");
        pA = new mfem::PetscParMatrix(hA, wrap_ ? PETSC_MATSHELL : PETSC_MATAIJ);
      }
    } else if (oA) {
      // fallback to general operator
      // Create MATSHELL or MATNEST (if oA is a BlockOperator) object
      // If oA is a BlockOperator, Operator::Type is relevant to the subblocks
      SLIC_WARNING_ROOT(
          "PetscKSPSolver::SetOperator(...) - Converting operator, consider using PetscParMatrix to avoid conversion "
          "costs");
      pA = new mfem::PetscParMatrix(GetComm(), oA, wrap_ ? PETSC_MATSHELL : PETSC_MATAIJ);
    }
    pA_ = pA;
  }
  smith::logger::flush();
  MFEM_VERIFY(pA, "PetscKSPSolver::SetOperator(...) - Unsupported operation!");

  // Set operators into PETSc KSP
  KSP ksp = *this;
  Mat A = *pA;
  if (operatorset) {
    Mat C;
    PetscInt nheight, nwidth, oheight, owidth;

    PetscCallAbort(GetComm(), KSPGetOperators(ksp, &C, nullptr));
    PetscCallAbort(GetComm(), MatGetSize(A, &nheight, &nwidth));
    PetscCallAbort(GetComm(), MatGetSize(C, &oheight, &owidth));
    if (nheight != oheight || nwidth != owidth) {
      // reinit without destroying the KSP
      // communicator remains the same
      SLIC_WARNING_ROOT("PetscKSPSolver::SetOperator(...) - Rebuilding KSP");
      PetscCallAbort(GetComm(), KSPReset(ksp));
      delete X;
      delete B;
      X = B = nullptr;
    }
  }
  PetscBool is_nest;
  MatType type;
  MatGetType(*pA, &type);
  PetscCallAbort(GetComm(), PetscObjectTypeCompare(*pA, MATNEST, &is_nest));
  SLIC_DEBUG_ROOT(axom::fmt::format("PetscKSPSolver::SetOperator(...) - Mat type: {}", type));

  PetscObjectTypeCompare(*pA, MATNEST, &is_nest);
  SLIC_DEBUG_ROOT_IF(is_nest, "Using MATNEST");
  smith::logger::flush();

  // mfem::PetscParMatrix op_wrapped(GetComm(), &op);
  PetscCallAbort(GetComm(), KSPSetOperators(ksp, A, A));

  // Update PetscSolver
  operatorset = true;

  // Update the Operator fields.
  IterativeSolver::height = pA->Height();
  PetscLinearSolver::height = pA->Height();
  IterativeSolver::width = pA->Width();
  PetscLinearSolver::width = pA->Width();

  if (petsc_pc) {
    prec->SetOperator(*pA);
    if (is_nest) {
      SLIC_ERROR_IF(!petsc_pc->fieldsplit_pc_, "Failed to create fieldsplit preconditioner");
      PetscCallAbort(GetComm(), KSPSetPC(*this, *petsc_pc->fieldsplit_pc_));
    }
  } else if (prec) {
    prec->SetOperator(op);
  }
}

void PetscKSPSolver::SetPreconditioner(mfem::Solver& pc)
{
  mfem::PetscLinearSolver::SetPreconditioner(pc);
  prec = &pc;
}

// PetscNewtonSolver methods

/**
 * @brief Callback function passed to SNESLineSearchSetPreCheck which backs-off step size to prevent nan or inf values
 *
 * @param[in] linesearch Line search context
 * @param[in] X Solution vector pre-step
 * @param[in,out] Y Candidate Newton step
 * @param[out] changed Output flag indicating whether the step @a Y was changed
 * @param[in] ctx Context passed to SNESLineSearchSetPreCheck, unused
 *
 * @return Error code, or 0 on success.
 */
PetscErrorCode linesearchPreCheckBackoffOnNan(SNESLineSearch linesearch, Vec X, Vec Y, PetscBool* changed,
                                              [[maybe_unused]] void* ctx)
{
  SNES snes;
  PetscReal lambda_orig, lambda, min_lambda;
  PetscScalar fty;
  PetscInt max_failures, num_failures = 0;
  Vec W, Ftemp;
  PetscViewer monitor;
  PetscObject linesearch_obj = reinterpret_cast<PetscObject>(linesearch);

  PetscFunctionBeginUser;
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchGetVecs(linesearch, nullptr, nullptr, nullptr, &W, &Ftemp));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda_orig));
  lambda = lambda_orig;
  PetscCall(SNESGetMaxNonlinearStepFailures(snes, &max_failures));
  PetscCall(SNESLineSearchGetTolerances(linesearch, &min_lambda, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCall(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));
  // If -snes_max_fail 0, don't check at all
  // This is faster, but will fail if the step leads to a nan or inf value
  while (num_failures++ < max_failures) {
    PetscCall(VecWAXPY(W, -lambda_orig, Y, X));
    if (linesearch->ops->viproject) {
      PetscCall((*linesearch->ops->viproject)(snes, W));
    }
    PetscCall((*linesearch->ops->snesfunc)(snes, W, Ftemp));
    PetscCall(VecDot(Ftemp, Y, &fty));
    if (!PetscIsInfOrNanScalar(fty)) {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, linesearch_obj->tablevel));
        auto msg = axom::fmt::format("    Line search: dot(F,Y) = {}, no back-off steps needed", fty);
        PetscCall(PetscViewerASCIIPrintf(monitor, "%s\n", msg.c_str()));
        PetscCall(PetscViewerASCIISubtractTab(monitor, linesearch_obj->tablevel));
      }
      break;
    }
    lambda *= 0.5;
    if (lambda < min_lambda) {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, linesearch_obj->tablevel));
        auto msg =
            axom::fmt::format("    Line search: step size too small ({} < {}) after {} failures, exiting recovery",
                              lambda, min_lambda, num_failures);
        PetscCall(PetscViewerASCIIPrintf(monitor, "%s\n", msg.c_str()));
        PetscCall(PetscViewerASCIISubtractTab(monitor, linesearch_obj->tablevel));
      }
      break;
    }
    if (monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, linesearch_obj->tablevel));
      auto msg = axom::fmt::format(
          "    Line search: dot(F,Y) = {}, scaling back step size to {} ({} of a maximum {} back-off steps)", fty,
          lambda, num_failures, max_failures);
      PetscCall(PetscViewerASCIIPrintf(monitor, "%s\n", msg.c_str()));
      PetscCall(PetscViewerASCIISubtractTab(monitor, linesearch_obj->tablevel));
    }
    PetscCall(VecScale(Y, 0.5));
    *changed = PETSC_TRUE;
  }
  // If we didn't find a sufficiently small step, PETSc will fail for us
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscNewtonSolver::PetscNewtonSolver(MPI_Comm comm_, SNESType snes_type, SNESLineSearchType linesearch_type,
                                     const std::string& prefix)
    : mfem::NewtonSolver(comm_),
      mfem::PetscNonlinearSolver(comm_, prefix),
      snes_type_(snes_type),
      linesearch_type_(linesearch_type)
{
  rel_tol = PETSC_DEFAULT;
  abs_tol = PETSC_DEFAULT;
  max_iter = PETSC_DEFAULT;
  SetJacobianType(ANY_TYPE);
  PetscCallVoid(SNESSetType(*this, snes_type_));
  PetscCallVoid(SNESSetMaxNonlinearStepFailures(*this, 5));
  clcustom = false;
  Customize();
  NewtonSolver::iterative_mode = PetscNonlinearSolver::iterative_mode = true;
}

PetscNewtonSolver::PetscNewtonSolver(MPI_Comm comm_, NonlinearSolverOptions nonlinear_opts, const std::string& prefix)
    : PetscNewtonSolver(comm_, SNESTypeFromOptions(nonlinear_opts), SNESLineSearchTypeFromOptions(nonlinear_opts),
                        prefix)
{
  nonlinear_options_ = nonlinear_opts;
  SetAbsTol(nonlinear_options_.absolute_tol);
  SetRelTol(nonlinear_options_.relative_tol);
  SetMaxIter(nonlinear_options_.max_iterations);
  SetPrintLevel(nonlinear_options_.print_level);

  rel_tol = nonlinear_options_.relative_tol;
  abs_tol = nonlinear_options_.absolute_tol;
  max_iter = nonlinear_options_.max_iterations;
  if (nonlinear_options_.min_iterations > 0) {
    PetscCallAbort(GetComm(), SNESSetForceIteration(*this, PETSC_TRUE));
  }
}

void PetscNewtonSolver::SetTolerances()
{
  PetscCallAbort(GetComm(), SNESSetTolerances(*this, abs_tol, rel_tol, step_tol_, max_iter, PETSC_DEFAULT));
  // Fix specifically the absolute tolerance for CP linesearch, since a PETSc bug will erroneously lead to early
  // "convergence". See: https://gitlab.com/petsc/petsc/-/issues/1583
  if (operatorset) {
    PetscBool is_newtonls, is_cp;
    SNESLineSearch linesearch;
    PetscCallAbort(GetComm(), PetscObjectTypeCompare(*this, SNESNEWTONLS, &is_newtonls));
    PetscCallAbort(GetComm(), SNESGetLineSearch(*this, &linesearch));
    PetscCallAbort(GetComm(),
                   PetscObjectTypeCompare(reinterpret_cast<PetscObject>(linesearch), SNESLINESEARCHCP, &is_cp));

    auto max_ls_iters = nonlinear_options_.max_line_search_iterations > 0
                            ? nonlinear_options_.max_line_search_iterations
                            : PETSC_DEFAULT;
    auto abs_ls_tol = is_newtonls && is_cp ? 1e-30 : PETSC_DEFAULT;
    // min step, max step, rel tol, abs tol, delta step tol, max iters
    PetscCallAbort(GetComm(), SNESLineSearchSetTolerances(linesearch, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT,
                                                          abs_ls_tol, PETSC_DEFAULT, max_ls_iters));
    // ensure we don't fail immediately if a nan occurs
    PetscCallAbort(GetComm(), SNESLineSearchSetPreCheck(linesearch, linesearchPreCheckBackoffOnNan, nullptr));
  }
}

void PetscNewtonSolver::SetNonPetscSolver(mfem::Solver& solver)
{
  // Set the KSP object associated with the SNES to be PREONLY so no linear solver is used
  KSP ksp;
  PetscCallAbort(GetComm(), SNESGetKSP(*this, &ksp));
  PetscCallAbort(GetComm(), KSPSetType(ksp, KSPPREONLY));
  // Place the non-PETSc solver into a shell PC
  PC pc_shell;
  PetscCallAbort(GetComm(), KSPGetPC(ksp, &pc_shell));
  PetscCallAbort(GetComm(), wrapSolverInShellPC(pc_shell, solver, false));
}

void PetscNewtonSolver::SetLineSearchType(SNESLineSearchType linesearch_type)
{
  linesearch_type_ = linesearch_type;
  if (operatorset) {
    SNESLineSearch linesearch;
    PetscCallAbort(GetComm(), SNESGetLineSearch(*this, &linesearch));
    PetscCallAbort(GetComm(), SNESLineSearchSetType(linesearch, linesearch_type_));
  }
  SetTolerances();
  clcustom = false;
  Customize();
}

void PetscNewtonSolver::SetSNESType(SNESType snes_type)
{
  snes_type_ = snes_type;
  PetscCallAbort(GetComm(), SNESSetType(*this, snes_type_));
  clcustom = false;
  Customize();
}

void PetscNewtonSolver::SetSolver(mfem::Solver& solver)
{
  auto petsc_solver = dynamic_cast<mfem::PetscLinearSolver*>(&solver);
  if (petsc_solver) {
    PetscCallAbort(GetComm(), SNESSetKSP(*this, *petsc_solver));
    prec = &solver;
    auto* ksp_solver = dynamic_cast<PetscKSPSolver*>(&solver);
    if (ksp_solver) {
      auto* inner_prec = ksp_solver->GetPreconditioner();
      auto* petsc_inner_prec = dynamic_cast<mfem::PetscPreconditioner*>(inner_prec);
      if (petsc_inner_prec) {
        SLIC_DEBUG_ROOT("PetscNewtonSolver::SetSolver(...) - Set Jacobian type to PETSC_MATAIJ");
        SetJacobianType(PETSC_MATAIJ);
      } else {
        SLIC_DEBUG_ROOT("PetscNewtonSolver::SetSolver(...) - Set Jacobian type to PETSC_MATHYPRE");
        SetJacobianType(ANY_TYPE);
      }
    }
  } else {
    SetNonPetscSolver(solver);
  }
}

void PetscNewtonSolver::SetOperator(const mfem::Operator& op)
{
  bool first_set = !operatorset;
  mfem::PetscNonlinearSolver::SetOperator(op);
  oper = &op;
  // mfem::NewtonSolver::SetOperator sets defaults, we need to override them
  if (first_set) {
    SetSNESType(snes_type_);
    SetLineSearchType(linesearch_type_);
  }
  SetTolerances();
  clcustom = false;
  Customize();
}

void PetscNewtonSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const
{
  bool b_nonempty = b.Size();
  if (!B) {
    B = new mfem::PetscParVector(GetComm(), *oper, false, false);
  }
  if (!X) {
    X = new mfem::PetscParVector(GetComm(), *oper, false, false);
  }
  X->PlaceMemory(x.GetMemory(), NewtonSolver::iterative_mode);

  if (b_nonempty) {
    B->PlaceMemory(b.GetMemory());
  } else {
    *B = 0.0;
  }

  KSP ksp;
  PetscCallAbort(GetComm(), SNESGetKSP(*this, &ksp));
  PetscBool is_set;
  PetscCallAbort(GetComm(), KSPGetOperatorsSet(ksp, &is_set, nullptr));
  if (is_set) {
    Mat A;
    PetscInt nheight, nwidth;
    int oheight = B->GlobalSize();
    int owidth = X->GlobalSize();
    PetscCallAbort(GetComm(), KSPGetOperators(ksp, &A, nullptr));
    PetscCallAbort(GetComm(), MatGetSize(A, &nheight, &nwidth));
    if (nheight != oheight || nwidth != owidth) {
      // reinit without destroying the KSP
      // communicator remains the same
      SLIC_WARNING_ROOT("PetscKSPSolver::SetOperator(...) - Rebuilding KSP");
      PetscCallAbort(GetComm(), KSPReset(ksp));
    }
  }

  Customize();

  if (!NewtonSolver::iterative_mode) {
    *X = 0.;
  }

  // Solve the system.
  PetscCallAbort(GetComm(), SNESSolve(*this, *B, *X));
  X->ResetMemory();
  if (b_nonempty) {
    B->ResetMemory();
  }
}

}  // namespace smith::mfem_ext
#endif
