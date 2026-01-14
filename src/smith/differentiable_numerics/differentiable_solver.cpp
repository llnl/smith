// Copyright (c) Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "mfem.hpp"

namespace smith {

using smith::FiniteElementDual;
using smith::FiniteElementState;

/// @brief Utility to compute the matrix norm
double matrixNorm(std::unique_ptr<mfem::HypreParMatrix>& K)
{
  mfem::HypreParMatrix* H = K.get();
  hypre_ParCSRMatrix* Hhypre = static_cast<hypre_ParCSRMatrix*>(*H);
  double Hfronorm;
  hypre_ParCSRMatrixNormFro(Hhypre, &Hfronorm);
  return Hfronorm;
}

/// @brief Utility to compute 0.5*norm(K-K.T)
double skewMatrixNorm(std::unique_ptr<mfem::HypreParMatrix>& K)
{
  auto K_T = std::unique_ptr<mfem::HypreParMatrix>(K->Transpose());
  K_T->Add(-1.0, *K);
  (*K_T) *= 0.5;
  mfem::HypreParMatrix* H = K_T.get();
  hypre_ParCSRMatrix* Hhypre = static_cast<hypre_ParCSRMatrix*>(*H);
  double Hfronorm;
  hypre_ParCSRMatrixNormFro(Hhypre, &Hfronorm);
  return Hfronorm;
}

/// @brief Initialize mfem solver if near-nullspace is needed
void initializeSolver(mfem::Solver* mfem_solver, const smith::FiniteElementState& u)
{
  // If the user wants the AMG preconditioner with a linear solver, set the pfes
  // to be the displacement
  auto* amg_prec = dynamic_cast<mfem::HypreBoomerAMG*>(mfem_solver);
  if (amg_prec) {
    amg_prec->SetSystemsOptions(u.space().GetVDim(), smith::ordering == mfem::Ordering::byNODES);
  }

#ifdef SMITH_USE_PETSC
  auto* space_dep_pc = dynamic_cast<smith::mfem_ext::PetscPreconditionerSpaceDependent*>(mfem_solver);
  if (space_dep_pc) {
    // This call sets the displacement ParFiniteElementSpace used to get the spatial coordinates and to
    // generate the near null space for the PCGAMG preconditioner
    mfem::ParFiniteElementSpace* space = const_cast<mfem::ParFiniteElementSpace*>(&u.space());
    space_dep_pc->SetFESpace(space);
  }
#endif
}

LinearDifferentiableSolver::LinearDifferentiableSolver(std::unique_ptr<mfem::Solver> s, std::unique_ptr<mfem::Solver> p)
    : mfem_solver(std::move(s)), mfem_preconditioner(std::move(p))
{
}

void LinearDifferentiableSolver::completeSetup(const smith::FiniteElementState& u)
{
  initializeSolver(mfem_preconditioner.get(), u);
}

std::shared_ptr<FiniteElementState> LinearDifferentiableSolver::solve(
    const FiniteElementState& u,  // initial guess
    std::function<mfem::Vector(const FiniteElementState&)> equation,
    std::function<std::unique_ptr<mfem::HypreParMatrix>(const FiniteElementState&)> jacobian) const
{
  SMITH_MARK_FUNCTION;
  auto r = equation(u);
  auto du = std::make_shared<FiniteElementState>(u.space(), "u");
  *du = 0.0;
  auto Jptr = jacobian(u);
  mfem_solver->SetOperator(*Jptr);
  mfem_solver->Mult(r, *du);
  *du -= u;
  *du *= -1.0;
  return du;  // return u - K^{-1}r
}

std::shared_ptr<FiniteElementState> LinearDifferentiableSolver::solveAdjoint(
    const FiniteElementDual& u_bar, std::unique_ptr<mfem::HypreParMatrix> jacobian_transposed) const
{
  SMITH_MARK_FUNCTION;

  auto ds = std::make_shared<FiniteElementState>(u_bar.space(), "ds");
  mfem_solver->SetOperator(*jacobian_transposed);
  mfem_solver->Mult(u_bar, *ds);
  return ds;
}

NonlinearDifferentiableSolver::NonlinearDifferentiableSolver(std::unique_ptr<EquationSolver> s)
    : nonlinear_solver_(std::move(s))
{
}

void NonlinearDifferentiableSolver::completeSetup(const smith::FiniteElementState& u)
{
  initializeSolver(&nonlinear_solver_->preconditioner(), u);
}

std::shared_ptr<FiniteElementState> NonlinearDifferentiableSolver::solve(
    const FiniteElementState& u_guess,  // initial guess
    std::function<mfem::Vector(const FiniteElementState&)> equation,
    std::function<std::unique_ptr<mfem::HypreParMatrix>(const FiniteElementState&)> jacobian) const
{
  SMITH_MARK_FUNCTION;

  auto u = std::make_shared<FiniteElementState>(u_guess);

  auto residual_op_ = std::make_unique<mfem_ext::StdFunctionOperator>(
      u->space().TrueVSize(),

      [&u, &equation](const mfem::Vector& u_, mfem::Vector& r_) {
        FiniteElementState uu(u->space(), "uu");
        uu = u_;
        r_ = equation(uu);
      },

      [&u, &jacobian, this](const mfem::Vector& u_) -> mfem::Operator& {
        FiniteElementState uu(u->space(), "uu");
        uu = u_;
        J_.reset();
        J_ = jacobian(uu);
        return *J_;
      });

  nonlinear_solver_->setOperator(*residual_op_);
  nonlinear_solver_->solve(*u);

  return u;
}

std::shared_ptr<FiniteElementState> NonlinearDifferentiableSolver::solveAdjoint(
    const FiniteElementDual& x_bar, std::unique_ptr<mfem::HypreParMatrix> jacobian_transposed) const
{
  SMITH_MARK_FUNCTION;

  auto ds = std::make_shared<FiniteElementState>(x_bar.space(), "ds");
  auto& linear_solver = nonlinear_solver_->linearSolver();
  linear_solver.SetOperator(*jacobian_transposed);
  linear_solver.Mult(x_bar, *ds);

  return ds;
}

void NonlinearDifferentiableSolver::clearMemory() const { J_.reset(); }

LinearDifferentiableBlockSolver::LinearDifferentiableBlockSolver(std::unique_ptr<mfem::Solver> s,
                                                                 std::unique_ptr<mfem::Solver> p)
    : mfem_solver(std::move(s)), mfem_preconditioner(std::move(p))
{
}

void LinearDifferentiableBlockSolver::completeSetup(const std::vector<FieldT>& us)
{
  initializeSolver(mfem_preconditioner.get(), us[0]);
}

std::vector<DifferentiableBlockSolver::FieldPtr> LinearDifferentiableBlockSolver::solve(
    const std::vector<FieldPtr>& u_guesses,
    std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)> residual_funcs,
    std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)> jacobian_funcs) const
{
  SMITH_MARK_FUNCTION;

  int num_rows = static_cast<int>(u_guesses.size());
  SLIC_ERROR_IF(num_rows < 0, "Number of residual rows must be non-negative");

  mfem::Array<int> block_offsets;
  block_offsets.SetSize(num_rows + 1);
  block_offsets[0] = 0;
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_offsets[row_i + 1] = u_guesses[static_cast<size_t>(row_i)]->space().TrueVSize();
  }
  block_offsets.PartialSum();

  auto block_du = std::make_unique<mfem::BlockVector>(block_offsets);
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_du->GetBlock(row_i) = *u_guesses[static_cast<size_t>(row_i)];
  }

  auto residuals = residual_funcs(u_guesses);
  auto block_r = std::make_unique<mfem::BlockVector>(block_offsets);
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_r->GetBlock(row_i) = residuals[static_cast<size_t>(row_i)];
  }

  auto jacs = jacobian_funcs(u_guesses);
  auto block_jac = std::make_unique<mfem::BlockOperator>(block_offsets);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_rows; ++j) {
      block_jac->SetBlock(i, j, jacs[static_cast<size_t>(i)][static_cast<size_t>(j)].get());
    }
  }

  mfem_solver->SetOperator(*block_jac);

  mfem_solver->Mult(*block_r, *block_du);

  for (int row_i = 0; row_i < num_rows; ++row_i) {
    *u_guesses[static_cast<size_t>(row_i)] -= block_du->GetBlock(row_i);
  }
  *block_du = 0.0;

  return u_guesses;
}

std::vector<DifferentiableBlockSolver::FieldPtr> LinearDifferentiableBlockSolver::solveAdjoint(
    const std::vector<DualPtr>& u_bars, std::vector<std::vector<MatrixPtr>>& jacobian_transposed) const
{
  SMITH_MARK_FUNCTION;

  int num_rows = static_cast<int>(u_bars.size());
  SLIC_ERROR_IF(num_rows < 0, "Number of residual rows must be non-negative");

  std::vector<DifferentiableBlockSolver::FieldPtr> u_duals(static_cast<size_t>(num_rows));
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    u_duals[static_cast<size_t>(row_i)] = std::make_shared<DifferentiableBlockSolver::FieldT>(
        u_bars[static_cast<size_t>(row_i)]->space(), "u_dual_" + std::to_string(row_i));
  }

  mfem::Array<int> block_offsets;
  block_offsets.SetSize(num_rows + 1);
  block_offsets[0] = 0;
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_offsets[row_i + 1] = u_bars[static_cast<size_t>(row_i)]->space().TrueVSize();
  }
  block_offsets.PartialSum();

  auto block_ds = std::make_unique<mfem::BlockVector>(block_offsets);
  *block_ds = 0.0;

  auto block_r = std::make_unique<mfem::BlockVector>(block_offsets);
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_r->GetBlock(row_i) = *u_bars[static_cast<size_t>(row_i)];
  }

  auto block_jac = std::make_unique<mfem::BlockOperator>(block_offsets);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_rows; ++j) {
      block_jac->SetBlock(i, j, jacobian_transposed[static_cast<size_t>(i)][static_cast<size_t>(j)].get());
    }
  }

  mfem_solver->SetOperator(*block_jac);

  mfem_solver->Mult(*block_r, *block_ds);

  for (int row_i = 0; row_i < num_rows; ++row_i) {
    *u_duals[static_cast<size_t>(row_i)] = block_ds->GetBlock(row_i);
  }

  return u_duals;
}

std::shared_ptr<LinearDifferentiableSolver> buildDifferentiableLinearSolver(LinearSolverOptions linear_opts,
                                                                            const smith::Mesh& mesh)
{
  auto [linear_solver, precond] = smith::buildLinearSolverAndPreconditioner(linear_opts, mesh.getComm());
  return std::make_shared<smith::LinearDifferentiableSolver>(std::move(linear_solver), std::move(precond));
}

std::shared_ptr<NonlinearDifferentiableSolver> buildDifferentiableNonlinearSolver(
    smith::NonlinearSolverOptions nonlinear_opts, LinearSolverOptions linear_opts, const smith::Mesh& mesh)
{
  auto solid_solver = std::make_unique<smith::EquationSolver>(nonlinear_opts, linear_opts, mesh.getComm());
  return std::make_shared<smith::NonlinearDifferentiableSolver>(std::move(solid_solver));
}

}  // namespace smith
