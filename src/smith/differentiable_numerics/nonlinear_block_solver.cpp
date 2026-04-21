// Copyright (c) Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
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

NonlinearBlockSolver::NonlinearBlockSolver(std::unique_ptr<EquationSolver> s, MPI_Comm comm, double abs_tol,
                                           double rel_tol,
                                           std::optional<NonlinearSolverOptions> retained_nonlinear_options,
                                           std::optional<LinearSolverOptions> retained_linear_options)
    : nonlinear_solver_(std::move(s)),
      comm_(comm),
      abs_tol_(abs_tol),
      rel_tol_(rel_tol),
      retained_nonlinear_options_(std::move(retained_nonlinear_options)),
      retained_linear_options_(std::move(retained_linear_options))
{
}

std::shared_ptr<NonlinearBlockSolver> NonlinearBlockSolver::cloneFresh() const
{
  if (!retained_nonlinear_options_ || !retained_linear_options_) {
    return nullptr;
  }

  auto nonlinear_opts = *retained_nonlinear_options_;
  const auto linear_opts = *retained_linear_options_;

  auto solver = std::make_unique<EquationSolver>(nonlinear_opts, linear_opts, comm_);
  return std::make_shared<NonlinearBlockSolver>(std::move(solver), comm_, nonlinear_opts.absolute_tol,
                                                nonlinear_opts.relative_tol, nonlinear_opts, linear_opts);
}

void NonlinearBlockSolver::completeSetup(const std::vector<FieldT>& us)
{
  if (us.empty() || !nonlinear_solver_) return;
  if (!retained_linear_options_ || retained_linear_options_->preconditioner == Preconditioner::None) return;

  // Pick the field with the highest vdim (typically the displacement field for vector problems).
  const FieldT* best = &us[0];
  for (const auto& u : us) {
    if (u.space().GetVDim() > best->space().GetVDim()) best = &u;
  }

  auto* mfem_solver = &nonlinear_solver_->preconditioner();

  auto* amg_prec = dynamic_cast<mfem::HypreBoomerAMG*>(mfem_solver);
  if (amg_prec) {
    amg_prec->SetSystemsOptions(best->space().GetVDim(), smith::ordering == mfem::Ordering::byNODES);
  }

#ifdef SMITH_USE_PETSC
  auto* space_dep_pc = dynamic_cast<smith::mfem_ext::PetscPreconditionerSpaceDependent*>(mfem_solver);
  if (space_dep_pc) {
    // This call sets the displacement ParFiniteElementSpace used to get the spatial coordinates and to
    // generate the near null space for the PCGAMG preconditioner
    auto* space = const_cast<mfem::ParFiniteElementSpace*>(&best->space());
    space_dep_pc->SetFESpace(space);
  }
#endif
}

ConvergenceStatus NonlinearBlockSolver::convergenceStatus(double tolerance_multiplier,
                                                          const std::vector<mfem::Vector>& residuals,
                                                          NonlinearConvergenceContext& context) const
{
  auto block_norms = computeResidualBlockNorms(residuals, comm_);
  return evaluateResidualConvergence(tolerance_multiplier, abs_tol_, rel_tol_, block_norms, context);
}

void NonlinearBlockSolver::primeConvergenceContext(const std::vector<mfem::Vector>& residuals,
                                                   NonlinearConvergenceContext& context) const
{
  static_cast<void>(convergenceStatus(1.0, residuals, context));
}

std::vector<NonlinearBlockSolverBase::FieldPtr> NonlinearBlockSolver::solve(
    const std::vector<FieldPtr>& u_guesses,
    std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)> residual_funcs,
    std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)> jacobian_funcs) const
{
  SMITH_MARK_FUNCTION;
  SLIC_ERROR_IF(!nonlinear_solver_, "NonlinearBlockSolver requires an EquationSolver instance before solve()");

  nonlinear_solver_->resetConvergenceState();

  int num_rows = static_cast<int>(u_guesses.size());
  SLIC_ERROR_IF(num_rows < 0, "Number of residual rows must be non-negative");

  mfem::Array<int> block_offsets;
  block_offsets.SetSize(num_rows + 1);
  block_offsets[0] = 0;
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_offsets[row_i + 1] = u_guesses[static_cast<size_t>(row_i)]->space().TrueVSize();
  }
  block_offsets.PartialSum();
  auto block_u = std::make_unique<mfem::BlockVector>(block_offsets);
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_u->GetBlock(row_i) = *u_guesses[static_cast<size_t>(row_i)];
  }

  auto block_r = std::make_unique<mfem::BlockVector>(block_offsets);

  auto residual_op_ = std::make_unique<mfem_ext::StdFunctionOperator>(
      block_u->Size(),
      [&residual_funcs, num_rows, &u_guesses, &block_r, &block_offsets](const mfem::Vector& u_, mfem::Vector& r_) {
        const mfem::BlockVector* u = dynamic_cast<const mfem::BlockVector*>(&u_);
        mfem::BlockVector u_block_wrapper;
        if (!u) {
          u_block_wrapper.Update(const_cast<double*>(u_.GetData()), block_offsets);
          u = &u_block_wrapper;
        }
        for (int row_i = 0; row_i < num_rows; ++row_i) {
          *u_guesses[static_cast<size_t>(row_i)] = u->GetBlock(row_i);
        }
        auto residuals = residual_funcs(u_guesses);
        SLIC_ERROR_IF(!block_r, "Invalid residual block cast to an mfem::BlockVector");
        for (int row_i = 0; row_i < num_rows; ++row_i) {
          auto r = residuals[static_cast<size_t>(row_i)];
          block_r->GetBlock(row_i) = r;
        }
        r_ = *block_r;
      },
      [this, &block_offsets, &u_guesses, jacobian_funcs, num_rows](const mfem::Vector& u_) -> mfem::Operator& {
        const mfem::BlockVector* u = dynamic_cast<const mfem::BlockVector*>(&u_);
        mfem::BlockVector u_block_wrapper;
        if (!u) {
          u_block_wrapper.Update(const_cast<double*>(u_.GetData()), block_offsets);
          u = &u_block_wrapper;
        }
        for (int row_i = 0; row_i < num_rows; ++row_i) {
          *u_guesses[static_cast<size_t>(row_i)] = u->GetBlock(row_i);
        }
        matrix_of_jacs_ = jacobian_funcs(u_guesses);
        if (num_rows == 1) {
          auto& J = matrix_of_jacs_[0][0];
          SLIC_ERROR_IF(!J, "Jacobian block (0,0) is null in single-block solve");
          return *J;
        }
        block_jac_ = std::make_unique<mfem::BlockOperator>(block_offsets);
        for (int i = 0; i < num_rows; ++i) {
          for (int j = 0; j < num_rows; ++j) {
            auto& J = matrix_of_jacs_[static_cast<size_t>(i)][static_cast<size_t>(j)];
            if (J) {
              block_jac_->SetBlock(i, j, J.get());
            }
          }
        }
        return *block_jac_;
      });
  nonlinear_solver_->setConvergenceTolerances(inner_tol_multiplier_ * abs_tol_, inner_tol_multiplier_ * rel_tol_,
                                              comm_);
  nonlinear_solver_->setOperator(*residual_op_);
  nonlinear_solver_->solve(*block_u);

  for (int row_i = 0; row_i < num_rows; ++row_i) {
    *u_guesses[static_cast<size_t>(row_i)] = block_u->GetBlock(row_i);
  }

  return u_guesses;
}

std::vector<NonlinearBlockSolverBase::FieldPtr> NonlinearBlockSolver::solveAdjoint(
    const std::vector<DualPtr>& u_bars, std::vector<std::vector<MatrixPtr>>& jacobian_transposed) const
{
  SMITH_MARK_FUNCTION;

  int num_rows = static_cast<int>(u_bars.size());
  SLIC_ERROR_IF(num_rows < 0, "Number of residual rows must be non-negative");

  std::vector<NonlinearBlockSolverBase::FieldPtr> u_duals(static_cast<size_t>(num_rows));
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    u_duals[static_cast<size_t>(row_i)] = std::make_shared<NonlinearBlockSolverBase::FieldT>(
        u_bars[static_cast<size_t>(row_i)]->space(), "u_dual_" + std::to_string(row_i));
  }

  auto& linear_solver = nonlinear_solver_->linearSolver();

  // If it's a 1x1 system, pass the operator directly to avoid potential BlockOperator issues with smoothers
  if (num_rows == 1) {
    linear_solver.SetOperator(*jacobian_transposed[0][0]);
    linear_solver.Mult(*u_bars[0], *u_duals[0]);
    return u_duals;
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

  linear_solver.SetOperator(*block_jac);
  linear_solver.Mult(*block_r, *block_ds);

  for (int row_i = 0; row_i < num_rows; ++row_i) {
    *u_duals[static_cast<size_t>(row_i)] = block_ds->GetBlock(row_i);
  }

  return u_duals;
}

std::shared_ptr<NonlinearBlockSolver> buildNonlinearBlockSolver(NonlinearSolverOptions nonlinear_opts,
                                                                LinearSolverOptions linear_opts,
                                                                const smith::Mesh& mesh)
{
  auto solid_solver = std::make_unique<EquationSolver>(nonlinear_opts, linear_opts, mesh.getComm());
  return std::make_shared<NonlinearBlockSolver>(std::move(solid_solver), mesh.getComm(), nonlinear_opts.absolute_tol,
                                                nonlinear_opts.relative_tol, nonlinear_opts, linear_opts);
}

}  // namespace smith
