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

inline int to_int(size_t i) { return static_cast<int>(i); }

inline double matrixNorm(std::unique_ptr<mfem::HypreParMatrix>& K)
{
  mfem::HypreParMatrix* H = K.get();
  hypre_ParCSRMatrix* Hhypre = static_cast<hypre_ParCSRMatrix*>(*H);
  double Hfronorm;
  hypre_ParCSRMatrixNormFro(Hhypre, &Hfronorm);
  return Hfronorm;
}

inline double skewMatrixNorm(std::unique_ptr<mfem::HypreParMatrix>& K)
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
    // ZRA - Iterative refinement tends to be more expensive than it is worth
    // We should add a flag allowing users to enable it

    // bool iterative_refinement = false;
    // amg_prec->SetElasticityOptions(&displacement_.space(), iterative_refinement);

    // SetElasticityOptions only works with byVDIM ordering, some evidence that it is not often optimal
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

  // std::cout << "solution norm = " << u->Norml2() << std::endl;

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

std::shared_ptr<LinearDifferentiableSolver> buildDifferentiableLinearSolve(LinearSolverOptions linear_opts,
                                                                           const smith::Mesh& mesh)
{
  auto [linear_solver, precond] = smith::buildLinearSolverAndPreconditioner(linear_opts, mesh.getComm());
  return std::make_shared<smith::LinearDifferentiableSolver>(std::move(linear_solver), std::move(precond));
}

std::shared_ptr<NonlinearDifferentiableSolver> buildDifferentiableNonlinearSolve(
    smith::NonlinearSolverOptions nonlinear_opts, LinearSolverOptions linear_opts, const smith::Mesh& mesh)
{
  auto solid_solver = std::make_unique<smith::EquationSolver>(nonlinear_opts, linear_opts, mesh.getComm());
  return std::make_shared<smith::NonlinearDifferentiableSolver>(std::move(solid_solver));
}

}  // namespace smith