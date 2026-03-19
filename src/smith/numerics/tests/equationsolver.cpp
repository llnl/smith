// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <string>
#include <tuple>

#include "mpi.h"
#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/differentiate_wrt.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/geometry.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/smith_config.hpp"

using namespace smith;
using namespace smith::mfem_ext;

using param_t = std::tuple<NonlinearSolver, LinearSolver, Preconditioner>;

namespace {

class TwoBlockQuadraticOperator : public mfem::Operator {
 public:
  explicit TwoBlockQuadraticOperator(double second_scale = 100.0) : mfem::Operator(2), second_scale_(second_scale) {}

  void Mult(const mfem::Vector& x, mfem::Vector& r) const override
  {
    r.SetSize(2);
    r(0) = x(0) * x(0);
    r(1) = second_scale_ * x(1) * x(1);
  }

  mfem::Operator& GetGradient(const mfem::Vector& x) const override
  {
    jacobian_diag_ = std::make_unique<mfem::SparseMatrix>(2);
    jacobian_diag_->Add(0, 0, 2.0 * x(0));
    jacobian_diag_->Add(1, 1, 2.0 * second_scale_ * x(1));
    jacobian_diag_->Finalize();
    jacobian_ = std::make_unique<mfem::HypreParMatrix>(MPI_COMM_WORLD, 2, 2, offsets_, offsets_, jacobian_diag_.get());
    return *jacobian_;
  }

 private:
  double second_scale_;
  mutable std::unique_ptr<mfem::SparseMatrix> jacobian_diag_ = nullptr;
  mutable std::unique_ptr<mfem::HypreParMatrix> jacobian_ = nullptr;
  mutable HYPRE_BigInt offsets_[2] = {0, 2};
};

class ManagedHalvingSolver : public mfem::NewtonSolver, public smith::ConvergenceManagedNonlinearSolver {
 public:
  ManagedHalvingSolver() : mfem::NewtonSolver(MPI_COMM_WORLD) {}

  void setConvergenceManager(std::shared_ptr<smith::EquationSolverConvergenceManager> convergence_manager) override
  {
    convergence_manager_ = std::move(convergence_manager);
  }

  void Mult(const mfem::Vector&, mfem::Vector& x) const override
  {
    mfem::Vector residual(x.Size());
    oper->Mult(x, residual);
    initial_norm = residual.Norml2();

    int it = 0;
    for (;; ++it) {
      oper->Mult(x, residual);

      smith::ConvergenceStatus status;
      if (convergence_manager_) {
        status = convergence_manager_->evaluate(1.0, residual);
      } else {
        status.global_norm = residual.Norml2();
        status.global_goal = std::max(abs_tol, rel_tol * initial_norm);
        status.global_converged = status.global_norm <= status.global_goal;
        status.converged = status.global_converged;
      }

      if (status.converged) {
        converged = true;
        final_iter = it;
        final_norm = status.global_norm;
        return;
      }
      if (it >= max_iter) {
        converged = false;
        final_iter = it;
        final_norm = status.global_norm;
        return;
      }

      x *= 0.5;
    }
  }

 private:
  std::shared_ptr<smith::EquationSolverConvergenceManager> convergence_manager_ = nullptr;
};

void expectBlockOnlyConvergence(const smith::EquationSolver& eq_solver, const TwoBlockQuadraticOperator& op,
                                const mfem::Vector& solution, double global_rel_tol)
{
  mfem::Vector residual(2);
  op.Mult(solution, residual);

  EXPECT_TRUE(eq_solver.nonlinearSolver().GetConverged());
  EXPECT_LT(std::abs(residual(0)), 0.2);
  EXPECT_LT(std::abs(residual(1)), 20.0);
  EXPECT_GT(eq_solver.nonlinearSolver().GetFinalNorm(), global_rel_tol * eq_solver.nonlinearSolver().GetInitialNorm());
}

}  // namespace

class EquationSolverSuite : public testing::TestWithParam<param_t> {
 protected:
  void SetUp() override { std::tie(nonlin_solver, lin_solver, precond) = GetParam(); }
  NonlinearSolver nonlin_solver;
  LinearSolver lin_solver;
  Preconditioner precond;
};

TEST_P(EquationSolverSuite, All)
{
  auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL);
  auto pmesh = mfem::ParMesh(MPI_COMM_WORLD, mesh);

  pmesh.EnsureNodes();
  pmesh.ExchangeFaceNbrData();

  constexpr int p = 1;
  constexpr int dim = 2;

  // Define the types for the test and trial spaces using the function arguments
  using test_space = H1<p>;
  using trial_space = H1<p>;

  // Create standard MFEM bilinear and linear forms on H1
  auto [fes, fec] = smith::generateParFiniteElementSpace<test_space>(&pmesh);

  mfem::HypreParVector x_exact(fes.get());
  mfem::HypreParVector x_computed(fes.get());

  std::unique_ptr<mfem::HypreParMatrix> J;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(fes.get(), {fes.get()});

  x_exact.Randomize(0);

  Domain domain = EntireDomain(pmesh);

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [&](double /*t*/, auto, auto scalar) {
        auto [u, du_dx] = scalar;
        auto source = 0.5 * sin(u);
        auto flux = du_dx;
        return smith::tuple{source, flux};
      },
      domain);

  StdFunctionOperator residual_opr(
      fes->TrueVSize(),
      [&x_exact, &residual](const mfem::Vector& x, mfem::Vector& r) {
        // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
        // tracking strategy
        // See https://github.com/mfem/mfem/issues/3531

        double dummy_time = 0.0;

        const mfem::Vector res = residual(dummy_time, x);

        r = res;
        r -= residual(dummy_time, x_exact);
      },
      [&residual, &J](const mfem::Vector& x) -> mfem::Operator& {
        double dummy_time = 0.0;
        auto [val, grad] = residual(dummy_time, differentiate_wrt(x));
        J = assemble(grad);
        return *J;
      });

  const LinearSolverOptions lin_opts = {.linear_solver = lin_solver,
                                        .preconditioner = precond,
                                        .relative_tol = 1.0e-10,
                                        .absolute_tol = 1.0e-12,
                                        .max_iterations = 500,
                                        .print_level = 1};

  const NonlinearSolverOptions nonlin_opts = {.nonlin_solver = nonlin_solver,
                                              .relative_tol = 1.0e-10,
                                              .absolute_tol = 1.0e-12,
                                              .max_iterations = 100,
                                              .print_level = 1};

  EquationSolver eq_solver(nonlin_opts, lin_opts);

  eq_solver.setOperator(residual_opr);

  eq_solver.solve(x_computed);

  EXPECT_EQ(x_computed.Size(), x_exact.Size());
  for (int i = 0; i < x_computed.Size(); ++i) {
    EXPECT_LT(std::abs((x_computed(i) - x_exact(i))) / x_exact(i), 1.0e-6);
  }
}

class BlockConvergenceEquationSolverSuite : public testing::TestWithParam<NonlinearSolver> {};

TEST_P(BlockConvergenceEquationSolverSuite, StopsOnPerBlockInnerConvergence)
{
  TwoBlockQuadraticOperator residual_opr;

  const LinearSolverOptions lin_opts = {.linear_solver = LinearSolver::CG,
                                        .preconditioner = Preconditioner::HypreJacobi,
                                        .relative_tol = 1.0e-14,
                                        .absolute_tol = 1.0e-14,
                                        .max_iterations = 20,
                                        .print_level = 0};

  const double global_rel_tol = 1.0e-2;
  const NonlinearSolverOptions nonlin_opts = {.nonlin_solver = GetParam(),
                                              .relative_tol = global_rel_tol,
                                              .absolute_tol = 0.0,
                                              .max_iterations = 10,
                                              .print_level = 0};

  EquationSolver eq_solver(nonlin_opts, lin_opts);
  eq_solver.setConvergenceBlockData({0, 1, 2}, {.relative_tols = {0.2, 0.2}});
  eq_solver.setOperator(residual_opr);

  mfem::Vector x(2);
  x = 1.0;
  eq_solver.solve(x);

  expectBlockOnlyConvergence(eq_solver, residual_opr, x, global_rel_tol);
}

TEST(EquationSolverManualConvergence, InjectedManagedSolverSupportsPerBlockConvergence)
{
  auto nonlinear_solver = std::make_unique<ManagedHalvingSolver>();
  nonlinear_solver->SetRelTol(1.0e-2);
  nonlinear_solver->SetAbsTol(0.0);
  nonlinear_solver->SetMaxIter(10);
  nonlinear_solver->SetPrintLevel(0);

  auto linear_solver = std::make_unique<mfem::CGSolver>(MPI_COMM_WORLD);
  EquationSolver eq_solver(std::move(nonlinear_solver), std::move(linear_solver));

  TwoBlockQuadraticOperator residual_opr;
  eq_solver.setConvergenceBlockData({0, 1, 2}, {.relative_tols = {0.2, 0.2}}, 0.0, 1.0e-2, MPI_COMM_WORLD);
  eq_solver.setOperator(residual_opr);

  mfem::Vector x(2);
  x = 1.0;
  eq_solver.solve(x);

  expectBlockOnlyConvergence(eq_solver, residual_opr, x, 1.0e-2);
}

/**
 * @brief Nonlinear solvers to test. Always includes NonlinearSolver::Newton and NonlinearSolver::LBFGS
 * If SMITH_USE_SUNDIALS is set, adds: NonlinearSolver::KINFullStep, NonlinearSolver::KINBacktrackingLineSearch, and
 * NonlinearSolver::KINPicard.
 * If MFEM_USE_PETSC and SMITH_USE_PETSC are set, adds NonlinearSolver::PetscNewton,
 * NonlinearSolver::PetscNewtonBacktracking, and NonlinearSolver::PetscNewtonCriticalPoint
 */
auto nonlinear_solvers = testing::Values(
    NonlinearSolver::Newton, NonlinearSolver::NewtonLineSearch, NonlinearSolver::TrustRegion, NonlinearSolver::LBFGS
#ifdef SMITH_USE_SUNDIALS
    ,
    NonlinearSolver::KINFullStep, NonlinearSolver::KINBacktrackingLineSearch, NonlinearSolver::KINPicard
#endif
#ifdef SMITH_USE_PETSC
    ,
    NonlinearSolver::PetscNewton, NonlinearSolver::PetscNewtonBacktracking, NonlinearSolver::PetscNewtonCriticalPoint
#endif
);

/**
 * @brief Linear solvers to test. Always includes LinearSolver::CG, LinearSolver::GMRES, and LinearSolver::SuperLU.
 * If MFEM_USE_PETSC and SMITH_USE_PETSC are set, adds LinearSolver::PetscCG and LinearSolver::PetscGMRES.
 */
auto linear_solvers = testing::Values(LinearSolver::CG, LinearSolver::GMRES, LinearSolver::SuperLU
#ifdef SMITH_USE_PETSC
                                      ,
                                      LinearSolver::PetscCG, LinearSolver::PetscGMRES
#endif
);

auto preconditioners =
    testing::Values(Preconditioner::HypreJacobi, Preconditioner::HypreL1Jacobi, Preconditioner::HypreGaussSeidel,
                    Preconditioner::HypreAMG, Preconditioner::HypreILU
#ifdef SMITH_USE_PETSC
                    ,
                    Preconditioner::Petsc
#endif
    );

INSTANTIATE_TEST_SUITE_P(AllEquationSolverTests, EquationSolverSuite,
                         testing::Combine(nonlinear_solvers, linear_solvers, preconditioners),
                         [](const testing::TestParamInfo<EquationSolverSuite::ParamType>& test_info) {
                           std::string name =
                               axom::fmt::format("{}_{}_{}", std::get<0>(test_info.param), std::get<1>(test_info.param),
                                                 std::get<2>(test_info.param));
                           return name;
                         });

INSTANTIATE_TEST_SUITE_P(BlockConvergenceBackends, BlockConvergenceEquationSolverSuite,
                         testing::Values(NonlinearSolver::Newton, NonlinearSolver::NewtonLineSearch,
                                         NonlinearSolver::TrustRegion));

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
