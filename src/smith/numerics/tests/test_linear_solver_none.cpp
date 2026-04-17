// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "mfem.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace smith {

/**
 * @brief Simple identity-like operator: A*x = x
 */
class IdentityOperator : public mfem::Operator {
 public:
  IdentityOperator(int size) : mfem::Operator(size) {}
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override { y = x; }
  mfem::Operator& GetGradient(const mfem::Vector& /*x*/) const override { return const_cast<IdentityOperator&>(*this); }
};

/**
 * @brief Simple diagonal operator: A*x = d * x
 */
class DiagonalOperator : public mfem::Operator {
 public:
  DiagonalOperator(const mfem::Vector& d) : mfem::Operator(d.Size()), d_(d) {}
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    for (int i = 0; i < height; i++) {
      y(i) = d_(i) * x(i);
    }
  }
  mfem::Operator& GetGradient(const mfem::Vector& /*x*/) const override { return const_cast<DiagonalOperator&>(*this); }

 private:
  const mfem::Vector& d_;
};

TEST(LinearSolverNone, Identity)
{
  int size = 10;
  IdentityOperator op(size);

  LinearSolverOptions linear_opts;
  linear_opts.linear_solver = LinearSolver::None;
  linear_opts.preconditioner = Preconditioner::None;

  NonlinearSolverOptions nonlinear_opts;
  nonlinear_opts.nonlin_solver = NonlinearSolver::Newton;
  nonlinear_opts.max_iterations = 1;

  EquationSolver solver(nonlinear_opts, linear_opts, MPI_COMM_WORLD);
  solver.setOperator(op);

  mfem::Vector x(size);
  x = 1.0;  // Initial guess
  // Residual will be f(x) = x.
  // x_new = x - [df/dx]^-1 * f(x) = x - I^-1 * x = 0.

  solver.solve(x);

  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(x(i), 0.0, 1e-12);
  }
}

TEST(LinearSolverNone, Jacobi)
{
  int size = 10;
  mfem::Vector d(size);
  d = 2.0;
  DiagonalOperator op(d);

  LinearSolverOptions linear_opts;
  linear_opts.linear_solver = LinearSolver::None;
  linear_opts.preconditioner = Preconditioner::None;  // We'll set this manually if needed, but wait.
  // Actually, Preconditioner::None with LinearSolver::None should be Identity.
  // Let's test that first.

  NonlinearSolverOptions nonlinear_opts;
  nonlinear_opts.nonlin_solver = NonlinearSolver::Newton;
  nonlinear_opts.max_iterations = 1;

  EquationSolver solver(nonlinear_opts, linear_opts, MPI_COMM_WORLD);
  solver.setOperator(op);

  mfem::Vector x(size);
  x = 1.0;
  // f(x) = 2x. df/dx = 2I.
  // If LinearSolver::None and Preconditioner::None, it uses identity:
  // x_new = x - I * (2x) = x - 2x = -x.

  solver.solve(x);

  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(x(i), -1.0, 1e-12);
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
