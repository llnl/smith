// Copyright (c) Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace smith {

class MockEquationSolver : public EquationSolver {
 public:
  MockEquationSolver(MPI_Comm comm) : EquationSolver(NonlinearSolverOptions(), LinearSolverOptions(), comm) {}
  void solve(mfem::Vector& /*x*/) {}
};

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager app(argc, argv);
  return RUN_ALL_TESTS();
}

namespace smith {

TEST(DifferentiableBlockSolverTest, ConvergenceChecks)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  double abs_tol = 1e-4;
  double rel_tol = 1e-2;

  auto mock_solver = std::make_unique<MockEquationSolver>(comm);
  NonlinearDifferentiableBlockSolver solver(std::move(mock_solver), comm, abs_tol, rel_tol);

  // Create a synthetic vector
  // Local size = 2 for both ranks, so global size = 4
  mfem::Vector res(2);

  // Test case 1: Initial residual norm is relatively large, say 10.0 globally.
  // Each rank contributes sqrt(50) = 7.07
  // Norm = sqrt( 50 + 50 ) = 10.0
  res[0] = 5.0;  // 25
  res[1] = 5.0;  // 25

  // First check should initialize root tracking, then return false
  // Tolerance multiplier = 1.0
  // Target tol = max(abs_tol, rel_tol * initial_norm) = max(1e-4, 1e-2 * 10) = 0.1
  bool converged = solver.checkConvergence(1.0, {res});
  EXPECT_FALSE(converged);

  // Reduce residual to 1.0 (above tol)
  res[0] = 0.5;  // 0.25
  res[1] = 0.5;  // 0.25
  // Global norm = sqrt(0.5 * 4) = 1.414 > 0.1
  converged = solver.checkConvergence(1.0, {res});
  EXPECT_FALSE(converged);

  // Reduce residual to 0.05 (below target tol of 0.1)
  res[0] = 0.025;  // 0.000625
  res[1] = 0.025;  // 0.000625
  // Global norm = sqrt(0.00125 * 2) = sqrt(0.0025) = 0.05 < 0.1
  converged = solver.checkConvergence(1.0, {res});
  EXPECT_TRUE(converged);

  // Test reset
  solver.resetConvergenceState();

  // Test case 2: Initial residual is very small (abs_tol dominated)
  // Target tol = max(1e-4, 1e-2 * 1e-6) = 1e-4
  res[0] = 5e-7;
  res[1] = 5e-7;
  // Global norm = sqrt(5e-13 * 2) = 1e-6

  converged = solver.checkConvergence(1.0, {res});
  // The very first norm is 1e-6, which is already < 1e-4, so it should be true!
  EXPECT_TRUE(converged);

  // Just to make sure, checking again with a larger residual should fail
  res[0] = 1.0;
  res[1] = 1.0;
  converged = solver.checkConvergence(1.0, {res});
  EXPECT_FALSE(converged);

  // Test tolerance multiplier
  solver.resetConvergenceState();

  // Initial norm = 10.0 again
  res[0] = 5.0;
  res[1] = 5.0;

  // First check saves initial_residual_norm = 10.0.
  // Multiplier = 1e12 (similar to system_solver's 1e12 trick)
  // Target = 1e12 * 0.1 = 1e11
  converged = solver.checkConvergence(1e12, {res});
  EXPECT_TRUE(converged);  // Should trivially pass due to huge multiplier

  // Normal multiplier target is 0.1
  // Use a residual of exactly 0.08 (passing)
  res[0] = 0.04;
  res[1] = 0.04;
  // global norm = sqrt( (0.0016+0.0016)*2 ) = sqrt(0.0064) = 0.08
  converged = solver.checkConvergence(1.0, {res});
  EXPECT_TRUE(converged);

  // Wait, if multiplier was used, what if it was small?
  // Say multiplier is 0.5. Then target tol is 0.05.
  converged = solver.checkConvergence(0.5, {res});
  EXPECT_FALSE(converged);  // 0.08 > 0.05
}

}  // namespace smith
