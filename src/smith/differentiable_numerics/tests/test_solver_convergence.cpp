// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <initializer_list>
#include <memory>

#include "mfem.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"

namespace smith {

namespace {

class FakeNonlinearBlockSolver : public NonlinearBlockSolverBase {
 public:
  using NonlinearBlockSolverBase::convergenceStatus;

  FakeNonlinearBlockSolver(double abs_tol, double rel_tol) : abs_tol_(abs_tol), rel_tol_(rel_tol) {}

  std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses, std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)>,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)>) const override
  {
    return u_guesses;
  }

  std::vector<FieldPtr> solveAdjoint(const std::vector<DualPtr>&, std::vector<std::vector<MatrixPtr>>&) const override
  {
    return {};
  }

  ConvergenceStatus convergenceStatus(double tolerance_multiplier, const std::vector<mfem::Vector>& residuals,
                                      NonlinearConvergenceContext& context) const override
  {
    return evaluateResidualConvergence(tolerance_multiplier, abs_tol_, rel_tol_,
                                       computeResidualBlockNorms(residuals, MPI_COMM_SELF), context);
  }

  void primeConvergenceContext(const std::vector<mfem::Vector>& residuals,
                               NonlinearConvergenceContext& context) const override
  {
    static_cast<void>(convergenceStatus(1.0, residuals, context));
  }

  void setInnerToleranceMultiplier(double) override {}

 private:
  double abs_tol_;
  double rel_tol_;
};

std::vector<mfem::Vector> makeResiduals(std::initializer_list<double> values)
{
  std::vector<mfem::Vector> residuals;
  for (double value : values) {
    mfem::Vector residual(1);
    residual(0) = value;
    residuals.push_back(residual);
  }
  return residuals;
}

}  // namespace

TEST(SolverConvergence, ScalarConvergenceUsesCombinedResidualNorm)
{
  FakeNonlinearBlockSolver solver(1.0e-12, 0.1);

  auto status = solver.convergenceStatus(1.0, makeResiduals({1.0, 1.0}));
  EXPECT_FALSE(status.converged);

  status = solver.convergenceStatus(1.0, makeResiduals({0.1001, 0.1001}));
  EXPECT_FALSE(status.global_converged);
  EXPECT_FALSE(status.converged);

  status = solver.convergenceStatus(1.0, makeResiduals({0.05, 0.05}));
  EXPECT_TRUE(status.global_converged);
  EXPECT_TRUE(status.converged);
}

TEST(SolverConvergence, ResetConvergenceStateRefreshesInitialNorm)
{
  FakeNonlinearBlockSolver solver(1.0e-12, 0.01);
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({10.0, 10.0})));
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({0.1001, 0.1001})));
  EXPECT_TRUE(solver.checkConvergence(1.0, makeResiduals({0.0999, 0.0999})));

  solver.resetConvergenceState();
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({1.0, 1.0})));
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({0.0101, 0.0101})));
  EXPECT_TRUE(solver.checkConvergence(1.0, makeResiduals({0.0099, 0.0099})));
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
