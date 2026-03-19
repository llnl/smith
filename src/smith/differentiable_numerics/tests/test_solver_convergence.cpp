// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <memory>

#include "axom/fmt.hpp"
#include "axom/slic.hpp"
#include "mfem.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/coupled_system_solver.hpp"
#include "smith/numerics/equation_solver.hpp"

namespace smith {

namespace {

class FakeNonlinearBlockSolver : public NonlinearBlockSolverBase {
 public:
  using NonlinearBlockSolverBase::convergenceStatus;

  FakeNonlinearBlockSolver(double abs_tol, double rel_tol, BlockConvergenceTolerances block_tolerances = {})
      : abs_tol_(abs_tol), rel_tol_(rel_tol), block_tolerances_(std::move(block_tolerances))
  {
  }

  void completeSetup(const std::vector<FieldT>&) override {}

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
                                      const BlockConvergenceTolerances& tolerance_overrides,
                                      NonlinearConvergenceContext& context) const override
  {
    auto relative_tols = effectiveRelativeTolerances(residuals.size(), tolerance_overrides);
    auto absolute_tols = effectiveAbsoluteTolerances(residuals.size(), tolerance_overrides);
    bool block_path_enabled = !tolerance_overrides.relative_tols.empty() || !tolerance_overrides.absolute_tols.empty() ||
                              !block_tolerances_.relative_tols.empty() || !block_tolerances_.absolute_tols.empty();
    auto block_norms = computeResidualBlockNorms(residuals, MPI_COMM_SELF);
    return evaluateResidualConvergence(tolerance_multiplier, abs_tol_, rel_tol_, absolute_tols, relative_tols,
                                       block_path_enabled, block_norms, context);
  }

  void primeConvergenceContext(const std::vector<mfem::Vector>& residuals,
                               const BlockConvergenceTolerances& tolerance_overrides,
                               NonlinearConvergenceContext& context) const override
  {
    static_cast<void>(convergenceStatus(1.0, residuals, tolerance_overrides, context));
  }

  std::vector<double> effectiveRelativeTolerances(size_t num_blocks,
                                                  const BlockConvergenceTolerances& tolerance_overrides) const override
  {
    return expandPerBlockTolerances(
        tolerance_overrides.relative_tols.empty() ? block_tolerances_.relative_tols : tolerance_overrides.relative_tols,
        num_blocks, 0.0, "relative block tolerances");
  }

  std::vector<double> effectiveAbsoluteTolerances(size_t num_blocks,
                                                  const BlockConvergenceTolerances& tolerance_overrides) const override
  {
    return expandPerBlockTolerances(
        tolerance_overrides.absolute_tols.empty() ? block_tolerances_.absolute_tols : tolerance_overrides.absolute_tols,
        num_blocks, 0.0, "absolute block tolerances");
  }

 private:
  double abs_tol_;
  double rel_tol_;
  BlockConvergenceTolerances block_tolerances_;
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

TEST(SolverConvergence, PerBlockTolerancesRequireAllBlocksToPass)
{
  FakeNonlinearBlockSolver solver(1.0e-12, 0.1, {.relative_tols = {0.5, 0.01}, .absolute_tols = {0.0, 0.0}});
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({1.0, 1.0})));
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({0.49, 0.25})));
  EXPECT_TRUE(solver.checkConvergence(1.0, makeResiduals({0.49, 0.009})));
  EXPECT_TRUE(solver.checkConvergence(1.0, makeResiduals({0.49, 0.0})));
}

TEST(SolverConvergence, EmptyBlockTolerancesPreserveScalarOnlyBehavior)
{
  FakeNonlinearBlockSolver solver(0.2, 0.1);
  auto status = solver.convergenceStatus(1.0, makeResiduals({0.15, 0.15}));
  EXPECT_FALSE(status.block_path_enabled);
  EXPECT_FALSE(status.block_converged);
  EXPECT_FALSE(status.converged);

  solver.resetConvergenceState();
  status = solver.convergenceStatus(1.0, makeResiduals({0.05, 0.05}));
  EXPECT_TRUE(status.global_converged);
  EXPECT_TRUE(status.converged);
}

TEST(SolverConvergence, OrSemanticsAllowGlobalOrBlockConvergence)
{
  FakeNonlinearBlockSolver solver_global(0.3, 0.1, {.relative_tols = {0.2, 0.2}, .absolute_tols = {0.0, 0.0}});

  auto status = solver_global.convergenceStatus(1.0, makeResiduals({1.0, 1.0}));
  EXPECT_FALSE(status.converged);

  status = solver_global.convergenceStatus(1.0, makeResiduals({0.05, 0.25}));
  EXPECT_TRUE(status.global_converged);
  EXPECT_FALSE(status.block_converged);
  EXPECT_TRUE(status.converged);

  FakeNonlinearBlockSolver solver_block(0.05, 0.01, {.relative_tols = {0.2, 0.2}, .absolute_tols = {0.0, 0.0}});
  status = solver_block.convergenceStatus(1.0, makeResiduals({1.0, 1.0}));
  EXPECT_FALSE(status.converged);

  status = solver_block.convergenceStatus(1.0, makeResiduals({0.19, 0.19}));
  EXPECT_FALSE(status.global_converged);
  EXPECT_TRUE(status.block_converged);
  EXPECT_TRUE(status.converged);
}

TEST(SolverConvergence, ResetConvergenceStateRefreshesPerBlockInitialNorms)
{
  FakeNonlinearBlockSolver solver(1.0e-12, 0.01, {.relative_tols = {0.1, 0.1}, .absolute_tols = {0.0, 0.0}});
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({10.0, 10.0})));
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({2.0, 2.0})));
  solver.resetConvergenceState();
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({2.0, 2.0})));
  EXPECT_TRUE(solver.checkConvergence(1.0, makeResiduals({0.19, 0.19})));
  EXPECT_TRUE(solver.checkConvergence(1.0, makeResiduals({0.19, 0.0})));
}

TEST(SolverConvergence, StageConstructorRejectsMismatchedToleranceSizes)
{
  auto solver = std::make_shared<FakeNonlinearBlockSolver>(1.0e-12, 1.0e-8);
  EXPECT_DEATH(
      {
        CoupledSystemSolver staggered_solver(2);
        staggered_solver.addSubsystemSolver({0, 1}, solver, {.relative_tols = {1.0e-3}, .absolute_tols = {1.0e-6}});
      },
      "does not match number of stage blocks");
}

TEST(SolverConvergence, StageConstructorAllowsOverridesWhenSolverHasNoBlockTolerances)
{
  auto solver =
      std::make_shared<NonlinearBlockSolver>(std::unique_ptr<EquationSolver>{}, MPI_COMM_SELF, 1.0e-6, 1.0e-3);
  CoupledSystemSolver staggered_solver(2);
  EXPECT_NO_THROW(staggered_solver.addSubsystemSolver({0}, solver, {.relative_tols = {1.0e-4}}));
  EXPECT_NO_THROW(staggered_solver.addSubsystemSolver({0}, solver, {.absolute_tols = {1.0e-7}}));
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
