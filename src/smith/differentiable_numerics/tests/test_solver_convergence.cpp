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

  bool checkConvergence(double tolerance_multiplier, const std::vector<mfem::Vector>& residuals) const override
  {
    return checkConvergence(tolerance_multiplier, residuals, {});
  }

  bool checkConvergence(double tolerance_multiplier, const std::vector<mfem::Vector>& residuals,
                        const BlockConvergenceTolerances& tolerance_overrides) const override
  {
    auto relative_tols = effectiveRelativeTolerances(residuals.size(), tolerance_overrides);
    auto absolute_tols = effectiveAbsoluteTolerances(residuals.size(), tolerance_overrides);

    if (initial_residual_norms_.empty()) {
      initial_residual_norms_.resize(residuals.size(), 0.0);
    }

    for (size_t i = 0; i < residuals.size(); ++i) {
      double residual_norm = residuals[i].Norml2();
      if (initial_residual_norms_[i] == 0.0) {
        initial_residual_norms_[i] = residual_norm;
      }
      double tol = std::max(absolute_tols[i], relative_tols[i] * initial_residual_norms_[i]);
      if (residual_norm > tolerance_multiplier * tol) {
        return false;
      }
    }

    return true;
  }

  void resetConvergenceState() const override { initial_residual_norms_.clear(); }

  std::vector<double> effectiveRelativeTolerances(size_t num_blocks,
                                                  const BlockConvergenceTolerances& tolerance_overrides) const override
  {
    return expandTolerances(
        tolerance_overrides.relative_tols.empty() ? block_tolerances_.relative_tols : tolerance_overrides.relative_tols,
        rel_tol_, num_blocks, "relative block tolerances");
  }

  std::vector<double> effectiveAbsoluteTolerances(size_t num_blocks,
                                                  const BlockConvergenceTolerances& tolerance_overrides) const override
  {
    return expandTolerances(
        tolerance_overrides.absolute_tols.empty() ? block_tolerances_.absolute_tols : tolerance_overrides.absolute_tols,
        abs_tol_, num_blocks, "absolute block tolerances");
  }

 private:
  static std::vector<double> expandTolerances(const std::vector<double>& block_tols, double scalar_tol,
                                              size_t num_blocks, const std::string& name)
  {
    if (block_tols.empty()) {
      return std::vector<double>(num_blocks, scalar_tol);
    }

    SLIC_ERROR_IF(block_tols.size() != num_blocks,
                  axom::fmt::format("{} size {} does not match number of residual blocks {}", name, block_tols.size(),
                                    num_blocks));
    return block_tols;
  }

  double abs_tol_;
  double rel_tol_;
  BlockConvergenceTolerances block_tolerances_;
  mutable std::vector<double> initial_residual_norms_;
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
  FakeNonlinearBlockSolver solver(1.0e-12, 1.0, {.relative_tols = {0.5, 0.01}, .absolute_tols = {0.0, 0.0}});
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({1.0, 1.0})));
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({0.49, 0.25})));
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({0.49, 0.009})));
  EXPECT_TRUE(solver.checkConvergence(1.0, makeResiduals({0.49, 0.0})));
}

TEST(SolverConvergence, ResetConvergenceStateRefreshesPerBlockInitialNorms)
{
  FakeNonlinearBlockSolver solver(1.0e-12, 1.0, {.relative_tols = {0.1, 0.1}, .absolute_tols = {0.0, 0.0}});
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({10.0, 10.0})));
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({2.0, 2.0})));
  solver.resetConvergenceState();
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({2.0, 2.0})));
  EXPECT_FALSE(solver.checkConvergence(1.0, makeResiduals({0.19, 0.19})));
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

TEST(SolverConvergence, StageConstructorRejectsTighterRelativeTolerance)
{
  auto solver =
      std::make_shared<NonlinearBlockSolver>(std::unique_ptr<EquationSolver>{}, MPI_COMM_SELF, 1.0e-6, 1.0e-3);
  EXPECT_DEATH(
      {
        CoupledSystemSolver staggered_solver(2);
        staggered_solver.addSubsystemSolver({0}, solver, {.relative_tols = {1.0e-4}});
      },
      "relative tolerance");
}

TEST(SolverConvergence, StageConstructorRejectsTighterAbsoluteTolerance)
{
  auto solver =
      std::make_shared<NonlinearBlockSolver>(std::unique_ptr<EquationSolver>{}, MPI_COMM_SELF, 1.0e-6, 1.0e-3);
  EXPECT_DEATH(
      {
        CoupledSystemSolver staggered_solver(2);
        staggered_solver.addSubsystemSolver({0}, solver, {.absolute_tols = {1.0e-7}});
      },
      "absolute tolerance");
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
