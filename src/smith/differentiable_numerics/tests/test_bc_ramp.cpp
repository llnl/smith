// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <exception>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/functional_weak_form.hpp"

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"

class SlicErrorException : public std::exception {};

namespace smith {

namespace {

class RecordingRampSolver : public NonlinearBlockSolverBase {
 public:
  using NonlinearBlockSolverBase::convergenceStatus;

  RecordingRampSolver(std::vector<bool> converged_by_call, mfem::Array<int> constrained_tdofs,
                      bool warm_start_enabled = false, bool prediction_succeeds = true)
      : converged_by_call_(std::move(converged_by_call)),
        constrained_tdofs_(std::move(constrained_tdofs)),
        warm_start_enabled_(warm_start_enabled),
        prediction_succeeds_(prediction_succeeds)
  {
  }

  std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses, std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)>,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)>) const override
  {
    attempt_alphas_.push_back(constrainedValue(*u_guesses[0]));
    const size_t call = attempt_alphas_.size() - 1;
    last_solve_converged_ = call < converged_by_call_.size() ? converged_by_call_[call] : true;

    std::vector<FieldPtr> results;
    for (const auto& guess : u_guesses) results.push_back(std::make_shared<FiniteElementState>(*guess));
    return results;
  }

  std::vector<FieldPtr> solveAdjoint(const std::vector<DualPtr>&, std::vector<std::vector<MatrixPtr>>&) const override
  {
    return {};
  }

  ConvergenceStatus convergenceStatus(double, const std::vector<mfem::Vector>& residuals,
                                      NonlinearConvergenceContext&) const override
  {
    ConvergenceStatus status;
    status.global_converged = true;
    status.converged = true;
    status.block_norms.resize(residuals.size(), 0.0);
    return status;
  }

  void primeConvergenceContext(const std::vector<mfem::Vector>&, NonlinearConvergenceContext&) const override {}

  void setInnerToleranceMultiplier(double multiplier) override { static_cast<void>(multiplier); }

  bool warmStartEnabled() const override { return warm_start_enabled_; }

  FieldPtr predictBcStep(const FieldPtr& base_state, const FieldPtr& target_bc, mfem::HypreParMatrix&,
                         const mfem::Array<int>&) const override
  {
    prediction_base_alphas_.push_back(constrainedValue(*base_state));
    predicted_alphas_.push_back(constrainedValue(*target_bc));
    return prediction_succeeds_ ? std::make_shared<FiniteElementState>(*base_state) : nullptr;
  }

  const std::vector<double>& attemptAlphas() const { return attempt_alphas_; }
  const std::vector<double>& predictionBaseAlphas() const { return prediction_base_alphas_; }
  const std::vector<double>& predictedAlphas() const { return predicted_alphas_; }

 private:
  double constrainedValue(const FiniteElementState& field) const
  {
    EXPECT_GT(constrained_tdofs_.Size(), 0);
    return field[constrained_tdofs_[0]];
  }

  std::vector<bool> converged_by_call_;
  mfem::Array<int> constrained_tdofs_;
  mutable std::vector<double> attempt_alphas_;
  mutable std::vector<double> prediction_base_alphas_;
  mutable std::vector<double> predicted_alphas_;
  bool warm_start_enabled_ = false;
  bool prediction_succeeds_ = true;
};

template <typename FieldTypeT>
auto buildScalarDiffusionWeakForm(const std::string& name, std::shared_ptr<Mesh> mesh, std::shared_ptr<FieldStore> fs,
                                  FieldTypeT field_type)
{
  using WeakFormType = FunctionalWeakForm<2, H1<1>, Parameters<H1<1>>>;
  auto weak_form = std::make_shared<WeakFormType>(name, mesh, fs->getField(field_type.name).get()->space(),
                                                  fs->createSpaces(name, field_type.name, field_type));
  weak_form->addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(),
                             [](auto, auto, auto u) { return smith::tuple{0.0 * get<VALUE>(u), get<DERIVATIVE>(u)}; });
  return weak_form;
}

struct ScalarRampHarness {
  axom::sidre::DataStore datastore;
  std::shared_ptr<Mesh> mesh;
  std::shared_ptr<FieldStore> field_store;
  std::shared_ptr<WeakForm> weak_form;
  std::vector<WeakForm*> residuals;
  std::vector<std::string> residual_names;
  std::vector<std::vector<size_t>> block_indices;
  std::vector<std::vector<FieldState>> states;
  std::vector<std::vector<FieldState>> params;
  std::unique_ptr<DirichletBoundaryConditions> boundary_conditions;
  std::vector<const BoundaryConditionManager*> bc_managers;

  ScalarRampHarness()
  {
    StateManager::initialize(datastore, "bc_ramp");
    mesh = std::make_shared<Mesh>(mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0),
                                  "bc_ramp_mesh");
    field_store = std::make_shared<FieldStore>(mesh, 20);

    FieldType<H1<1, 2>> shape_disp_type("shape_displacement");
    field_store->addShapeDisp(shape_disp_type);

    auto quasi_static = std::make_shared<QuasiStaticRule>();
    FieldType<H1<1>> field_type("temperature");
    field_store->addIndependent(field_type, quasi_static);

    weak_form = buildScalarDiffusionWeakForm("temperature_main", mesh, field_store, field_type);
    residuals = {weak_form.get()};
    residual_names = {"temperature_main"};
    block_indices = field_store->indexMap(residual_names);
    states = {field_store->getStates("temperature_main")};
    params = std::vector<std::vector<FieldState>>(residuals.size());
    boundary_conditions =
        std::make_unique<DirichletBoundaryConditions>(*mesh, field_store->getField("temperature").get()->space());
    boundary_conditions->setScalarBCs<2>(mesh->entireBoundary(), [](double time, tensor<double, 2>) { return time; });
    bc_managers = {&boundary_conditions->getBoundaryConditionManager()};
  }

  const mfem::Array<int>& constrainedDofs() const { return bc_managers[0]->allEssentialTrueDofs(); }

  std::vector<FieldState> solve(const std::shared_ptr<RecordingRampSolver>& solver)
  {
    SystemSolver system_solver(solver);
    return system_solver.solve(residuals, block_indices, field_store->getShapeDisp(), states, params,
                               TimeInfo(0.0, 1.0, 1), bc_managers);
  }

  ~ScalarRampHarness() { StateManager::reset(); }
};

}  // namespace

TEST(BcRampOptionsTest, Defaults)
{
  BcRampOptions opts{};
  EXPECT_GT(opts.shrink_factor, 0.0);
  EXPECT_LT(opts.shrink_factor, 1.0);
  EXPECT_EQ(opts.max_cutbacks, 0);
}

TEST(BcRampOptionsTest, SetGetRoundTrip)
{
  ScalarRampHarness harness;
  RecordingRampSolver solver(std::vector<bool>{true}, harness.constrainedDofs());
  BcRampOptions opts{.shrink_factor = 0.25, .max_cutbacks = 8};
  solver.setBcRampOptions(opts);

  const auto& got = solver.bcRampOptions();
  EXPECT_DOUBLE_EQ(got.shrink_factor, 0.25);
  EXPECT_EQ(got.max_cutbacks, 8);
}

TEST(BcRamp, DisabledPerformsSingleFailedAttempt)
{
  ScalarRampHarness harness;
  auto solver = std::make_shared<RecordingRampSolver>(std::vector<bool>{false}, harness.constrainedDofs());

  auto solved_states = harness.solve(solver);

  EXPECT_EQ(solved_states.size(), 1);
  EXPECT_EQ(solver->attemptAlphas(), std::vector<double>({1.0}));
  EXPECT_TRUE(solver->predictedAlphas().empty());
}

TEST(BcRamp, WarmStartAlonePreservesRequestedFraction)
{
  ScalarRampHarness harness;
  auto solver = std::make_shared<RecordingRampSolver>(std::vector<bool>{false}, harness.constrainedDofs(), true);

  auto solved_states = harness.solve(solver);

  EXPECT_EQ(solved_states.size(), 1);
  EXPECT_EQ(solver->attemptAlphas(), std::vector<double>({1.0}));
  EXPECT_EQ(solver->predictedAlphas(), std::vector<double>({1.0}));
}

TEST(BcRamp, CutbackRetriesReducedThenRequestedFraction)
{
  ScalarRampHarness harness;
  auto solver = std::make_shared<RecordingRampSolver>(std::vector<bool>{false, true, true}, harness.constrainedDofs());
  solver->setBcRampOptions(BcRampOptions{.max_cutbacks = 2});

  auto solved_states = harness.solve(solver);

  ASSERT_EQ(solved_states.size(), 1);
  EXPECT_EQ(solver->attemptAlphas(), std::vector<double>({1.0, 0.5, 1.0}));
  EXPECT_TRUE(solver->predictedAlphas().empty());
  for (int dof : harness.constrainedDofs()) EXPECT_DOUBLE_EQ((*solved_states[0].get())[dof], 1.0);
}

TEST(BcRamp, CombinedModePredictsEveryRequestedFraction)
{
  ScalarRampHarness harness;
  auto solver =
      std::make_shared<RecordingRampSolver>(std::vector<bool>{false, true, true}, harness.constrainedDofs(), true);
  solver->setBcRampOptions(BcRampOptions{.max_cutbacks = 2});

  auto solved_states = harness.solve(solver);

  EXPECT_EQ(solved_states.size(), 1);
  EXPECT_EQ(solver->attemptAlphas(), std::vector<double>({1.0, 0.5, 1.0}));
  EXPECT_EQ(solver->predictionBaseAlphas(), std::vector<double>({0.0, 0.0, 0.5}));
  EXPECT_EQ(solver->predictedAlphas(), solver->attemptAlphas());
}

TEST(BcRamp, SuccessfulStepBecomesCutbackAnchor)
{
  ScalarRampHarness harness;
  auto solver = std::make_shared<RecordingRampSolver>(std::vector<bool>{false, true, false, true, true},
                                                      harness.constrainedDofs());
  solver->setBcRampOptions(BcRampOptions{.max_cutbacks = 2});

  auto solved_states = harness.solve(solver);

  EXPECT_EQ(solved_states.size(), 1);
  EXPECT_EQ(solver->attemptAlphas(), std::vector<double>({1.0, 0.5, 1.0, 0.75, 1.0}));
}

TEST(BcRamp, FailedPredictionFallsBackWithoutChangingFraction)
{
  ScalarRampHarness harness;
  auto solver = std::make_shared<RecordingRampSolver>(std::vector<bool>{false, true, true}, harness.constrainedDofs(),
                                                      true, false);
  solver->setBcRampOptions(BcRampOptions{.max_cutbacks = 1});

  auto solved_states = harness.solve(solver);

  EXPECT_EQ(solved_states.size(), 1);
  EXPECT_EQ(solver->attemptAlphas(), std::vector<double>({1.0, 0.5, 1.0}));
  EXPECT_EQ(solver->predictedAlphas(), solver->attemptAlphas());
}

TEST(BcRamp, ExhaustionReportsFailure)
{
  ScalarRampHarness harness;
  auto solver =
      std::make_shared<RecordingRampSolver>(std::vector<bool>{false, false, false}, harness.constrainedDofs());
  solver->setBcRampOptions(BcRampOptions{.max_cutbacks = 2});

  EXPECT_THROW(harness.solve(solver), SlicErrorException);
  EXPECT_EQ(solver->attemptAlphas(), std::vector<double>({1.0, 0.5, 0.25}));
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
  return RUN_ALL_TESTS();
}
