// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/functional_weak_form.hpp"

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"

namespace smith {

namespace {

class RecordingRampSolver : public NonlinearBlockSolverBase {
 public:
  using NonlinearBlockSolverBase::convergenceStatus;

  explicit RecordingRampSolver(std::vector<bool> converged_by_call) : converged_by_call_(std::move(converged_by_call))
  {
  }

  std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses, std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)>,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)>) const override
  {
    intermediate_flags_.push_back(intermediate_enabled_);
    abs_tol_factors_.push_back(intermediate_abs_tol_factor_);
    rel_tol_floors_.push_back(intermediate_rel_tol_floor_);
    max_iterations_.push_back(intermediate_max_iterations_);

    const size_t idx = intermediate_flags_.size() - 1;
    last_solve_converged_ = idx < converged_by_call_.size() ? converged_by_call_[idx] : true;
    return u_guesses;
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

  void setInnerToleranceMultiplier(double multiplier) override { inner_tol_multiplier_ = multiplier; }

  void setIntermediateTolerancePolicy(bool enabled, double abs_tol_factor, double rel_tol_floor,
                                      int max_iterations) override
  {
    intermediate_enabled_ = enabled;
    intermediate_abs_tol_factor_ = abs_tol_factor;
    intermediate_rel_tol_floor_ = rel_tol_floor;
    intermediate_max_iterations_ = max_iterations;
  }

  const std::vector<bool>& intermediateFlags() const { return intermediate_flags_; }

  const std::vector<double>& absTolFactors() const { return abs_tol_factors_; }

  const std::vector<double>& relTolFloors() const { return rel_tol_floors_; }

  const std::vector<int>& maxIterations() const { return max_iterations_; }

  double innerToleranceMultiplier() const { return inner_tol_multiplier_; }

 private:
  std::vector<bool> converged_by_call_;
  mutable std::vector<bool> intermediate_flags_;
  mutable std::vector<double> abs_tol_factors_;
  mutable std::vector<double> rel_tol_floors_;
  mutable bool intermediate_enabled_ = false;
  mutable double intermediate_abs_tol_factor_ = 1.0;
  mutable double intermediate_rel_tol_floor_ = 0.0;
  mutable int intermediate_max_iterations_ = 0;
  mutable double inner_tol_multiplier_ = 1.0;
  mutable std::vector<int> max_iterations_;
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
    bc_managers = field_store->getBoundaryConditionManagers(residual_names);
  }

  ~ScalarRampHarness() { StateManager::reset(); }
};

}  // namespace

TEST(BcRampOptionsTest, DefaultsAreDisabled)
{
  BcRampOptions opts{};
  EXPECT_FALSE(opts.enabled);
  EXPECT_GT(opts.shrink_factor, 0.0);
  EXPECT_LT(opts.shrink_factor, 1.0);
  EXPECT_GT(opts.max_cutbacks, 0);
  EXPECT_EQ(opts.intermediate_max_iterations, 10);
  EXPECT_DOUBLE_EQ(opts.intermediate_relative_tol, 0.05);
  EXPECT_DOUBLE_EQ(opts.intermediate_absolute_tol_fac, 1e3);
}

TEST(BcRampOptionsTest, SetGetRoundTrip)
{
  SystemSolver solver(/*max_staggered_iterations=*/1);
  BcRampOptions opts{.enabled = true,
                     .shrink_factor = 0.25,
                     .j_floor = 1e-6,
                     .max_cutbacks = 8,
                     .intermediate_max_iterations = 7,
                     .intermediate_relative_tol = 0.75,
                     .intermediate_absolute_tol_fac = 42.0};
  solver.setBcRampOptions(opts);

  const auto& got = solver.bcRampOptions();
  EXPECT_TRUE(got.enabled);
  EXPECT_DOUBLE_EQ(got.shrink_factor, 0.25);
  EXPECT_DOUBLE_EQ(got.j_floor, 1e-6);
  EXPECT_EQ(got.max_cutbacks, 8);
  EXPECT_EQ(got.intermediate_max_iterations, 7);
  EXPECT_DOUBLE_EQ(got.intermediate_relative_tol, 0.75);
  EXPECT_DOUBLE_EQ(got.intermediate_absolute_tol_fac, 42.0);
}

TEST(BcRampOptionsTest, HiddenSolveCounterStartsZero)
{
  SystemSolver solver(/*max_staggered_iterations=*/1);
  EXPECT_EQ(solver.lastHiddenSolveCount(), 0);
}

TEST(BcRampOptionsTest, ClearCacheCompiles)
{
  SystemSolver solver(/*max_staggered_iterations=*/1);
  solver.clearBcRampCache();
}

TEST(BcRamp, CutbackUsesIntermediateTolerancePolicy)
{
  ScalarRampHarness harness;

  auto recording_solver = std::make_shared<RecordingRampSolver>(std::vector<bool>{false, true, true});
  auto system_solver = std::make_shared<SystemSolver>(recording_solver);
  BcRampOptions opts;
  opts.enabled = true;
  opts.max_cutbacks = 4;
  opts.intermediate_max_iterations = 10;
  opts.intermediate_relative_tol = 0.9;
  opts.intermediate_absolute_tol_fac = 1e3;
  system_solver->setBcRampOptions(opts);

  auto solved_states =
      system_solver->solve(harness.residuals, harness.block_indices, harness.field_store->getShapeDisp(),
                           harness.states, harness.params, TimeInfo(0.0, 1.0, 0), harness.bc_managers);

  EXPECT_EQ(solved_states.size(), 1);
  EXPECT_EQ(recording_solver->intermediateFlags().size(), 3);
  EXPECT_EQ(recording_solver->intermediateFlags(), std::vector<bool>({false, true, false}));
  EXPECT_EQ(recording_solver->maxIterations(), std::vector<int>({10, 10, 10}));
  EXPECT_DOUBLE_EQ(recording_solver->absTolFactors()[1], 1e3);
  EXPECT_DOUBLE_EQ(recording_solver->relTolFloors()[1], 0.9);
  EXPECT_EQ(system_solver->lastHiddenSolveCount(), 2);
}

TEST(BcRamp, RampDisabledIsBitIdentical)
{
  ScalarRampHarness harness;

  auto base_solver = std::make_shared<RecordingRampSolver>(std::vector<bool>{true});
  auto disabled_solver = std::make_shared<RecordingRampSolver>(std::vector<bool>{true});

  SystemSolver base_system(base_solver);
  SystemSolver explicit_disabled_system(disabled_solver);
  explicit_disabled_system.setBcRampOptions(BcRampOptions{.enabled = false});

  auto base_states = base_system.solve(harness.residuals, harness.block_indices, harness.field_store->getShapeDisp(),
                                       harness.states, harness.params, TimeInfo(0.0, 1.0, 0), harness.bc_managers);
  auto disabled_states =
      explicit_disabled_system.solve(harness.residuals, harness.block_indices, harness.field_store->getShapeDisp(),
                                     harness.states, harness.params, TimeInfo(0.0, 1.0, 0), harness.bc_managers);

  ASSERT_EQ(base_states.size(), disabled_states.size());
  for (size_t i = 0; i < base_states.size(); ++i) {
    ASSERT_EQ(base_states[i].get()->Size(), disabled_states[i].get()->Size());
    for (int j = 0; j < base_states[i].get()->Size(); ++j) {
      EXPECT_DOUBLE_EQ((*base_states[i].get())[j], (*disabled_states[i].get())[j]);
    }
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
