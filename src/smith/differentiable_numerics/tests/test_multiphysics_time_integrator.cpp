// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <cmath>
#include <memory>

#include "gtest/gtest.h"

#include "gretl/data_store.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/physics/functional_weak_form.hpp"

namespace smith {

namespace {

class NoOpNonlinearBlockSolver : public NonlinearBlockSolverBase {
 public:
  std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses, std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)>,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)>) const override
  {
    ++solve_calls_;
    last_num_unknowns_ = static_cast<int>(u_guesses.size());
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

  int solveCalls() const { return solve_calls_; }

  int lastNumUnknowns() const { return last_num_unknowns_; }

  void setInnerToleranceMultiplier(double) override {}

 private:
  mutable int solve_calls_ = 0;
  mutable int last_num_unknowns_ = -1;
};

class CountingNoOpNonlinearBlockSolver : public NoOpNonlinearBlockSolver {
 public:
  std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses, std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)> f,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)> j) const override
  {
    return NoOpNonlinearBlockSolver::solve(u_guesses, std::move(f), std::move(j));
  }
};

class NeedsInitialSolveRule : public QuasiStaticRule {
 public:
  bool requiresInitialAccelerationSolve() const override { return true; }
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

template <typename DispFieldType, typename DispOldFieldType, typename VelocityFieldType, typename AccelerationFieldType>
auto buildSecondOrderMainWeakForm(const std::string& name, std::shared_ptr<Mesh> mesh, std::shared_ptr<FieldStore> fs,
                                  DispFieldType displacement_type, DispOldFieldType displacement_old_type,
                                  VelocityFieldType velocity_type, AccelerationFieldType acceleration_type)
{
  using WeakFormType = FunctionalWeakForm<2, H1<1>, Parameters<H1<1>, H1<1>, H1<1>, H1<1>>>;
  auto weak_form =
      std::make_shared<WeakFormType>(name, mesh, fs->getField(displacement_type.name).get()->space(),
                                     fs->createSpaces(name, displacement_type.name, displacement_type,
                                                      displacement_old_type, velocity_type, acceleration_type));
  weak_form->addBodySource(mesh->entireBodyName(), [](auto, auto, auto...) { return 0.0; });
  return weak_form;
}

template <typename DispFieldType, typename VelocityFieldType, typename AccelerationFieldType>
auto buildSecondOrderCycleZeroWeakForm(const std::string& name, std::shared_ptr<Mesh> mesh,
                                       std::shared_ptr<FieldStore> fs, DispFieldType displacement_type,
                                       VelocityFieldType velocity_type, AccelerationFieldType acceleration_type)
{
  using WeakFormType = FunctionalWeakForm<2, H1<1>, Parameters<H1<1>, H1<1>, H1<1>>>;
  auto weak_form = std::make_shared<WeakFormType>(
      name, mesh, fs->getField(acceleration_type.name).get()->space(),
      fs->createSpaces(name, acceleration_type.name, displacement_type, velocity_type, acceleration_type));
  weak_form->addBodySource(mesh->entireBodyName(), [](auto, auto, auto...) { return 0.0; });
  return weak_form;
}

bool allNodesOnBoundary(const std::vector<vec2>& nodes, double x_target)
{
  constexpr double tol = 1.0e-12;
  return std::all_of(nodes.begin(), nodes.end(), [x_target](const vec2& x) { return std::abs(x[0] - x_target) < tol; });
}

}  // namespace

TEST(MultiphysicsTimeIntegrator, CycleZeroUsesBcsForReactionFieldNotUnknownZero)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "multiphysics_time_integrator");

  auto mesh = std::make_shared<Mesh>(mfem::Mesh::MakeCartesian2D(8, 8, mfem::Element::QUADRILATERAL, true, 1.0, 1.0),
                                     "integrator_mesh");
  mesh->addDomainOfBoundaryElements("left",
                                    [](std::vector<vec2> nodes, int) { return allNodesOnBoundary(nodes, 0.0); });
  mesh->addDomainOfBoundaryElements("right",
                                    [](std::vector<vec2> nodes, int) { return allNodesOnBoundary(nodes, 1.0); });

  auto field_store = std::make_shared<FieldStore>(mesh, 20);
  FieldType<H1<1, 2>> shape_disp_type("shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  auto quasi_static = std::make_shared<NeedsInitialSolveRule>();
  FieldType<H1<1>> temperature_type("temperature");
  auto temperature_bc = field_store->addIndependent(temperature_type, quasi_static);
  FieldType<H1<1>> displacement_type("displacement");
  auto displacement_bc = field_store->addIndependent(displacement_type, quasi_static);

  ASSERT_TRUE(temperature_type.is_unknown);
  ASSERT_TRUE(displacement_type.is_unknown);

  temperature_bc->setScalarBCs<2>(mesh->domain("left"), [](double, tensor<double, 2>) { return 0.0; });
  displacement_bc->setScalarBCs<2>(mesh->domain("right"), [](double, tensor<double, 2>) { return 1.0; });

  auto temperature_wf = buildScalarDiffusionWeakForm("temperature_main", mesh, field_store, temperature_type);
  auto displacement_wf = buildScalarDiffusionWeakForm("displacement_main", mesh, field_store, displacement_type);
  auto cycle_zero_wf = buildScalarDiffusionWeakForm("cycle_zero_displacement", mesh, field_store, displacement_type);

  auto main_solver = std::make_shared<SystemSolver>(std::make_shared<NoOpNonlinearBlockSolver>());

  LinearSolverOptions lin_opts{.linear_solver = LinearSolver::SuperLU};
  NonlinearSolverOptions nonlin_opts{
      .nonlin_solver = NonlinearSolver::Newton, .relative_tol = 1.0e-12, .absolute_tol = 1.0e-12, .max_iterations = 8};
  auto cycle_zero_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh);
  auto cycle_zero_solver = std::make_shared<SystemSolver>(cycle_zero_block_solver);

  auto main_system = std::make_shared<SystemBase>();
  main_system->field_store = field_store;
  main_system->weak_forms = {temperature_wf, displacement_wf};
  main_system->solver = main_solver;

  auto cycle_zero_system = std::make_shared<SystemBase>();
  cycle_zero_system->field_store = field_store;
  cycle_zero_system->weak_forms = {cycle_zero_wf};
  cycle_zero_system->solver = cycle_zero_solver;

  MultiphysicsTimeIntegrator advancer(main_system, {cycle_zero_system});

  auto [new_states, reactions] =
      advancer.advanceState(TimeInfo(0.0, 1.0, 0), field_store->getShapeDisp(), field_store->getAllFields(), {});
  static_cast<void>(reactions);

  const auto& displacement = *new_states[field_store->getFieldIndex("displacement")].get();
  auto essential_dofs = displacement_bc->getBoundaryConditionManager().allEssentialTrueDofs();
  ASSERT_GT(essential_dofs.Size(), 0);
  for (int i = 0; i < essential_dofs.Size(); ++i) {
    EXPECT_NEAR(displacement(essential_dofs[i]), 1.0, 1.0e-10);
  }

  StateManager::reset();
}

TEST(MultiphysicsTimeIntegrator, CycleZeroSkippedForQuasiStaticSecondOrderRule)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "multiphysics_time_integrator_quasistatic_second_order");

  auto mesh = std::make_shared<Mesh>(mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 1.0),
                                     "integrator_mesh");

  auto field_store = std::make_shared<FieldStore>(mesh, 20);
  FieldType<H1<1, 2>> shape_disp_type("shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  auto quasi_static = std::make_shared<QuasiStaticSecondOrderTimeIntegrationRule>();
  FieldType<H1<1>> displacement_type("displacement_solve_state");
  field_store->addIndependent(displacement_type, quasi_static);
  auto displacement_old_type =
      field_store->addDependent(displacement_type, FieldStore::TimeDerivative::VAL, "displacement");
  auto velocity_type = field_store->addDependent(displacement_type, FieldStore::TimeDerivative::DOT, "velocity");
  auto acceleration_type =
      field_store->addDependent(displacement_type, FieldStore::TimeDerivative::DDOT, "acceleration");

  auto main_wf = buildSecondOrderMainWeakForm("displacement_main", mesh, field_store, displacement_type,
                                              displacement_old_type, velocity_type, acceleration_type);
  auto cycle_zero_wf = buildSecondOrderCycleZeroWeakForm("cycle_zero_acceleration", mesh, field_store,
                                                         displacement_type, velocity_type, acceleration_type);

  auto main_solver = std::make_shared<SystemSolver>(std::make_shared<NoOpNonlinearBlockSolver>());
  auto cycle_zero_block_solver = std::make_shared<CountingNoOpNonlinearBlockSolver>();
  auto cycle_zero_solver = std::make_shared<SystemSolver>(cycle_zero_block_solver);

  auto main_system = std::make_shared<SystemBase>();
  main_system->field_store = field_store;
  main_system->weak_forms = {main_wf};
  main_system->solver = main_solver;

  auto cycle_zero_system = std::make_shared<SystemBase>();
  cycle_zero_system->field_store = field_store;
  cycle_zero_system->weak_forms = {cycle_zero_wf};
  cycle_zero_system->solver = cycle_zero_solver;

  MultiphysicsTimeIntegrator advancer(main_system, {cycle_zero_system});

  auto [new_states, reactions] =
      advancer.advanceState(TimeInfo(0.0, 1.0, 0), field_store->getShapeDisp(), field_store->getAllFields(), {});
  static_cast<void>(new_states);
  static_cast<void>(reactions);

  EXPECT_EQ(cycle_zero_block_solver->solveCalls(), 0);

  StateManager::reset();
}

TEST(MultiphysicsTimeIntegrator, CycleZeroAccelerationBcUsesDisplacementSecondTimeDerivative)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "cycle_zero_acceleration_bc_uses_displacement_second_time_derivative");

  auto mesh = std::make_shared<Mesh>(mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 1.0),
                                     "cycle_zero_bc_mesh");
  mesh->addDomainOfBoundaryElements("left",
                                    [](std::vector<vec2> nodes, int) { return allNodesOnBoundary(nodes, 0.0); });
  mesh->addDomainOfBoundaryElements("right",
                                    [](std::vector<vec2> nodes, int) { return allNodesOnBoundary(nodes, 1.0); });

  auto field_store = std::make_shared<FieldStore>(mesh, 20);
  FieldType<H1<1, 2>> shape_disp_type("shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  auto time_rule = std::make_shared<QuasiStaticSecondOrderTimeIntegrationRule>();
  FieldType<H1<1>> displacement_type("displacement_solve_state");
  auto displacement_bc = field_store->addIndependent(displacement_type, time_rule);
  auto displacement_old_type =
      field_store->addDependent(displacement_type, FieldStore::TimeDerivative::VAL, "displacement");
  auto velocity_type = field_store->addDependent(displacement_type, FieldStore::TimeDerivative::DOT, "velocity");
  auto acceleration_type =
      field_store->addDependent(displacement_type, FieldStore::TimeDerivative::DDOT, "acceleration");

  auto cycle_zero_wf = buildSecondOrderCycleZeroWeakForm("cycle_zero_acceleration", mesh, field_store,
                                                         displacement_old_type, velocity_type, acceleration_type);

  displacement_bc->setScalarBCs<2>(mesh->domain("right"), [](double t, tensor<double, 2>) { return t * t; });

  auto bc_managers = field_store->getBoundaryConditionManagers({cycle_zero_wf->name()});
  ASSERT_EQ(bc_managers.size(), 1);
  ASSERT_NE(bc_managers[0], nullptr);
  const int right_dofs = bc_managers[0]->allEssentialTrueDofs().Size();
  ASSERT_GT(right_dofs, 0);

  displacement_bc->setScalarBCs<2>(mesh->domain("left"), [](double t, tensor<double, 2>) { return t * t; });

  bc_managers = field_store->getBoundaryConditionManagers({cycle_zero_wf->name()});
  ASSERT_NE(bc_managers[0], nullptr);
  EXPECT_GT(bc_managers[0]->allEssentialTrueDofs().Size(), right_dofs);

  mfem::Vector acceleration(field_store->getField("acceleration").get()->space().GetTrueVSize());
  acceleration = 0.0;
  for (const auto& bc : bc_managers[0]->essentials()) {
    bc.setDofs(acceleration, 0.0);
  }

  for (int dof : bc_managers[0]->allEssentialTrueDofs()) {
    EXPECT_NEAR(acceleration[dof], 2.0, 1.0e-7);
  }

  StateManager::reset();
}

TEST(SystemSolver, SingleBlockSolverFromMonolithicStageNarrowsToRequestedBlock)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "coupled_system_solver_single_block_characterization");

  auto mesh = std::make_shared<Mesh>(mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 1.0),
                                     "single_block_characterization_mesh");

  auto field_store = std::make_shared<FieldStore>(mesh, 20);
  FieldType<H1<1, 2>> shape_disp_type("shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  auto quasi_static = std::make_shared<QuasiStaticRule>();
  FieldType<H1<1>> temperature_type("temperature");
  field_store->addIndependent(temperature_type, quasi_static);
  FieldType<H1<1>> displacement_type("displacement");
  field_store->addIndependent(displacement_type, quasi_static);

  auto temperature_wf = buildScalarDiffusionWeakForm("temperature_main", mesh, field_store, temperature_type);
  auto displacement_wf = buildScalarDiffusionWeakForm("displacement_main", mesh, field_store, displacement_type);

  auto recording_solver = std::make_shared<NoOpNonlinearBlockSolver>();
  auto monolithic_solver = std::make_shared<SystemSolver>(recording_solver);
  auto derived_single_block_solver = monolithic_solver->singleBlockSolver(0);

  ASSERT_NE(derived_single_block_solver, nullptr);

  const std::vector<WeakForm*> residuals = {temperature_wf.get(), displacement_wf.get()};
  const std::vector<std::string> residual_names = {"temperature_main", "displacement_main"};
  const auto block_indices = field_store->indexMap(residual_names);
  const std::vector<std::vector<FieldState>> states = {field_store->getStates("temperature_main"),
                                                       field_store->getStates("displacement_main")};
  const std::vector<std::vector<FieldState>> params(residuals.size());
  const auto bc_managers = field_store->getBoundaryConditionManagers(residual_names);

  auto solved_states = derived_single_block_solver->solve(residuals, block_indices, field_store->getShapeDisp(), states,
                                                          params, TimeInfo(0.0, 1.0, 0), bc_managers);

  EXPECT_EQ(solved_states.size(), 2);
  EXPECT_EQ(recording_solver->solveCalls(), 1);
  EXPECT_EQ(recording_solver->lastNumUnknowns(), 1);

  StateManager::reset();
}

TEST(SystemSolver, AppendsStagesWithBlockMappingForCombinedSubsystems)
{
  auto first_solver = std::make_shared<NoOpNonlinearBlockSolver>();
  auto second_solver = std::make_shared<NoOpNonlinearBlockSolver>();

  SystemSolver subsystem_a(3, false);
  subsystem_a.addSubsystemSolver({0}, first_solver, 0.5);

  SystemSolver subsystem_b(3, false);
  subsystem_b.addSubsystemSolver({0, 1}, second_solver, 1.0);

  SystemSolver combined_solver(3, false);
  combined_solver.appendStagesWithBlockMapping(subsystem_a, {0});
  combined_solver.appendStagesWithBlockMapping(subsystem_b, {1, 2});

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "combined_solver_stage_mapping");

  auto mesh = std::make_shared<Mesh>(mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0),
                                     "combined_solver_stage_mapping_mesh");

  auto field_store = std::make_shared<FieldStore>(mesh, 20);
  FieldType<H1<1, 2>> shape_disp_type("shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  auto static_rule = std::make_shared<StaticTimeIntegrationRule>();
  FieldType<H1<1>> field0_type("field0");
  FieldType<H1<1>> field1_type("field1");
  FieldType<H1<1>> field2_type("field2");
  field_store->addIndependent(field0_type, static_rule);
  field_store->addIndependent(field1_type, static_rule);
  field_store->addIndependent(field2_type, static_rule);

  auto wf0 = buildScalarDiffusionWeakForm("wf0", mesh, field_store, field0_type);
  auto wf1 = buildScalarDiffusionWeakForm("wf1", mesh, field_store, field1_type);
  auto wf2 = buildScalarDiffusionWeakForm("wf2", mesh, field_store, field2_type);

  const std::vector<WeakForm*> residuals = {wf0.get(), wf1.get(), wf2.get()};
  const std::vector<std::string> residual_names = {"wf0", "wf1", "wf2"};
  const auto block_indices = field_store->indexMap(residual_names);
  const std::vector<std::vector<FieldState>> states = {field_store->getStates("wf0"), field_store->getStates("wf1"),
                                                       field_store->getStates("wf2")};
  const std::vector<std::vector<FieldState>> params(residuals.size());
  const auto bc_managers = field_store->getBoundaryConditionManagers(residual_names);

  auto solved_states = combined_solver.solve(residuals, block_indices, field_store->getShapeDisp(), states, params,
                                             TimeInfo(0.0, 1.0, 0), bc_managers);

  EXPECT_EQ(solved_states.size(), 3);
  EXPECT_EQ(first_solver->solveCalls(), 1);
  EXPECT_EQ(first_solver->lastNumUnknowns(), 1);
  EXPECT_EQ(second_solver->solveCalls(), 1);
  EXPECT_EQ(second_solver->lastNumUnknowns(), 2);

  StateManager::reset();
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager app(argc, argv);
  return RUN_ALL_TESTS();
}
