// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include "gtest/gtest.h"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/green_saint_venant_thermoelastic.hpp"
#include "smith/physics/materials/solid_material.hpp"

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"
#include "smith/differentiable_numerics/thermo_mechanics_system.hpp"
#include "smith/differentiable_numerics/combined_system.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/evaluate_objective.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/time_info_solid_materials.hpp"
#include "smith/differentiable_numerics/time_info_thermo_mechanical_materials.hpp"
#include "smith/physics/functional_objective.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"

namespace smith {

static constexpr int dim = 3;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;

using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;

TEST(CouplingTimeRuleInterpolation, AppliesEachForeignPhysicsRuleBeforeCallback)
{
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "dcoupling_interpolation");
  auto mesh =
      std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON), "mesh", 0, 0);
  auto field_store = std::make_shared<FieldStore>(mesh, 100, "");
  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store);
  auto scale_params = registerParameterFields(FieldType<L2<0>>("scale"));

  auto solid_coupling = detail::collectCouplingFields(field_store, couplingFields(thermal_fields), scale_params);
  auto saw_thermal_values = detail::applyCouplingTimeRules(
      solid_coupling, TimeInfo(0.0, 2.0, 0),
      [](auto temperature, auto temperature_dot, auto scale) {
        EXPECT_DOUBLE_EQ(temperature, 7.0);
        EXPECT_DOUBLE_EQ(temperature_dot, 3.0);
        EXPECT_DOUBLE_EQ(scale, 11.0);
        return true;
      },
      7.0, 1.0, 11.0);
  EXPECT_TRUE(saw_thermal_values);

  auto thermal_coupling = detail::collectCouplingFields(field_store, couplingFields(solid_fields));
  auto saw_solid_values = detail::applyCouplingTimeRules(
      thermal_coupling, TimeInfo(0.0, 2.0, 0),
      [](auto displacement, auto velocity, auto /*acceleration*/) {
        EXPECT_DOUBLE_EQ(displacement, 10.0);
        EXPECT_DOUBLE_EQ(velocity, 3.0);
        return true;
      },
      10.0, 4.0, 0.0, 0.0);
  EXPECT_TRUE(saw_solid_values);
}

TEST(CouplingTimeRuleInterpolation, PreservesForeignPacksWithSameTimeRuleType)
{
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "same_rule_coupling");
  auto mesh =
      std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON), "mesh", 0, 0);
  auto field_store = std::make_shared<FieldStore>(mesh, 100, "");

  using ScalarSpace = H1<temperature_order>;
  PhysicsFields<TempRule, ScalarSpace, ScalarSpace> thermal_a{
      field_store, FieldType<ScalarSpace>("temperature_a_solve_state"), FieldType<ScalarSpace>("temperature_a")};
  PhysicsFields<TempRule, ScalarSpace, ScalarSpace> thermal_b{
      field_store, FieldType<ScalarSpace>("temperature_b_solve_state"), FieldType<ScalarSpace>("temperature_b")};

  auto same_rule_coupling = detail::collectCouplingFields(field_store, couplingFields(thermal_a, thermal_b));
  auto saw_all_values = detail::applyCouplingTimeRules(
      same_rule_coupling, TimeInfo(0.0, 2.0, 0),
      [](auto temperature_a, auto temperature_a_dot, auto temperature_b, auto temperature_b_dot) {
        EXPECT_DOUBLE_EQ(temperature_a, 7.0);
        EXPECT_DOUBLE_EQ(temperature_a_dot, 3.0);
        EXPECT_DOUBLE_EQ(temperature_b, 5.0);
        EXPECT_DOUBLE_EQ(temperature_b_dot, 1.75);
        return true;
      },
      7.0, 1.0, 5.0, 1.5);
  EXPECT_TRUE(saw_all_values);
}

struct ThermoMechanicsMeshFixture : public testing::Test {
  void SetUp()
  {
    datastore_ = std::make_unique<axom::sidre::DataStore>();
    smith::StateManager::initialize(*datastore_, "solid");
    mesh_ = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(24, 2, 2, mfem::Element::HEXAHEDRON, 1.2, 0.03, 0.03), "mesh", 0, 0);
    mesh_->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
    mesh_->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
    field_store_ = std::make_shared<FieldStore>(mesh_, 100, "");
  }

  std::shared_ptr<SystemSolver> makeSolver(const NonlinearSolverOptions& nonlin, const LinearSolverOptions& lin)
  {
    return std::make_shared<SystemSolver>(buildNonlinearBlockSolver(nonlin, lin, *mesh_));
  }

  // Advance one step, return final states + lateral deflection.
  template <typename System>
  double advanceOneStepAndGetLateralDeflection(std::shared_ptr<System> coupled_system, double dt = 1.0)
  {
    auto shape_disp = field_store_->getShapeDisp();
    auto params = field_store_->getParameterFields();
    std::vector<ReactionState> reactions;
    std::tie(std::ignore, reactions) =
        makeAdvancer(coupled_system)
            ->advanceState(smith::TimeInfo(0.0, dt, 0), shape_disp, field_store_->getStateFields(), params);

    mfem::Vector final_disp(*field_store_->getField("displacement").get());
    double deflection = 0.0;
    for (int i = 1; i < final_disp.Size(); i += dim) {
      deflection = std::max(deflection, std::abs(final_disp(i)));
    }
    return deflection;
  }

  template <typename Solid, typename Thermal>
  void applyBucklingLoads(Solid& solid, Thermal& thermal, double compressive_traction, double lateral_body_force,
                          double thermal_source)
  {
    solid->setDisplacementBC(mesh_->domain("left"));
    thermal->setTemperatureBC(mesh_->domain("left"));
    thermal->setTemperatureBC(mesh_->domain("right"));

    solid->addTraction("right", [=](auto, auto X, auto, auto, auto, auto, auto... /*args*/) {
      auto traction = 0.0 * X;
      traction[0] = -compressive_traction;
      return traction;
    });
    solid->addBodyForce(mesh_->entireBodyName(), [=](auto, auto X, auto, auto, auto, auto, auto... /*args*/) {
      auto force = 0.0 * X;
      force[1] = lateral_body_force;
      return force;
    });
    thermal->addHeatSource(mesh_->entireBodyName(), [=](auto, auto, auto, auto... /*args*/) { return thermal_source; });
  }

  std::unique_ptr<axom::sidre::DataStore> datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
  std::shared_ptr<FieldStore> field_store_;
};

// Defaults used by multiple tests.
static const LinearSolverOptions directLinOpts{.linear_solver = LinearSolver::SuperLU};
static const NonlinearSolverOptions newtonNonlinOpts{
    .nonlin_solver = NonlinearSolver::Newton, .relative_tol = 1e-10, .absolute_tol = 1e-10, .max_iterations = 4};

// 1. CreateDifferentiablePhysicsAllocatesReactionInfo
TEST_F(ThermoMechanicsMeshFixture, CreateDifferentiablePhysicsAllocatesReactionInfo)
{
  FieldType<L2<0>> thermal_expansion_scaling("thermal_expansion_scaling");

  auto param_fields = registerParameterFields(thermal_expansion_scaling);
  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store_);

  auto solid_system = buildSolidMechanicsSystem<dim, displacement_order>(makeSolver(newtonNonlinOpts, directLinOpts),
                                                                         SolidMechanicsOptions{}, solid_fields,
                                                                         couplingFields(thermal_fields), param_fields);

  auto thermal_system =
      buildThermalSystem<dim, temperature_order>(makeSolver(newtonNonlinOpts, directLinOpts), ThermalOptions{},
                                                 thermal_fields, couplingFields(solid_fields), param_fields);

  auto coupled_system = combineSystems(solid_system, thermal_system);
  auto physics = makeDifferentiablePhysics(coupled_system, "coupled_physics");
  const auto& solid_dual_space = physics->dual("reactions").space();
  const auto& solid_state_space = physics->state("displacement").space();
  const auto& thermal_dual_space = physics->dual("thermal_flux").space();
  const auto& thermal_state_space = physics->state("temperature").space();

  EXPECT_EQ(physics->dualNames().size(), 2);
  EXPECT_EQ(physics->dualNames()[0], "reactions");
  EXPECT_EQ(physics->dualNames()[1], "thermal_flux");
  EXPECT_EQ(solid_dual_space.GetMesh(), solid_state_space.GetMesh());
  EXPECT_STREQ(solid_dual_space.FEColl()->Name(), solid_state_space.FEColl()->Name());
  EXPECT_EQ(solid_dual_space.GetVDim(), solid_state_space.GetVDim());
  EXPECT_EQ(solid_dual_space.TrueVSize(), solid_state_space.TrueVSize());
  EXPECT_EQ(thermal_dual_space.GetMesh(), thermal_state_space.GetMesh());
  EXPECT_STREQ(thermal_dual_space.FEColl()->Name(), thermal_state_space.FEColl()->Name());
  EXPECT_EQ(thermal_dual_space.GetVDim(), thermal_state_space.GetVDim());
  EXPECT_EQ(thermal_dual_space.TrueVSize(), thermal_state_space.TrueVSize());
}

// 2. BackpropagateThroughPhysics
TEST_F(ThermoMechanicsMeshFixture, BackpropagateThroughPhysics)
{
  FieldType<L2<0>> youngs_modulus("youngs_modulus");

  auto param_fields = registerParameterFields(youngs_modulus);
  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store_);

  auto solid_system = buildSolidMechanicsSystem<dim, displacement_order>(makeSolver(newtonNonlinOpts, directLinOpts),
                                                                         SolidMechanicsOptions{}, solid_fields,
                                                                         couplingFields(thermal_fields), param_fields);

  auto thermal_system =
      buildThermalSystem<dim, temperature_order>(makeSolver(newtonNonlinOpts, directLinOpts), ThermalOptions{},
                                                 thermal_fields, couplingFields(solid_fields), param_fields);

  auto coupled_system = combineSystems(solid_system, thermal_system);
  thermomechanics::ParameterizedGreenSaintVenantThermoelasticMaterialWithTimeInfo material{1.0,    100.0, 0.25, 1.0,
                                                                                           0.0025, 0.0,   0.05};
  setCoupledThermoMechanicsMaterial(solid_system, thermal_system, material, mesh_->entireBodyName());

  coupled_system->field_store->getParameterFields()[0].get()->setFromFieldFunction(
      [=](smith::tensor<double, dim>) { return 1.0; });

  solid_system->setDisplacementBC(mesh_->domain("left"));
  thermal_system->setTemperatureBC(mesh_->domain("left"), [](auto, auto) { return 1.0; });

  solid_system->addTraction("right", [=](double, auto X, auto, auto, auto, auto, auto... /*params*/) {
    // If X is a dual number, we need to create a dual number for traction with zero derivative wrt all active
    // parameters. For now, returning a value works perfectly fine with smith AD system! But since X might be a dual
    // number, we must strip its dual part if we just want a value.
    auto traction = 0.0 * X;
    traction[0] = -0.015;
    return traction;
  });

  auto physics = makeDifferentiablePhysics(coupled_system, "coupled_physics");

  double dt = 1.0;
  for (int step = 0; step < 2; ++step) {
    physics->advanceTimestep(dt);
  }

  auto reactions = physics->getReactionStates();
  auto obj = 0.5 * (innerProduct(reactions[0], reactions[0]) + innerProduct(reactions[1], reactions[1]));

  gretl::set_as_objective(obj);
  obj.data_store().back_prop();

  auto param_sens = coupled_system->field_store->getParameterFields()[0].get_dual();
  EXPECT_TRUE(param_sens->Norml2() > 0.0);
}

TEST_F(ThermoMechanicsMeshFixture, BackpropagateThroughStaggeredPhysics)
{
  FieldType<L2<0>> thermal_expansion_scaling("thermal_expansion_scaling");

  auto param_fields = registerParameterFields(thermal_expansion_scaling);
  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store_);

  LinearSolverOptions solid_lin_opts{.linear_solver = LinearSolver::CG,
                                     .preconditioner = Preconditioner::HypreAMG,
                                     .relative_tol = 1e-6,
                                     .absolute_tol = 1e-10,
                                     .max_iterations = 120};
  NonlinearSolverOptions solid_nonlin_opts{
      .nonlin_solver = NonlinearSolver::TrustRegion, .relative_tol = 1e-6, .absolute_tol = 1e-7, .max_iterations = 25};
  LinearSolverOptions thermal_lin_opts{
      .linear_solver = LinearSolver::SuperLU, .relative_tol = 1e-8, .absolute_tol = 1e-10, .max_iterations = 80};
  NonlinearSolverOptions thermal_nonlin_opts{.nonlin_solver = NonlinearSolver::NewtonLineSearch,
                                             .relative_tol = 1e-7,
                                             .absolute_tol = 1e-8,
                                             .max_iterations = 12,
                                             .max_line_search_iterations = 6};

  auto solid_system = buildSolidMechanicsSystem<dim, displacement_order>(nullptr, SolidMechanicsOptions{}, solid_fields,
                                                                         couplingFields(thermal_fields), param_fields);
  auto thermal_system = buildThermalSystem<dim, temperature_order>(nullptr, ThermalOptions{}, thermal_fields,
                                                                   couplingFields(solid_fields), param_fields);

  auto coupled_solver = std::make_shared<SystemSolver>(10);
  coupled_solver->addSubsystemSolver({0}, buildNonlinearBlockSolver(solid_nonlin_opts, solid_lin_opts, *mesh_));
  coupled_solver->addSubsystemSolver({1}, buildNonlinearBlockSolver(thermal_nonlin_opts, thermal_lin_opts, *mesh_));
  auto coupled_system = combineSystems(coupled_solver, solid_system, thermal_system);

  thermomechanics::ParameterizedGreenSaintVenantThermoelasticMaterialWithTimeInfo material{1.0,    100.0, 0.25, 1.0,
                                                                                           0.0025, 0.0,   0.05};
  setCoupledThermoMechanicsMaterial(solid_system, thermal_system, material, mesh_->entireBodyName());

  coupled_system->field_store->getParameterFields()[0].get()->setFromFieldFunction(
      [=](smith::tensor<double, dim>) { return 1.0; });

  solid_system->setDisplacementBC(mesh_->domain("left"));
  thermal_system->setTemperatureBC(mesh_->domain("left"), [](auto, auto) { return 1.0; });
  thermal_system->setTemperatureBC(mesh_->domain("right"), [](auto, auto) { return 0.0; });

  solid_system->addTraction("right", [=](double, auto X, auto, auto, auto, auto, auto... /*params*/) {
    auto traction = 0.0 * X;
    traction[0] = -0.005;
    return traction;
  });
  thermal_system->addHeatSource(mesh_->entireBodyName(), [=](auto, auto, auto, auto... /*args*/) { return 0.1; });

  auto physics = makeDifferentiablePhysics(coupled_system, "staggered_coupled_physics");

  FunctionalObjective<dim, Parameters<H1<displacement_order, dim>, H1<temperature_order>>> qoi(
      "staggered_qoi", mesh_,
      spaces({coupled_system->field_store->getField("displacement"),
              coupled_system->field_store->getField("temperature")}));
  qoi.addBodyIntegral(mesh_->entireBodyName(), [](auto, auto, auto U, auto Theta) {
    auto u = get<VALUE>(U);
    auto theta = get<VALUE>(Theta);
    return 0.5 * u[0] * u[0] + 0.05 * theta * theta;
  });

  physics->advanceTimestep(0.5);
  auto qoi_fields = std::vector<FieldState>{coupled_system->field_store->getField("displacement"),
                                            coupled_system->field_store->getField("temperature")};
  auto obj = smith::evaluateObjective(qoi, physics->getShapeDispFieldState(), qoi_fields,
                                      TimeInfo(physics->time(), 0.5, static_cast<size_t>(physics->cycle())));

  gretl::set_as_objective(obj);
  obj.data_store().back_prop();

  auto param_sens = coupled_system->field_store->getParameterFields()[0].get_dual();
  EXPECT_TRUE(param_sens->Norml2() > 0.0);
}

// Shared buckling-load magnitudes (used by staggered + monolithic buckling tests).
static constexpr double kBucklingTraction = 0.015;
static constexpr double kBucklingBodyForce = 2.5e-5;
static constexpr double kBucklingHeatSource = 1.0;

// 3. StaggeredBucklingChallenge
TEST_F(ThermoMechanicsMeshFixture, StaggeredBucklingChallenge)
{
  LinearSolverOptions mech_lin_opts{.linear_solver = LinearSolver::CG,
                                    .preconditioner = Preconditioner::HypreAMG,
                                    .relative_tol = 1e-6,
                                    .absolute_tol = 1e-10,
                                    .max_iterations = 120};
  NonlinearSolverOptions mech_nonlin_opts{
      .nonlin_solver = NonlinearSolver::TrustRegion, .relative_tol = 1e-6, .absolute_tol = 1e-7, .max_iterations = 25};
  LinearSolverOptions therm_lin_opts{.linear_solver = LinearSolver::GMRES,
                                     .preconditioner = Preconditioner::HypreAMG,
                                     .relative_tol = 1e-6,
                                     .absolute_tol = 1e-10,
                                     .max_iterations = 80};
  NonlinearSolverOptions therm_nonlin_opts{.nonlin_solver = NonlinearSolver::NewtonLineSearch,
                                           .relative_tol = 1e-7,
                                           .absolute_tol = 1e-7,
                                           .max_iterations = 12,
                                           .max_line_search_iterations = 6};

  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store_);

  auto solid_system = buildSolidMechanicsSystem<dim, displacement_order>(makeSolver(mech_nonlin_opts, mech_lin_opts),
                                                                         SolidMechanicsOptions{}, solid_fields,
                                                                         couplingFields(thermal_fields));
  auto thermal_system = buildThermalSystem<dim, temperature_order>(
      makeSolver(therm_nonlin_opts, therm_lin_opts), ThermalOptions{}, thermal_fields, couplingFields(solid_fields));

  auto coupled_system = combineSystems(solid_system, thermal_system);
  thermomechanics::GreenSaintVenantThermoelasticMaterialWithTimeInfo material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  setCoupledThermoMechanicsMaterial(solid_system, thermal_system, material, mesh_->entireBodyName());

  applyBucklingLoads(solid_system, thermal_system, kBucklingTraction, kBucklingBodyForce, kBucklingHeatSource);

  double deflection = advanceOneStepAndGetLateralDeflection(coupled_system);

  EXPECT_GT(deflection, 1e-5);
}

// 4. MonolithicBucklingChallenge
TEST_F(ThermoMechanicsMeshFixture, MonolithicBucklingChallenge)
{
  LinearSolverOptions lin_opts{
      .linear_solver = LinearSolver::SuperLU, .relative_tol = 1e-6, .absolute_tol = 1e-10, .max_iterations = 80};
  NonlinearSolverOptions nonlin_opts{
      .nonlin_solver = NonlinearSolver::Newton, .relative_tol = 1e-7, .absolute_tol = 1e-7, .max_iterations = 12};

  auto solver_ptr = makeSolver(nonlin_opts, lin_opts);

  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store_);

  auto solid_system = buildSolidMechanicsSystem<dim, displacement_order>(nullptr, SolidMechanicsOptions{}, solid_fields,
                                                                         couplingFields(thermal_fields));
  auto thermal_system = buildThermalSystem<dim, temperature_order>(nullptr, ThermalOptions{}, thermal_fields,
                                                                   couplingFields(solid_fields));

  auto coupled_system = combineSystems(solver_ptr, solid_system, thermal_system);
  thermomechanics::GreenSaintVenantThermoelasticMaterialWithTimeInfo material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  setCoupledThermoMechanicsMaterial(solid_system, thermal_system, material, mesh_->entireBodyName());

  applyBucklingLoads(solid_system, thermal_system, kBucklingTraction, kBucklingBodyForce, kBucklingHeatSource);

  double deflection = advanceOneStepAndGetLateralDeflection(coupled_system);

  EXPECT_GT(deflection, 1e-5);
}

// 5. CauchyStressOutput
TEST_F(ThermoMechanicsMeshFixture, CauchyStressOutput)
{
  SolidMechanicsOptions solid_opts{.enable_stress_output = true, .output_cauchy_stress = true};

  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_, solid_opts);
  EXPECT_TRUE(field_store_->hasField(field_store_->prefix("stress")));
  auto solid_system = buildSolidMechanicsSystem<dim, displacement_order>(makeSolver(newtonNonlinOpts, directLinOpts),
                                                                         solid_opts, solid_fields);
  ASSERT_EQ(solid_system->post_solve_systems.size(), 1u);

  constexpr double E = 100.0;
  constexpr double nu = 0.25;
  constexpr double G = E / (2.0 * (1.0 + nu));
  constexpr double K = E / (3.0 * (1.0 - 2.0 * nu));

  solid_system->setMaterial(solid_mechanics::TimeInfoNeoHookean{.density = 1.0, .K = K, .G = G},
                            mesh_->entireBodyName());

  solid_system->setDisplacementBC(mesh_->domain("left"));
  solid_system->addTraction("right", [](double, auto X, auto, auto, auto, auto, auto... /*params*/) {
    auto t = 0.0 * X;
    t[0] = -0.01;
    return t;
  });

  auto physics = makeDifferentiablePhysics(solid_system, "cauchy_physics");
  physics->advanceTimestep(1.0);

  ASSERT_FALSE(solid_system->post_solve_systems.empty()) << "Stress output system should be present";
  auto states = physics->getFieldStates();
  size_t stress_idx = field_store_->getFieldIndex("stress");
  double stress_norm = norm(*states[stress_idx].get());
  EXPECT_GT(stress_norm, 1e-8) << "Cauchy stress field should be non-zero after deformation";
}

TEST_F(ThermoMechanicsMeshFixture, StressOutputRegistrationDisabledByDefault)
{
  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_);
  EXPECT_FALSE(field_store_->hasField(field_store_->prefix("stress")));

  auto solid_system = buildSolidMechanicsSystem<dim, displacement_order>(makeSolver(newtonNonlinOpts, directLinOpts),
                                                                         SolidMechanicsOptions{}, solid_fields);
  EXPECT_TRUE(solid_system->post_solve_systems.empty());
}

TEST_F(ThermoMechanicsMeshFixture, CombinedSystemCarriesPostSolveSystems)
{
  SolidMechanicsOptions solid_opts{.enable_stress_output = true};

  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_, solid_opts);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store_);

  auto solid_system = buildSolidMechanicsSystem<dim, displacement_order>(
      makeSolver(newtonNonlinOpts, directLinOpts), solid_opts, solid_fields, couplingFields(thermal_fields));
  auto thermal_system = buildThermalSystem<dim, temperature_order>(
      makeSolver(newtonNonlinOpts, directLinOpts), ThermalOptions{}, thermal_fields, couplingFields(solid_fields));

  auto combined_system = combineSystems(solid_system, thermal_system);

  ASSERT_EQ(combined_system->post_solve_systems.size(), solid_system->post_solve_systems.size());
  EXPECT_FALSE(combined_system->post_solve_systems.empty());
}

TEST_F(ThermoMechanicsMeshFixture, CombinedSystemCarriesCycleZeroSystems)
{
  using DynamicDispRule = ImplicitNewmarkSecondOrderTimeIntegrationRule;

  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DynamicDispRule>(field_store_);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store_);

  auto solid_system = buildSolidMechanicsSystem<dim, displacement_order>(makeSolver(newtonNonlinOpts, directLinOpts),
                                                                         SolidMechanicsOptions{}, solid_fields,
                                                                         couplingFields(thermal_fields));
  auto thermal_system = buildThermalSystem<dim, temperature_order>(
      makeSolver(newtonNonlinOpts, directLinOpts), ThermalOptions{}, thermal_fields, couplingFields(solid_fields));

  auto combined_system = combineSystems(solid_system, thermal_system);

  ASSERT_EQ(solid_system->cycle_zero_systems.size(), 1u);
  EXPECT_EQ(combined_system->cycle_zero_systems.size(), solid_system->cycle_zero_systems.size());
  EXPECT_EQ(combined_system->cycle_zero_systems[0]->solve_result_field_names, std::vector<std::string>{"acceleration"});
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
