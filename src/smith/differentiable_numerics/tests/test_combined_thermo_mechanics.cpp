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
#include "smith/differentiable_numerics/thermo_mechanical_system.hpp"
#include "smith/differentiable_numerics/combined_system.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/physics/functional_objective.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"

namespace smith {

static constexpr int dim = 3;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;

using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;

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
  double advanceOneStepAndGetLateralDeflection(std::shared_ptr<System> coupled,
                                               double dt = 1.0)
  {
    auto shape_disp = field_store_->getShapeDisp();
    auto params = field_store_->getParameterFields();
    std::vector<ReactionState> reactions;
    std::tie(std::ignore, reactions) =
        makeAdvancer(coupled)->advanceState(smith::TimeInfo(0.0, dt, 0), shape_disp, field_store_->getStateFields(),
                                            params);

    mfem::Vector final_disp(
        *field_store_->getStateFields()[field_store_->getFieldIndex("displacement_solve_state")].get());
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

  auto solid = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      makeSolver(newtonNonlinOpts, directLinOpts), SolidMechanicsOptions{}, solid_fields, param_fields, thermal_fields);

  auto thermal = buildThermalSystem<dim, temperature_order, TempRule>(
      makeSolver(newtonNonlinOpts, directLinOpts), ThermalOptions{}, thermal_fields, param_fields, solid_fields);

  auto coupled = combineSystems(solid, thermal);
  auto physics = makeDifferentiablePhysics(coupled, "coupled_physics");
  const auto& solid_dual_space = physics->dual("reactions").space();
  const auto& solid_state_space = physics->state("displacement_solve_state").space();
  const auto& thermal_dual_space = physics->dual("thermal_flux").space();
  const auto& thermal_state_space = physics->state("temperature_solve_state").space();

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

  auto solid = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      makeSolver(newtonNonlinOpts, directLinOpts), SolidMechanicsOptions{}, solid_fields, param_fields, thermal_fields);

  auto thermal = buildThermalSystem<dim, temperature_order, TempRule>(
      makeSolver(newtonNonlinOpts, directLinOpts), ThermalOptions{}, thermal_fields, param_fields, solid_fields);

  auto coupled = combineSystems(solid, thermal);
  thermomechanics::ParameterizedGreenSaintVenantThermoelasticMaterial material{1.0,    100.0, 0.25, 1.0,
                                                                               0.0025, 0.0,   0.05};
  setCoupledThermoMechanicsMaterial(solid, thermal, material, mesh_->entireBodyName());

  coupled->field_store->getParameterFields()[0].get()->setFromFieldFunction(
      [=](smith::tensor<double, dim>) { return 1.0; });

  solid->setDisplacementBC(mesh_->domain("left"));
  thermal->setTemperatureBC(mesh_->domain("left"), [](auto, auto) { return 1.0; });

  solid->addTraction("right", [=](double, auto X, auto, auto, auto, auto, auto, auto, auto) {
    auto traction = 0.0 * X;
    traction[0] = -0.015;
    return traction;
  });

  auto physics = makeDifferentiablePhysics(coupled, "coupled_physics");

  double dt = 1.0;
  for (int step = 0; step < 2; ++step) {
    physics->advanceTimestep(dt);
  }

  auto reactions = physics->getReactionStates();
  auto obj = 0.5 * (innerProduct(reactions[0], reactions[0]) + innerProduct(reactions[1], reactions[1]));

  gretl::set_as_objective(obj);
  obj.data_store().back_prop();

  auto param_sens = coupled->field_store->getParameterFields()[0].get_dual();
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

  auto solid = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      makeSolver(mech_nonlin_opts, mech_lin_opts), SolidMechanicsOptions{}, solid_fields, thermal_fields);
  auto thermal = buildThermalSystem<dim, temperature_order, TempRule>(
      makeSolver(therm_nonlin_opts, therm_lin_opts), ThermalOptions{}, thermal_fields, solid_fields);

  auto coupled = combineSystems(solid, thermal);
  thermomechanics::GreenSaintVenantThermoelasticMaterial material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  setCoupledThermoMechanicsMaterial(solid, thermal, material, mesh_->entireBodyName());

  applyBucklingLoads(solid, thermal, kBucklingTraction, kBucklingBodyForce, kBucklingHeatSource);

  double deflection = advanceOneStepAndGetLateralDeflection(coupled);

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

  auto solid =
      buildSolidMechanicsSystem<dim, displacement_order, DispRule>(nullptr, SolidMechanicsOptions{}, solid_fields,
                                                                   thermal_fields);
  auto thermal = buildThermalSystem<dim, temperature_order, TempRule>(nullptr, ThermalOptions{}, thermal_fields,
                                                                       solid_fields);

  auto coupled = combineSystems(solver_ptr, solid, thermal);
  thermomechanics::GreenSaintVenantThermoelasticMaterial material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  setCoupledThermoMechanicsMaterial(solid, thermal, material, mesh_->entireBodyName());

  applyBucklingLoads(solid, thermal, kBucklingTraction, kBucklingBodyForce, kBucklingHeatSource);

  double deflection = advanceOneStepAndGetLateralDeflection(coupled);

  EXPECT_GT(deflection, 1e-5);
}

// 5. CauchyStressOutput
TEST_F(ThermoMechanicsMeshFixture, CauchyStressOutput)
{
  SolidMechanicsOptions solid_opts{.enable_stress_output = true, .output_cauchy_stress = true};

  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_, solid_opts);
  EXPECT_TRUE(field_store_->hasField(field_store_->prefix("stress_solve_state")));
  EXPECT_TRUE(field_store_->hasField(field_store_->prefix("stress")));
  auto sys = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      makeSolver(newtonNonlinOpts, directLinOpts), solid_opts, solid_fields);
  ASSERT_EQ(sys->post_solve_systems.size(), 1u);

  constexpr double E = 100.0;
  constexpr double nu = 0.25;
  constexpr double G = E / (2.0 * (1.0 + nu));
  constexpr double K = E / (3.0 * (1.0 - 2.0 * nu));

  sys->setMaterial(solid_mechanics::NeoHookean{.density = 1.0, .K = K, .G = G}, mesh_->entireBodyName());

  sys->setDisplacementBC(mesh_->domain("left"));
  sys->addTraction("right", [](double, auto X, auto, auto, auto, auto) {
    auto t = 0.0 * X;
    t[0] = -0.01;
    return t;
  });

  auto physics = makeDifferentiablePhysics(sys, "cauchy_physics");
  physics->advanceTimestep(1.0);

  ASSERT_FALSE(sys->post_solve_systems.empty()) << "Stress output system should be present";
  auto states = physics->getFieldStates();
  size_t stress_idx = field_store_->getFieldIndex("stress_solve_state");
  double stress_norm = norm(*states[stress_idx].get());
  EXPECT_GT(stress_norm, 1e-8) << "Cauchy stress field should be non-zero after deformation";
}

TEST_F(ThermoMechanicsMeshFixture, StressOutputRegistrationDisabledByDefault)
{
  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_);
  EXPECT_FALSE(field_store_->hasField(field_store_->prefix("stress_solve_state")));
  EXPECT_FALSE(field_store_->hasField(field_store_->prefix("stress")));

  auto sys = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      makeSolver(newtonNonlinOpts, directLinOpts), SolidMechanicsOptions{}, solid_fields);
  EXPECT_TRUE(sys->post_solve_systems.empty());
}

TEST_F(ThermoMechanicsMeshFixture, CombinedSystemCarriesPostSolveSystems)
{
  SolidMechanicsOptions solid_opts{.enable_stress_output = true};

  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store_, solid_opts);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store_);

  auto solid = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      makeSolver(newtonNonlinOpts, directLinOpts), solid_opts, solid_fields, thermal_fields);
  auto thermal = buildThermalSystem<dim, temperature_order, TempRule>(
      makeSolver(newtonNonlinOpts, directLinOpts), ThermalOptions{}, thermal_fields, solid_fields);

  auto combined = combineSystems(solid, thermal);

  ASSERT_EQ(combined->post_solve_systems.size(), solid->post_solve_systems.size());
  EXPECT_FALSE(combined->post_solve_systems.empty());
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
