// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "gretl/data_store.hpp"
#include <tuple>
#include <algorithm>
#include <memory>

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/functional_objective.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"

namespace smith {

/**
 * @brief Verify that reaction forces are zero at non-Dirichlet DOFs.
 * @param reaction The reaction field to check.
 * @param bc_manager Boundary condition manager to identify Dirichlet DOFs.
 * @param tolerance Absolute tolerance for zero check.
 */
inline void checkUnconstrainedReactions(const FiniteElementDual& reaction, const BoundaryConditionManager& bc_manager,
                                        double tolerance = 1e-8)
{
  FiniteElementState unconstrained_reactions(reaction.space(), "unconstrained_reactions");
  unconstrained_reactions = reaction;
  unconstrained_reactions.SetSubVector(bc_manager.allEssentialTrueDofs(), 0.0);

  double max_unconstrained = unconstrained_reactions.Normlinf();
  EXPECT_LT(max_unconstrained, tolerance)
      << "Reaction forces should be zero at non-Dirichlet DOFs. Max violation: " << max_unconstrained;
}

LinearSolverOptions solid_linear_options{.linear_solver = LinearSolver::CG,
                                         .preconditioner = Preconditioner::HypreJacobi,
                                         .relative_tol = 1e-11,
                                         .absolute_tol = 1e-11,
                                         .max_iterations = 10000,
                                         .print_level = 0};

NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                            .relative_tol = 1.0e-10,
                                            .absolute_tol = 1.0e-10,
                                            .max_iterations = 500,
                                            .print_level = 1};

static constexpr int dim = 3;
static constexpr int order = 1;

using ShapeDispSpace = H1<1, dim>;
using VectorSpace = H1<order, dim>;
using ScalarParameterSpace = L2<0>;

struct SolidMechanicsMeshFixture : public testing::Test {
  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 12;
  int num_elements_y = 2;
  int num_elements_z = 2;
  double elem_size = length / num_elements_x;

  void SetUp()
  {
    StateManager::initialize(datastore, "solid");
    auto mfem_shape = mfem::Element::HEXAHEDRON;
    mesh = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
        "mesh", 0, 0);
    mesh->addDomainOfBoundaryElements("left", by_attr<dim>(3));
    mesh->addDomainOfBoundaryElements("right", by_attr<dim>(5));
  }

  static constexpr double total_simulation_time_ = 1.1;
  static constexpr size_t num_steps_ = 4;
  static constexpr double dt_ = total_simulation_time_ / num_steps_;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
};

// Verifies the cycle-zero contract: rules that report requiresInitialAccelerationSolve()
// produce a non-null cycle_zero_system; rules that don't (QuasiStatic) produce nullptr.
TEST_F(SolidMechanicsMeshFixture, CycleZeroSystemPresenceMatchesRuleContract)
{
  auto solver =
      std::make_shared<SystemSolver>(buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh));

  {
    auto field_store = std::make_shared<FieldStore>(mesh, 100, "impl");
    using ImplicitRule = ImplicitNewmarkSecondOrderTimeIntegrationRule;
    auto solid_fields = registerSolidMechanicsFields<dim, order, ImplicitRule>(field_store);
    auto sys = buildSolidMechanicsSystem<dim, order, ImplicitRule>(solver, SolidMechanicsOptions{}, solid_fields);
    EXPECT_NE(sys->cycle_zero_system, nullptr) << "ImplicitNewmark should emit a cycle-zero initial acceleration solve";
  }
  {
    auto field_store = std::make_shared<FieldStore>(mesh, 100, "qs");
    using QsRule = QuasiStaticSecondOrderTimeIntegrationRule;
    auto solid_fields = registerSolidMechanicsFields<dim, order, QsRule>(field_store);
    auto sys = buildSolidMechanicsSystem<dim, order, QsRule>(solver, SolidMechanicsOptions{}, solid_fields);
    EXPECT_EQ(sys->cycle_zero_system, nullptr) << "QuasiStatic has no initial acceleration solve";
  }
}

TEST_F(SolidMechanicsMeshFixture, TransientConstantGravity)
{
  SMITH_MARK_FUNCTION;

  auto solid_block_solver = buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);

  auto coupled_solver = std::make_shared<SystemSolver>(solid_block_solver);
  auto field_store = std::make_shared<FieldStore>(mesh, 100, "");

  using TimeRule = ImplicitNewmarkSecondOrderTimeIntegrationRule;
  auto param_fields =
      registerParameterFields(FieldType<ScalarParameterSpace>("bulk"), FieldType<ScalarParameterSpace>("shear"));
  auto solid_fields = registerSolidMechanicsFields<dim, order, TimeRule>(
      field_store, SolidMechanicsOptions{.enable_stress_output = true});

  auto system = buildSolidMechanicsSystem<dim, order, TimeRule>(
      coupled_solver, SolidMechanicsOptions{.enable_stress_output = true}, solid_fields, param_fields);

  static constexpr double gravity = -9.0;

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2.0 * nu));
  auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = solid_mechanics::ParameterizedNeoHookeanSolid;
  MaterialType material{.density = 1.0, .K0 = K, .G0 = G};

  // Set parameters
  auto params = system->field_store->getParameterFields();
  params[0].get()->setFromFieldFunction([=](tensor<double, dim>) { return material.K0; });
  params[1].get()->setFromFieldFunction([=](tensor<double, dim>) { return material.G0; });

  system->setMaterial(material, mesh->entireBodyName());

  // Add gravity body force
  system->addBodyForce(mesh->entireBodyName(),
                       [](double /*time*/, auto /*X*/, auto /*u*/, auto /*v*/, auto /*a*/, auto... /*params*/) {
                         tensor<double, dim> b{};
                         b[1] = gravity;
                         return b;
                       });

  // Add dummy traction to test compilation
  system->addTraction("right", [](double /*time*/, auto /*X*/, auto /*n*/, auto /*u*/, auto /*v*/, auto /*a*/,
                                  auto... /*params*/) { return tensor<double, dim>{}; });

  auto shape_disp = system->field_store->getShapeDisp();
  auto states = system->field_store->getStateFields();
  auto output_states = system->field_store->getOutputFieldStates();

  std::string pv_dir = "paraview_solid";
  auto pv_writer = createParaviewWriter(*mesh, {output_states[0], params[0], params[1]}, pv_dir);
  pv_writer.write(0, 0.0, {output_states[0], params[0], params[1]});

  double time = 0.0;
  size_t cycle = 0;
  std::vector<ReactionState> reactions;

  auto advancer = makeAdvancer(system);

  for (size_t m = 0; m < num_steps_; ++m) {
    TimeInfo t_info(time, dt_, cycle);
    std::tie(states, reactions) = advancer->advanceState(t_info, shape_disp, states, params);
    output_states = system->field_store->getOutputFieldStates();
    time += dt_;
    cycle++;
    pv_writer.write(m + 1, time, {output_states[0], params[0], params[1]});
  }

  double a_exact = gravity;
  double v_exact = gravity * total_simulation_time_;
  double u_exact = 0.5 * gravity * total_simulation_time_ * total_simulation_time_;

  TimeInfo endTimeInfo(time, dt_, cycle);

  // Test acceleration (states[3] is acceleration)
  FunctionalObjective<dim, Parameters<VectorSpace>> accel_error("accel_error", mesh, spaces({states[3]}));
  accel_error.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [a_exact](auto /*t*/, auto /*X*/, auto A) {
    auto a = get<VALUE>(A);
    auto da0 = a[0];
    auto da1 = a[1] - a_exact;
    return da0 * da0 + da1 * da1;
  });
  double a_err = accel_error.evaluate(endTimeInfo, shape_disp.get().get(), getConstFieldPointers({states[3]}));
  EXPECT_NEAR(0.0, a_err, 1e-12);

  // Test velocity (states[2] is velocity)
  FunctionalObjective<dim, Parameters<VectorSpace>> velo_error("velo_error", mesh, spaces({states[2]}));
  velo_error.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [v_exact](auto /*t*/, auto /*X*/, auto V) {
    auto v = get<VALUE>(V);
    auto dv0 = v[0];
    auto dv1 = v[1] - v_exact;
    return dv0 * dv0 + dv1 * dv1;
  });
  double v_err = velo_error.evaluate(TimeInfo(0.0, 1.0, 0), shape_disp.get().get(), getConstFieldPointers({states[2]}));
  EXPECT_NEAR(0.0, v_err, 1e-12);

  // Test displacement (states[1] is displacement)
  FunctionalObjective<dim, Parameters<VectorSpace>> disp_error("disp_error", mesh, spaces({states[1]}));
  disp_error.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [u_exact](auto /*t*/, auto /*X*/, auto U) {
    auto u = get<VALUE>(U);
    auto du0 = u[0];
    auto du1 = u[1] - u_exact;
    return du0 * du0 + du1 * du1;
  });
  double u_err = disp_error.evaluate(TimeInfo(0.0, 1.0, 0), shape_disp.get().get(), getConstFieldPointers({states[0]}));
  EXPECT_NEAR(0.0, u_err, 1e-14);
}

auto createSolidMechanicsBasePhysics(std::string physics_name, std::shared_ptr<smith::Mesh> mesh)
{
  std::shared_ptr<NonlinearBlockSolver> solid_block_solver =
      buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);

  auto coupled_solver = std::make_shared<SystemSolver>(solid_block_solver);
  auto field_store = std::make_shared<FieldStore>(mesh, 100, physics_name);

  using TimeRule = ImplicitNewmarkSecondOrderTimeIntegrationRule;
  auto param_fields =
      registerParameterFields(FieldType<ScalarParameterSpace>("bulk"), FieldType<ScalarParameterSpace>("shear"));
  auto solid_fields = registerSolidMechanicsFields<dim, order, TimeRule>(field_store);

  auto system = buildSolidMechanicsSystem<dim, order, TimeRule>(coupled_solver, SolidMechanicsOptions{}, solid_fields,
                                                                param_fields);

  auto physics = makeDifferentiablePhysics(system, physics_name);
  auto bcs = system->disp_bc;

  bcs->setFixedVectorBCs<dim>(mesh->domain("right"));
  bcs->setVectorBCs<dim>(mesh->domain("left"), [](double t, tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = 0.01 * t;
    bc[1] = -0.05 * t;
    return bc;
  });

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2 * nu));
  auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = solid_mechanics::ParameterizedNeoHookeanSolid;
  MaterialType material{.density = 10.0, .K0 = K, .G0 = G};

  system->setMaterial(material, mesh->entireBodyName());

  auto shape_disp = physics->getShapeDispFieldState();
  auto params = physics->getFieldParams();
  auto states = physics->getInitialFieldStates();

  params[0].get()->setFromFieldFunction([=](tensor<double, dim>) {
    double scaling = 1.0;
    return scaling * material.K0;
  });

  params[1].get()->setFromFieldFunction([=](tensor<double, dim>) {
    double scaling = 1.0;
    return scaling * material.G0;
  });

  physics->resetStates();

  return std::make_tuple(std::move(physics), shape_disp, states, params, bcs);
}

TEST_F(SolidMechanicsMeshFixture, SensitivitiesGretl)
{
  SMITH_MARK_FUNCTION;
  std::string physics_name = "solid";
  auto [physics, shape_disp, initial_states, params, bcs] = createSolidMechanicsBasePhysics(physics_name, mesh);

  auto pv_writer = smith::createParaviewWriter(*mesh, physics->getFieldStatesAndParamStates(), physics_name);
  pv_writer.write(0, physics->time(), physics->getFieldStatesAndParamStates());
  for (size_t m = 0; m < num_steps_; ++m) {
    physics->advanceTimestep(dt_);
    pv_writer.write(m + 1, physics->time(), physics->getFieldStatesAndParamStates());
  }

  auto reactions = physics->getReactionStates();

  // Check that reaction forces are zero away from Dirichlet DOFs
  checkUnconstrainedReactions(*reactions[0].get(), bcs->getBoundaryConditionManager());

  auto reaction_squared = 0.5 * innerProduct(reactions[0], reactions[0]);

  gretl::set_as_objective(reaction_squared);

  EXPECT_GT(checkGradWrt(reaction_squared, shape_disp, 1.1e-2, 4, true), 0.7);
  EXPECT_GT(checkGradWrt(reaction_squared, params[0], 6.2e-1, 4, true), 0.7);
  EXPECT_GT(checkGradWrt(reaction_squared, params[1], 6.2e-1, 4, true), 0.7);

  // re-evaluate the final objective value, and backpropagate again
  reaction_squared.get();
  gretl::set_as_objective(reaction_squared);
  reaction_squared.data_store().back_prop();

  for (auto s : initial_states) {
    SLIC_INFO_ROOT(axom::fmt::format("{} {} {}", s.get()->name(), s.get()->Norml2(), s.get_dual()->Norml2()));
  }

  SLIC_INFO_ROOT(axom::fmt::format("{} {} {}", shape_disp.get()->name(), shape_disp.get()->Norml2(),
                                   shape_disp.get_dual()->Norml2()));

  for (size_t p = 0; p < params.size(); ++p) {
    SLIC_INFO_ROOT(axom::fmt::format("{} {} {}", params[p].get()->name(), params[p].get()->Norml2(),
                                     params[p].get_dual()->Norml2()));
  }
}

// these functions mimic the BasePhysics style of running smith

void resetAndApplyInitialConditions(BasePhysics& physics) { physics.resetStates(); }

double integrateForward(BasePhysics& physics, size_t num_steps, double dt, std::string reaction_name)
{
  resetAndApplyInitialConditions(physics);
  for (size_t m = 0; m < num_steps; ++m) {
    physics.advanceTimestep(dt);
  }
  FiniteElementDual reaction = physics.dual(reaction_name);

  return 0.5 * innerProduct(reaction, reaction);
}

void adjointBackward(BasePhysics& physics, smith::FiniteElementDual& shape_sensitivity,
                     std::vector<smith::FiniteElementDual>& parameter_sensitivities, std::string reaction_name)
{
  smith::FiniteElementDual reaction = physics.dual(reaction_name);
  smith::FiniteElementState reaction_dual(reaction.space(), reaction_name + "_dual");
  reaction_dual = reaction;

  physics.resetAdjointStates();

  physics.setDualAdjointBcs({{reaction_name, reaction_dual}});

  while (physics.cycle() > 0) {
    physics.reverseAdjointTimestep();
    shape_sensitivity += physics.computeTimestepShapeSensitivity();
    for (size_t param_index = 0; param_index < parameter_sensitivities.size(); ++param_index) {
      parameter_sensitivities[param_index] += physics.computeTimestepSensitivity(param_index);
    }
  }
}

TEST_F(SolidMechanicsMeshFixture, SensitivitiesBasePhysics)
{
  SMITH_MARK_FUNCTION;
  std::string physics_name = "solid";
  auto [physics, shape_disp, initial_states, params, bcs] = createSolidMechanicsBasePhysics(physics_name, mesh);

  double qoi = integrateForward(*physics, num_steps_, dt_, physics_name + "_reactions");
  SLIC_INFO_ROOT(axom::fmt::format("{}", qoi));

  // Check that reaction forces are zero away from Dirichlet DOFs
  auto reactions = physics->getReactionStates();
  checkUnconstrainedReactions(*reactions[0].get(), bcs->getBoundaryConditionManager());

  size_t num_params = physics->parameterNames().size();

  smith::FiniteElementDual shape_sensitivity(*shape_disp.get_dual());
  std::vector<smith::FiniteElementDual> parameter_sensitivities;
  for (size_t p = 0; p < num_params; ++p) {
    parameter_sensitivities.emplace_back(*params[p].get_dual());
  }

  adjointBackward(*physics, shape_sensitivity, parameter_sensitivities, physics_name + "_reactions");

  auto state_sensitivities = physics->computeInitialConditionSensitivity();
  for (auto name_and_state_sensitivity : state_sensitivities) {
    SLIC_INFO_ROOT(
        axom::fmt::format("{} {}", name_and_state_sensitivity.first, name_and_state_sensitivity.second.Norml2()));
  }

  SLIC_INFO_ROOT(axom::fmt::format("{} {}", shape_sensitivity.name(), shape_sensitivity.Norml2()));

  for (size_t p = 0; p < num_params; ++p) {
    SLIC_INFO_ROOT(axom::fmt::format("{} {}", parameter_sensitivities[p].name(), parameter_sensitivities[p].Norml2()));
  }
}

TEST_F(SolidMechanicsMeshFixture, SensitivitiesComparison)
{
  SMITH_MARK_FUNCTION;
  std::string physics_name = "solid";

  // 1. Calculate sensitivities using Gretl
  auto [physicsGretl, shape_dispG, initial_statesG, paramsG, bcsG] =
      createSolidMechanicsBasePhysics(physics_name + "_gretl", mesh);

  // Forward pass
  for (size_t m = 0; m < num_steps_; ++m) {
    physicsGretl->advanceTimestep(dt_);
  }

  auto reactionsG = physicsGretl->getReactionStates();
  auto reaction_squaredG = 0.5 * innerProduct(reactionsG[0], reactionsG[0]);

  // Backprop
  gretl::set_as_objective(reaction_squaredG);
  reaction_squaredG.data_store().back_prop();

  // 2. Calculate sensitivities using BasePhysics manual adjoint
  auto [physicsBase, shape_dispB, initial_statesB, paramsB, bcsB] =
      createSolidMechanicsBasePhysics(physics_name + "_base", mesh);

  // Forward pass
  double qoiB = integrateForward(*physicsBase, num_steps_, dt_, physics_name + "_base_reactions");

  // Adjoint pass
  size_t num_params = physicsBase->parameterNames().size();
  smith::FiniteElementDual shape_sensitivityB(*shape_dispB.get_dual());
  shape_sensitivityB = 0.0;
  std::vector<smith::FiniteElementDual> parameter_sensitivitiesB;
  for (size_t p = 0; p < num_params; ++p) {
    parameter_sensitivitiesB.emplace_back(*paramsB[p].get_dual());
    parameter_sensitivitiesB.back() = 0.0;
  }

  adjointBackward(*physicsBase, shape_sensitivityB, parameter_sensitivitiesB, physics_name + "_base_reactions");
  auto initial_condition_sensitivitiesB = physicsBase->computeInitialConditionSensitivity();

  // 3. Compare sensitivities
  double tol = 1e-12;

  // Compare objective values
  EXPECT_NEAR(reaction_squaredG.get(), qoiB, tol);

  auto diff_norm = [](const mfem::Vector& a, const mfem::Vector& b) {
    mfem::Vector diff = a;
    diff -= b;
    return diff.Norml2();
  };

  // Compare shape sensitivities
  double shape_diff = diff_norm(*shape_dispG.get_dual(), shape_sensitivityB);
  SLIC_INFO_ROOT(axom::fmt::format("Shape sensitivity difference: {:.6e}", shape_diff));
  EXPECT_LT(shape_diff, tol);

  // Compare parameter sensitivities
  for (size_t p = 0; p < num_params; ++p) {
    double param_diff = diff_norm(*paramsG[p].get_dual(), parameter_sensitivitiesB[p]);
    SLIC_INFO_ROOT(axom::fmt::format("Parameter {} sensitivity difference: {:.6e}", p, param_diff));
    EXPECT_LT(param_diff, tol);
  }

  // Compare initial condition sensitivities
  std::vector<std::string> state_suffixes = {"displacement_solve_state", "displacement", "velocity", "acceleration"};
  for (const auto& suffix : state_suffixes) {
    std::string nameG = physics_name + "_gretl_" + suffix;
    std::string nameB = physics_name + "_base_" + suffix;
    SLIC_INFO_ROOT(axom::fmt::format("Comparing sensitivity for {}: {} vs {}", suffix, nameG, nameB));

    // Find Gretl dual
    const FiniteElementDual* dualG = nullptr;
    for (auto const& s : initial_statesG) {
      if (s.get()->name() == nameG) {
        dualG = s.get_dual().get();
        break;
      }
    }
    ASSERT_NE(dualG, nullptr) << "Could not find Gretl dual for " << nameG;

    double state_diff = diff_norm(*dualG, initial_condition_sensitivitiesB.at(nameB));
    SLIC_INFO_ROOT(axom::fmt::format("Initial state {} sensitivity difference: {:.6e}", suffix, state_diff));
    EXPECT_LT(state_diff, tol);
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
