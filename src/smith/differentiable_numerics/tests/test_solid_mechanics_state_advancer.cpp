// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "gretl/data_store.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/functional_objective.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"

#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/differentiable_solid_mechanics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"

namespace smith {

/**
 * @brief Verify that reaction forces are zero at non-Dirichlet DOFs.
 * @param reaction_field The reaction field to check.
 * @param bc Boundary conditions to identify Dirichlet DOFs.
 * @param tolerance Absolute tolerance for zero check.
 */
void checkUnconstrainedReactionForces(const FiniteElementDual& reaction_field,
                                       const DirichletBoundaryConditions& bc, double tolerance = 1e-10)
{
  const auto& true_dofs = bc.getBoundaryConditionManager().allEssentialTrueDofs();
  const mfem::Array<int>& ess_tdof_list = true_dofs;

  // Get reaction data
  const mfem::Vector& reaction_data = reaction_field;

  // Create a set of essential DOFs for fast lookup
  std::set<int> ess_dof_set(ess_tdof_list.begin(), ess_tdof_list.end());

  // Check all DOFs
  double max_non_bc_reaction = 0.0;
  int num_non_bc_violations = 0;

  for (int i = 0; i < reaction_data.Size(); ++i) {
    if (ess_dof_set.find(i) == ess_dof_set.end()) {
      // This is NOT a Dirichlet DOF, reaction should be zero
      double reaction_value = std::abs(reaction_data[i]);
      if (reaction_value > tolerance) {
        max_non_bc_reaction = std::max(max_non_bc_reaction, reaction_value);
        ++num_non_bc_violations;
      }
    }
  }

  if (num_non_bc_violations > 0) {
    SLIC_WARNING_ROOT(axom::fmt::format("Found {} non-Dirichlet DOFs with non-zero reactions (max: {:.6e})",
                                        num_non_bc_violations, max_non_bc_reaction));
  }

  // Expect all non-Dirichlet DOFs to have zero reactions
  EXPECT_EQ(num_non_bc_violations, 0) << "Reaction forces should be zero at non-Dirichlet DOFs. Max violation: "
                                      << max_non_bc_reaction;

  SLIC_INFO_ROOT(axom::fmt::format("Reaction force check passed. {} Dirichlet DOFs, {} free DOFs", ess_dof_set.size(),
                                   static_cast<size_t>(reaction_data.Size()) - ess_dof_set.size()));
}

smith::LinearSolverOptions solid_linear_options{.linear_solver = smith::LinearSolver::CG,
                                                .preconditioner = smith::Preconditioner::HypreJacobi,
                                                .relative_tol = 1e-11,
                                                .absolute_tol = 1e-11,
                                                .max_iterations = 10000,
                                                .print_level = 1};

smith::NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                   .relative_tol = 1.0e-10,
                                                   .absolute_tol = 1.0e-10,
                                                   .max_iterations = 500,
                                                   .print_level = 1,
                                                   .force_monolithic = true};

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
    smith::StateManager::initialize(datastore, "solid");
    auto mfem_shape = mfem::Element::HEXAHEDRON;
    mesh = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
        "mesh", 0, 0);
    mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
    mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  }

  static constexpr double total_simulation_time_ = 1.1;
  static constexpr size_t num_steps_ = 4;
  static constexpr double dt_ = total_simulation_time_ / num_steps_;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
};

TEST_F(SolidMechanicsMeshFixture, TransientConstantGravity)
{
  SMITH_MARK_FUNCTION;

  auto d_solid_nonlinear_solver =
      buildDifferentiableNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);

  auto system = buildSolidMechanicsSystem<dim, order>(
      mesh, d_solid_nonlinear_solver, ImplicitNewmarkSecondOrderTimeIntegrationRule{},
      FieldType<ScalarParameterSpace>("bulk"), FieldType<ScalarParameterSpace>("shear"));

  static constexpr double gravity = -9.0;

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2.0 * nu));
  auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = solid_mechanics::ParameterizedNeoHookeanSolid;
  MaterialType material{.density = 1.0, .K0 = K, .G0 = G};

  // Set parameters
  auto params = system.getParameterFields();
  params[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return material.K0; });
  params[1].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return material.G0; });

  // Set material
  system.setMaterial(material, mesh->entireBodyName());

  // Add gravity body force
  system.solid_weak_form->addBodyIntegral(
      smith::DependsOn<0, 1, 2, 3>{}, mesh->entireBodyName(),
      [](auto /*t_info*/, auto /*X*/, auto /*disp*/, auto /*disp_old*/, auto /*velo_old*/, auto accel_old) {
        smith::tensor<double, dim> b{};
        b[1] = gravity;
        auto zero_grad = 0.0 * get<DERIVATIVE>(accel_old);
        return smith::tuple{-b, zero_grad};
      });

  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();

  // Set initial acceleration to gravity
  states[3].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    smith::tensor<double, dim> a{};
    a[1] = gravity;
    return a;
  });

  std::string pv_dir = "paraview_solid";
  auto pv_writer = createParaviewWriter(*mesh, {states[0], params[0], params[1]}, pv_dir);
  pv_writer.write(0, 0.0, {states[0], params[0], params[1]});

  double time = 0.0;
  for (size_t m = 0; m < num_steps_; ++m) {
    TimeInfo t_info(time, dt_, m);
    auto [new_states, reactions] = system.advancer->advanceState(t_info, shape_disp, states, params);
    states = new_states;
    time += dt_;
    pv_writer.write(m + 1, time, {states[0], params[0], params[1]});
  }

  double a_exact = gravity;
  double v_exact = gravity * total_simulation_time_;
  double u_exact = 0.5 * gravity * total_simulation_time_ * total_simulation_time_;

  TimeInfo endTimeInfo(time, dt_, num_steps_);

  // Test acceleration (states[3] is acceleration_old)
  FunctionalObjective<dim, Parameters<VectorSpace>> accel_error("accel_error", mesh, spaces({states[3]}));
  accel_error.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [a_exact](auto /*t*/, auto /*X*/, auto A) {
    auto a = get<VALUE>(A);
    auto da0 = a[0];
    auto da1 = a[1] - a_exact;
    return da0 * da0 + da1 * da1;
  });
  double a_err = accel_error.evaluate(endTimeInfo, shape_disp.get().get(), getConstFieldPointers({states[3]}));
  EXPECT_NEAR(0.0, a_err, 1e-14);

  // Test velocity (states[2] is velocity_old)
  FunctionalObjective<dim, Parameters<VectorSpace>> velo_error("velo_error", mesh, spaces({states[2]}));
  velo_error.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [v_exact](auto /*t*/, auto /*X*/, auto V) {
    auto v = get<VALUE>(V);
    auto dv0 = v[0];
    auto dv1 = v[1] - v_exact;
    return dv0 * dv0 + dv1 * dv1;
  });
  double v_err = velo_error.evaluate(TimeInfo(0.0, 1.0, 0), shape_disp.get().get(), getConstFieldPointers({states[2]}));
  EXPECT_NEAR(0.0, v_err, 1e-14);

  // Test displacement (states[0] is current displacement)
  FunctionalObjective<dim, Parameters<VectorSpace>> disp_error("disp_error", mesh, spaces({states[0]}));
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
  size_t num_checkpoints = 200;
  std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver =
      buildDifferentiableNonlinearSolver(solid_nonlinear_opts, solid_linear_options, *mesh);

  auto [physics, solid_weak_form, bcs] =
      buildSolidMechanics<dim, ShapeDispSpace, VectorSpace, ScalarParameterSpace, ScalarParameterSpace>(
          mesh, d_solid_nonlinear_solver, ImplicitNewmarkSecondOrderTimeIntegrationRule(), num_checkpoints,
          physics_name, {"bulk", "shear"});

  bcs->setFixedVectorBCs<dim>(mesh->domain("right"));
  bcs->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
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

  solid_weak_form->addBodyIntegral(
      smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
      [material](const auto& /*time_info*/, auto /*X*/, auto u, auto /*v*/, auto a, auto bulk, auto shear) {
        MaterialType::State state;
        auto pk_stress = material(state, get<DERIVATIVE>(u), bulk, shear);
        return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
      });

  auto shape_disp = physics->getShapeDispFieldState();
  auto params = physics->getFieldParams();
  auto states = physics->getInitialFieldStates();

  params[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double scaling = 1.0;
    return scaling * material.K0;
  });

  params[1].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double scaling = 1.0;
    return scaling * material.G0;
  });

  physics->resetStates();

  return std::make_tuple(physics, shape_disp, states, params, bcs);
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
  checkUnconstrainedReactionForces(*reactions[0].get(), *bcs);

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

void resetAndApplyInitialConditions(std::shared_ptr<BasePhysics> physics) { physics->resetStates(); }

double integrateForward(std::shared_ptr<BasePhysics> physics, size_t num_steps, double dt)
{
  resetAndApplyInitialConditions(physics);
  for (size_t m = 0; m < num_steps; ++m) {
    physics->advanceTimestep(dt);
  }
  FiniteElementDual reaction = physics->dual("reactions");

  return 0.5 * innerProduct(reaction, reaction);
}

void adjointBackward(std::shared_ptr<BasePhysics> physics, smith::FiniteElementDual& shape_sensitivity,
                     std::vector<smith::FiniteElementDual>& parameter_sensitivities)
{
  smith::FiniteElementDual reaction = physics->dual("reactions");
  smith::FiniteElementState reaction_dual(reaction.space(), "reactions_dual");
  reaction_dual = reaction;

  physics->resetAdjointStates();

  physics->setDualAdjointBcs({{"reactions", reaction_dual}});

  while (physics->cycle() > 0) {
    physics->reverseAdjointTimestep();
    shape_sensitivity += physics->computeTimestepShapeSensitivity();
    for (size_t param_index = 0; param_index < parameter_sensitivities.size(); ++param_index) {
      parameter_sensitivities[param_index] += physics->computeTimestepSensitivity(param_index);
    }
  }
}

TEST_F(SolidMechanicsMeshFixture, SensitivitiesBasePhysics)
{
  SMITH_MARK_FUNCTION;
  std::string physics_name = "solid";
  auto [physics, shape_disp, initial_states, params, bcs] = createSolidMechanicsBasePhysics(physics_name, mesh);

  double qoi = integrateForward(physics, num_steps_, dt_);
  SLIC_INFO_ROOT(axom::fmt::format("{}", qoi));

  // Check that reaction forces are zero away from Dirichlet DOFs
  auto reactions = physics->getReactionStates();
  checkUnconstrainedReactionForces(*reactions[0].get(), *bcs);

  size_t num_params = physics->parameterNames().size();

  smith::FiniteElementDual shape_sensitivity(*shape_disp.get_dual());
  std::vector<smith::FiniteElementDual> parameter_sensitivities;
  for (size_t p = 0; p < num_params; ++p) {
    parameter_sensitivities.emplace_back(*params[p].get_dual());
  }

  adjointBackward(physics, shape_sensitivity, parameter_sensitivities);

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
  auto [physicsGretl, shape_dispG, initial_statesG, paramsG, bcsG] = createSolidMechanicsBasePhysics(physics_name + "_gretl", mesh);

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
  auto [physicsBase, shape_dispB, initial_statesB, paramsB, bcsB] = createSolidMechanicsBasePhysics(physics_name + "_base", mesh);

  // Forward pass
  double qoiB = integrateForward(physicsBase, num_steps_, dt_);

  // Adjoint pass
  size_t num_params = physicsBase->parameterNames().size();
  smith::FiniteElementDual shape_sensitivityB(*shape_dispB.get_dual());
  shape_sensitivityB = 0.0;
  std::vector<smith::FiniteElementDual> parameter_sensitivitiesB;
  for (size_t p = 0; p < num_params; ++p) {
    parameter_sensitivitiesB.emplace_back(*paramsB[p].get_dual());
    parameter_sensitivitiesB.back() = 0.0;
  }

  adjointBackward(physicsBase, shape_sensitivityB, parameter_sensitivitiesB);
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
  std::vector<std::string> state_suffixes = {"displacement", "velocity", "acceleration"};
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
