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

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"
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

template <typename T, int dim_>
auto greenStrain(const tensor<T, dim_, dim_>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

// Material with E parameter
struct ParameterizedGreenSaintVenantThermoelasticMaterial {
  double density;
  double E0;
  double nu;
  double C_v;
  double alpha;
  double theta_ref;
  double kappa;
  using State = Empty;
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta, const T5& E_param) const
  {
    auto E = E0 + get<0>(E_param);
    const auto K = E / (3.0 * (1.0 - 2.0 * nu));
    const auto G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain<T1, dim>(grad_u);
    const auto trEg = tr(Eg);
    static constexpr auto I = Identity<dim>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - dim * alpha * (theta - theta_ref)) * I;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate) + 0.0 * E;
    const auto q0 = -kappa * grad_theta;
    return smith::tuple{Piola, C_v, s0, q0};
  }
};

// Material without user parameters
struct ThermoelasticMaterialNoParam {
  double density;
  double E;
  double nu;
  double C_v;
  double alpha;
  double theta_ref;
  double kappa;
  using State = Empty;
  template <typename T1, typename T2, typename T3, typename T4>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    const auto K = E / (3.0 * (1.0 - 2.0 * nu));
    const auto G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain<T1, dim>(grad_u);
    const auto trEg = tr(Eg);
    static constexpr auto I = Identity<dim>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - dim * alpha * (theta - theta_ref)) * I;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);
    const auto q0 = -kappa * grad_theta;
    return smith::tuple{Piola, C_v, s0, q0};
  }
};

template <int disp_order_, int temp_order_, typename DispRule, typename TempRule, typename SolidCoupling,
          typename ThermalCoupling, typename MaterialType>
void setCoupledThermoMechanicsMaterial(
    std::shared_ptr<SolidMechanicsSystem<dim, disp_order_, DispRule, SolidCoupling>> solid,
    std::shared_ptr<ThermalSystem<dim, temp_order_, TempRule, ThermalCoupling>> thermal, const MaterialType& material,
    const std::string& domain_name)
{
  auto captured_disp_rule = solid->disp_time_rule;
  auto captured_temp_rule = thermal->temperature_time_rule;

  // Solid contribution: inertia + PK1 stress
  solid->solid_weak_form->addBodyIntegral(
      domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old, auto a_old, auto temperature,
                       auto temperature_old, auto... params) {
        auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
        auto T = captured_temp_rule->value(t_info, temperature, temperature_old);

        typename MaterialType::State state;
        auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current),
                                          get<VALUE>(T), get<DERIVATIVE>(T), params...);
        return smith::tuple{get<VALUE>(a_current) * material.density, pk};
      });

  // Cycle-zero
  if (solid->cycle_zero_solid_weak_form) {
    solid->cycle_zero_solid_weak_form->addBodyIntegral(
        domain_name,
        [=](auto t_info, auto /*X*/, auto u, auto v, auto a, auto temperature, auto temperature_old, auto... params) {
          auto T = captured_temp_rule->value(t_info, temperature, temperature_old);
          typename MaterialType::State state;
          auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T),
                                            get<DERIVATIVE>(T), params...);
          return smith::tuple{get<VALUE>(a) * material.density, pk};
        });
  }

  // Thermal contribution
  thermal->thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto T, auto T_old, auto disp,
                                                               auto disp_old, auto v_old, auto a_old, auto... params) {
    auto [T_current, T_dot] = captured_temp_rule->interpolate(t_info, T, T_old);
    auto [u, v, a] = captured_disp_rule->interpolate(t_info, disp, disp_old, v_old, a_old);

    typename MaterialType::State state;
    auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T_current),
                                      get<DERIVATIVE>(T_current), params...);
    return smith::tuple{C_v * get<VALUE>(T_dot) - s0, -q0};
  });
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
  }
  std::unique_ptr<axom::sidre::DataStore> datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
};

// 1. CreateDifferentiablePhysicsAllocatesReactionInfo
TEST_F(ThermoMechanicsMeshFixture, CreateDifferentiablePhysicsAllocatesReactionInfo)
{
  smith::LinearSolverOptions lin_opts{.linear_solver = smith::LinearSolver::SuperLU};
  smith::NonlinearSolverOptions nonlin_opts{.nonlin_solver = smith::NonlinearSolver::Newton,
                                            .relative_tol = 1e-10,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 4};

  auto solid_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);
  auto thermal_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);

  auto field_store = std::make_shared<FieldStore>(mesh_, 100, "");
  FieldType<L2<0>> youngs_modulus("youngs_modulus");

  using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;
  SolidMechanicsOptions solid_opts;
  ThermalOptions thermal_opts;

  auto solid_coupling_fields =
      registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store, youngs_modulus);
  auto thermal_coupling_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store, youngs_modulus);

  auto solid_res = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      field_store, thermal_coupling_fields, std::make_shared<SystemSolver>(solid_block_solver), solid_opts);
  auto solid = solid_res.system;

  auto thermal_res = buildThermalSystem<dim, temperature_order, TempRule>(
      field_store, solid_coupling_fields, std::make_shared<SystemSolver>(thermal_block_solver), thermal_opts);
  auto thermal = thermal_res.system;

  auto [coupled, coupled_cz] = combineSystems(solid, thermal);

  auto physics = makeDifferentiablePhysics(coupled, "coupled_physics", coupled_cz);
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
  smith::LinearSolverOptions lin_opts{.linear_solver = smith::LinearSolver::SuperLU};
  smith::NonlinearSolverOptions nonlin_opts{.nonlin_solver = smith::NonlinearSolver::Newton,
                                            .relative_tol = 1e-10,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 4};

  auto solid_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);
  auto thermal_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);

  auto field_store = std::make_shared<FieldStore>(mesh_, 100, "");
  FieldType<L2<0>> youngs_modulus("youngs_modulus");

  using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;
  SolidMechanicsOptions solid_opts;
  ThermalOptions thermal_opts;

  auto solid_coupling_fields =
      registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store, youngs_modulus);
  auto thermal_coupling_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store, youngs_modulus);

  auto solid_res = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      field_store, thermal_coupling_fields, std::make_shared<SystemSolver>(solid_block_solver), solid_opts);
  auto solid = solid_res.system;

  auto thermal_res = buildThermalSystem<dim, temperature_order, TempRule>(
      field_store, solid_coupling_fields, std::make_shared<SystemSolver>(thermal_block_solver), thermal_opts);
  auto thermal = thermal_res.system;

  auto [coupled, coupled_cz] = combineSystems(solid, thermal);

  ParameterizedGreenSaintVenantThermoelasticMaterial material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  setCoupledThermoMechanicsMaterial(solid, thermal, material, mesh_->entireBodyName());

  coupled->field_store->getParameterFields()[0].get()->setFromFieldFunction(
      [=](smith::tensor<double, dim>) { return 100.0; });

  solid->disp_bc->setFixedVectorBCs<dim>(mesh_->domain("left"));
  thermal->temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));

  solid->addTraction("right", [=](double, auto X, auto, auto, auto, auto, auto, auto, auto) {
    auto traction = 0.0 * X;
    traction[0] = -0.015;
    return traction;
  });

  auto physics = makeDifferentiablePhysics(coupled, "coupled_physics", coupled_cz);

  // Run forward
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

// 3. StaggeredBucklingChallenge
// Replaces MonolithicBucklingChallenge: verifies convergence and displacement of the staggered combined solver.
TEST_F(ThermoMechanicsMeshFixture, StaggeredBucklingChallenge)
{
  constexpr double compressive_traction = 0.015;
  constexpr double lateral_body_force = 2.5e-5;
  constexpr double thermal_source = 1.0;

  smith::LinearSolverOptions mech_lin_opts{.linear_solver = smith::LinearSolver::CG,
                                           .preconditioner = smith::Preconditioner::HypreAMG,
                                           .relative_tol = 1e-6,
                                           .absolute_tol = 1e-10,
                                           .max_iterations = 120,
                                           .print_level = 0};
  smith::NonlinearSolverOptions mech_nonlin_opts{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                 .relative_tol = 1e-6,
                                                 .absolute_tol = 1e-7,
                                                 .max_iterations = 25,
                                                 .print_level = 0};

  smith::LinearSolverOptions therm_lin_opts{.linear_solver = smith::LinearSolver::GMRES,
                                            .preconditioner = smith::Preconditioner::HypreAMG,
                                            .relative_tol = 1e-6,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 80,
                                            .print_level = 0};
  smith::NonlinearSolverOptions therm_nonlin_opts{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                                  .relative_tol = 1e-7,
                                                  .absolute_tol = 1e-7,
                                                  .max_iterations = 12,
                                                  .max_line_search_iterations = 6,
                                                  .print_level = 0};

  auto solid_block_solver = buildNonlinearBlockSolver(mech_nonlin_opts, mech_lin_opts, *mesh_);
  auto thermal_block_solver = buildNonlinearBlockSolver(therm_nonlin_opts, therm_lin_opts, *mesh_);

  auto field_store = std::make_shared<FieldStore>(mesh_, 100, "");

  using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;
  SolidMechanicsOptions solid_opts;
  ThermalOptions thermal_opts;

  auto solid_coupling_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store);
  auto thermal_coupling_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store);

  auto solid_res = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      field_store, thermal_coupling_fields, std::make_shared<SystemSolver>(solid_block_solver), solid_opts);
  auto solid = solid_res.system;

  auto thermal_res = buildThermalSystem<dim, temperature_order, TempRule>(
      field_store, solid_coupling_fields, std::make_shared<SystemSolver>(thermal_block_solver), thermal_opts);
  auto thermal = thermal_res.system;

  auto [coupled, coupled_cz] = combineSystems(solid, thermal);

  ThermoelasticMaterialNoParam material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  setCoupledThermoMechanicsMaterial(solid, thermal, material, mesh_->entireBodyName());

  solid->disp_bc->setFixedVectorBCs<dim>(mesh_->domain("left"));
  thermal->temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));
  thermal->temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("right"));

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

  SLIC_INFO_ROOT("Starting staggered thermo-mechanics solve");

  double dt = 1.0;
  double time = 0.0;
  auto shape_disp = field_store->getShapeDisp();
  auto states = field_store->getStateFields();
  auto params = field_store->getParameterFields();
  std::vector<ReactionState> reactions;
  for (size_t step = 0; step < 1; ++step) {
    std::tie(states, reactions) =
        makeAdvancer(coupled, coupled_cz)->advanceState(smith::TimeInfo(time, dt, step), shape_disp, states, params);
    time += dt;
  }

  mfem::Vector final_disp(*states[field_store->getFieldIndex("displacement_solve_state")].get());

  bool staggered_solid_converged = solid_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged();
  int staggered_solid_iterations = solid_block_solver->nonlinear_solver_->nonlinearSolver().GetNumIterations();
  bool staggered_thermal_converged = thermal_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged();
  int staggered_thermal_iterations = thermal_block_solver->nonlinear_solver_->nonlinearSolver().GetNumIterations();

  double staggered_lateral_deflection = 0.0;
  for (int i = 1; i < final_disp.Size(); i += dim) {
    staggered_lateral_deflection = std::max(staggered_lateral_deflection, std::abs(final_disp(i)));
  }

  SLIC_INFO_ROOT("Staggered solid converged: " << staggered_solid_converged
                                               << ", iterations: " << staggered_solid_iterations);
  SLIC_INFO_ROOT("Staggered thermal converged: " << staggered_thermal_converged
                                                 << ", iterations: " << staggered_thermal_iterations);
  SLIC_INFO_ROOT("Staggered max lateral deflection: " << staggered_lateral_deflection);

  EXPECT_TRUE(staggered_solid_converged);
  EXPECT_TRUE(staggered_thermal_converged);
  EXPECT_GT(staggered_lateral_deflection, 1e-5);
}

// 4. MonolithicBucklingChallenge
// Verifies convergence and displacement of the monolithic combined solver.
TEST_F(ThermoMechanicsMeshFixture, MonolithicBucklingChallenge)
{
  constexpr double compressive_traction = 0.015;
  constexpr double lateral_body_force = 2.5e-5;
  constexpr double thermal_source = 1.0;

  smith::LinearSolverOptions lin_opts{.linear_solver = smith::LinearSolver::SuperLU,
                                      .relative_tol = 1e-6,
                                      .absolute_tol = 1e-10,
                                      .max_iterations = 80,
                                      .print_level = 0};
  smith::NonlinearSolverOptions nonlin_opts{.nonlin_solver = smith::NonlinearSolver::Newton,
                                            .relative_tol = 1e-7,
                                            .absolute_tol = 1e-7,
                                            .max_iterations = 12,
                                            .print_level = 0};

  auto block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);
  auto solver_ptr = std::make_shared<SystemSolver>(block_solver);

  auto field_store = std::make_shared<FieldStore>(mesh_, 100, "");

  // Notice that the block_solver is the SAME solver for the whole system
  using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;
  SolidMechanicsOptions solid_opts;
  ThermalOptions thermal_opts;

  auto solid_coupling_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store);
  auto thermal_coupling_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store);

  auto solid_res = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      field_store, thermal_coupling_fields, nullptr, solid_opts);
  auto solid = solid_res.system;

  auto thermal_res = buildThermalSystem<dim, temperature_order, TempRule>(
      field_store, solid_coupling_fields, nullptr, thermal_opts);
  auto thermal = thermal_res.system;

  auto [coupled, coupled_cz] = combineSystems(solver_ptr, solid, thermal);

  ThermoelasticMaterialNoParam material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  setCoupledThermoMechanicsMaterial(solid, thermal, material, mesh_->entireBodyName());

  solid->disp_bc->setFixedVectorBCs<dim>(mesh_->domain("left"));
  thermal->temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));
  thermal->temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("right"));

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

  SLIC_INFO_ROOT("Starting monolithic thermo-mechanics solve");

  double dt = 1.0;
  double time = 0.0;
  auto shape_disp = field_store->getShapeDisp();
  auto states = field_store->getStateFields();
  auto params = field_store->getParameterFields();
  std::vector<ReactionState> reactions;
  for (size_t step = 0; step < 1; ++step) {
    std::tie(states, reactions) =
        makeAdvancer(coupled, coupled_cz)->advanceState(smith::TimeInfo(time, dt, step), shape_disp, states, params);
    time += dt;
  }

  mfem::Vector final_disp(*states[field_store->getFieldIndex("displacement_solve_state")].get());

  bool converged = block_solver->nonlinear_solver_->nonlinearSolver().GetConverged();
  int iterations = block_solver->nonlinear_solver_->nonlinearSolver().GetNumIterations();

  double lateral_deflection = 0.0;
  for (int i = 1; i < final_disp.Size(); i += dim) {
    lateral_deflection = std::max(lateral_deflection, std::abs(final_disp(i)));
  }

  SLIC_INFO_ROOT("Monolithic converged: " << converged << ", iterations: " << iterations);
  SLIC_INFO_ROOT("Monolithic max lateral deflection: " << lateral_deflection);

  EXPECT_TRUE(converged);
  EXPECT_GT(lateral_deflection, 1e-5);
}

// 5. NewAPIBackpropagateThroughPhysics
// Same as test 2 but using the new variadic build API with PhysicsFields and registerParameterFields.
TEST_F(ThermoMechanicsMeshFixture, NewAPIBackpropagateThroughPhysics)
{
  smith::LinearSolverOptions lin_opts{.linear_solver = smith::LinearSolver::SuperLU};
  smith::NonlinearSolverOptions nonlin_opts{.nonlin_solver = smith::NonlinearSolver::Newton,
                                            .relative_tol = 1e-10,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 4};

  auto solid_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);
  auto thermal_block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);

  auto field_store = std::make_shared<FieldStore>(mesh_, 100, "");
  FieldType<L2<0>> youngs_modulus("youngs_modulus");

  using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;
  SolidMechanicsOptions solid_opts;
  ThermalOptions thermal_opts;

  auto param_fields = registerParameterFields(youngs_modulus);
  auto solid_fields = registerSolidMechanicsFields<dim, displacement_order, DispRule>(field_store);
  auto thermal_fields = registerThermalFields<dim, temperature_order, TempRule>(field_store);

  auto [solid, solid_cz, solid_end_steps] = buildSolidMechanicsSystem<dim, displacement_order, DispRule>(
      std::make_shared<SystemSolver>(solid_block_solver), solid_opts, param_fields, solid_fields, thermal_fields);

  auto [thermal, thermal_cz, thermal_end_steps] = buildThermalSystem<dim, temperature_order, TempRule>(
      std::make_shared<SystemSolver>(thermal_block_solver), thermal_opts, param_fields, thermal_fields, solid_fields);

  auto [coupled, coupled_cz] = combineSystems(solid, thermal);

  ParameterizedGreenSaintVenantThermoelasticMaterial material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  setCoupledThermoMechanicsMaterial(solid, thermal, material, mesh_->entireBodyName());

  coupled->field_store->getParameterFields()[0].get()->setFromFieldFunction(
      [=](smith::tensor<double, dim>) { return 100.0; });

  solid->setDisplacementBC(mesh_->domain("left"));
  thermal->setTemperatureBC(mesh_->domain("left"));

  solid->addTraction("right", [=](double, auto X, auto, auto, auto, auto, auto, auto, auto) {
    auto traction = 0.0 * X;
    traction[0] = -0.015;
    return traction;
  });

  auto physics = makeDifferentiablePhysics(coupled, "coupled_physics", coupled_cz);

  // Run forward
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

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
