// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_combined_thermo_mechanics.cpp
 * @brief Demo of combineSystems: same coupled thermoelastic problem as test_thermo_mechanics
 * but written with the new modular API.  Results are diffed against the old
 * buildThermoMechanicsSystem path to confirm correctness.
 *
 * Code-size comparison (staggered case):
 *
 * OLD (buildThermoMechanicsSystem):
 *   - Build system: 1 call (buildThermoMechanicsSystem)
 *   - Stagger config: manual addSubsystemSolver({0}, ...) + addSubsystemSolver({0,1}, ...)
 *   - BCs:  system->disp_bc / system->temperature_bc
 *   - BCs:  system->addSolidTraction, system->addSolidBodyForce, system->addHeatSource
 *   - Material: single system->setMaterial call
 *
 * NEW (combineSystems):
 *   - Register fields: registerSolidMechanicsFields + registerThermalFields
 *   - CouplingSpec: 2 declarations (solid borrows temp, thermal borrows disp)
 *   - Build systems: buildSolidMechanicsSystemFromStore + buildThermalSystemFromStore
 *   - Combine:  combineSystems(solid, thermal)
 *   - Stagger:  automatically handled by CombinedSystem::solve — no block-index wiring
 *   - BCs:  solid->disp_bc / thermal->temperature_bc  (cleaner naming)
 *   - BCs:  solid->addTraction, solid->addBodyForce, thermal->addHeatSource
 *   - Material: setCoupledThermoMechanicsMaterial(solid, thermal, mat, domain)
 *
 * Win: no manual addSubsystemSolver block indices; each physics solver is independently
 * configured and passed directly to its factory.
 */

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
#include "smith/differentiable_numerics/thermo_mechanics_system.hpp"  // for reference solution
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"

namespace smith {

static constexpr int dim = 3;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;

// ---------------------------------------------------------------------------
// Coupled thermoelastic material — no user parameter fields; E is hardcoded.
// This avoids the shared-parameter registration issue (both physics needing the
// same L2 param in the same FieldStore) which requires a dedicated combineSystems
// API beyond this demo.  All other coupling (solid<->thermal fields) is exercised.
// ---------------------------------------------------------------------------
template <typename T, int dim_>
auto greenStrainCombined(const tensor<T, dim_, dim_>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

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
    const auto Eg = greenStrainCombined<T1, dim>(grad_u);
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

// Also keep the parameterized version (with L2<0> E_param) for the OLD path comparison only.
template <typename T, int dim_>
auto greenStrainCombinedParam(const tensor<T, dim_, dim_>& grad_u)
{
  return greenStrainCombined<T, dim_>(grad_u);
}

struct GreenSaintVenantThermoelasticMaterialCombined {
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
    const auto Eg = greenStrainCombinedParam<T1, dim>(grad_u);
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

// ---------------------------------------------------------------------------
// Free function: register material integrands on separate solid + thermal systems.
// This is the new "coupled setMaterial" pattern when using combineSystems.
//
// Solid closure sees:  (t_info, X, u, u_old, v_old, a_old, temp_solve_state, temperature, params...)
// Thermal closure sees: (t_info, X, T, T_old, disp, disp_old, velo, accel, params...)
// ---------------------------------------------------------------------------
template <int disp_order_, int temp_order_, typename DispRule, typename TempRule, typename... P,
          typename MaterialType>
void setCoupledThermoMechanicsMaterial(
    std::shared_ptr<SolidMechanicsSystem<dim, disp_order_,
                                         DispRule,
                                         CouplingSpec<H1<temp_order_>, H1<temp_order_>>,
                                         P...>> solid,
    std::shared_ptr<ThermalSystem<dim, temp_order_,
                                   TempRule,
                                   CouplingSpec<H1<disp_order_, dim>, H1<disp_order_, dim>,
                                                H1<disp_order_, dim>, H1<disp_order_, dim>>,
                                   P...>> thermal,
    const MaterialType& material,
    const std::string& domain_name)
{
  auto captured_disp_rule = solid->disp_time_rule;
  auto captured_temp_rule = thermal->temperature_time_rule;

  // Solid contribution: inertia + PK1 stress
  solid->solid_weak_form->addBodyIntegral(
      domain_name,
      [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old, auto a_old,
          auto temperature, auto temperature_old, auto... params) {
        auto [u_current, v_current, a_current] =
            captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
        auto T = captured_temp_rule->value(t_info, temperature, temperature_old);

        typename MaterialType::State state;
        auto [pk, C_v, s0, q0] =
            material(t_info.dt(), state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current),
                     get<VALUE>(T), get<DERIVATIVE>(T), params...);
        return smith::tuple{get<VALUE>(a_current) * material.density, pk};
      });

  // Thermal contribution: heat capacity * dT/dt - volumetric source, and heat flux
  thermal->thermal_weak_form->addBodyIntegral(
      domain_name,
      [=](auto t_info, auto /*X*/, auto T, auto T_old,
          auto disp, auto disp_old, auto v_old, auto a_old, auto... params) {
        auto [T_current, T_dot] = captured_temp_rule->interpolate(t_info, T, T_old);
        auto [u, v, a] = captured_disp_rule->interpolate(t_info, disp, disp_old, v_old, a_old);

        typename MaterialType::State state;
        auto [pk, C_v, s0, q0] =
            material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v),
                     get<VALUE>(T_current), get<DERIVATIVE>(T_current), params...);
        return smith::tuple{C_v * get<VALUE>(T_dot) - s0, -q0};
      });
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
struct CombinedThermoMechanicsMeshFixture : public testing::Test {
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

// ---------------------------------------------------------------------------
// Helper: build and run the problem using the OLD buildThermoMechanicsSystem path
// ---------------------------------------------------------------------------
static auto runOldStaggered(std::shared_ptr<Mesh> mesh,
                            std::shared_ptr<NonlinearBlockSolverBase> solid_block_solver,
                            std::shared_ptr<NonlinearBlockSolverBase> thermal_block_solver)
{
  ThermoelasticMaterialNoParam material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};

  // OLD: manually wire block indices for stagger
  auto staggered_solver = std::make_shared<SystemSolver>(10);
  staggered_solver->addSubsystemSolver({0}, solid_block_solver);
  staggered_solver->addSubsystemSolver({1}, thermal_block_solver);

  auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(
      mesh, staggered_solver, QuasiStaticSecondOrderTimeIntegrationRule{},
      BackwardEulerFirstOrderTimeIntegrationRule{}, {});

  system->setMaterial(material, mesh->entireBodyName());
  system->disp_bc->setFixedVectorBCs<dim>(mesh->domain("left"));
  system->temperature_bc->setFixedScalarBCs<dim>(mesh->domain("left"));
  system->temperature_bc->setFixedScalarBCs<dim>(mesh->domain("right"));

  constexpr double compressive_traction = 0.015;
  constexpr double lateral_body_force = 2.5e-5;
  constexpr double thermal_source = 1.0;

  system->addSolidTraction("right", [=](auto, auto X, auto... /*args*/) {
    auto traction = 0.0 * X;
    traction[0] = -compressive_traction;
    return traction;
  });
  system->addSolidBodyForce(mesh->entireBodyName(), [=](auto, auto X, auto... /*args*/) {
    auto force = 0.0 * X;
    force[1] = lateral_body_force;
    return force;
  });
  system->addHeatSource(mesh->entireBodyName(),
                        [=](auto, auto, auto, auto, auto, auto, auto) { return thermal_source; });

  auto shape_disp = system->field_store->getShapeDisp();
  auto states = system->field_store->getStateFields();
  auto params = system->field_store->getParameterFields();
  std::vector<ReactionState> reactions;
  double time = 0.0;
  std::tie(states, reactions) =
      makeAdvancer(system)->advanceState(smith::TimeInfo(time, 1.0, 0), shape_disp, states, params);

  return std::make_pair(
      mfem::Vector(*states[system->field_store->getFieldIndex("displacement_solve_state")].get()),
      mfem::Vector(*states[system->field_store->getFieldIndex("temperature_solve_state")].get()));
}

// ---------------------------------------------------------------------------
// Helper: build and run the problem using the NEW combineSystems path
// ---------------------------------------------------------------------------
static auto runNewCombined(std::shared_ptr<Mesh> mesh,
                           std::shared_ptr<NonlinearBlockSolverBase> solid_block_solver,
                           std::shared_ptr<NonlinearBlockSolverBase> thermal_block_solver)
{
  // Use no-param material — hardcoded E avoids the shared-parameter registration issue.
  ThermoelasticMaterialNoParam material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};

  // ---- Phase 1: register all fields into a shared store ----
  auto field_store = std::make_shared<FieldStore>(mesh, 100, "");
  auto solid_info = registerSolidMechanicsFields<dim, displacement_order>(
      field_store, QuasiStaticSecondOrderTimeIntegrationRule{});
  auto thermal_info = registerThermalFields<dim, temperature_order>(
      field_store, BackwardEulerFirstOrderTimeIntegrationRule{});

  // ---- Declare coupling: each physics borrows fields from the other ----
  CouplingSpec solid_coupling{FieldType<H1<temperature_order>>("temperature_solve_state"),
                              FieldType<H1<temperature_order>>("temperature")};
  CouplingSpec thermal_coupling{FieldType<H1<displacement_order, dim>>("displacement_solve_state"),
                                FieldType<H1<displacement_order, dim>>("displacement"),
                                FieldType<H1<displacement_order, dim>>("velocity"),
                                FieldType<H1<displacement_order, dim>>("acceleration")};

  // ---- Phase 2: build each system with its own solver — no block-index wiring ----
  using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = BackwardEulerFirstOrderTimeIntegrationRule;

  SolidMechanicsOptions<dim, displacement_order, DispRule> solid_opts{};
  ThermalOptions<dim, temperature_order, TempRule> thermal_opts{};

  auto solid = buildSolidMechanicsSystemFromStore<dim, displacement_order, DispRule>(
      solid_info, std::make_shared<SystemSolver>(solid_block_solver), solid_opts,
      solid_coupling);
  auto thermal = buildThermalSystemFromStore<dim, temperature_order, TempRule>(
      thermal_info, std::make_shared<SystemSolver>(thermal_block_solver), thermal_opts,
      thermal_coupling);

  // ---- Combine — stagger is automatic ----
  auto coupled = combineSystems(solid, thermal);

  // ---- Configure the problem ----
  setCoupledThermoMechanicsMaterial(solid, thermal, material, mesh->entireBodyName());
  solid->disp_bc->setFixedVectorBCs<dim>(mesh->domain("left"));
  thermal->temperature_bc->setFixedScalarBCs<dim>(mesh->domain("left"));
  thermal->temperature_bc->setFixedScalarBCs<dim>(mesh->domain("right"));

  constexpr double compressive_traction = 0.015;
  constexpr double lateral_body_force = 2.5e-5;
  constexpr double thermal_source = 1.0;

  // Coupling fields appear as leading auto... after time-rule state args; absorb with /**/
  solid->addTraction("right", [=](auto, auto X, auto, auto, auto, auto, auto... /*coupling+params*/) {
    auto traction = 0.0 * X;
    traction[0] = -compressive_traction;
    return traction;
  });
  solid->addBodyForce(mesh->entireBodyName(), [=](auto, auto X, auto, auto, auto, auto, auto... /*coupling+params*/) {
    auto force = 0.0 * X;
    force[1] = lateral_body_force;
    return force;
  });
  thermal->addHeatSource(mesh->entireBodyName(),
                         [=](auto, auto, auto, auto... /*coupling+params*/) { return thermal_source; });

  // ---- Solve ----
  auto shape_disp = field_store->getShapeDisp();
  auto states = field_store->getStateFields();
  auto params = field_store->getParameterFields();
  std::vector<ReactionState> reactions;
  double time = 0.0;
  std::tie(states, reactions) =
      makeAdvancer(coupled)->advanceState(smith::TimeInfo(time, 1.0, 0), shape_disp, states, params);

  return std::make_pair(
      mfem::Vector(*states[field_store->getFieldIndex("displacement_solve_state")].get()),
      mfem::Vector(*states[field_store->getFieldIndex("temperature_solve_state")].get()));
}

// ---------------------------------------------------------------------------
// Test: new combineSystems result matches old buildThermoMechanicsSystem
// ---------------------------------------------------------------------------
TEST_F(CombinedThermoMechanicsMeshFixture, StaggeredResultsMatchOldPath)
{
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

  auto solid_solver_old = buildNonlinearBlockSolver(mech_nonlin_opts, mech_lin_opts, *mesh_);
  auto thermal_solver_old = buildNonlinearBlockSolver(therm_nonlin_opts, therm_lin_opts, *mesh_);
  auto old_result = runOldStaggered(mesh_, solid_solver_old, thermal_solver_old);

  // Reset state manager between runs (same pattern as test_thermo_mechanics)
  mesh_.reset();
  smith::StateManager::reset();
  SetUp();

  auto solid_solver_new = buildNonlinearBlockSolver(mech_nonlin_opts, mech_lin_opts, *mesh_);
  auto thermal_solver_new = buildNonlinearBlockSolver(therm_nonlin_opts, therm_lin_opts, *mesh_);
  auto new_result = runNewCombined(mesh_, solid_solver_new, thermal_solver_new);

  double disp_diff =
      mfem::Vector(old_result.first).Add(-1.0, new_result.first).Normlinf();
  double temp_diff =
      mfem::Vector(old_result.second).Add(-1.0, new_result.second).Normlinf();

  SLIC_INFO_ROOT("Old vs new displacement diff: " << disp_diff);
  SLIC_INFO_ROOT("Old vs new temperature diff:  " << temp_diff);

  // Both approaches run the same number of stagger iterations so results should be
  // numerically identical (same tolerances, same solvers, same mesh).
  EXPECT_LT(disp_diff, 1e-4);
  EXPECT_LT(temp_diff, 1e-4);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
