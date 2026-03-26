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
#include "smith/differentiable_numerics/coupled_system_solver.hpp"
#include "smith/differentiable_numerics/thermo_mechanics_system.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
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

struct GreenSaintVenantThermoelasticMaterial {
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
  static constexpr int numParameters() { return 1; }
};

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

TEST_F(ThermoMechanicsMeshFixture, CreateDifferentiablePhysicsAllocatesExportedDuals)
{
  smith::LinearSolverOptions lin_opts{.linear_solver = smith::LinearSolver::SuperLU};
  smith::NonlinearSolverOptions nonlin_opts{.nonlin_solver = smith::NonlinearSolver::Newton,
                                            .relative_tol = 1e-10,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 4};

  auto block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);
  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(
      mesh_, std::make_shared<CoupledSystemSolver>(block_solver), QuasiStaticSecondOrderTimeIntegrationRule{},
      BackwardEulerFirstOrderTimeIntegrationRule{}, "thermo", youngs_modulus);

  auto physics = system.createDifferentiablePhysics("thermo_physics");
  const auto& solid_dual_space = physics->dual(system.prefix("solid_force")).space();
  const auto& solid_state_space = physics->state(system.prefix("displacement")).space();
  const auto& thermal_dual_space = physics->dual(system.prefix("thermal_flux")).space();
  const auto& thermal_state_space = physics->state(system.prefix("temperature")).space();

  EXPECT_EQ(physics->dualNames().size(), 2);
  EXPECT_EQ(physics->dualNames()[0], system.prefix("solid_force"));
  EXPECT_EQ(physics->dualNames()[1], system.prefix("thermal_flux"));
  EXPECT_EQ(solid_dual_space.GetMesh(), solid_state_space.GetMesh());
  EXPECT_STREQ(solid_dual_space.FEColl()->Name(), solid_state_space.FEColl()->Name());
  EXPECT_EQ(solid_dual_space.GetVDim(), solid_state_space.GetVDim());
  EXPECT_EQ(solid_dual_space.TrueVSize(), solid_state_space.TrueVSize());
  EXPECT_EQ(thermal_dual_space.GetMesh(), thermal_state_space.GetMesh());
  EXPECT_STREQ(thermal_dual_space.FEColl()->Name(), thermal_state_space.FEColl()->Name());
  EXPECT_EQ(thermal_dual_space.GetVDim(), thermal_state_space.GetVDim());
  EXPECT_EQ(thermal_dual_space.TrueVSize(), thermal_state_space.TrueVSize());
}

TEST_F(ThermoMechanicsMeshFixture, BackpropagateThroughPhysics)
{
  smith::LinearSolverOptions lin_opts{.linear_solver = smith::LinearSolver::SuperLU};
  smith::NonlinearSolverOptions nonlin_opts{.nonlin_solver = smith::NonlinearSolver::Newton,
                                            .relative_tol = 1e-10,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 4};

  auto block_solver = buildNonlinearBlockSolver(nonlin_opts, lin_opts, *mesh_);
  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(
      mesh_, std::make_shared<CoupledSystemSolver>(block_solver), QuasiStaticSecondOrderTimeIntegrationRule{},
      BackwardEulerFirstOrderTimeIntegrationRule{}, "thermo", youngs_modulus);

  GreenSaintVenantThermoelasticMaterial material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  system.setMaterial(material, mesh_->entireBodyName());
  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return 100.0; });
  system.disp_bc->setFixedVectorBCs<dim>(mesh_->domain("left"));
  system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));

  system.addSolidTraction("right", [=](double, auto X, auto, auto, auto, auto, auto, auto, auto) {
    auto traction = 0.0 * X;
    traction[0] = -0.015;
    return traction;
  });

  auto physics = system.createDifferentiablePhysics("thermo_physics");

  // Run forward
  double dt = 1.0;
  for (int step = 0; step < 2; ++step) {
    physics->advanceTimestep(dt);
  }

  auto reactions = physics->getReactionStates();
  auto obj = 0.5 * (innerProduct(reactions[0], reactions[0]) + innerProduct(reactions[1], reactions[1]));

  gretl::set_as_objective(obj);
  obj.data_store().back_prop();

  auto param_sens = system.getParameterFields()[0].get_dual();
  EXPECT_TRUE(param_sens->Norml2() > 0.0);
}

TEST_F(ThermoMechanicsMeshFixture, MonolithicBucklingChallenge)
{
  constexpr double compressive_traction = 0.015;
  constexpr double lateral_body_force = 2.5e-5;
  constexpr double thermal_source = 1.0;

  auto run_problem = [&](const std::string& label, std::shared_ptr<CoupledSystemSolver> coupled_solver) {
    GreenSaintVenantThermoelasticMaterial material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
    FieldType<L2<0>> youngs_modulus("youngs_modulus");
    auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(
        mesh_, coupled_solver, QuasiStaticSecondOrderTimeIntegrationRule{},
        BackwardEulerFirstOrderTimeIntegrationRule{}, youngs_modulus);
    system.setMaterial(material, mesh_->entireBodyName());
    system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return 100.0; });
    system.disp_bc->setFixedVectorBCs<dim>(mesh_->domain("left"));
    system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));
    system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("right"));
    system.addSolidTraction("right", [=](auto, auto X, auto... /*args*/) {
      auto traction = 0.0 * X;
      traction[0] = -compressive_traction;
      return traction;
    });
    system.addSolidBodyForce(mesh_->entireBodyName(), [=](auto, auto X, auto... /*args*/) {
      auto force = 0.0 * X;
      force[1] = lateral_body_force;
      return force;
    });
    system.addHeatSource(mesh_->entireBodyName(),
                         [=](auto, auto, auto, auto, auto, auto, auto, auto) { return thermal_source; });

    SLIC_INFO_ROOT("Starting " << label << " thermo-mechanics solve");

    double dt = 1.0;
    double time = 0.0;
    auto shape_disp = system.field_store->getShapeDisp();
    auto states = system.getStateFields();
    auto params = system.getParameterFields();
    std::vector<ReactionState> reactions;
    for (size_t step = 0; step < 1; ++step) {
      std::tie(states, reactions) =
          system.advancer->advanceState(smith::TimeInfo(time, dt, step), shape_disp, states, params);
      time += dt;
    }

    return std::make_pair(mfem::Vector(*states[system.field_store->getFieldIndex("displacement_solve_state")].get()),
                          mfem::Vector(*states[system.field_store->getFieldIndex("temperature_solve_state")].get()));
  };

  smith::LinearSolverOptions monolithic_lin_opts{.linear_solver = smith::LinearSolver::GMRES,
                                                 .preconditioner = smith::Preconditioner::BlockDiagonal,
                                                 .relative_tol = 1e-10,
                                                 .absolute_tol = 1e-10,
                                                 .max_iterations = 100,
                                                 .print_level = 0};
  smith::LinearSolverOptions block_opt{.linear_solver = smith::LinearSolver::SuperLU};
  monolithic_lin_opts.sub_block_linear_solver_options.push_back(block_opt);
  monolithic_lin_opts.sub_block_linear_solver_options.push_back(block_opt);

  smith::BlockConvergenceTolerances monolithic_block_tolerances{.relative_tols = {1e-10, 1e-7},
                                                                .absolute_tols = {1e-10, 1e-8}};
  smith::NonlinearSolverOptions monolithic_nonlin_opts{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                                       .relative_tol = 1e-10,
                                                       .absolute_tol = 1e-10,
                                                       .max_iterations = 5,
                                                       .max_line_search_iterations = 6,
                                                       .print_level = 2,
                                                       .block_tolerances = monolithic_block_tolerances};

  auto monolithic_block_solver = buildNonlinearBlockSolver(monolithic_nonlin_opts, monolithic_lin_opts, *mesh_);
  auto monolithic_result = run_problem("monolithic", std::make_shared<CoupledSystemSolver>(monolithic_block_solver));
  bool monolithic_converged = monolithic_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged();
  int monolithic_iterations = monolithic_block_solver->nonlinear_solver_->nonlinearSolver().GetNumIterations();

  this->mesh_.reset();
  smith::StateManager::reset();
  this->SetUp();

  auto staggered_coupled_solver = std::make_shared<CoupledSystemSolver>(10);

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
                                                 .print_level = 1};

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
                                                  .print_level = 1};

  auto solid_block_solver = buildNonlinearBlockSolver(mech_nonlin_opts, mech_lin_opts, *mesh_);
  auto thermal_block_solver = buildNonlinearBlockSolver(therm_nonlin_opts, therm_lin_opts, *mesh_);
  staggered_coupled_solver->addSubsystemSolver({0}, solid_block_solver,
                                               {.relative_tols = {5e-6}, .absolute_tols = {1e-6}});
  staggered_coupled_solver->addSubsystemSolver({0, 1}, thermal_block_solver,
                                               {.relative_tols = {5e-6, 1e-6}, .absolute_tols = {1e-7, 1e-7}});

  auto staggered_result = run_problem("staggered", staggered_coupled_solver);
  bool staggered_solid_converged = solid_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged();
  int staggered_solid_iterations = solid_block_solver->nonlinear_solver_->nonlinearSolver().GetNumIterations();
  bool staggered_thermal_converged = thermal_block_solver->nonlinear_solver_->nonlinearSolver().GetConverged();
  int staggered_thermal_iterations = thermal_block_solver->nonlinear_solver_->nonlinearSolver().GetNumIterations();

  double disp_diff = mfem::Vector(monolithic_result.first).Add(-1.0, staggered_result.first).Normlinf();
  double temp_diff = mfem::Vector(monolithic_result.second).Add(-1.0, staggered_result.second).Normlinf();
  double monolithic_lateral_deflection = 0.0;
  for (int i = 1; i < monolithic_result.first.Size(); i += dim) {
    monolithic_lateral_deflection = std::max(monolithic_lateral_deflection, std::abs(monolithic_result.first(i)));
  }
  double staggered_lateral_deflection = 0.0;
  for (int i = 1; i < staggered_result.first.Size(); i += dim) {
    staggered_lateral_deflection = std::max(staggered_lateral_deflection, std::abs(staggered_result.first(i)));
  }

  SLIC_INFO_ROOT("Monolithic converged: " << monolithic_converged << ", iterations: " << monolithic_iterations);
  SLIC_INFO_ROOT("Monolithic per-block tolerances (solid, thermal): rel = [1e-10, 1e-7], abs = [1e-10, 1e-8]");
  SLIC_INFO_ROOT("Monolithic max lateral deflection: " << monolithic_lateral_deflection);
  SLIC_INFO_ROOT("Staggered solid converged: " << staggered_solid_converged
                                               << ", iterations: " << staggered_solid_iterations);
  SLIC_INFO_ROOT("Staggered thermal converged: " << staggered_thermal_converged
                                                 << ", iterations: " << staggered_thermal_iterations);
  SLIC_INFO_ROOT(
      "Staggered outer per-weak-form tolerances: solid rel/abs = 5e-6/1e-6, combined thermo-mechanics rel = [5e-6, "
      "1e-6], abs = [1e-7, 1e-7]");
  SLIC_INFO_ROOT("Buckling displacement discrepancy: " << disp_diff);
  SLIC_INFO_ROOT("Buckling temperature discrepancy: " << temp_diff);
  SLIC_INFO_ROOT("Staggered max lateral deflection: " << staggered_lateral_deflection);

  EXPECT_TRUE(monolithic_converged);
  EXPECT_TRUE(staggered_solid_converged);
  EXPECT_TRUE(staggered_thermal_converged);
  EXPECT_GE(monolithic_iterations, staggered_thermal_iterations);
  EXPECT_GT(staggered_lateral_deflection, 1e-5);
  EXPECT_GT(disp_diff, 1e-8);
  EXPECT_GT(temp_diff, 1e-8);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
