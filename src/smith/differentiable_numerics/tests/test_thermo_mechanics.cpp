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

#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
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
    auto greenStrainRate = 0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate) + 0.0 * E;
    const auto q0 = -kappa * grad_theta;
    return smith::tuple{Piola, C_v, s0, q0};
  }
  static constexpr int numParameters() { return 1; }
};

struct ThermoMechanicsMeshFixture : public testing::Test {
  void SetUp() {
    datastore_ = std::make_unique<axom::sidre::DataStore>();
    smith::StateManager::initialize(*datastore_, "solid");
    mesh_ = std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian3D(4, 1, 1, mfem::Element::QUADRILATERAL, 1.0, 0.04, 0.04), "mesh", 0, 0);
    mesh_->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
    mesh_->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  }
  std::unique_ptr<axom::sidre::DataStore> datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
};

TEST_F(ThermoMechanicsMeshFixture, MonolithicVsStaggered)
{
  auto run_problem = [&](std::shared_ptr<SystemSolver> sys_solver, bool check_grad) {
    GreenSaintVenantThermoelasticMaterial material{1.0, 100.0, 0.25, 1.0, 0.001, 0.0, 0.1};
    FieldType<L2<0>> youngs_modulus("youngs_modulus");
    auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, sys_solver, youngs_modulus);
    system.setMaterial(material, mesh_->entireBodyName());
    system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return 100.0; });
    system.disp_bc->setVectorBCs<dim>(mesh_->domain("left"), [](double t, smith::tensor<double, dim> X) { auto bc = 0.0 * X; bc[0] = 0.01 * t; return bc; });
    system.disp_bc->setFixedVectorBCs<dim>(mesh_->domain("right"));
    system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));
    system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("right"));
    system.addThermalHeatSource(mesh_->entireBodyName(), [](auto, auto, auto, auto, auto, auto, auto) { return 100.0; });

    double dt = 0.001;
    double time = 0.0;
    auto shape_disp = system.field_store->getShapeDisp();
    auto states = system.getStateFields();
    auto params = system.getParameterFields();
    std::vector<ReactionState> reactions;
    for (size_t step = 0; step < 2; ++step) {
      std::tie(states, reactions) = system.advancer->advanceState(smith::TimeInfo(time, dt, step), shape_disp, states, params);
      time += dt;
    }

    if (check_grad) {
      auto reaction_squared = innerProduct(reactions[0], reactions[0]);
      gretl::set_as_objective(reaction_squared);
      EXPECT_GT(checkGradWrt(reaction_squared, shape_disp, 1e-3, 4, true), 0.9);
    }

    return std::make_pair(mfem::Vector(*states[system.field_store->getFieldIndex("displacement_predicted")].get()),
                          mfem::Vector(*states[system.field_store->getFieldIndex("temperature_predicted")].get()));
  };

  // 1. Monolithic with BlockDiagonal Preconditioner (GMRES + AMG)
  smith::LinearSolverOptions mono_lin_opts{.linear_solver = smith::LinearSolver::GMRES,
                                           .preconditioner = smith::Preconditioner::BlockDiagonal,
                                           .relative_tol = 1e-12, .absolute_tol = 1e-12, .max_iterations = 200};
  smith::LinearSolverOptions block_opt{.linear_solver = smith::LinearSolver::GMRES,
                                       .preconditioner = smith::Preconditioner::HypreAMG,
                                       .relative_tol = 1e-6, .absolute_tol = 1e-12, .max_iterations = 50};
  mono_lin_opts.block_options.push_back(block_opt);
  mono_lin_opts.block_options.push_back(block_opt);

  smith::NonlinearSolverOptions mono_nonlin_opts{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                                 .relative_tol = 1e-12, .absolute_tol = 1e-12, .max_iterations = 50};

  auto mono_solver = buildDifferentiableSolver(mono_nonlin_opts, mono_lin_opts, *mesh_);
  auto mono_result = run_problem(std::make_shared<SystemSolver>(mono_solver), true);

  // Reset
  this->mesh_.reset(); smith::StateManager::reset(); this->SetUp();

  // 2. Staggered with specific block solvers (TrustRegion/CG for solid, NewtonLineSearch/GMRES for thermal)
  auto stag_sys_solver = std::make_shared<SystemSolver>(100);
  stag_sys_solver->setRelaxationFactor(0.5);

  smith::LinearSolverOptions mech_lin_opts{.linear_solver = smith::LinearSolver::CG,
                                           .preconditioner = smith::Preconditioner::HypreAMG,
                                           .relative_tol = 1e-12, .absolute_tol = 1e-12, .max_iterations = 200};
  smith::NonlinearSolverOptions mech_nonlin_opts{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                 .relative_tol = 1e-12, .absolute_tol = 1e-12, .max_iterations = 100};

  smith::LinearSolverOptions therm_lin_opts{.linear_solver = smith::LinearSolver::GMRES,
                                            .preconditioner = smith::Preconditioner::HypreAMG,
                                            .relative_tol = 1e-12, .absolute_tol = 1e-12, .max_iterations = 200};
  smith::NonlinearSolverOptions therm_nonlin_opts{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                                  .relative_tol = 1e-12, .absolute_tol = 1e-12, .max_iterations = 100};

  auto solver_disp = buildDifferentiableSolver(mech_nonlin_opts, mech_lin_opts, *mesh_);
  auto solver_temp = buildDifferentiableSolver(therm_nonlin_opts, therm_lin_opts, *mesh_);
  stag_sys_solver->addStage({0}, solver_disp);
  stag_sys_solver->addStage({1}, solver_temp);
  auto stag_result = run_problem(stag_sys_solver, true);

  double disp_diff = mfem::Vector(mono_result.first).Add(-1.0, stag_result.first).Normlinf();
  double temp_diff = mfem::Vector(mono_result.second).Add(-1.0, stag_result.second).Normlinf();
  SLIC_INFO_ROOT("Displacement discrepancy: " << disp_diff);
  SLIC_INFO_ROOT("Temperature discrepancy: " << temp_diff);

  EXPECT_LT(disp_diff, 1e-4);
  EXPECT_LT(temp_diff, 1e-2);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
