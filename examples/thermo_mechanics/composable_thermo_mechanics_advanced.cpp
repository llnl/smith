// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file composable_thermo_mechanics_advanced.cpp
 * @brief Advanced composable thermo-mechanics example with staged solves, a differentiable QoI,
 *        finite-difference verification, and ParaView output including Cauchy stress.
 */

#include <iostream>
#include <memory>
#include <vector>

// _includes_start
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/functional_objective.hpp"

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"
#include "smith/differentiable_numerics/thermo_mechanical_system.hpp"
#include "smith/differentiable_numerics/combined_system.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/evaluate_objective.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/time_info_thermo_mechanical_materials.hpp"
#include "smith/physics/materials/green_saint_venant_thermoelastic.hpp"
// _includes_end

namespace {

std::vector<smith::FieldState> outputFields(const smith::FieldStore& field_store)
{
  return field_store.getOutputFieldStates();
}

std::vector<smith::FieldState> qoiFields(const smith::FieldStore& field_store)
{
  return {field_store.getField(field_store.prefix("displacement")),
          field_store.getField(field_store.prefix("temperature"))};
}

}  // namespace

int main(int argc, char* argv[])
{
  // _init_start
  smith::ApplicationManager application_manager(argc, argv);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "composable_thermo_mechanics_advanced");
  // _init_end

  // _mesh_start
  constexpr int dim = 3;
  constexpr int order = 1;
  using DispSpace = smith::H1<order, dim>;
  using TempSpace = smith::H1<order>;

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(8, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 0.1, 0.1), "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  // _mesh_end

  // _solver_start
  auto field_store = std::make_shared<smith::FieldStore>(mesh, 200);

  smith::SolidMechanicsOptions solid_options{.enable_stress_output = true, .output_cauchy_stress = true};
  auto solid_fields = smith::registerSolidMechanicsFields<dim, order, smith::QuasiStaticSecondOrderTimeIntegrationRule>(
      field_store, solid_options);
  auto thermal_fields =
      smith::registerThermalFields<dim, order, smith::BackwardEulerFirstOrderTimeIntegrationRule>(field_store);
  auto param_fields = smith::registerParameterFields(smith::FieldType<smith::L2<0>>("thermal_expansion_scaling"));

  auto solid_system =
      smith::buildSolidMechanicsSystem<dim, order>(nullptr, solid_options, solid_fields, param_fields, thermal_fields);
  auto thermal_system = smith::buildThermalSystem<dim, order>(nullptr, smith::ThermalOptions{}, thermal_fields,
                                                              param_fields, solid_fields);

  smith::thermomechanics::TimeInfoParameterizedGreenSaintVenantThermoelasticMaterial material{1.0,    100.0, 0.25, 1.0,
                                                                                              0.0025, 0.0,   0.05};
  smith::setCoupledThermoMechanicsMaterial(solid_system, thermal_system, material, mesh->entireBodyName());

  field_store->getParameterFields()[0].get()->setFromFieldFunction([](smith::tensor<double, dim>) { return 1.0; });

  // _bc_start
  solid_system->setDisplacementBC(mesh->domain("left"));
  thermal_system->setTemperatureBC(mesh->domain("left"), [](auto, auto) { return 1.0; });
  thermal_system->setTemperatureBC(mesh->domain("right"), [](auto, auto) { return 0.0; });

  solid_system->addTraction(smith::DependsOn<>{}, "right",
                            [](double, auto X, auto /*n*/, auto /*u*/, auto /*v*/, auto /*a*/) {
                              auto traction = 0.0 * X;
                              traction[0] = -0.005;
                              return traction;
                            });

  thermal_system->addHeatSource(mesh->entireBodyName(), [](auto, auto... /*unused*/) { return 0.1; });
  // _bc_end

  smith::LinearSolverOptions trust_region_linear{.linear_solver = smith::LinearSolver::CG,
                                                 .preconditioner = smith::Preconditioner::HypreAMG,
                                                 .relative_tol = 1e-6,
                                                 .absolute_tol = 1e-10,
                                                 .max_iterations = 80,
                                                 .print_level = 0};
  smith::NonlinearSolverOptions trust_region_nonlin{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                    .relative_tol = 1e-7,
                                                    .absolute_tol = 1e-8,
                                                    .max_iterations = 15,
                                                    .print_level = 0};

  smith::LinearSolverOptions coupled_linear{.linear_solver = smith::LinearSolver::SuperLU,
                                            .relative_tol = 1e-8,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 200,
                                            .print_level = 0};
  smith::NonlinearSolverOptions coupled_nonlin{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                               .relative_tol = 1e-8,
                                               .absolute_tol = 1e-8,
                                               .max_iterations = 12,
                                               .max_line_search_iterations = 6,
                                               .print_level = 0};
  // _solver_end

  // _build_start
  size_t max_staggered_iterations = 10;
  auto custom_solver = std::make_shared<smith::SystemSolver>(max_staggered_iterations);
  custom_solver->addSubsystemSolver(
      {0}, smith::buildNonlinearBlockSolver(trust_region_nonlin, trust_region_linear, *mesh), 1.0);
  custom_solver->addSubsystemSolver({1}, smith::buildNonlinearBlockSolver(coupled_nonlin, coupled_linear, *mesh), 1.0);

  auto coupled_system = smith::combineSystems(custom_solver, solid_system, thermal_system);
  std::string physics_name = "composable_thermo_mechanics_advanced";
  auto physics = smith::makeDifferentiablePhysics(coupled_system, physics_name);
  auto output_states = outputFields(*field_store);
  auto output_writer =
      smith::createParaviewWriter(*mesh, output_states, "paraview_composable_thermo_mechanics_advanced",
                                  smith::ParaviewWriter::Options{.write_duals = false});
  // _build_end

  // _qoi_start
  smith::FunctionalObjective<dim, smith::Parameters<DispSpace, TempSpace>> qoi("thermo_mechanical_energy_proxy", mesh,
                                                                               smith::spaces(qoiFields(*field_store)));
  qoi.addBodyIntegral(smith::DependsOn<0, 1>(), mesh->entireBodyName(), [](double, auto /*X*/, auto U, auto Theta) {
    auto u = smith::get<smith::VALUE>(U);
    auto theta = smith::get<smith::VALUE>(Theta);
    return 0.5 * u[0] * u[0] + 0.05 * theta * theta;
  });

  auto qoi_state =
      0.0 * smith::evaluateObjective(qoi, physics->getShapeDispFieldState(), qoiFields(*field_store),
                                     smith::TimeInfo(physics->time(), 1.0, static_cast<size_t>(physics->cycle())));
  // _qoi_end

  // _run_start
  constexpr double dt = 0.5;
  constexpr int qoi_steps = 1;
  for (int step = 0; step < qoi_steps; ++step) {
    physics->advanceTimestep(dt);
    qoi_state = qoi_state +
                smith::evaluateObjective(qoi, physics->getShapeDispFieldState(), qoiFields(*field_store),
                                         smith::TimeInfo(physics->time(), dt, static_cast<size_t>(physics->cycle())));
  }
  // _run_end

  // _output_start
  output_writer.write(physics->cycle(), physics->time(), outputFields(*field_store));

  std::cout << "ParaView output: paraview_composable_thermo_mechanics_advanced\n";
  // _output_end

  // _sensitivity_start
  gretl::set_as_objective(qoi_state);
  auto qoi_value = qoi_state.get();
  std::cout << "QoI value: " << qoi_value << '\n';
  qoi_state.data_store().back_prop();

  auto parameter_state = field_store->getParameterFields()[0];
  auto parameter_sensitivity = parameter_state.get_dual()->Norml2();

  std::cout << "dQoI/d(thermal_expansion_scaling) norm: " << parameter_sensitivity << '\n';
  SLIC_ERROR_ROOT_IF(parameter_sensitivity <= 0.0, "Expected non-zero QoI sensitivity.");

  auto fd_order = smith::checkGradWrt(qoi_state, parameter_state, 5.0e-2, 4, true);
  std::cout << "finite-difference convergence rate: " << fd_order << '\n';
  SLIC_ERROR_ROOT_IF(fd_order < 0.7, "Finite-difference check did not converge.");
  // _sensitivity_end

  return 0;
}
