// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file composable_thermo_mechanics.cpp
 * @brief Minimal composable thermo-mechanics example using differentiable numerics.
 */

#include <memory>

// _includes_start
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"
#include "smith/differentiable_numerics/thermo_mechanical_system.hpp"
#include "smith/differentiable_numerics/combined_system.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/materials/green_saint_venant_thermoelastic.hpp"
// _includes_end

int main(int argc, char* argv[])
{
  // _init_start
  smith::ApplicationManager application_manager(argc, argv);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "composable_thermo_mechanics");
  // _init_end

  // _mesh_start
  constexpr int dim = 3;
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(8, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 0.1, 0.1), "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  // _mesh_end

  // _solver_start
  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::SuperLU,
                                            .relative_tol = 1e-8,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 200,
                                            .print_level = 0};
  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                                  .relative_tol = 1e-7,
                                                  .absolute_tol = 1e-8,
                                                  .max_iterations = 20,
                                                  .max_line_search_iterations = 6,
                                                  .print_level = 0};

  auto field_store = std::make_shared<smith::FieldStore>(mesh, 100, "tutorial_");

  using DispRule = smith::QuasiStaticSecondOrderTimeIntegrationRule;
  using TempRule = smith::BackwardEulerFirstOrderTimeIntegrationRule;

  auto solid_fields = smith::registerSolidMechanicsFields<dim, 1, DispRule>(field_store);
  auto thermal_fields = smith::registerThermalFields<dim, 1, TempRule>(field_store);
  // _solver_end

  // _build_start
  auto solid_solver =
      std::make_shared<smith::SystemSolver>(smith::buildNonlinearBlockSolver(nonlinear_options, linear_options, *mesh));
  auto thermal_solver =
      std::make_shared<smith::SystemSolver>(smith::buildNonlinearBlockSolver(nonlinear_options, linear_options, *mesh));

  auto solid = smith::buildSolidMechanicsSystem<dim, 1, DispRule>(solid_solver, smith::SolidMechanicsOptions{},
                                                                  solid_fields, thermal_fields);
  auto thermal = smith::buildThermalSystem<dim, 1, TempRule>(thermal_solver, smith::ThermalOptions{}, thermal_fields,
                                                             solid_fields);

  smith::thermomechanics::GreenSaintVenantThermoelasticMaterial material{1.0, 100.0, 0.25, 1.0, 0.0025, 0.0, 0.05};
  smith::setCoupledThermoMechanicsMaterial(solid, thermal, material, mesh->entireBodyName());

  auto coupled = smith::combineSystems(solid, thermal);
  // _build_end

  // _bc_start
  solid->setDisplacementBC(mesh->domain("left"));
  thermal->setTemperatureBC(mesh->domain("left"), [](auto, auto) { return 1.0; });
  thermal->setTemperatureBC(mesh->domain("right"), [](auto, auto) { return 0.0; });

  solid->addTraction("right", [](double, auto X, auto, auto, auto, auto, auto... /*unused*/) {
    auto traction = 0.0 * X;
    traction[0] = -0.01;
    return traction;
  });

  thermal->addHeatSource(mesh->entireBodyName(), [](auto, auto, auto, auto... /*unused*/) { return 0.5; });
  // _bc_end

  // _run_start
  auto physics = smith::makeDifferentiablePhysics(coupled, "composable_thermo_mechanics");
  for (int step = 0; step < 2; ++step) {
    physics->advanceTimestep(1.0);
  }
  // _run_end

  return 0;
}
