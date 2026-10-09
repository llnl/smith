// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dynamic_temperature_quasistatic_mechanics.cpp
 * @brief Transient heat conduction coupled to quasistatic mechanics.
 */

#include <format>
#include <memory>

#include "axom/slic.hpp"
#include "mfem.hpp"

#include "smith/smith_config.hpp"

#include "smith/differentiable_numerics/combined_system.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/thermal_system.hpp"
#include "smith/differentiable_numerics/thermo_mechanics_system.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "thermoelastic_material.hpp"

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "dynamic_temperature_quasistatic_mechanics");

  constexpr int dim = 3;
  constexpr int order = 1;

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(8, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 0.1, 0.1), "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::SuperLU,
                                            .relative_tol = 1.0e-10,
                                            .absolute_tol = 1.0e-12,
                                            .max_iterations = 200,
                                            .print_level = 0};
  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = 20,
                                                  .max_line_search_iterations = 6,
                                                  .print_level = 0};

  auto field_store = std::make_shared<smith::FieldStore>(mesh, 100);
  // The generic Cauchy-stress projector only sees the incremental displacement, so retain PK1 output when using an
  // initial-displacement offset.
  smith::SolidMechanicsOptions solid_options{.enable_stress_output = true, .output_cauchy_stress = false};
  auto solid_fields = smith::registerSolidMechanicsFields<dim, order, smith::QuasiStaticSecondOrderTimeIntegrationRule>(
      field_store, solid_options);
  auto thermal_fields =
      smith::registerThermalFields<dim, order, smith::BackwardEulerFirstOrderTimeIntegrationRule>(field_store);
  auto parameter_fields =
      smith::registerParameterFields(field_store, smith::FieldType<smith::H1<order, dim>>("initial_displacement"));

  auto make_solver = [&]() {
    return std::make_shared<smith::SystemSolver>(
        smith::buildNonlinearBlockSolver(nonlinear_options, linear_options, *mesh));
  };
  auto solid_system = smith::buildSolidMechanicsSystem(make_solver(), solid_options, solid_fields,
                                                       smith::couplingFields(thermal_fields), parameter_fields);
  auto thermal_system = smith::buildThermalSystem(make_solver(), smith::ThermalOptions{}, thermal_fields,
                                                  smith::couplingFields(solid_fields), parameter_fields);

  smith::examples::kevin::ThermoelasticMaterial material;
  smith::setCoupledThermoMechanicsMaterial(solid_system, thermal_system, material, mesh->entireBodyName());

  solid_system->setDisplacementBC(mesh->domain("left"));
  solid_system->addTraction("right", [](double, auto X, auto, auto, auto, auto, auto... /*unused*/) {
    auto traction = 0.0 * X;
    traction[0] = 0.05;
    return traction;
  });

  // The heat equation, rather than user projection, now determines the interior temperature field.
  thermal_system->setTemperatureBC(mesh->domain("left"), [](double time, auto) { return 20.0 * time; });
  thermal_system->setTemperatureBC(mesh->domain("right"), [](double, auto) { return 0.0; });

  // Solve heat conduction first so that the quasistatic mechanics stage responds to the new temperature immediately.
  auto coupled_system = smith::combineSystems(thermal_system, solid_system);

  constexpr double dt = 0.25;
  constexpr int num_steps = 4;
  auto initial_displacement = field_store->getParameter("param_initial_displacement");

  // TimeInfo evaluates loads at time + dt. Start at -dt to solve the t = 0 coupled equilibrium without advancing the
  // clock, then store the total displacement as a fixed offset and reset the evolving incremental fields to zero. The
  // combined system returns the thermal unknown first and the mechanical unknown second.
  {
    const smith::TimeInfo initial_time_info(-dt, dt);
    const auto initial_fields = coupled_system->solve(initial_time_info);
    *field_store->getField("temperature_solve_state").get() = *initial_fields[0].get();
    *field_store->getField("temperature").get() = *initial_fields[0].get();
    *initial_displacement.get() = *initial_fields[1].get();

    *field_store->getField("displacement_solve_state").get() = 0.0;
    *field_store->getField("displacement").get() = 0.0;
    *field_store->getField("velocity").get() = 0.0;
    *field_store->getField("acceleration").get() = 0.0;

    const auto initial_stress = solid_system->stress_output_system->solve(initial_time_info).front();
    *field_store->getField("stress").get() = *initial_stress.get();
  }
  field_store->graph()->reset_graph();

  auto physics = smith::makeDifferentiablePhysics(coupled_system, "dynamic_temperature_quasistatic_mechanics");
  auto writer = smith::createParaviewWriter(*mesh, field_store->getOutputFieldStates(),
                                            "paraview_dynamic_temperature_quasistatic_mechanics",
                                            smith::ParaviewWriter::Options{.write_duals = false});

  writer.write(physics->cycle(), physics->time(), field_store->getOutputFieldStates());

  for (int step = 0; step < num_steps; ++step) {
    physics->advanceTimestep(dt);
    writer.write(physics->cycle(), physics->time(), field_store->getOutputFieldStates());
    const double max_temperature = smith::max(physics->state("temperature"));
    SLIC_INFO_ROOT(std::format("step {}: time = {}, max temperature = {}", step + 1, physics->time(), max_temperature));
  }

  return 0;
}
