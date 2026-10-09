// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file prescribed_temperature_mechanics.cpp
 * @brief Quasistatic mechanics with a temperature field updated manually from a lambda.
 */

#include <format>
#include <memory>

#include "axom/slic.hpp"
#include "mfem.hpp"

#include "smith/infrastructure/logger.hpp"
#include "smith/smith_config.hpp"

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "thermoelastic_material.hpp"

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "prescribed_temperature_mechanics");

  constexpr int dim = 3;
  constexpr int order = 2;
  constexpr double beam_length = 1.0;

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(80, 20, 20, mfem::Element::HEXAHEDRON, beam_length, 0.1 * beam_length, 0.1 * beam_length), "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::CG,
                                            .relative_tol = 1.0e-10,
                                            .absolute_tol = 1.0e-12,
                                            .max_iterations = 200,
                                            .print_level = 1};
  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = 200,
                                                  .max_line_search_iterations = 20,
                                                  .print_level = 2};

  // The generic Cauchy-stress projector only sees the incremental displacement, so retain PK1 output when using an
  // initial-displacement offset.
  smith::SolidMechanicsOptions solid_options{.enable_stress_output = true, .output_cauchy_stress = false};
  auto solid_system = smith::buildSolidMechanicsSystem<dim, order, smith::QuasiStaticSecondOrderTimeIntegrationRule>(
      nonlinear_options, linear_options, solid_options, mesh, smith::FieldType<smith::H1<order>>("temperature"),
      smith::FieldType<smith::H1<order, dim>>("initial_displacement"));

  using material = smith::examples::kevin::ThermomechanicalLCE<dim>;

  solid_system->setMaterial(material{}, mesh->entireBodyName());
  solid_system->setDisplacementBC(mesh->domain("right"));
  // solid_system->addTraction("left", [](double, auto X, auto, auto, auto, auto, auto... /*unused*/) {
  //   auto traction = 0.0 * X;
  //   traction[0] = 0.05;
  //   return traction;
  // });

  constexpr double T_stop = 1.0;
  constexpr double dt = 0.01;
  auto field_store = solid_system->field_store;
  auto temperature = field_store->getParameter("param_temperature");
  auto initial_displacement = field_store->getParameter("param_initial_displacement");

  // This lambda is entirely user controlled. Re-project it before each mechanics solve to prescribe any desired
  // spatial and temporal temperature history.
  auto update_temperature = [&](double time) {
    temperature.get()->setFromFieldFunction([=](smith::tensor<double, dim> X) {
      constexpr double heating_rate = 100.0;
      return heating_rate * time * (1.0 + 0.0 * X[0] / beam_length) + 300.0;
    });
  };

  // TimeInfo evaluates loads at time + dt. Start at -dt to solve the t = 0 equilibrium without advancing the clock,
  // then store that total displacement as a fixed offset and reset the evolving incremental fields to zero.
  update_temperature(0.0);
  {
    const smith::TimeInfo initial_time_info(-dt, dt);
    SLIC_INFO_ROOT("Starting initial solve");
    const auto initial_solution = solid_system->solve(initial_time_info).front();
    SLIC_INFO_ROOT("Finished initial solve");
    *initial_displacement.get() = *initial_solution.get();

    *field_store->getField("displacement_solve_state").get() = 0.0;
    *field_store->getField("displacement").get() = 0.0;
    *field_store->getField("velocity").get() = 0.0;
    *field_store->getField("acceleration").get() = 0.0;

    const auto initial_stress = solid_system->stress_output_system->solve(initial_time_info).front();
    *field_store->getField("stress").get() = *initial_stress.get();
  }
  field_store->graph()->reset_graph();

  auto physics = smith::makeDifferentiablePhysics(solid_system, "prescribed_temperature_mechanics");
  auto writer = smith::createParaviewWriter(*mesh, physics->getFieldStatesAndParamStates(),
                                            "paraview_prescribed_temperature_mechanics",
                                            smith::ParaviewWriter::Options{.write_duals = false});

  writer.write(physics->cycle(), physics->time(), physics->getFieldStatesAndParamStates());

  double T = 0.0;
  int step = 0;
  while (T <= T_stop) {
    update_temperature(physics->time() + dt);
    physics->advanceTimestep(dt);
    writer.write(physics->cycle(), physics->time(), physics->getFieldStatesAndParamStates());
    const double max_temperature = smith::max(physics->parameter("param_temperature"));
    SLIC_INFO_ROOT(std::format("step {}: time = {}, max temperature = {}", step + 1, physics->time(), max_temperature));
    smith::logger::flush();
    T += dt;
    step++;
  }

  return 0;
}
