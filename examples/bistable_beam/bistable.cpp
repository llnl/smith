// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file bistable.cpp
 *
 * @brief A curved, fixed-fixed beam loaded at center span through a cycle of snap-through and back to unloaded
 *
 */

#include <cmath>
#include <fstream>
#include <memory>
#include <string>

#include "axom/slic.hpp"
#include "axom/inlet.hpp"
#include "axom/CLI11.hpp"
#include "serac/serac.hpp"

constexpr int dim = 2;


int main(int argc, char* argv[])
{
  // polynomial degree of finite element to use
  constexpr int p = 2;

  // Command line arguments
  // Mesh options
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // number of time steps (max load is fixed)
  int steps = 40;

  // Solver options
  serac::NonlinearSolverOptions nonlinear_options = serac::solid_mechanics::default_nonlinear_options;
  nonlinear_options.nonlin_solver = serac::NonlinearSolver::TrustRegion;
  nonlinear_options.relative_tol = 1e-6;
  nonlinear_options.absolute_tol = 1e-10;
  nonlinear_options.min_iterations = 1;
  nonlinear_options.max_iterations = 500;
  nonlinear_options.print_level = 1;

  serac::LinearSolverOptions linear_options = serac::solid_mechanics::default_linear_options;
  linear_options.linear_solver = serac::LinearSolver::CG;
  linear_options.preconditioner = serac::Preconditioner::HypreAMG;
  linear_options.relative_tol = 1e-8;
  linear_options.absolute_tol = 1e-16;
  linear_options.max_iterations = 2000;

  // Initialize and automatically finalize MPI and other libraries
  serac::ApplicationManager applicationManager(argc, argv);
  auto [num_ranks, rank] = serac::getMPIInfo(MPI_COMM_WORLD);

  // Handle command line arguments
  axom::CLI::App app{"Hollow cylinder buckling example"};
  // Mesh options
  app.add_option("--serial-refinement", serial_refinement, "Serial mesh refinements")->check(axom::CLI::PositiveNumber);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel mesh refinements")
      ->check(axom::CLI::PositiveNumber);
  // Solver options
  app.add_option("--nonlinear-solver", nonlinear_options.nonlin_solver,
                 "Nonlinear solver (Index of enum serac::NonlinearSolver)")
      ->expected(0, 10);
  app.add_option("--linear-solver", linear_options.linear_solver, "Linear solver (Index of enum serac::LinearSolver)")
      ->expected(0, 5);
  app.add_option("--preconditioner", linear_options.preconditioner,
                 "Preconditioner (Index of enum serac::NonlinearSolver)")
      ->expected(0, 7);
  app.add_option("--petsc-pc-type", linear_options.petsc_preconditioner,
                 "Petsc preconditioner (Index of enum serac::PetscPCType)")
      ->expected(0, 14);
  // Time stepping options
  app.add_option("--time-steps", steps, "Number of time steps to take")->check(axom::CLI::PositiveNumber);

  // Need to allow extra arguments for PETSc support
  app.set_help_flag("--help");
  app.allow_extras()->parse(argc, argv);

  double dt = 1.0/steps;

  // Create DataStore
  std::string name = "bistable_beam";
  std::string mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Create and refine mesh
  std::string filename = "./bistable_beam/sinusoidal_beam.g";
  auto mesh = std::make_shared<serac::Mesh>(filename, mesh_tag, serial_refinement, parallel_refinement);

  // Surfaces for boundary conditions
  mesh->addDomainOfBoundaryElements("left", serac::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("right", serac::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("load_surface", serac::by_attr<dim>(3));

  SLIC_INFO_ROOT(axom::fmt::format("edges in domain left: {}", mesh->domain("left").total_elements()));
  SLIC_INFO_ROOT(axom::fmt::format("edges in domain right: {}", mesh->domain("right").total_elements()));
  SLIC_INFO_ROOT(axom::fmt::format("edges in domain load_surface: {}", mesh->domain("load_surface").total_elements()));

  // Create solver
  auto solid_solver = std::make_unique<serac::SolidMechanics<p, dim>>(
        nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, mesh);
  
  // Define the time-dependent load
  constexpr double peak_traction = 0.03;
  auto traction = [peak_traction](auto, auto, double t) { 
    return serac::tensor<double, dim> {0.0, -peak_traction*std::sin(M_PI*t)};
  };

  solid_solver->setTraction(traction,
                            mesh->domain("load_surface"));

  // Define the material
  double E = 10.0;
  double nu = 0.25;
  serac::solid_mechanics::NeoHookean mat{.density = 1.0, .K = E / 3 / (1 - 2*nu), .G = 0.5*E/(1 + nu)};
  solid_solver->setMaterial(mat, mesh->entireBody());

  // Set up essential boundary conditions
  solid_solver->setFixedBCs(mesh->domain("left"));
  solid_solver->setFixedBCs(mesh->domain("right"));

  // Finalize the data structures
  solid_solver->completeSetup();

  // Output functionals
  using DisplacementSpace = serac::H1<p, dim>;
  // Compute the area of the surface over which the traction is applied
  auto compute_load_surface_area = serac::Functional<double(DisplacementSpace)>({&solid_solver->displacement().space()});
  compute_load_surface_area.AddBoundaryIntegral(
    serac::Dimension<dim - 1>{}, serac::DependsOn<>{},
    [](auto, auto) {
      return 1.0;
    },
    mesh->domain("load_surface"));
  double area = compute_load_surface_area(solid_solver->time(), solid_solver->displacement());

  // Define function to compute the average downward displacement of the loading surface
  auto u_integral = serac::Functional<double(DisplacementSpace)>({&solid_solver->displacement().space()});
  u_integral.AddBoundaryIntegral(
      serac::Dimension<dim - 1>{}, serac::DependsOn<0>{},
      [](auto, auto, auto displacement) {
        auto u = serac::get<0>(displacement);
        return u[1];
      },
      mesh->domain("load_surface"));
  
  auto compute_average_displacement = [&u_integral, area](serac::FiniteElementState u) {
    double t = 0;
    return -u_integral(t, u) / area;
  };

  // Define function to compute the total downward force applied to the beam
  auto compute_force = [&traction, area](double t) {
    return -traction(serac::tensor<double, dim>{}, serac::tensor<double, dim>{}, t)[1] * area;
  };

  // Save initial state
  std::string paraview_name = name + "_paraview";
  solid_solver->outputStateToDisk(paraview_name);

  std::ofstream force_displacement_history("force_displacement_history.txt");
  force_displacement_history << compute_average_displacement(solid_solver->displacement());
  force_displacement_history << " " << compute_force(solid_solver->time());
  force_displacement_history << std::endl;
 
  // Perform the quasi-static solve
  SLIC_INFO_ROOT(axom::fmt::format("Running curved bistable beam example with {} displacement dofs",
                                   solid_solver->displacement().GlobalSize()));
  SLIC_INFO_ROOT("Starting pseudo-timestepping.");
  serac::logger::flush();
  for (int time_step = 0; time_step < steps; ++time_step) {
    SLIC_INFO_ROOT("----------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", time_step));
    SLIC_INFO_ROOT(axom::fmt::format("start time = {}, end time = {}\n", solid_solver->time(), solid_solver->time() + dt));
    serac::logger::flush();

    solid_solver->advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver->outputStateToDisk(paraview_name);

    // Output the displacement and force to the text file
    if (rank == 0) {
      force_displacement_history << compute_average_displacement(solid_solver->displacement());
      force_displacement_history << " " << compute_force(solid_solver->time());
      force_displacement_history << std::endl;
    }
  }
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver->time()));

  force_displacement_history.close();

  return 0;
}
