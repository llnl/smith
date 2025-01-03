// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>

#include "axom/inlet.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

int main(int argc, char* argv[])
{
  serac::initialize(argc, argv);

  constexpr int p = 2;
  constexpr int dim = 3;
  constexpr double x_length = 1.0;
  constexpr double y_length = 0.1;
  constexpr double z_length = 0.1;
  constexpr int elements_in_x = 10;
  constexpr int elements_in_y = 1;
  constexpr int elements_in_z = 1;
  
  int serial_refinements = 0;
  int parallel_refinements = 0;
  int time_steps = 20;
  double strain_rate = 1e-3;

  constexpr double E = 1.0;
  constexpr double nu = 0.25;

  constexpr double sigma_y = 0.001;
  constexpr double sigma_sat = 3.0*sigma_y;
  constexpr double strain_constant = 10*sigma_y/E;
  constexpr double eta = 1e-2;

  constexpr double max_strain = 3*strain_constant;

  // Handle command line arguments
  axom::CLI::App app{"Plane strain uniaxial extension of a bar."};
  // Mesh options
  app.add_option("--serial-refinements", serial_refinements, "Serial refinement steps", true);
  app.add_option("--parallel-refinements", parallel_refinements, "Parallel refinement steps", true);
  app.add_option("--time-steps", time_steps, 
    "Number of time steps to divide simulation", true);
  app.add_option("--strain_rate", strain_rate, "Nominal strain rate", true);

  double max_time = max_strain/strain_rate;
  
  // Create DataStore
  const std::string simulation_tag = "uniaxial";
  const std::string mesh_tag = simulation_tag + "mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, simulation_tag + "_data");

  auto serial_mesh = std::make_unique<mfem::Mesh>(
    mfem::Mesh::MakeCartesian3D(elements_in_x, elements_in_y, elements_in_z,
                                mfem::Element::QUADRILATERAL, true, x_length,
                                y_length, z_length));
  serial_mesh->Print();
  auto  mesh  = serac::mesh::refineAndDistribute(std::move(*serial_mesh), serial_refinements, parallel_refinements);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // create boundary domains for boundary conditions
  auto fix_x = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(1));
  auto fix_y = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(2));
  auto fix_z = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(4));
  auto apply_displacement = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(3));
  serac::Domain whole_mesh = serac::EntireDomain(pmesh);

  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 1};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Uniaxial app requires MFEM built with strumpack.");
  return 1;
#endif

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = serac::NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-10,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 200,
                                                  .print_level    = 1};

  serac::SolidMechanics<p, dim> solid_solver(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, simulation_tag,
      mesh_tag);

  using Hardening = serac::solid_mechanics::VoceHardening;
  using Material = serac::solid_mechanics::J2<Hardening>;

  Hardening hardening{sigma_y, sigma_sat, strain_constant, eta};
  Material material{E, nu, hardening, .density = 1.0};

  auto internal_states = solid_solver.createQuadratureDataBuffer(Material::State{}, whole_mesh);

  solid_solver.setMaterial(material, whole_mesh, internal_states);

  solid_solver.setFixedBCs(fix_x, serac::Component::X);
  solid_solver.setFixedBCs(fix_y, serac::Component::Y);
  solid_solver.setFixedBCs(fix_z, serac::Component::Z);
  auto applied_displacement = [strain_rate](serac::vec3, double t){ return serac::vec3{strain_rate*x_length*t, 0., 0.}; };
  solid_solver.setDisplacementBCs(applied_displacement, apply_displacement, serac::Component::X);

  solid_solver.completeSetup();

  std::string paraview_tag = simulation_tag + "_paraview";
  solid_solver.outputStateToDisk(paraview_tag);

  double dt = max_time/(time_steps - 1);

  for (int i = 0; i < time_steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", solid_solver.time(), max_time));
    serac::logger::flush();

    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_tag);
  }

  serac::exitGracefully();

  return 0;
}
