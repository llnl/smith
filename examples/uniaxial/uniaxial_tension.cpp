// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <fstream>

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

#include <cfenv>

template <class Physics>
void output(double u, double f, const Physics& solid, const std::string& paraview_tag, std::ofstream& file)
{
  solid.outputStateToDisk(paraview_tag);
  file << solid.time() << " " << u << " " << f << std::endl;
}

int main(int argc, char* argv[])
{
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
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
  int time_steps = 100;
  double strain_rate = 1e-3;

  constexpr double E = 1.0;
  constexpr double nu = 0.25;
  constexpr double density = 1.0;

  constexpr double sigma_y = 0.001;
  constexpr double sigma_sat = 3.0 * sigma_y;
  constexpr double strain_constant = 10 * sigma_y / E;
  constexpr double eta = 1e-2;

  constexpr double max_strain = 3 * strain_constant;
  std::string output_filename = "uniaxial_fd.txt";

  // Handle command line arguments
  axom::CLI::App app{"Plane strain uniaxial extension of a bar."};
  // Mesh options
  app.add_option("--serial-refinements", serial_refinements, "Serial refinement steps", true);
  app.add_option("--parallel-refinements", parallel_refinements, "Parallel refinement steps", true);
  app.add_option("--time-steps", time_steps, "Number of time steps to divide simulation", true);
  app.add_option("--strain-rate", strain_rate, "Nominal strain rate", true);
  app.add_option("--output-file", output_filename, "Name for force-displacement output file", true);
  app.set_help_flag("--help");

  CLI11_PARSE(app, argc, argv);

  SLIC_INFO_ROOT(axom::fmt::format("strain rate: {}", strain_rate));
  SLIC_INFO_ROOT(axom::fmt::format("time_steps: {}", time_steps));

  double max_time = max_strain / strain_rate;

  // Create DataStore
  const std::string simulation_tag = "uniaxial";
  const std::string mesh_tag = simulation_tag + "mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, simulation_tag + "_data");

  auto mesh = serac::mesh::refineAndDistribute(
      serac::buildCuboidMesh(elements_in_x, elements_in_y, elements_in_z, x_length, y_length, z_length),
      serial_refinements, parallel_refinements);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // create boundary domains for boundary conditions
  auto fix_x = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(5));
  auto fix_y = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(2));
  auto fix_z = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(1));
  auto apply_displacement = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(3));
  serac::Domain whole_mesh = serac::EntireDomain(pmesh);

  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 0};

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = serac::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-10,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 200,
                                                  .print_level = 1};

  serac::SolidMechanics<p, dim> solid_solver(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, simulation_tag, mesh_tag);

  using Hardening = serac::solid_mechanics::VoceHardening;
  using Material = serac::solid_mechanics::J2<Hardening>;

  Hardening hardening{sigma_y, sigma_sat, strain_constant, eta};
  Material material{E, nu, hardening, density};

  auto internal_states = solid_solver.createQuadratureDataBuffer(Material::State{}, whole_mesh);

  solid_solver.setRateDependentMaterial(material, whole_mesh, internal_states);

  solid_solver.setFixedBCs(fix_x, serac::Component::X);
  solid_solver.setFixedBCs(fix_y, serac::Component::Y);
  solid_solver.setFixedBCs(fix_z, serac::Component::Z);
  auto applied_displacement = [strain_rate](serac::vec3, double t) {
    return serac::vec3{strain_rate * x_length * t, 0., 0.};
  };
  solid_solver.setDisplacementBCs(applied_displacement, apply_displacement, serac::Component::X);

  solid_solver.completeSetup();

  double dt = max_time / (time_steps - 1);

  // get nodes and dofs to compute total force
  mfem::Array<int> dof_list = apply_displacement.dof_list(&solid_solver.displacement().space());
  solid_solver.displacement().space().DofsToVDofs(0, dof_list);

  auto compute_net_force = [&dof_list](const serac::FiniteElementDual& reaction) -> double {
    double R{};
    for (int i = 0; i < dof_list.Size(); i++) {
      R += reaction(dof_list[i]);
    }
    return R;
  };

  std::string paraview_tag = simulation_tag + "_paraview";
  std::ofstream file(output_filename);
  file << "# time displacement force" << std::endl;
  {
    double u = applied_displacement(serac::vec3{}, solid_solver.time())[0];
    double f = compute_net_force(solid_solver.dual("reactions"));
    output(u, f, solid_solver, paraview_tag, file);
  }

  for (int i = 1; i < time_steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", solid_solver.time() + dt, max_time));
    serac::logger::flush();

    solid_solver.advanceTimestep(dt);

    double u = applied_displacement(serac::vec3{}, solid_solver.time())[0];
    double f = compute_net_force(solid_solver.dual("reactions"));
    output(u, f, solid_solver, paraview_tag, file);
  }

  file.close();
  serac::exitGracefully();

  return 0;
}
