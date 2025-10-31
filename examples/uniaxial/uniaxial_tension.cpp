// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <fstream>
#include <memory>

#include "axom/inlet.hpp"
#include "axom/slic.hpp"
#include "mfem.hpp"
#include "smith/smith.hpp"

template <class Physics>
void output(double u, double f, const Physics& solid, const std::string& paraview_tag, std::ofstream& file)
{
  solid.outputStateToDisk(paraview_tag);
  file << solid.time() << " " << u << " " << f << std::endl;
}

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  constexpr int p = 2;
  constexpr int dim = 3;
  constexpr double x_length = 1.0;
  constexpr double y_length = 0.1;
  constexpr double z_length = 0.1;
  constexpr int elements_in_x = 10;
  constexpr int elements_in_y = 1;
  constexpr int elements_in_z = 1;

  int serial_refinement = 0;
  int parallel_refinement = 0;
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
  app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps")
      ->default_val("0")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps")
      ->default_val("0")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--time-steps", time_steps, "Number of time steps to divide simulation")
      ->default_val("100")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--strain-rate", strain_rate, "Nominal strain rate")
      ->default_val("1e-3")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--output-file", output_filename, "Name for force-displacement output file")
      ->default_val(output_filename);
  app.set_help_flag("--help");

  CLI11_PARSE(app, argc, argv);

  SLIC_INFO_ROOT(axom::fmt::format("strain rate: {}", strain_rate));
  SLIC_INFO_ROOT(axom::fmt::format("time_steps: {}", time_steps));

  double max_time = max_strain / strain_rate;

  // Create DataStore
  const std::string simulation_tag = "uniaxial";
  const std::string mesh_tag = simulation_tag + "mesh";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, simulation_tag + "_data");

  auto mesh = std::make_shared<smith::Mesh>(
      smith::buildCuboidMesh(elements_in_x, elements_in_y, elements_in_z, x_length, y_length, z_length), mesh_tag,
      serial_refinement, parallel_refinement);

  // create boundary domains for boundary conditions
  mesh->addDomainOfBoundaryElements("fix_x", smith::by_attr<dim>(5));
  mesh->addDomainOfBoundaryElements("fix_y", smith::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("fix_z", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("apply_displacement", smith::by_attr<dim>(3));

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack, .print_level = 0};

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-10,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 200,
                                                  .print_level = 1};

  smith::SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options,
                                             smith::solid_mechanics::default_quasistatic_options, simulation_tag, mesh);

  using Hardening = smith::solid_mechanics::VoceHardening;
  using Material = smith::solid_mechanics::J2<Hardening>;

  Hardening hardening{sigma_y, sigma_sat, strain_constant, eta};
  Material material{E, nu, hardening, density};

  auto internal_states = solid_solver.createQuadratureDataBuffer(Material::State{}, mesh->entireBody());

  solid_solver.setRateDependentMaterial(material, mesh->entireBody(), internal_states);

  solid_solver.setFixedBCs(mesh->domain("fix_x"), smith::Component::X);
  solid_solver.setFixedBCs(mesh->domain("fix_y"), smith::Component::Y);
  solid_solver.setFixedBCs(mesh->domain("fix_z"), smith::Component::Z);
  auto applied_displacement = [strain_rate](smith::vec3, double t) {
    return smith::vec3{strain_rate * x_length * t, 0., 0.};
  };
  solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("apply_displacement"), smith::Component::X);

  solid_solver.completeSetup();

  double dt = max_time / (time_steps - 1);

  // get nodes and dofs to compute total force
  mfem::Array<int> dof_list = mesh->domain("apply_displacement").dof_list(&solid_solver.displacement().space());
  solid_solver.displacement().space().DofsToVDofs(0, dof_list);

  auto compute_net_force = [&dof_list](const smith::FiniteElementDual& reaction) -> double {
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
    double u = applied_displacement(smith::vec3{}, solid_solver.time())[0];
    double f = compute_net_force(solid_solver.dual("reactions"));
    output(u, f, solid_solver, paraview_tag, file);
  }

  for (int i = 1; i < time_steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", solid_solver.time() + dt, max_time));
    smith::logger::flush();

    solid_solver.advanceTimestep(dt);

    double u = applied_displacement(smith::vec3{}, solid_solver.time())[0];
    double f = compute_net_force(solid_solver.dual("reactions"));
    output(u, f, solid_solver, paraview_tag, file);
  }

  file.close();

  return 0;
}
