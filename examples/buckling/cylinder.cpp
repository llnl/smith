// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file cylinder.cpp
 *
 * @brief A buckling cylinder under compression, run with or without contact
 *
 * @note Run with mortar contact and PETSc preconditioners:
 * @code{.sh}
 * ./build/examples/buckling_cylinder --contact --contact-type 1 --preconditioner 6 \
 *    -options_file examples/buckling/cylinder_petsc_options.yml
 * @endcode
 * @note Run with penalty contact and HYPRE BoomerAMG preconditioner
 * @code{.sh}
 * ./build/examples/buckling_cylinder
 * @endcode
 * @note Run without contact:
 * @code{.sh}
 * ./build/examples/buckling_cylinder --no-contact
 * @endcode
 */

#include <set>
#include <string>
#include <cmath>
#include <memory>
#include <utility>

#include "axom/slic.hpp"
#include "axom/inlet.hpp"
#include "axom/CLI11.hpp"
#include "mfem.hpp"
#include "smith/smith.hpp"

using namespace smith;

/**
 * @brief Run buckling cylinder example
 *
 * @note Based on doi:10.1016/j.cma.2014.08.012
 */
int main(int argc, char* argv[])
{
  constexpr int dim = 3;
  constexpr int p = 1;

  // Command line arguments
  // Mesh options
  int serial_refinement = 0;
  int parallel_refinement = 0;
  double dt = 0.1;

  // Solver options
  NonlinearSolverOptions nonlinear_options = solid_mechanics::default_nonlinear_options;
  nonlinear_options.nonlin_solver = smith::NonlinearSolver::TrustRegion;
  nonlinear_options.relative_tol = 1e-6;
  nonlinear_options.absolute_tol = 1e-10;
  nonlinear_options.min_iterations = 1;
  nonlinear_options.max_iterations = 500;
  nonlinear_options.max_line_search_iterations = 20;
  nonlinear_options.print_level = 1;

  LinearSolverOptions linear_options = solid_mechanics::default_linear_options;
  linear_options.linear_solver = smith::LinearSolver::CG;
  linear_options.preconditioner = smith::Preconditioner::HypreAMG;
  linear_options.relative_tol = 1e-8;
  linear_options.absolute_tol = 1e-16;
  linear_options.max_iterations = 2000;

  // Contact specific options
  double penalty = 1e3;
#ifdef SMITH_USE_TRIBOL
  bool use_contact = true;
#else
  bool use_contact = false;
#endif

  auto contact_type = smith::ContactEnforcement::Penalty;

  // Option for testing purposes only, to reduce runtime
  bool use_fast_options = false;

  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // Handle command line arguments
  axom::CLI::App app{"Hollow cylinder buckling example"};
  // Mesh options
  app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps")->check(axom::CLI::PositiveNumber);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps")
      ->check(axom::CLI::PositiveNumber);
  // Solver options
  app.add_option("--nonlinear-solver", nonlinear_options.nonlin_solver,
                 "Nonlinear solver (Index of enum smith::NonlinearSolver)")
      ->expected(0, 10);
  app.add_option("--linear-solver", linear_options.linear_solver, "Linear solver (Index of enum smith::LinearSolver)")
      ->expected(0, 5);
  app.add_option("--preconditioner", linear_options.preconditioner,
                 "Preconditioner (Index of enum smith::NonlinearSolver)")
      ->expected(0, 7);
  app.add_option("--petsc-pc-type", linear_options.petsc_preconditioner,
                 "Petsc preconditioner (Index of enum smith::PetscPCType)")
      ->expected(0, 14);
  app.add_option("--dt", dt, "Size of pseudo-time step pre-contact")->check(axom::CLI::PositiveNumber);
  // Contact options
  auto opt_contact =
      app.add_flag("--contact,!--no-contact", use_contact, "Use contact for the inner faces of the cylinder");
  app.add_option("--contact-type", contact_type,
                 "Type of contact enforcement, 0 for penalty or 1 for Lagrange multipliers (Index of enum "
                 "smith::ContactEnforcement)")
      ->needs(opt_contact)
      ->expected(0, 1);
  app.add_option("--penalty", penalty, "Penalty for contact")->needs(opt_contact)->check(axom::CLI::PositiveNumber);
  // Misc options
  app.add_flag("--fast", use_fast_options, "Reduce max iterations and delta-time for testing purposes.");

  // Need to allow extra arguments for PETSc support
  app.set_help_flag("--help");
  app.allow_extras();
  CLI11_PARSE(app, argc, argv);

  if (use_fast_options) {
    dt = 1;
    nonlinear_options.max_iterations = 5;
    linear_options.max_iterations = 5;
  }

  // Create DataStore
  std::string name = use_contact ? "buckling_cylinder_contact" : "buckling_cylinder";
  std::string mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Create and refine mesh
  std::string filename = SMITH_REPO_DIR "/data/meshes/hollow-cylinder.mesh";
  auto mesh = std::make_shared<smith::Mesh>(filename, mesh_tag, serial_refinement, parallel_refinement);

  // Surfaces for boundary conditions
  constexpr int xneg_attr{2};
  constexpr int xpos_attr{3};
  mesh->addDomainOfBoundaryElements("xneg", smith::by_attr<dim>(xneg_attr));
  mesh->addDomainOfBoundaryElements("xpos", smith::by_attr<dim>(xpos_attr));
  mesh->addDomainOfBoundaryElements("bottom", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("top", smith::by_attr<dim>(4));

  // Create solver, either with or without contact
  std::unique_ptr<SolidMechanics<p, dim>> solid_solver;
  if (use_contact) {
#ifdef SMITH_USE_TRIBOL
    auto solid_contact_solver = std::make_unique<smith::SolidMechanicsContact<p, dim>>(
        nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh);

    // Add the contact interaction
    smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                          .enforcement = contact_type,
                                          .type = smith::ContactType::Frictionless,
                                          .penalty = penalty,
                                          .jacobian = smith::ContactJacobian::Exact};
    auto contact_interaction_id = 0;
    solid_contact_solver->addContactInteraction(contact_interaction_id, {xpos_attr}, {xneg_attr}, contact_options);
    solid_solver = std::move(solid_contact_solver);
#else
    SLIC_ERROR("Smith built without Tribol enabled!");
#endif
  } else {
    solid_solver = std::make_unique<smith::SolidMechanics<p, dim>>(
        nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh);
    solid_solver->setPressure([&](auto&, double t) { return 0.01 * t; }, mesh->domain("xpos"));
  }

  // Define a Neo-Hookean material
  auto lambda = 1.0;
  auto G = 0.1;
  solid_mechanics::NeoHookean mat{.density = 1.0, .K = (3 * lambda + 2 * G) / 3, .G = G};
  solid_solver->setMaterial(mat, mesh->entireBody());

  // Set up essential boundary conditions
  // Bottom of cylinder is fixed
  solid_solver->setFixedBCs(mesh->domain("bottom"));

  // Top of cylinder has prescribed displacement of magnitude in x-z direction
  auto compress = [&](const smith::tensor<double, dim>, double t) {
    smith::tensor<double, dim> u{};
    u[0] = u[2] = -1.35 / std::sqrt(2.0) * t;
    return u;
  };
  solid_solver->setDisplacementBCs(compress, mesh->domain("top"), Component::X + Component::Z);
  solid_solver->setDisplacementBCs(compress, mesh->domain("top"),
                                   Component::Y);  // BT: Would it be better to leave this component free?

  // Finalize the data structures
  solid_solver->completeSetup();

  // Save initial state
  std::string paraview_name = name + "_paraview";
  solid_solver->outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  SLIC_INFO_ROOT(axom::fmt::format("Running hollow cylinder bucking example with {} displacement dofs",
                                   solid_solver->displacement().GlobalSize()));
  SLIC_INFO_ROOT("Starting pseudo-timestepping.");
  smith::logger::flush();
  while (solid_solver->time() < 1.0 && std::abs(solid_solver->time() - 1) > DBL_EPSILON) {
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of 1.0)", solid_solver->time()));
    smith::logger::flush();

    // Refine dt as contact starts
    auto next_dt = solid_solver->time() < 0.65 ? dt : dt * 0.1;
    solid_solver->advanceTimestep(next_dt);

    // Output the sidre-based plot files
    solid_solver->outputStateToDisk(paraview_name);
  }
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver->time()));

  return 0;
}
