// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
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
 * ./build/examples/buckling_cylinder --contact-type 1 --preconditioner 6 \
 *     -options_file examples/buckling/cylinder_petsc_options.yml
 * @endcode
 * @note Run with penalty contact and HYPRE BoomerAMG preconditioner
 * @code{.sh}
 * ./build/examples/buckling_cylinder --penalty 1e3
 * @endcode
 * @note Run without contact:
 * @code{.sh}
 * ./build/examples/buckling_cylinder --no-contact
 * @endcode
 */

#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include "axom/inlet.hpp"

#include "mfem.hpp"

#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/solid_mechanics_contact.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

using namespace serac;

std::function<std::string(const std::string&)> petscPCTypeValidator = [](const std::string& in) -> std::string {
  return std::to_string(static_cast<int>(mfem_ext::stringToPetscPCType(in)));
};

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
  LinearSolverOptions linear_options = solid_mechanics::default_linear_options;
  nonlinear_options.nonlin_solver = serac::NonlinearSolver::TrustRegion;
  nonlinear_options.relative_tol = 1e-6;
  nonlinear_options.absolute_tol = 1e-10;
  nonlinear_options.min_iterations = 1;
  nonlinear_options.max_iterations = 500;
  nonlinear_options.max_line_search_iterations = 20;
  nonlinear_options.print_level = 1;
#ifdef SERAC_USE_PETSC
  linear_options.linear_solver = serac::LinearSolver::GMRES;
  linear_options.preconditioner = serac::Preconditioner::HypreAMG;
  linear_options.relative_tol = 1e-8;
  linear_options.absolute_tol = 1e-16;
  linear_options.max_iterations = 2000;
#endif

  // Contact specific options
  double penalty = 1e3;
  bool use_contact = true;
  auto contact_type = serac::ContactEnforcement::Penalty;

  // Initialize Serac and all of the external libraried
  serac::initialize(argc, argv);

  // Handle command line arguments
  axom::CLI::App app{"Hollow cylinder buckling example"};
  // Mesh options
  app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps", true);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps", true);
  // Solver options
  app.add_option("--nonlinear-solver", nonlinear_options.nonlin_solver, "Nonlinear solver", true);
  app.add_option("--linear-solver", linear_options.linear_solver, "Linear solver", true);
  app.add_option("--preconditioner", linear_options.preconditioner, "Preconditioner", true);
  app.add_option("--petsc-pc-type", linear_options.petsc_preconditioner, "Petsc preconditioner", true)
      ->transform(
          [](const std::string& in) -> std::string {
            return std::to_string(static_cast<int>(mfem_ext::stringToPetscPCType(in)));
          },
          "Convert string to PetscPCType", "PetscPCTypeTransform");
  app.add_option("--dt", dt, "Size of pseudo-time step pre-contact", true);
  // Contact options
  app.add_flag("--contact,!--no-contact", use_contact, "Use contact for the inner faces of the cylinder");
  app.add_option("--contact-type", contact_type,
                 "Type of contact enforcement, 0 for penalty or 1 for Lagrange multipliers", true)
      ->needs("--contact");
  app.add_option("--penalty", penalty, "Penalty for contact", true)->needs("--contact");

  // Need to allow extra arguments for PETSc support
  app.set_help_flag("--help");
  app.allow_extras()->parse(argc, argv);

  nonlinear_options.force_monolithic = linear_options.preconditioner != Preconditioner::Petsc;

  // Create DataStore
  std::string name = use_contact ? "buckling_cylinder_contact" : "buckling_cylinder";
  std::string mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Create and refine mesh
  std::string filename = SERAC_REPO_DIR "/data/meshes/hollow-cylinder.mesh";
  auto mesh = serac::buildMeshFromFile(filename);
  auto refined_mesh = mesh::refineAndDistribute(std::move(mesh), serial_refinement, parallel_refinement);
  auto& pmesh = serac::StateManager::setMesh(std::move(refined_mesh), mesh_tag);

  // Surfaces for boundary conditions
  constexpr int xneg_attr{2};
  constexpr int xpos_attr{3};
  auto xneg = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(xneg_attr));
  auto xpos = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(xpos_attr));
  auto bottom = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(1));
  auto top = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(4));

  // Create solver, either with or without contact
  std::unique_ptr<SolidMechanics<p, dim>> solid_solver;
  if (use_contact) {
    auto solid_contact_solver = std::make_unique<serac::SolidMechanicsContact<p, dim>>(
        nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, mesh_tag);

    // Add the contact interaction
    serac::ContactOptions contact_options{.method = serac::ContactMethod::SingleMortar,
                                          .enforcement = contact_type,
                                          .type = serac::ContactType::Frictionless,
                                          .penalty = penalty};
    auto contact_interaction_id = 0;
    solid_contact_solver->addContactInteraction(contact_interaction_id, {xpos_attr}, {xneg_attr}, contact_options);
    solid_solver = std::move(solid_contact_solver);
  } else {
    solid_solver = std::make_unique<serac::SolidMechanics<p, dim>>(
        nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, mesh_tag);
    solid_solver->setPressure([&](auto&, double t) { return 0.01 * t; }, xpos);
  }

  // Define a Neo-Hookean material
  auto lambda = 1.0;
  auto G = 0.1;
  solid_mechanics::NeoHookean mat{.density = 1.0, .K = (3 * lambda + 2 * G) / 3, .G = G};
  Domain whole_mesh = EntireDomain(pmesh);
  solid_solver->setMaterial(mat, whole_mesh);

  // Set up essential boundary conditions
  // Bottom of cylinder is fixed
  solid_solver->setFixedBCs(bottom);

  // Top of cylinder has prescribed displacement of magnitude in x-z direction
  auto compress = [&](const serac::tensor<double, dim>, double t) {
    serac::tensor<double, dim> u{};
    u[0] = u[2] = -1.5 / std::sqrt(2.0) * t;
    return u;
  };
  solid_solver->setDisplacementBCs(compress, top, Component::X + Component::Z);
  solid_solver->setDisplacementBCs(compress, top,
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
  serac::logger::flush();
  while (solid_solver->time() < 1.0 && std::abs(solid_solver->time() - 1) > DBL_EPSILON) {
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of 1.0)", solid_solver->time()));
    serac::logger::flush();

    // Refine dt as contact starts
    auto next_dt = solid_solver->time() < 0.65 ? dt : dt * 0.1;
    solid_solver->advanceTimestep(next_dt);

    // Output the sidre-based plot files
    solid_solver->outputStateToDisk(paraview_name);
  }
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver->time()));

  // Exit without error
  serac::exitGracefully();
}
