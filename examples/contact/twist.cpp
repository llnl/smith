// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>

#include <set>
#include <string>

#include "axom/slic.hpp"

#include "mfem.hpp"

#include "serac/serac.hpp"

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  serac::ApplicationManager applicationManager(argc, argv);

  // NOTE: p must be equal to 1 to work with Tribol's mortar method
  constexpr int p = 1;
  // NOTE: dim must be equal to 3
  constexpr int dim = 3;

  // Create DataStore
  std::string name = "contact_twist_example";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/twohex_for_contact.mesh";

  auto pmesh = std::make_shared<serac::Mesh>(serac::buildMeshFromFile(filename), "twist_mesh", 3, 0);

  pmesh->addDomainOfBoundaryElements("fixed_surface", serac::by_attr<dim>(3));
  pmesh->addDomainOfBoundaryElements("driven_surface", serac::by_attr<dim>(6));

  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 0};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = serac::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-13,
                                                  .absolute_tol = 1.0e-13,
                                                  .max_iterations = 200,
                                                  .print_level = 1};

  serac::ContactOptions contact_options{.method = serac::ContactMethod::SingleMortar,
                                        .enforcement = serac::ContactEnforcement::Penalty,
                                        .type = serac::ContactType::Frictionless,
                                        .penalty = 1.0e4,
                                        .jacobian = serac::ContactJacobian::Exact};

  serac::SolidMechanicsContact<p, dim> solid_solver(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, "twist_mesh");

  serac::solid_mechanics::NeoHookean mat{1.0, 10.0, 10.0};
  solid_solver.setMaterial(mat, pmesh->entireBody());

  // Pass the BC information to the solver object
  solid_solver.setFixedBCs(pmesh->domain("fixed_surface"));

  auto applied_displacement = [](serac::tensor<double, dim> x, double t) {
    serac::tensor<double, dim> u{};
    if (t <= 3.0 + 1.0e-12) {
      u[2] = -t * 0.02;
    } else {
      u[0] =
          (std::cos(M_PI / 40.0 * (t - 3.0)) - 1.0) * (x[0] - 0.5) - std::sin(M_PI / 40.0 * (t - 3.0)) * (x[1] - 0.5);
      u[1] =
          std::sin(M_PI / 40.0 * (t - 3.0)) * (x[0] - 0.5) + (std::cos(M_PI / 40.0 * (t - 3.0)) - 1.0) * (x[1] - 0.5);
      u[2] = -0.06;
    }
    return u;
  };

  solid_solver.setDisplacementBCs(applied_displacement, pmesh->domain("driven_surface"));

  // Add the contact interaction
  auto contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes({4});
  std::set<int> surface_2_boundary_attributes({5});
  solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                     surface_2_boundary_attributes, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;

  for (int i{0}; i < 23; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
