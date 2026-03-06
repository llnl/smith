// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <set>
#include <string>
#include <memory>

#include "axom/slic.hpp"
#include "mfem.hpp"
#include "smith/smith.hpp"

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // NOTE: p must be equal to 1 to work with Tribol's mortar method
  constexpr int p = 1;
  // NOTE: dim must be equal to 3
  constexpr int dim = 3;

  // Create DataStore
  std::string name = "contact_beam_example";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex-with-contact-block.mesh";

  auto mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(filename), "beam_mesh", 2, 0);

  // create boundary domains for boundary conditions
  mesh->addDomainOfBoundaryElements("support", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("applied_displacement_surface", smith::by_attr<dim>(6));

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack, .print_level = 0};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-13,
                                                  .absolute_tol = 1.0e-13,
                                                  .max_iterations = 200,
                                                  .print_level = 1};

  smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = 8.0e2,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", "beam_mesh"));
  // each vector value corresponds to a different element attribute:
  // [0] (element attribute 1) : the beam
  // [1] (element attribute 2) : indenter block
  mfem::Vector K_values({10.0, 100.0});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", "beam_mesh"));
  // each vector value corresponds to a different element attribute:
  // [0] (element attribute 1) : the beam
  // [1] (element attribute 2) : indenter block
  mfem::Vector G_values({0.25, 2.5});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Pass the BC information to the solver object
  solid_solver.setFixedBCs(mesh->domain("support"));

  auto applied_displacement = [](smith::tensor<double, dim>, double t) {
    smith::tensor<double, dim> u{};
    u[2] = -0.05 * t;
    return u;
  };

  solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("applied_displacement_surface"));

  // Add the contact interaction
  auto contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes({7});
  std::set<int> surface_2_boundary_attributes({5});
  solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                     surface_2_boundary_attributes, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;

  for (int i{0}; i < 12; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
