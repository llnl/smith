// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <set>
#include <string>
#include <vector>
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
  std::string name = "contact_sphere_example";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  int num_refinements{3};

  mfem::Mesh ball_mesh{SMITH_REPO_DIR "/data/meshes/ball-nurbs.mesh"};
  for (int i{0}; i < num_refinements; ++i) {
    ball_mesh.UniformRefinement();
  }
  ball_mesh.SetCurvature(p);

  mfem::Mesh cube_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements; ++i) {
    cube_mesh.UniformRefinement();
  }
  cube_mesh.SetCurvature(p);

  std::vector<mfem::Mesh*> mesh_ptrs{&ball_mesh, &cube_mesh};
  auto mesh = std::make_shared<smith::Mesh>(mfem::Mesh(mesh_ptrs.data(), static_cast<int>(mesh_ptrs.size())),
                                            "sphere_mesh", 0, 0);

  mesh->addDomainOfBoundaryElements("fixed_boundary", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("driven_surface", smith::by_attr<dim>(12));

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
                                        .penalty = 1.0e4,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                                    smith::solid_mechanics::default_quasistatic_options, name, mesh);

  smith::solid_mechanics::NeoHookean mat{1.0, 10.0, 0.25};
  solid_solver.setMaterial(mat, mesh->entireBody());

  // Pass the BC information to the solver object
  solid_solver.setFixedBCs(mesh->domain("fixed_boundary"));

  auto applied_displacement = [](smith::tensor<double, dim> x, double t) {
    smith::tensor<double, dim> u{};
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

  solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("driven_surface"));

  // Add the contact interaction
  auto contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes({5});
  std::set<int> surface_2_boundary_attributes({7});
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
