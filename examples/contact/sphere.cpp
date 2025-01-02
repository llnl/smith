// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>

#include <set>
#include <string>
#include <vector>

#include "axom/slic/core/SimpleLogger.hpp"

#include "mfem.hpp"

#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/solid_mechanics_contact.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

int main(int argc, char* argv[])
{
  serac::initialize(argc, argv);

  // NOTE: p must be equal to 1 to work with Tribol's mortar method
  constexpr int p = 1;
  // NOTE: dim must be equal to 3
  constexpr int dim = 3;

  // Create DataStore
  std::string            name = "contact_sphere_example";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  int num_refinements{3};

  mfem::Mesh ball_mesh{SERAC_REPO_DIR "/data/meshes/ball-nurbs.mesh"};
  for (int i{0}; i < num_refinements; ++i) {
    ball_mesh.UniformRefinement();
  }
  ball_mesh.SetCurvature(p);

  mfem::Mesh cube_mesh{SERAC_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements; ++i) {
    cube_mesh.UniformRefinement();
  }
  cube_mesh.SetCurvature(p);

  std::vector<mfem::Mesh*> mesh_ptrs{&ball_mesh, &cube_mesh};
  auto  mesh = serac::mesh::refineAndDistribute(mfem::Mesh(mesh_ptrs.data(), static_cast<int>(mesh_ptrs.size())), 0, 0);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), "sphere_mesh");

  auto fixed_boundary = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(3));
  auto driven_surface = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(12));

  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 1};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = serac::NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-8,
                                                  .absolute_tol   = 1.0e-5,
                                                  .max_iterations = 200,
                                                  .print_level    = 1};

  serac::ContactOptions contact_options{.method      = serac::ContactMethod::SingleMortar,
                                        .enforcement = serac::ContactEnforcement::Penalty,
                                        .type        = serac::ContactType::Frictionless,
                                        .penalty     = 1.0e4};

  serac::SolidMechanicsContact<p, dim> solid_solver(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, "sphere_mesh");

  serac::solid_mechanics::NeoHookean mat{1.0, 10.0, 0.25};
  serac::Domain                      whole_mesh = serac::EntireDomain(pmesh);
  solid_solver.setMaterial(mat, whole_mesh);

  // Pass the BC information to the solver object
  solid_solver.setFixedBCs(fixed_boundary);

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

  solid_solver.setDisplacementBCs(applied_displacement, driven_surface);

  // Add the contact interaction
  auto          contact_interaction_id = 0;
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

  serac::exitGracefully();

  return 0;
}
