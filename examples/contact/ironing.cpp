// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"

#include "mfem.hpp"

#include "serac/physics/solid_mechanics_contact.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/domain.hpp"
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
  std::string name = "contact_ironing_example";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/ironing.mesh";

  auto mesh = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(filename), 2, 0);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), "ironing_mesh");

  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 1};

#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = serac::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-7,
                                                  .absolute_tol = 1.0e-7,
                                                  .max_iterations = 200,
                                                  .print_level = 1};

  serac::ContactOptions contact_options{.method = serac::ContactMethod::SingleMortar,
                                        .enforcement = serac::ContactEnforcement::Penalty,
                                        .type = serac::ContactType::TiedNormal,
                                        .penalty = 1.0e3};

  serac::SolidMechanicsContact<p, dim, serac::Parameters<serac::L2<0>, serac::L2<0>>> solid_solver(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, "ironing_mesh",
      {"bulk_mod", "shear_mod"});

  serac::FiniteElementState K_field(serac::StateManager::newState(serac::L2<0>{}, "bulk_mod", "ironing_mesh"));
  // each vector value corresponds to a different element attribute:
  // [0] (element attribute 1) : the substrate
  // [1] (element attribute 2) : indenter block
  mfem::Vector K_values({10.0, 100.0});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  serac::FiniteElementState G_field(serac::StateManager::newState(serac::L2<0>{}, "shear_mod", "ironing_mesh"));
  // each vector value corresponds to a different element attribute:
  // [0] (element attribute 1) : the substrate
  // [1] (element attribute 2) : indenter block
  mfem::Vector G_values({0.25, 2.5});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  serac::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  serac::Domain whole_mesh = serac::EntireDomain(pmesh);
  solid_solver.setMaterial(serac::DependsOn<0, 1>{}, mat, whole_mesh);

  // Pass the BC information to the solver object
  serac::Domain bottom_of_substrate = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(5));
  solid_solver.setFixedBCs(bottom_of_substrate);

  serac::Domain top_of_indenter = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(12));
  auto applied_displacement = [](serac::tensor<double, dim>, double t) {
    constexpr double init_steps = 2.0;
    serac::tensor<double, dim> u{};
    if (t <= init_steps + 1.0e-12) {
      u[2] = -t * 0.3 / init_steps;
    } else {
      u[0] = -(t - init_steps) * 0.25;
      u[2] = -0.3;
    }
    return u;
  };
  solid_solver.setDisplacementBCs(applied_displacement, top_of_indenter);

  // Add the contact interaction
  auto contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes({6});
  std::set<int> surface_2_boundary_attributes({11});
  solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                     surface_2_boundary_attributes, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;

  for (int i{0}; i < 26; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  serac::exitGracefully();

  return 0;
}
