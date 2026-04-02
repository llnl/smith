// Copyright (c) Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <set>
#include <src/smith/physics/contact/contact_config.hpp>
#include <string>

#include "axom/slic.hpp"

#include "mfem.hpp"

#include "smith/numerics/solver_config.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith.hpp"

#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"

#include <cfenv>
#include <fem/datacollection.hpp>
#include <functional>
#include <mesh/vtk.hpp>
#include <set>
#include <string>
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "shared/mesh/MeshBuilder.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include <fenv.h>

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // NOTE: p must be equal to 1 to work with Tribol's mortar method
  constexpr int p = 1;

  // NOTE: dim must be equal to 2
  constexpr int dim = 2;

  // COARSE
  // constexpr auto mesh_factor = 1;
  // std::string name_postfix = "coarse";

  // REFINED
  constexpr auto mesh_factor = 4;
  std::string name_postfix = "refined";

  // Create DataStore
  std::string name = "contact_ironing_2D_example";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Construct the appropiate dimension mesh and give it to the data store
  //  std::string filename = smith_REPO_DIR "data/meshes/ironing_2D.mesh";
  //  std::shared_ptr<smith::Mesh> mesh = std::make_shared<smith::Mesh>(filename, "ironing_2D_mesh", 2, 0);

  auto mesh = std::make_shared<smith::Mesh>(
      shared::MeshBuilder::Unify(
          {shared::MeshBuilder::SquareMesh(32 * mesh_factor, 8 * mesh_factor)
               .updateBdrAttrib(1, 6)
               .updateBdrAttrib(3, 9)
               .bdrAttribInfo()
               .scale({1.0, 0.25}),
           shared::MeshBuilder::SemiCircularShell(mesh_factor * 3 / 2, 10 * mesh_factor, 0.075, 0.125)
               .translate({0.125, 0.375})
               .updateBdrAttrib(1, 5)
               .updateBdrAttrib(2, 8)
               .updateBdrAttrib(3, 5)
               .updateBdrAttrib(4, 5)
               .updateAttrib(1, 2)}),
      "ironing_2D_mesh_" + name_postfix, 0, 0);
  mesh->mfemParMesh().CheckElementOrientation(true);

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::CG,  // Strumpack,  // CG,
                                            .preconditioner = smith::Preconditioner::HypreAMG,
                                            .print_level = 0};

  mfem::VisItDataCollection visit_dc("contact_ironing_visit", &mesh->mfemParMesh());

  visit_dc.SetPrefixPath("visit_out");
  visit_dc.Save();

#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  smith::NonlinearSolverOptions nonlinear_options{
      .nonlin_solver = smith::NonlinearSolver::TrustRegion,  // NewtonLineSearch,  // TrustRegion,
      .relative_tol = 1.0e-8,
      .absolute_tol = 1.0e-8,
      .max_iterations = 5000,
      .max_line_search_iterations = 10,
      .print_level = 1};

  smith::ContactOptions contact_options{.method = smith::ContactMethod::EnergyMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = 30000.0,
                                        .penalty2 = 0,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"}, 0, 0.0, false, false);

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));

  mfem::Vector K_values({1.0, 100.0});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));

  mfem::Vector G_values({0.25, 25.0});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 100.0, 1.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Pass the BC information to the solver object
  mesh->addDomainOfBoundaryElements("bottom_of_subtrate", smith::by_attr<dim>(6));
  solid_solver.setFixedBCs((mesh->domain("bottom_of_subtrate")));

  mesh->addDomainOfBoundaryElements("top of indenter", smith::by_attr<dim>(5));
  auto applied_displacement = [](smith::tensor<double, dim>, double t) {
    constexpr double init_steps = 20.0;
    smith::tensor<double, dim> u{};
    // std::cout << "T ========= " << t << std::endl;
    if (t <= init_steps + 1.0e-12) {
      u[1] = -t * 0.101 / init_steps;
    } else {
      u[0] = (t - init_steps) * 0.005;
      u[1] = -0.101;
    }
    return u;
  };

  solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("top of indenter"));
  // std::cout << "top of indenter size: " << mesh->domain("top of indenter").size() << std::endl;

  // Add the contact interaction
  auto contact_interaction_id = 0;
  //   auto contact_interaction_id2 = 1;
  std::set<int> surface_1_boundary_attributes({9});
  std::set<int> surface_2_boundary_attributes({8});
  //   std::set<int> surface_3_boundary_attributes({2});
  solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                     surface_2_boundary_attributes, contact_options);
  //   solid_solver.addContactInteraction(contact_interaction_id2, surface_1_boundary_attributes,
  //                                      surface_3_boundary_attributes, contact_options);
  // Finalize the data structures
  //  solid_solver.completeSetup();

  std::string visit_name = name + "_visit";
  solid_solver.outputStateToDisk(visit_name);

  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;

  if (std::isnan(solid_solver.displacement().Norml2())) {
    std::cout << "NaN detected in displacement before first timestep!" << std::endl;
  }

  for (int i{0}; i < 175; ++i) {
    solid_solver.advanceTimestep(dt);
    visit_dc.SetCycle(i);
    visit_dc.SetTime((i + 1) * dt);
    visit_dc.Save();

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(visit_name);
  }

  return 0;
}
