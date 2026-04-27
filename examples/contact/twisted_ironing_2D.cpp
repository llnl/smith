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
#include <cmath>

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

  // Create DataStore
  std::string name = "twisted_contact_ironing_2D_example";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  auto mesh = std::make_shared<smith::Mesh>(
      shared::MeshBuilder::Unify(
          {shared::MeshBuilder::SquareMesh(32, 32).updateBdrAttrib(1, 6).updateBdrAttrib(3, 9).scale({1.0, 0.5}),
           shared::MeshBuilder::SquareMesh(8, 8)
               .scale({0.25, 0.25})
               .translate({0.0, 0.5})
               .updateBdrAttrib(3, 5)
               .updateBdrAttrib(1, 8)
               .updateBdrAttrib(2, 8)
               .updateBdrAttrib(4, 8)
               .updateAttrib(1, 2)}),
      "ironing_2D_mesh", 0, 0);

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack, .print_level = 0};

  mfem::VisItDataCollection visit_dc("contact_ironing_twist_vist", &mesh->mfemParMesh());

  visit_dc.SetPrefixPath("visit_out");
  visit_dc.Save();

#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = 50,
                                                  .print_level = 1};

  smith::ContactOptions contact_options{.method = smith::ContactMethod::EnergyMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .penalty = 4000,
                                        .penalty2 = 0,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));

  mfem::Vector K_values({10.0, 100.0});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));

  mfem::Vector G_values({0.25, 2.5});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Pass the BC information to the solver object
  mesh->addDomainOfBoundaryElements("bottom_of_subtrate", smith::by_attr<dim>(6));
  solid_solver.setFixedBCs((mesh->domain("bottom_of_subtrate")));

  mesh->addDomainOfBoundaryElements("top of indenter", smith::by_attr<dim>(5));
  const smith::tensor<double, dim> r0{{0.125, 0.625}};
  auto applied_displacement = [r0](smith::tensor<double, dim> x, double t) {
    constexpr double init_steps = 10.0;
    constexpr double theta_max = 80.0 * M_PI / 180.0;
    smith::tensor<double, dim> u{};
    if (t <= init_steps + 1.0e-12) {
      u[1] = -t * 0.05 / init_steps;
    } else {
      double hm = (t - init_steps) * 0.01;  // horizontal movement
      double theta = theta_max * hm;        // current rotation angle
      double cos_theta = std::cos(theta);
      double sin_theta = std::sin(theta);

      // Rotate about r0
      smith::tensor<double, dim> y{{x[0] - r0[0], x[1] - r0[1]}};
      smith::tensor<double, dim> y_rot{{cos_theta * y[0] - sin_theta * y[1], sin_theta * y[0] + cos_theta * y[1]}};

      u[0] = (y_rot[0] - y[0]) + 0.01 * (t - init_steps);
      u[1] = (y_rot[1] - y[1]) - 0.05;
    }
    return u;
  };

  solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("top of indenter"));

  // Add contact interaction

  auto contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes({8});
  std::set<int> surface_2_boundary_attributes({9});
  solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                     surface_2_boundary_attributes, contact_options);

  std::string visit_name = name + "_visit";
  solid_solver.outputStateToDisk(visit_name);

  solid_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;

  for (int i{0}; i < 110; ++i) {
    solid_solver.advanceTimestep(dt);
    visit_dc.SetCycle(i);
    visit_dc.SetTime((i + 1) * dt);
    visit_dc.Save();
    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(visit_name);
  }

  return 0;
}
