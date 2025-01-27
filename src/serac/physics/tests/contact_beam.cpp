// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics_contact.hpp"

#include <functional>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/numerics/functional/domain.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

class ContactTest : public testing::TestWithParam<std::tuple<ContactEnforcement, ContactType, std::string>> {};

TEST_P(ContactTest, beam)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p = 1;
  constexpr int dim = 3;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string name = "contact_beam_" + std::get<2>(GetParam());
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex-with-contact-block.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), 1, 0);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), "beam_mesh");

  LinearSolverOptions linear_options{.linear_solver = LinearSolver::Strumpack, .print_level = 1};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return;
#endif

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 1.0e-12,
                                           .absolute_tol = 1.0e-12,
                                           .max_iterations = 200,
                                           .print_level = 1};
#ifdef SERAC_USE_SUNDIALS
  // KINFullStep is preferred, but has issues when active set is enabled
  if (std::get<1>(GetParam()) == ContactType::TiedNormal) {
    nonlinear_options.nonlin_solver = NonlinearSolver::KINFullStep;
  }
#endif

  ContactOptions contact_options{.method = ContactMethod::SingleMortar,
                                 .enforcement = std::get<0>(GetParam()),
                                 .type = std::get<1>(GetParam()),
                                 .penalty = 1.0e2};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, "beam_mesh");

  double K = 10.0;
  double G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  Domain material_block = EntireDomain(pmesh);
  solid_solver.setMaterial(mat, material_block);

  // Pass the BC information to the solver object
  Domain support = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));
  solid_solver.setFixedBCs(support);
  auto applied_displacement = [](tensor<double, dim>, double) {
    tensor<double, dim> u{};
    u[2] = -0.15;
    return u;
  };
  auto driven_surface = Domain::ofBoundaryElements(pmesh, by_attr<dim>(6));
  solid_solver.setDisplacementBCs(applied_displacement, driven_surface);

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {7}, {5}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  // std::string paraview_name = name + "_paraview";
  // solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  // solid_solver.outputStateToDisk(paraview_name);

  // Check the l2 norm of the displacement dofs
  auto u_l2 = mfem::ParNormlp(solid_solver.displacement(), 2, MPI_COMM_WORLD);
  if (std::get<1>(GetParam()) == ContactType::TiedNormal) {
    EXPECT_NEAR(1.465, u_l2, 1.0e-2);
  } else if (std::get<1>(GetParam()) == ContactType::Frictionless) {
    EXPECT_NEAR(1.526, u_l2, 1.0e-2);
  }
}

// NOTE: if Penalty is first and Lagrange Multiplier is second, SuperLU gives a zero diagonal error
INSTANTIATE_TEST_SUITE_P(
    tribol, ContactTest,
    testing::Values(std::make_tuple(ContactEnforcement::Penalty, ContactType::TiedNormal, "penalty_tiednormal"),
                    std::make_tuple(ContactEnforcement::Penalty, ContactType::Frictionless, "penalty_frictionless"),
                    std::make_tuple(ContactEnforcement::LagrangeMultiplier, ContactType::TiedNormal,
                                    "lagrange_multiplier_tiednormal"),
                    std::make_tuple(ContactEnforcement::LagrangeMultiplier, ContactType::Frictionless,
                                    "lagrange_multiplier_frictionless")));

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
