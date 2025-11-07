// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <memory>
#include <tuple>

#include "mpi.h"
#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/contact/contact_config.hpp"

namespace smith {

class ContactTest
    : public testing::TestWithParam<std::tuple<ContactEnforcement, ContactType, ContactJacobian, std::string>> {};

TEST_P(ContactTest, beam)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p = 1;
  constexpr int dim = 3;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string name = "contact_beam_" + std::get<3>(GetParam());
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex-with-contact-block.mesh";

  auto mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), "beam_mesh", 1, 0);

  LinearSolverOptions linear_options{.linear_solver = LinearSolver::Strumpack, .print_level = 0};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return;
#endif

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 1.0e-13,
                                           .absolute_tol = 1.0e-13,
                                           .max_iterations = 20,
                                           .print_level = 1};

  ContactOptions contact_options{.method = ContactMethod::SingleMortar,
                                 .enforcement = std::get<0>(GetParam()),
                                 .type = std::get<1>(GetParam()),
                                 .penalty = 8.0e2,
                                 .jacobian = std::get<2>(GetParam())};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, mesh);

  double K = 10.0;
  double G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat, mesh->entireBody());

  // Pass the BC information to the solver object
  mesh->addDomainOfBoundaryElements("support", by_attr<dim>(1));
  solid_solver.setFixedBCs(mesh->domain("support"));
  auto applied_displacement = [](tensor<double, dim>, double) {
    tensor<double, dim> u{};
    u[2] = -0.15;
    return u;
  };
  mesh->addDomainOfBoundaryElements("driven_surface", by_attr<dim>(6));
  solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("driven_surface"));

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {7}, {5}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

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
    testing::Values(std::make_tuple(ContactEnforcement::Penalty, ContactType::TiedNormal, ContactJacobian::Approximate,
                                    "penalty_tiednormal_Japprox"),
                    std::make_tuple(ContactEnforcement::Penalty, ContactType::Frictionless,
                                    ContactJacobian::Approximate, "penalty_frictionless_Japprox"),
                    std::make_tuple(ContactEnforcement::LagrangeMultiplier, ContactType::TiedNormal,
                                    ContactJacobian::Approximate, "lagrange_multiplier_tiednormal_Japprox"),
                    std::make_tuple(ContactEnforcement::LagrangeMultiplier, ContactType::Frictionless,
                                    ContactJacobian::Approximate, "lagrange_multiplier_frictionless_Japprox"),
                    std::make_tuple(ContactEnforcement::Penalty, ContactType::TiedNormal, ContactJacobian::Exact,
                                    "penalty_tiednormal_Jexact"),
                    std::make_tuple(ContactEnforcement::Penalty, ContactType::Frictionless, ContactJacobian::Exact,
                                    "penalty_frictionless_Jexact"),
                    std::make_tuple(ContactEnforcement::LagrangeMultiplier, ContactType::TiedNormal,
                                    ContactJacobian::Exact, "lagrange_multiplier_tiednormal_Jexact"),
                    std::make_tuple(ContactEnforcement::LagrangeMultiplier, ContactType::Frictionless,
                                    ContactJacobian::Exact, "lagrange_multiplier_frictionless_Jexact")));

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
