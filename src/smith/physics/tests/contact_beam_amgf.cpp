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

#include "smith/smith_config.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/contact/contact_config.hpp"

namespace smith {

class ContactTestAMGF
    : public testing::TestWithParam<std::tuple<ContactEnforcement, ContactType, ContactJacobian, std::string, bool>> {};

TEST_P(ContactTestAMGF, beam)
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

#ifdef MFEM_USE_STRUMPACK
  LinearSolverOptions linear_options{.linear_solver = LinearSolver::GMRES,
                                     .preconditioner = Preconditioner::AMGFContact,
                                     .relative_tol = 0.0,
                                     .absolute_tol = 1.0e-13,
                                     .print_level = 2};
#else
  LinearSolverOptions linear_options{};
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return;
#endif

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 0.0,
                                           .absolute_tol = 1.0e-8,
                                           .max_iterations = 20,
                                           .print_level = 1};

  ContactOptions contact_options{.method = ContactMethod::SingleMortar,
                                 .enforcement = std::get<0>(GetParam()),
                                 .type = std::get<1>(GetParam()),
                                 .penalty = 1.0e4,
                                 .jacobian = std::get<2>(GetParam())};

  std::vector<std::string> parameter_names = {};
  int cycle = 0;
  double time = 0.0;
  bool checkpoint_to_disk = false;
  bool use_warm_start = std::get<4>(GetParam());

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, mesh, parameter_names,
                                             cycle, time, checkpoint_to_disk, use_warm_start);

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
    tribol, ContactTestAMGF,
    testing::Values(std::make_tuple(ContactEnforcement::Penalty, ContactType::TiedNormal, ContactJacobian::Approximate,
                                    "penalty_tiednormal_Japprox_amgf", true),
                    std::make_tuple(ContactEnforcement::Penalty, ContactType::Frictionless,
                                    ContactJacobian::Approximate, "penalty_frictionless_Japprox_amgf", true),
                    std::make_tuple(ContactEnforcement::Penalty, ContactType::TiedNormal, ContactJacobian::Approximate,
                                    "penalty_tiednormal_Japprox_amgf_nowarmstart", false),
                    std::make_tuple(ContactEnforcement::Penalty, ContactType::Frictionless,
                                    ContactJacobian::Approximate, "penalty_frictionless_Japprox_amgf_nowarmstart",
                                    false)

                        ));

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
