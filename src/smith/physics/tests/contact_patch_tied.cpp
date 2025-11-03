// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <memory>
#include <utility>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/solid_mechanics.hpp"

namespace smith {

class ContactPatchTied : public testing::TestWithParam<std::pair<ContactEnforcement, std::string>> {};

TEST_P(ContactPatchTied, patch)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p = 1;
  constexpr int dim = 3;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string name = "contact_patch_" + GetParam().second;
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/twohex_for_contact.mesh";

  auto mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), "patch_mesh", 3, 0);

  mesh->addDomainOfBoundaryElements("x0_faces", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("y0_faces", smith::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("z0_face", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("zmax_face", smith::by_attr<dim>(6));

// TODO: investigate performance with Petsc
// #ifdef SMITH_USE_PETSC
//   LinearSolverOptions linear_options{
//       .linear_solver        = LinearSolver::PetscGMRES,
//       .preconditioner       = Preconditioner::Petsc,
//       .petsc_preconditioner = PetscPCType::HMG,
//       .absolute_tol         = 1e-12,
//       .print_level          = 1,
//   };
// #elif defined(MFEM_USE_STRUMPACK)
#ifdef MFEM_USE_STRUMPACK
  LinearSolverOptions linear_options{.linear_solver = LinearSolver::Strumpack, .print_level = 0};
#else
  LinearSolverOptions linear_options{};
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return;
#endif

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 1.0e-13,
                                           .absolute_tol = 1.0e-13,
                                           .max_iterations = 20,
                                           .print_level = 1};

  ContactOptions contact_options{.method = ContactMethod::SingleMortar,
                                 .enforcement = GetParam().first,
                                 .type = ContactType::TiedNormal,
                                 .penalty = 8.0e2,
                                 .jacobian = ContactJacobian::Exact};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, mesh);

  double K = 10.0;
  double G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat, mesh->entireBody());

  // NOTE: Tribol will miss this contact if warm start doesn't account for contact
  constexpr double max_disp = 0.2;
  auto nonzero_disp_bc = [](vec3, double t) { return vec3{{0.0, 0.0, -max_disp * t}}; };

  // Define a boundary attribute set and specify initial / boundary conditions
  solid_solver.setFixedBCs(mesh->domain("x0_faces"), Component::X);
  solid_solver.setFixedBCs(mesh->domain("y0_faces"), Component::Y);
  solid_solver.setFixedBCs(mesh->domain("z0_face"), Component::Z);
  solid_solver.setDisplacementBCs(nonzero_disp_bc, mesh->domain("zmax_face"), Component::Z);

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {4}, {5}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  constexpr int n_steps = 1;
  double dt = 1.0 / static_cast<double>(n_steps);
  for (int i{0}; i < n_steps; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  // Check the l2 norm of the displacement dofs
  auto c = (3.0 * K - 2.0 * G) / (3.0 * K + G);
  mfem::VectorFunctionCoefficient elasticity_sol_coeff(3, [c](const mfem::Vector& x, mfem::Vector& u) {
    u[0] = 0.25 * max_disp * c * x[0];
    u[1] = 0.25 * max_disp * c * x[1];
    u[2] = -0.5 * max_disp * x[2];
  });
  mfem::ParFiniteElementSpace elasticity_fes(solid_solver.displacement().space());
  mfem::ParGridFunction elasticity_sol(&elasticity_fes);
  elasticity_sol.ProjectCoefficient(elasticity_sol_coeff);
  mfem::ParGridFunction approx_error(elasticity_sol);
  approx_error -= solid_solver.displacement().gridFunction();
  auto approx_error_l2 = mfem::ParNormlp(approx_error, 2, MPI_COMM_WORLD);
  // At 10% strain, linear elastic approximation breaks down, so larger error is expected here.
  EXPECT_NEAR(0.0, approx_error_l2, 0.13);
}

INSTANTIATE_TEST_SUITE_P(tribol, ContactPatchTied,
                         testing::Values(std::make_pair(ContactEnforcement::Penalty, "penalty"),
                                         std::make_pair(ContactEnforcement::LagrangeMultiplier,
                                                        "lagrange_multiplier")));

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
