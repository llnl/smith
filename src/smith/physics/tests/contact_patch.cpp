// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <tuple>
#include <memory>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
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

class ContactTest : public testing::TestWithParam<std::tuple<ContactEnforcement, ContactJacobian, std::string>> {};

TEST_P(ContactTest, patch)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p = 1;
  constexpr int dim = 3;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string name = "contact_patch_" + std::get<2>(GetParam());
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/twohex_for_contact.mesh";

  auto mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), "patch_mesh", 2, 0);

  mesh->addDomainOfBoundaryElements("x0_faces", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("y0_faces", smith::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("z0_face", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("zmax_face", smith::by_attr<dim>(6));

  // TODO: investigate performance with Petsc
  // #ifdef SMITH_USE_PETSC
  //   LinearSolverOptions linear_options{
  //       .linear_solver = LinearSolver::PetscGMRES,
  //       .preconditioner = Preconditioner::Petsc,
  //       .petsc_preconditioner = PetscPCType::HMG,
  //       .absolute_tol = 1e-16,
  //       .print_level = 1,
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
                                 .enforcement = std::get<0>(GetParam()),
                                 .type = ContactType::Frictionless,
                                 .penalty = 8.0e2,
                                 .jacobian = std::get<1>(GetParam())};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, mesh);

  double K = 10.0;
  double G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat, mesh->entireBody());

  // Define the function for the initial displacement and boundary condition
  auto applied_disp_function = [](tensor<double, dim>, auto) { return tensor<double, dim>{{0, 0, -0.01}}; };

  // Define a boundary attribute set and specify initial / boundary conditions
  solid_solver.setFixedBCs(mesh->domain("x0_faces"), Component::X);
  solid_solver.setFixedBCs(mesh->domain("y0_faces"), Component::Y);
  solid_solver.setFixedBCs(mesh->domain("z0_face"), Component::Z);
  solid_solver.setDisplacementBCs(applied_disp_function, mesh->domain("zmax_face"), Component::Z);

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {4}, {5}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputStateToDisk(paraview_name);

  // Check the l2 norm of the displacement dofs
  auto c = (3.0 * K - 2.0 * G) / (3.0 * K + G);
  mfem::VectorFunctionCoefficient elasticity_sol_coeff(3, [c](const mfem::Vector& x, mfem::Vector& u) {
    u[0] = 0.25 * 0.01 * c * x[0];
    u[1] = 0.25 * 0.01 * c * x[1];
    u[2] = -0.5 * 0.01 * x[2];
  });
  mfem::ParFiniteElementSpace elasticity_fes(solid_solver.reactions().space());
  mfem::ParGridFunction elasticity_sol(&elasticity_fes);
  elasticity_sol.ProjectCoefficient(elasticity_sol_coeff);
  mfem::ParGridFunction approx_error(elasticity_sol);
  approx_error -= solid_solver.displacement().gridFunction();
  auto approx_error_l2 = mfem::ParNormlp(approx_error, 2, MPI_COMM_WORLD);
  EXPECT_NEAR(0.0, approx_error_l2, 1.0e-3);
}

INSTANTIATE_TEST_SUITE_P(
    tribol, ContactTest,
    testing::Values(std::make_tuple(ContactEnforcement::Penalty, ContactJacobian::Approximate, "penalty_approxJ"),
                    std::make_tuple(ContactEnforcement::LagrangeMultiplier, ContactJacobian::Approximate,
                                    "lagrange_multiplier_approxJ"),
                    std::make_tuple(ContactEnforcement::Penalty, ContactJacobian::Exact, "penalty_exactJ"),
                    std::make_tuple(ContactEnforcement::LagrangeMultiplier, ContactJacobian::Exact,
                                    "lagrange_multiplier_exactJ")));

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
