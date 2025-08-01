// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics_contact.hpp"

#include <cfenv>
#include <fem/datacollection.hpp>
#include <functional>
#include <mesh/vtk.hpp>
#include <set>
#include <string>
#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "shared/mesh/MeshBuilder.hpp"
#include "serac/mesh_utils/mesh_utils.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include <fenv.h>



namespace serac {

class ContactTest : public testing::TestWithParam<std::tuple<ContactEnforcement, ContactJacobian, std::string>> {};

TEST_P(ContactTest, patch)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p = 1;
  constexpr int dim = 2;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string name = "contact_patch_" + std::get<2>(GetParam());
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store

  auto mesh = std::make_shared<serac::Mesh>(shared::MeshBuilder::Unify({
    shared::MeshBuilder::SquareMesh(1 , 1).translate({0.0, 1.0}).bdrAttribInfo()
    .updateBdrAttrib(4, 7).updateBdrAttrib(3, 9).updateBdrAttrib(1, 6),
    shared::MeshBuilder::SquareMesh(1, 1).bdrAttribInfo().updateBdrAttrib(4, 7).updateBdrAttrib(1, 8).updateBdrAttrib(3, 5)}), "patch_mesh_2D", 0, 0);

  mfem::VisItDataCollection visit_dc("contact_patch_visit", &mesh->mfemParMesh());

  visit_dc.SetPrefixPath("visit_out");
  visit_dc.Save();


  

  mesh->addDomainOfBoundaryElements("x0_faces", serac::by_attr<dim>(7));
  mesh->addDomainOfBoundaryElements("y0_faces", serac::by_attr<dim>(8));
  mesh->addDomainOfBoundaryElements("Ymax_face", serac::by_attr<dim>(9));

  // TODO: investigate performance with Petsc
  // #ifdef SERAC_USE_PETSC
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

  ContactOptions contact_options{.method = ContactMethod::SmoothMortar,
                                 .enforcement = std::get<0>(GetParam()),
                                 .type = ContactType::Frictionless,
                                 .penalty = 0.2,
                                 .penalty2 = 0.0,
                                 .jacobian = std::get<1>(GetParam())};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, mesh);

  double K = 10.0;
  double G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat, mesh->entireBody());

  // Define the function for the initial displacement and boundary condition
  // constexpr int dim = 2;
  auto applied_disp_function = [](tensor<double, dim>, auto) { return tensor<double, dim>{{0, -0.1}}; };

  // Define a boundary attribute set and specify initial / boundary conditions
  solid_solver.setFixedBCs(mesh->domain("x0_faces"), Component::X);
  solid_solver.setFixedBCs(mesh->domain("y0_faces"), Component::Y);
  solid_solver.setDisplacementBCs(applied_disp_function, mesh->domain("Ymax_face"), Component::Y);

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {6}, {5}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);
  solid_solver.advanceTimestep(dt);
  // solid_solver.advanceTimestep(dt);
  // solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputStateToDisk(paraview_name);

  // Check the l2 norm of the displacement dofs
  // auto c = 1.0;
  auto c = (3.0 * K - 2.0 * G) / (3.0 * K + G);
  mfem::VectorFunctionCoefficient elasticity_sol_coeff(2, [c](const mfem::Vector& x, mfem::Vector& u) {
    // u[0] = 0.0;
    // u[1] = -0.01 * c * x[1];
    u[0] = c * -0.1 * x[0];
    u[1] = -0.1 * x[1];
    // u[2] = -0.5 * 0.01 * x[2];
  });
  mfem::ParFiniteElementSpace elasticity_fes(solid_solver.displacement().space());
  mfem::ParGridFunction elasticity_sol(&elasticity_fes);
  elasticity_sol.ProjectCoefficient(elasticity_sol_coeff);
  mfem::ParGridFunction approx_error(elasticity_sol);
  approx_error -= solid_solver.displacement().gridFunction();
  auto approx_error_l2 = mfem::ParNormlp(approx_error, 2, MPI_COMM_WORLD);
  EXPECT_NEAR(0.0, approx_error_l2, 1.0e-2);
}

INSTANTIATE_TEST_SUITE_P(
    tribol, ContactTest,
    testing::Values(
                    std::make_tuple(ContactEnforcement::Penalty, ContactJacobian::Exact, "penalty_exactJ")
));

}  // namespace serac

int main(int argc, char* argv[])
{

feenableexcept(FE_INVALID | FE_OVERFLOW);

  testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
