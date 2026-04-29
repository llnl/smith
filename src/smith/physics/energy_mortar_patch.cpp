// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/solid_mechanics_contact.hpp"

#include <fem/datacollection.hpp>
#include <functional>
#include <mesh/vtk.hpp>
#include <mfem/fem/coefficient.hpp>
#include <mfem/fem/fe_coll.hpp>
#include <mfem/fem/pfespace.hpp>
#include <mfem/fem/pgridfunc.hpp>
#include <mfem/linalg/hypre.hpp>
#include <set>
#include <src/smith/physics/contact/contact_config.hpp>
#include <string>
#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include <mpi.h>
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

// static void enable_fpe() {
//   // trap on invalid ops (NaN), divide-by-zero, and overflow
//   feclearexcept(FE_ALL_EXCEPT);
//   feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

// }

namespace smith {

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

  auto mesh = std::make_shared<smith::Mesh>(shared::MeshBuilder::Unify({shared::MeshBuilder::SquareMesh(20, 20)
                                                                            .translate({0.0, 1.0})
                                                                            .bdrAttribInfo()
                                                                            .updateBdrAttrib(4, 7)
                                                                            .updateBdrAttrib(3, 9)
                                                                            .updateBdrAttrib(1, 6),
                                                                        shared::MeshBuilder::SquareMesh(20, 20)
                                                                            .bdrAttribInfo()
                                                                            .updateBdrAttrib(4, 7)
                                                                            .updateBdrAttrib(1, 8)
                                                                            .updateBdrAttrib(3, 5)}),
                                            "patch_mesh_2D", 0, 0);

  mfem::VisItDataCollection visit_dc("contact_patch_visit", &mesh->mfemParMesh());

  visit_dc.SetPrefixPath("visit_out");
  visit_dc.Save();

  mesh->addDomainOfBoundaryElements("x0_faces", smith::by_attr<dim>(7));
  mesh->addDomainOfBoundaryElements("y0_faces", smith::by_attr<dim>(8));
  mesh->addDomainOfBoundaryElements("Ymax_face", smith::by_attr<dim>(9));

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


  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = 500,
                                                  .print_level = 1};

  smith::ContactOptions contact_options{.method = smith::ContactMethod::EnergyMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = 100000,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  //   SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
  //                                              solid_mechanics::default_quasistatic_options, name, mesh);

  double K = 1000.0;
  double G = 10;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat, mesh->entireBody());

  // Define the function for the initial displacement and boundary condition
  // constexpr int dim = 2;
  auto applied_disp_function = [](tensor<double, dim>, auto) { return tensor<double, dim>{{0, -0.01}}; };

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
  // solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  solid_solver.outputStateToDisk(paraview_name);

  // Check the l2 norm of the displacement dofs
  auto c = (3.0 * K - 2.0 * G) / ((3.0 * K + 2 * G));
  // auto c = 0.0;
  mfem::VectorFunctionCoefficient elasticity_sol_coeff(2, [c](const mfem::Vector& x, mfem::Vector& u) {
    u[0] = 0.005 * c * x[0];
    u[1] = -0.005 * x[1];
    // u[2] = -0.5 * 0.01 * x[2];
  });
  mfem::ParFiniteElementSpace elasticity_fes(solid_solver.reactions().space());
  mfem::ParGridFunction elasticity_sol(&elasticity_fes);
  elasticity_sol.ProjectCoefficient(elasticity_sol_coeff);

  // Set up test to only look at y component of error*********
  const mfem::ParFiniteElementSpace& u_space_const = solid_solver.displacement().space();
  auto& u_space = const_cast<mfem::ParFiniteElementSpace&>(u_space_const);
  mfem::ParGridFunction U_exact(&u_space);
  U_exact.ProjectCoefficient(elasticity_sol_coeff);

  // Numerical displacement
  const mfem::ParGridFunction& U_num = solid_solver.displacement().gridFunction();

  // Overall Error
  mfem::ParGridFunction U_err(U_exact);
  U_err -= U_num;
  const double L2_err_vec = mfem::ParNormlp(U_err, 2, MPI_COMM_WORLD);
  std::cout << "L2_err_vec = " << L2_err_vec << std::endl;

  // y-component error
  const mfem::FiniteElementCollection* fec = u_space.FEColl();
  mfem::ParFiniteElementSpace y_fes(&mesh->mfemParMesh(), fec, /*vdim=*/1,
                                    u_space.GetOrdering());  // builds scalar space on same mesh

  mfem::ParGridFunction uy_ex(&y_fes), uy_num(&y_fes);
  const int n = y_fes.GetNDofs();

  for (int i = 0; i < n; ++i) {
    uy_ex(i) = U_exact(n * 1 + i);
    uy_num(i) = U_num(n * 1 + i);
  }

  // Same thing for x forces.
  mfem::ParGridFunction ux_ex(&y_fes), ux_num(&y_fes);

  for (int i = 0; i < n; ++i) {
    ux_ex(i) = U_exact(i);
    ux_num(i) = U_num(i);
  }

  mfem::ParGridFunction uy_err(uy_ex);
  mfem::ParGridFunction ux_err(ux_ex);
  uy_err -= uy_num;
  ux_err -= ux_num;
  const double L2_err_y = mfem::ParNormlp(uy_err, 2, MPI_COMM_WORLD);
  const double L2_err_x = mfem::ParNormlp(ux_err, 2, MPI_COMM_WORLD);
  std::cout << "L2_err_y   = " << L2_err_y << std::endl;
  std::cout << "L2_err_x   = " << L2_err_x << std::endl;

  EXPECT_NEAR(0.0, L2_err_vec, 1e-2);
  EXPECT_NEAR(0.0, L2_err_y, 1e-2);
  EXPECT_NEAR(0.0, L2_err_x, 1e-2);

  std::cout << "check = " << std::abs(L2_err_vec * L2_err_vec - (L2_err_x * L2_err_x + L2_err_y * L2_err_y)) << "\n";
}

INSTANTIATE_TEST_SUITE_P(tribol, ContactTest,
                         testing::Values(std::make_tuple(ContactEnforcement::Penalty, ContactJacobian::Exact,
                                                         "penalty_approxJ")));
// std::make_tuple(ContactEnforcement::Penalty, ContactJacobian::Exact, "penalty_exactJ")));

}  // namespace smith

int main(int argc, char* argv[])
{
  // enable_fpe();
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
