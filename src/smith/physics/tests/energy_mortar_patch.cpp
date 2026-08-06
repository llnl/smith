// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <string>

#include "axom/sidre.hpp"
#include "axom/slic.hpp"
#include "gtest/gtest.h"
#include "mfem.hpp"
#include "mpi.h"
#include "shared/mesh/MeshBuilder.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "tribol/interface/tribol.hpp"

namespace smith {

TEST(EnergyMortarPatch, patch)
{
  constexpr int p = 1;
  constexpr int dim = 2;

  MPI_Barrier(MPI_COMM_WORLD);

  const std::string name = "energy_mortar_patch";
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  auto mesh = std::make_shared<smith::Mesh>(
      shared::MeshBuilder::Unify(
          {shared::MeshBuilder::SquareMesh(20, 20)
               .translate({0.0, 1.0})
               .updateBdrAttrib(4, 7)
               .updateBdrAttrib(3, 9)
               .updateBdrAttrib(1, 6),
           shared::MeshBuilder::SquareMesh(20, 20).updateBdrAttrib(4, 7).updateBdrAttrib(1, 8).updateBdrAttrib(3, 5)}),
      "patch_mesh_2D", 0, 0);

  mesh->addDomainOfBoundaryElements("x0_faces", smith::by_attr<dim>(7));
  mesh->addDomainOfBoundaryElements("y0_faces", smith::by_attr<dim>(8));
  mesh->addDomainOfBoundaryElements("ymax_face", smith::by_attr<dim>(9));

#ifdef MFEM_USE_STRUMPACK
  LinearSolverOptions linear_options{.linear_solver = LinearSolver::Strumpack, .print_level = 0};
#else
  LinearSolverOptions linear_options{};
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return;
#endif

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::TrustRegion,
                                           .relative_tol = 1.0e-8,
                                           .absolute_tol = 1.0e-10,
                                           .max_iterations = 500,
                                           .print_level = 0};

  ContactOptions contact_options{.method = ContactMethod::EnergyMortar,
                                 .enforcement = ContactEnforcement::Penalty,
                                 .type = ContactType::Frictionless,
                                 .penalty = 100000.0,
                                 .jacobian = ContactJacobian::Exact};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, mesh);

  constexpr double K = 1000.0;
  constexpr double G = 10.0;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat, mesh->entireBody());

  auto applied_disp_function = [](tensor<double, dim>, auto) { return tensor<double, dim>{{0.0, -0.01}}; };

  solid_solver.setFixedBCs(mesh->domain("x0_faces"), Component::X);
  solid_solver.setFixedBCs(mesh->domain("y0_faces"), Component::Y);
  solid_solver.setDisplacementBCs(applied_disp_function, mesh->domain("ymax_face"), Component::Y);
  solid_solver.addContactInteraction(0, {6}, {5}, contact_options);
  tribol::setEnergyMortarEnforcementOption(0, tribol::EnergyMortarEnforcementOption::NodalGap);
  solid_solver.completeSetup();

  solid_solver.advanceTimestep(1.0);

  constexpr double c = (3.0 * K - 2.0 * G) / (3.0 * K + 2.0 * G);
  mfem::VectorFunctionCoefficient elasticity_sol_coeff(2, [](const mfem::Vector& x, mfem::Vector& u) {
    u[0] = 0.005 * c * x[0];
    u[1] = -0.005 * x[1];
  });

  const mfem::ParFiniteElementSpace& u_space_const = solid_solver.displacement().space();
  auto& u_space = const_cast<mfem::ParFiniteElementSpace&>(u_space_const);
  mfem::ParGridFunction U_exact(&u_space);
  U_exact.ProjectCoefficient(elasticity_sol_coeff);

  const mfem::ParGridFunction& U_num = solid_solver.displacement().gridFunction();
  mfem::ParGridFunction U_err(U_exact);
  U_err -= U_num;
  const double L2_err_vec = mfem::ParNormlp(U_err, 2, MPI_COMM_WORLD);

  const mfem::FiniteElementCollection* fec = u_space.FEColl();
  mfem::ParFiniteElementSpace scalar_fes(&mesh->mfemParMesh(), fec, /*vdim=*/1, u_space.GetOrdering());
  mfem::ParGridFunction ux_ex(&scalar_fes), ux_num(&scalar_fes), uy_ex(&scalar_fes), uy_num(&scalar_fes);

  const int n = scalar_fes.GetNDofs();
  for (int i = 0; i < n; ++i) {
    ux_ex(i) = U_exact(i);
    ux_num(i) = U_num(i);
    uy_ex(i) = U_exact(n + i);
    uy_num(i) = U_num(n + i);
  }

  mfem::ParGridFunction ux_err(ux_ex);
  mfem::ParGridFunction uy_err(uy_ex);
  ux_err -= ux_num;
  uy_err -= uy_num;

  EXPECT_NEAR(0.0, L2_err_vec, 1.0e-2);
  EXPECT_NEAR(0.0, mfem::ParNormlp(ux_err, 2, MPI_COMM_WORLD), 1.0e-2);
  EXPECT_NEAR(0.0, mfem::ParNormlp(uy_err, 2, MPI_COMM_WORLD), 1.0e-2);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
