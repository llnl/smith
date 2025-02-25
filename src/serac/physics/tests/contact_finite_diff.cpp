// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics_contact.hpp"

#include <functional>
#include <iomanip>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"

#include "serac/numerics/functional/domain.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

class ContactFiniteDiff : public testing::TestWithParam<std::pair<ContactEnforcement, std::string>> {};

TEST_P(ContactFiniteDiff, patch)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p = 1;
  constexpr int dim = 3;

  constexpr double eps = 1.0e-8;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string name = "contact_fd_" + GetParam().second;
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store

  double shift = eps * 10;
  // clang-format off
  auto mesh = mesh::refineAndDistribute(shared::MeshBuilder::Unify({
    shared::MeshBuilder::CubeMesh(1, 1, 1),
    shared::MeshBuilder::CubeMesh(1, 1, 1)
      // shift up height of element
      .translate({0.0, 0.0, 0.999})
      // shift x and y so the element edges are not overlapping
      .translate({shift, shift, 0.0})
      // change the mesh1 boundary attribute from 1 to 7
      .updateBdrAttrib(1, 7)
      // change the mesh1 boundary attribute from 6 to 8
      .updateBdrAttrib(6, 8)
  }), 0, 0);
  // clang-format on

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), "patch_mesh");
  pmesh.GetNodes()->Print();

  Domain x0_faces = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(5));
  Domain y0_faces = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(2));
  Domain z0_face = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(1));
  Domain zmax_face = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(8));

// TODO: investigate performance with Petsc
// #ifdef SERAC_USE_PETSC
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

  // Do a single iteration per timestep to check gradient for each iteration
  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 1.0e-15,
                                           .absolute_tol = 1.0e-15,
                                           .max_iterations = 1,
                                           .print_level = 1};

  ContactOptions contact_options{.method = ContactMethod::SingleMortar,
                                 .enforcement = GetParam().first,
                                 .type = ContactType::TiedNormal,
                                 .penalty = 1.0e3,
                                 .jacobian = ContactJacobian::Exact};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, "patch_mesh", {}, 0,
                                             0.0, false, false);

  double K = 10.0;
  double G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  Domain material_block = EntireDomain(pmesh);
  solid_solver.setMaterial(mat, material_block);

  // NOTE: Tribol will miss this contact if warm start doesn't account for contact
  // constexpr double max_disp = 0.2;
  auto nonzero_disp_bc = [](vec3, double) { return vec3{{0.0, 0.0, 0.0}}; };

  // Define a boundary attribute set and specify initial / boundary conditions
  solid_solver.setFixedBCs(x0_faces, Component::X);
  solid_solver.setFixedBCs(y0_faces, Component::Y);
  solid_solver.setFixedBCs(z0_face, Component::Z);
  solid_solver.setDisplacementBCs(nonzero_disp_bc, zmax_face, Component::Z);

  // Create a list of vdofs from Domains
  auto x0_face_dofs = x0_faces.dof_list(&solid_solver.displacement().space());
  auto y0_face_dofs = y0_faces.dof_list(&solid_solver.displacement().space());
  auto z0_face_dofs = z0_face.dof_list(&solid_solver.displacement().space());
  auto zmax_face_dofs = zmax_face.dof_list(&solid_solver.displacement().space());
  mfem::Array<int> bc_vdofs(dim *
                            (x0_face_dofs.Size() + y0_face_dofs.Size() + z0_face_dofs.Size() + zmax_face_dofs.Size()));
  int dof_ct = 0;
  for (int i{0}; i < x0_face_dofs.Size(); ++i) {
    for (int d{0}; d < dim; ++d) {
      bc_vdofs[dof_ct++] = solid_solver.displacement().space().DofToVDof(x0_face_dofs[i], d);
    }
  }
  for (int i{0}; i < y0_face_dofs.Size(); ++i) {
    for (int d{0}; d < dim; ++d) {
      bc_vdofs[dof_ct++] = solid_solver.displacement().space().DofToVDof(y0_face_dofs[i], d);
    }
  }
  for (int i{0}; i < z0_face_dofs.Size(); ++i) {
    for (int d{0}; d < dim; ++d) {
      bc_vdofs[dof_ct++] = solid_solver.displacement().space().DofToVDof(z0_face_dofs[i], d);
    }
  }
  for (int i{0}; i < zmax_face_dofs.Size(); ++i) {
    for (int d{0}; d < dim; ++d) {
      bc_vdofs[dof_ct++] = solid_solver.displacement().space().DofToVDof(zmax_face_dofs[i], d);
    }
  }
  bc_vdofs.Sort();
  bc_vdofs.Unique();

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {6}, {7}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  constexpr int n_steps = 3;
  double dt = 1.0 / static_cast<double>(n_steps);
  for (int i{0}; i < n_steps; ++i) {
    double max_diff = 0.0;
    auto oper = solid_solver.buildQuasistaticOperator();
    auto& u = solid_solver.displacement();
    auto pressure = solid_solver.pressure();
    mfem::Vector merged_sol(u.Size() + pressure.Size());
    merged_sol.SetVector(u, 0);
    merged_sol.SetVector(pressure, u.Size());
    auto* J_op = &oper->GetGradient(u);
    mfem::Vector f(merged_sol.Size());
    f = 0.0;
    oper->Mult(merged_sol, f);
    mfem::Vector u_dot(merged_sol.Size());
    u_dot = 0.0;
    dof_ct = 0;
    for (int j{0}; j < merged_sol.Size(); ++j) {
      if (dof_ct < bc_vdofs.Size() && bc_vdofs[dof_ct] == j) {
        ++dof_ct;
        continue;
      }
      u_dot[j] = 1.0;
      mfem::Vector J_exact(merged_sol.Size());
      J_exact = 0.0;
      J_op->Mult(u_dot, J_exact);
      u_dot[j] = 0.0;
      merged_sol[j] += eps;
      mfem::Vector J_fd(merged_sol.Size());
      J_fd = 0.0;
      oper->Mult(merged_sol, J_fd);
      J_fd -= f;
      J_fd /= eps;
      merged_sol[j] -= eps;
      for (int k{0}; k < merged_sol.Size(); ++k) {
        if (std::abs(J_exact[k]) > 1.0e-15 || std::abs(J_fd[k]) > 1.0e-15) {
          auto diff = std::abs(J_exact[k] - J_fd[k]);
          if (diff > max_diff) {
            max_diff = diff;
          }
          if (diff > 1.0e-5) {  // eps) {
            std::cout << "(" << j << ", " << k << "):  J_exact = " << std::setprecision(15) << J_exact[k]
                      << "   J_fd = " << std::setprecision(15) << J_fd[k] << "   |diff| = " << std::setprecision(15)
                      << diff << std::endl;
          }
        }
      }
    }
    std::cout << "Max diff = " << std::setprecision(15) << max_diff << std::endl;

    solid_solver.advanceTimestep(dt);
    solid_solver.displacement().Print(mfem::out, 17);
  }
}

INSTANTIATE_TEST_SUITE_P(tribol, ContactFiniteDiff,
                         testing::Values(  // std::make_pair(ContactEnforcement::Penalty, "penalty"),
                             std::make_pair(ContactEnforcement::LagrangeMultiplier, "lm")));

}  // namespace serac

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  int result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
