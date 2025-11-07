// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/solid_mechanics_contact.hpp"

#include <functional>
#include <iomanip>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include "gtest/gtest.h"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"

#include "smith/smith_config.hpp"

#ifndef SMITH_USE_ENZYME
#error "This file requires Enzyme to be enabled
#endif

#include "smith/numerics/functional/domain.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace smith {

class ContactFiniteDiff : public testing::TestWithParam<std::pair<ContactEnforcement, std::string>> {};

TEST_P(ContactFiniteDiff, patch)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p = 1;
  constexpr int dim = 3;

  constexpr double eps = 1.0e-7;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string name = "contact_fd_" + GetParam().second;
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store

  double shift = eps * 10;
  // clang-format off
  auto mesh = std::make_shared<smith::Mesh>(shared::MeshBuilder::Unify({
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
  }), "patch_mesh", 0, 0);
  // clang-format on

  mesh->addDomainOfBoundaryElements("x0_faces", smith::by_attr<dim>(5));
  mesh->addDomainOfBoundaryElements("y0_faces", smith::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("z0_face", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("zmax_face", smith::by_attr<dim>(8));

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
                                 .penalty = 1.0,
                                 .jacobian = ContactJacobian::Exact};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, mesh, {}, 0, 0.0,
                                             false, false);

  double K = 10.0;
  double G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  solid_solver.setMaterial(mat, mesh->entireBody());

  auto nonzero_disp_bc = [](vec3, double) { return vec3{{0.0, 0.0, 0.0}}; };

  // Define a boundary attribute set and specify initial / boundary conditions
  solid_solver.setFixedBCs(mesh->domain("x0_faces"), Component::X);
  solid_solver.setFixedBCs(mesh->domain("y0_faces"), Component::Y);
  solid_solver.setFixedBCs(mesh->domain("z0_face"), Component::Z);
  solid_solver.setDisplacementBCs(nonzero_disp_bc, mesh->domain("zmax_face"), Component::Z);

  // Create a list of vdofs from Domains
  auto x0_face_dofs = mesh->domain("x0_faces").dof_list(&solid_solver.displacement().space());
  auto y0_face_dofs = mesh->domain("y0_faces").dof_list(&solid_solver.displacement().space());
  auto z0_face_dofs = mesh->domain("z0_face").dof_list(&solid_solver.displacement().space());
  auto zmax_face_dofs = mesh->domain("zmax_face").dof_list(&solid_solver.displacement().space());
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

  auto oper = solid_solver.buildQuasistaticOperator();

  // Perform the quasi-static solve
  constexpr int n_steps = 3;
  double dt = 1.0 / static_cast<double>(n_steps);
  for (int i{0}; i < n_steps; ++i) {
    double max_diff = 0.0;
    auto& u = solid_solver.displacement();
    auto pressure = solid_solver.pressure();
    mfem::Vector merged_sol(u.Size() + pressure.Size());
    merged_sol.SetVector(u, 0);
    merged_sol.SetVector(pressure, u.Size());
    mfem::Vector f(merged_sol.Size());
    f = 0.0;
    oper->Mult(merged_sol, f);
    auto* J_op = &oper->GetGradient(merged_sol);
    mfem::Vector u_dot(merged_sol.Size());
    u_dot = 0.0;
    // wiggle displacement (col = j)
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
      // loop through forces (row = k)
      for (int k{0}; k < merged_sol.Size(); ++k) {
        if (J_exact[k] != 1.0 && (std::abs(J_exact[k]) > 1.0e-15 || std::abs(J_fd[k]) > 1.0e-15)) {
          auto diff = std::abs(J_exact[k] - J_fd[k]);
          if (diff > max_diff) {
            max_diff = diff;
          }
          if (diff > eps) {
            std::cout << "(" << k << ", " << j << "):  J_exact = " << std::setprecision(15) << J_exact[k]
                      << "   J_fd = " << std::setprecision(15) << J_fd[k] << "   |diff| = " << std::setprecision(15)
                      << diff << std::endl;
          }
          EXPECT_NEAR(J_exact[k], J_fd[k], eps);
        }
      }
    }
    std::cout << "Max diff = " << std::setprecision(15) << max_diff << std::endl;

    solid_solver.advanceTimestep(dt);
  }
}

INSTANTIATE_TEST_SUITE_P(tribol, ContactFiniteDiff,
                         testing::Values(std::make_pair(ContactEnforcement::Penalty, "penalty"),
                                         std::make_pair(ContactEnforcement::LagrangeMultiplier, "lm")));

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
