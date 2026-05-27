// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file energy_mortar_hertzian.cpp
 *
 * Small-displacement 2D Hertzian-style contact smoke test for the smoothed
 * energy contact formulations.
 *
 * A stiff linear-elastic half-disk indenter (radius R = 1.0) is pressed into a
 * linear-elastic block using load control on the indenter's flat top. The
 * target is the 2D Hertz line-contact half-width:
 *
 *     a_hertz = sqrt(4 F R / (pi E*))
 *
 * with 1/E* = (1 - nu_block^2)/E_block + (1 - nu_indenter^2)/E_indenter and F
 * the applied line load.
 *
 * The mesh is finite, so this is an approximate regression test rather than a
 * strict analytic benchmark.
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <string>

#include <gtest/gtest.h>
#include <mpi.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith.hpp"
#include "smith/smith_config.hpp"

namespace smith {

class EnergyContactHertzian : public testing::TestWithParam<std::pair<ContactMethod, std::string>> {};

double boundaryLength(mfem::ParMesh& mesh, int boundary_attr)
{
  double local_length = 0.0;
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    if (mesh.GetBdrAttribute(be) != boundary_attr) continue;

    mfem::Array<int> verts;
    mesh.GetBdrElementVertices(be, verts);
    SLIC_ERROR_ROOT_IF(verts.Size() != 2, "Hertzian test expects linear boundary segments.");

    const auto* x0 = mesh.GetVertex(verts[0]);
    const auto* x1 = mesh.GetVertex(verts[1]);
    const double dx = x1[0] - x0[0];
    const double dy = x1[1] - x0[1];
    local_length += std::sqrt(dx * dx + dy * dy);
  }

  double length = 0.0;
  MPI_Allreduce(&local_length, &length, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
  return length;
}

double contactHalfWidthFromNodalForces(const FiniteElementDual& contact_forces, mfem::ParMesh& mesh,
                                       const mfem::ParFiniteElementSpace& space, int boundary_attr)
{
  const mfem::ParLinearForm& contact_force_lf = contact_forces.linearForm();

  double max_force_local = 0.0;
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    if (mesh.GetBdrAttribute(be) != boundary_attr) continue;

    mfem::Array<int> dofs;
    space.GetBdrElementDofs(be, dofs);
    for (int i = 0; i < dofs.Size(); ++i) {
      const int y_vdof = space.DofToVDof(dofs[i], 1);
      max_force_local = std::max(max_force_local, std::abs(contact_force_lf[y_vdof]));
    }
  }

  double max_force = 0.0;
  MPI_Allreduce(&max_force_local, &max_force, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());

  const double active_threshold = 1.0e-3 * max_force;
  double half_width_local = 0.0;
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    if (mesh.GetBdrAttribute(be) != boundary_attr) continue;

    mfem::Array<int> dofs;
    space.GetBdrElementDofs(be, dofs);
    for (int i = 0; i < dofs.Size(); ++i) {
      const int y_vdof = space.DofToVDof(dofs[i], 1);
      if (std::abs(contact_force_lf[y_vdof]) <= active_threshold) continue;

      const int dof = dofs[i];
      const double x = (*mesh.GetNodes())(mesh.GetNodalFESpace()->DofToVDof(dof, 0));
      half_width_local = std::max(half_width_local, std::abs(x));
    }
  }

  double half_width = 0.0;
  MPI_Allreduce(&half_width_local, &half_width, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
  return half_width;
}

double relativeContactSurfaceDisplacementError(const mfem::ParGridFunction& displacement, mfem::ParMesh& mesh,
                                               const mfem::ParFiniteElementSpace& space, int boundary_attr,
                                               double contact_half_width, double radius)
{
  std::set<int> contact_dofs;
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    if (mesh.GetBdrAttribute(be) != boundary_attr) continue;

    mfem::Array<int> dofs;
    space.GetBdrElementDofs(be, dofs);
    for (int i = 0; i < dofs.Size(); ++i) {
      const int dof = dofs[i];
      const double x = (*mesh.GetNodes())(mesh.GetNodalFESpace()->DofToVDof(dof, 0));
      if (std::abs(x) <= contact_half_width) {
        contact_dofs.insert(dof);
      }
    }
  }

  double local_count = 0.0;
  double local_offset_sum = 0.0;
  double local_profile_sum = 0.0;
  for (int dof : contact_dofs) {
    const double x = (*mesh.GetNodes())(mesh.GetNodalFESpace()->DofToVDof(dof, 0));
    const double uy = displacement(space.DofToVDof(dof, 1));
    const double profile = x * x / (2.0 * radius);
    local_count += 1.0;
    local_offset_sum += uy - profile;
    local_profile_sum += profile;
  }

  double count = 0.0;
  double offset_sum = 0.0;
  double profile_sum = 0.0;
  MPI_Allreduce(&local_count, &count, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
  MPI_Allreduce(&local_offset_sum, &offset_sum, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
  MPI_Allreduce(&local_profile_sum, &profile_sum, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
  SLIC_ERROR_ROOT_IF(count <= 1.0, "Not enough contact nodes for Hertzian displacement-profile error.");

  const double offset = offset_sum / count;
  const double mean_profile = profile_sum / count;

  double local_error2 = 0.0;
  double local_norm2 = 0.0;
  for (int dof : contact_dofs) {
    const double x = (*mesh.GetNodes())(mesh.GetNodalFESpace()->DofToVDof(dof, 0));
    const double uy = displacement(space.DofToVDof(dof, 1));
    const double profile = x * x / (2.0 * radius);
    const double exact_uy = offset + profile;
    local_error2 += (uy - exact_uy) * (uy - exact_uy);
    local_norm2 += (profile - mean_profile) * (profile - mean_profile);
  }

  double error2 = 0.0;
  double norm2 = 0.0;
  MPI_Allreduce(&local_error2, &error2, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
  MPI_Allreduce(&local_norm2, &norm2, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
  return std::sqrt(error2 / norm2);
}

TEST_P(EnergyContactHertzian, solves_indenter_contact)
{
  constexpr int p   = 1;
  constexpr int dim = 2;

  MPI_Barrier(MPI_COMM_WORLD);

  const std::string name = "hertzian_unit_" + GetParam().second;
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  const std::string mesh_file = std::string(SMITH_REPO_DIR) + "/data/meshes/hertzian_contact_large.msh";

  auto mesh = std::make_shared<smith::Mesh>(mesh_file, "hertzian_unit_mesh", 0, 0);
  mesh->mfemParMesh().CheckElementOrientation(true);

  LinearSolverOptions linear_options{
      .linear_solver = LinearSolver::CG, .preconditioner = Preconditioner::HypreJacobi, .max_iterations = 10000, .print_level = 0};

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::TrustRegion,
                                           .relative_tol = 1.0e-9,
                                           .absolute_tol = 1.0e-9,
                                           .max_iterations = 1000,
                                           .max_line_search_iterations = 12,
                                           .print_level = 2};

  ContactOptions contact_options{.method = GetParam().first,
                                 .enforcement = ContactEnforcement::Penalty,
                                 .type = ContactType::Frictionless,
                                 .penalty = 3e6, //2.0e5,
                                 .penalty2 = 0,
                                 .jacobian = ContactJacobian::Exact,
                                 .penalty_smoothing_del = 1e-3};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, mesh);

  // Block (attr 1): E ~ 100, nu ~ 0.286 (K=100, G=50)
  // Indenter (attr 2): 100x stiffer, but still included in E*.
  const double K_block = 100.0;
  const double G_block = 50.0;
  const double E_block = 9.0 * K_block * G_block / (3.0 * K_block + G_block);
  const double nu_block = (3.0 * K_block - 2.0 * G_block) / (2.0 * (3.0 * K_block + G_block));
  const double K_indenter = 100.0 * K_block;
  const double G_indenter = 100.0 * G_block;
  const double E_indenter = 9.0 * K_indenter * G_indenter / (3.0 * K_indenter + G_indenter);
  const double nu_indenter =
      (3.0 * K_indenter - 2.0 * G_indenter) / (2.0 * (3.0 * K_indenter + G_indenter));
  const double E_star =
      1.0 / ((1.0 - nu_block * nu_block) / E_block + (1.0 - nu_indenter * nu_indenter) / E_indenter);
  constexpr double top_traction = 2.0;
  const double applied_line_load = top_traction * boundaryLength(mesh->mfemParMesh(), 1);

  // Per-attribute materials: block is domain attr 1, indenter is attr 2.
  solid_mechanics::LinearIsotropic block_mat{1.0, K_block, G_block};
  solid_mechanics::LinearIsotropic indenter_mat{1.0, K_indenter, G_indenter};
  mesh->addDomainOfBodyElements(
      "block_body", [](std::vector<tensor<double, dim>>, int attr) { return attr == 1; });
  mesh->addDomainOfBodyElements(
      "indenter_body", [](std::vector<tensor<double, dim>>, int attr) { return attr == 2; });
  solid_solver.setMaterial(block_mat, mesh->domain("block_body"));
  solid_solver.setMaterial(indenter_mat, mesh->domain("indenter_body"));

  // Bottom of block (attr 4): fully fixed.
  mesh->addDomainOfBoundaryElements("block_bottom", smith::by_attr<dim>(4));
  solid_solver.setFixedBCs(mesh->domain("block_bottom"));

  // Block sides are far enough from the target contact patch to act as approximate rollers.
  mesh->addDomainOfBoundaryElements("block_sides", smith::by_attr<dim>(5));
  solid_solver.setFixedBCs(mesh->domain("block_sides"), Component::X);

  // Keep the indenter centered while load control determines the vertical displacement.
  mesh->addDomainOfBoundaryElements("indenter_top", smith::by_attr<dim>(1));
  solid_solver.setFixedBCs(mesh->domain("indenter_top"), Component::X);

  solid_solver.setTraction(
      [](auto /*x*/, auto n, double t) { return -(t * top_traction) * n; },
      mesh->domain("indenter_top"));

  // Contact: master = block_top (attr 3), slave = indenter_arc (attr 2)
  solid_solver.addContactInteraction(0, {3}, {2}, contact_options);

  solid_solver.completeSetup();

  const std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  solid_solver.advanceTimestep(1.0);
  solid_solver.outputStateToDisk(paraview_name);

  mfem::ParMesh& pmesh = mesh->mfemParMesh();
  const mfem::ParFiniteElementSpace& u_space = solid_solver.displacement().space();

  const double F = applied_line_load;
  const double R = 1.0;
  const double a_hertz = std::sqrt(4.0 * F * R / (M_PI * E_star));
  const double a_num = contactHalfWidthFromNodalForces(solid_solver.dual("contact_force_0"), pmesh, u_space, 3);
  const double displacement_profile_error = relativeContactSurfaceDisplacementError(
      solid_solver.displacement().gridFunction(), pmesh, u_space, 3, a_hertz, R);

  int rank = 0;
  MPI_Comm_rank(mesh->getComm(), &rank);
  if (rank == 0) {
    std::cout << "Hertzian unit test (" << GetParam().second << "):\n"
              << "  E_block    = " << E_block << "\n"
              << "  nu_block   = " << nu_block << "\n"
              << "  E*         = " << E_star << "\n"
              << "  F (line)   = " << F << "\n"
              << "  a_hertz    = " << a_hertz << "\n"
              << "  a_num      = " << a_num << "\n"
              << "  rel err    = " << std::abs(a_num - a_hertz) / a_hertz << "\n"
              << "  disp err   = " << displacement_profile_error << "\n";
  }

  EXPECT_GT(F, 0.0);
  EXPECT_GT(a_num, 0.0);
  EXPECT_TRUE(std::isfinite(a_hertz));
  EXPECT_NEAR(a_num, a_hertz, 0.1 * a_hertz);
  EXPECT_LT(displacement_profile_error, 0.1);
}

INSTANTIATE_TEST_SUITE_P(tribol, EnergyContactHertzian,
                         testing::Values(std::make_pair(ContactMethod::EnergyMortar, "energy_mortar"),
                                         std::make_pair(ContactMethod::EnergyAreaPenalty, "energy_area_penalty")));

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
