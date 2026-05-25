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
 * A linear-elastic half-disk indenter (radius R = 0.5) is pressed into a
 * linear-elastic block ([-1,1] x [0,0.5]) using a small displacement BC on
 * the indenter's flat top. The indenter is made much stiffer than the block.
 * The Hertz rigid-indenter estimate is computed as a diagnostic:
 *
 *     a_hertz = sqrt(4 F R / (pi E*))
 *
 * with 1/E* = (1 - nu_block^2)/E_block (rigid indenter limit) and F the
 * total line load (sum of vertical reactions on the indenter top).
 *
 * The test checks that both energy contact formulations solve this curved
 * contact case and produce finite, positive contact diagnostics.
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

TEST_P(EnergyContactHertzian, solves_indenter_contact)
{
  constexpr int p   = 1;
  constexpr int dim = 2;

  MPI_Barrier(MPI_COMM_WORLD);

  const std::string name = "hertzian_unit_" + GetParam().second;
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  const std::string mesh_file = std::string(SMITH_REPO_DIR) + "/data/meshes/hertzian_contact.msh";

  auto mesh = std::make_shared<smith::Mesh>(mesh_file, "hertzian_unit_mesh", 0, 0);
  mesh->mfemParMesh().CheckElementOrientation(true);

  LinearSolverOptions linear_options{
      .linear_solver = LinearSolver::CG, .preconditioner = Preconditioner::HypreAMG, .print_level = 0};

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::TrustRegion,
                                           .relative_tol = 1.0e-9,
                                           .absolute_tol = 1.0e-9,
                                           .max_iterations = 500,
                                           .max_line_search_iterations = 12,
                                           .print_level = 1};

  ContactOptions contact_options{.method = GetParam().first,
                                 .enforcement = ContactEnforcement::Penalty,
                                 .type = ContactType::Frictionless,
                                 .penalty = 1.0e5,
                                 .penalty2 = 0,
                                 .jacobian = ContactJacobian::Exact};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, mesh);

  // Block (attr 1): E ~ 100, nu ~ 0.286 (K=100, G=50)
  // Indenter (attr 2): ~100x stiffer to approximate rigid-indenter Hertz limit.
  const double K_block = 100.0;
  const double G_block = 50.0;
  const double E_block = 9.0 * K_block * G_block / (3.0 * K_block + G_block);
  const double nu_block = (3.0 * K_block - 2.0 * G_block) / (2.0 * (3.0 * K_block + G_block));
  const double E_star = E_block / (1.0 - nu_block * nu_block);

  // Per-attribute materials: block is domain attr 1, indenter is attr 2.
  solid_mechanics::LinearIsotropic block_mat{1.0, K_block, G_block};
  solid_mechanics::LinearIsotropic indenter_mat{1.0, 100.0 * K_block, 100.0 * G_block};
  mesh->addDomainOfBodyElements(
      "block_body", [](std::vector<tensor<double, dim>>, int attr) { return attr == 1; });
  mesh->addDomainOfBodyElements(
      "indenter_body", [](std::vector<tensor<double, dim>>, int attr) { return attr == 2; });
  solid_solver.setMaterial(block_mat, mesh->domain("block_body"));
  solid_solver.setMaterial(indenter_mat, mesh->domain("indenter_body"));

  // Bottom of block (attr 4): fully fixed
  mesh->addDomainOfBoundaryElements("block_bottom", smith::by_attr<dim>(4));
  solid_solver.setFixedBCs(mesh->domain("block_bottom"));

  // Block sides (attr 5): roller (fix x only)
  mesh->addDomainOfBoundaryElements("block_sides", smith::by_attr<dim>(5));
  solid_solver.setFixedBCs(mesh->domain("block_sides"), Component::X);

  // Small downward displacement on indenter flat top (attr 1)
  const double delta = 5.0e-3;  // small w.r.t. R = 0.5
  auto applied_disp = [delta](tensor<double, dim> /*x*/, double /*t*/) {
    return tensor<double, dim>{{0.0, -delta}};
  };
  mesh->addDomainOfBoundaryElements("indenter_top", smith::by_attr<dim>(1));
  solid_solver.setDisplacementBCs(applied_disp, mesh->domain("indenter_top"));

  // Contact: master = block_top (attr 3), slave = indenter_arc (attr 2)
  solid_solver.addContactInteraction(0, {3}, {2}, contact_options);

  solid_solver.completeSetup();

  const std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  solid_solver.advanceTimestep(1.0);
  solid_solver.outputStateToDisk(paraview_name);

  // ── Extract total vertical reaction force on the indenter flat top ──────
  // reactions() holds the residual at constrained dofs (i.e. nodal forces
  // that the BC supplies). Sum the y-component over nodes on bdr attr 1.
  const mfem::ParLinearForm& reactions_lf = solid_solver.reactions().linearForm();
  const mfem::ParFiniteElementSpace& r_space = solid_solver.reactions().space();
  mfem::ParMesh& pmesh = mesh->mfemParMesh();
  const int n_scalar_dofs = r_space.GetNDofs();

  std::set<int> indenter_top_vdofs;
  for (int be = 0; be < pmesh.GetNBE(); ++be) {
    if (pmesh.GetBdrAttribute(be) != 1) continue;
    mfem::Array<int> verts;
    pmesh.GetBdrElementVertices(be, verts);
    for (int v : verts) indenter_top_vdofs.insert(v);
  }

  double F_local = 0.0;
  for (int v : indenter_top_vdofs) {
    // Ordering byNODES: component c is at index c*n_scalar_dofs + v
    F_local += -reactions_lf[1 * n_scalar_dofs + v];  // downward push -> reaction is upward
  }
  double F = 0.0;
  MPI_Allreduce(&F_local, &F, 1, MPI_DOUBLE, MPI_SUM, mesh->getComm());

  // ── Extract contact half-width from block_top (attr 3) nodes ────────────
  const mfem::ParGridFunction& u_gf = solid_solver.displacement().gridFunction();
  const mfem::ParFiniteElementSpace& u_space = solid_solver.displacement().space();
  const int n_u_scalar = u_space.GetNDofs();

  const double uy_threshold = -0.05 * delta;  // 5% of applied indentation
  double a_local = 0.0;
  for (int be = 0; be < pmesh.GetNBE(); ++be) {
    if (pmesh.GetBdrAttribute(be) != 3) continue;
    mfem::Array<int> verts;
    pmesh.GetBdrElementVertices(be, verts);
    for (int v : verts) {
      const double x_v  = pmesh.GetVertex(v)[0];
      const double uy_v = u_gf(1 * n_u_scalar + v);
      if (uy_v < uy_threshold) {
        a_local = std::max(a_local, std::abs(x_v));
      }
    }
  }
  double a_num = 0.0;
  MPI_Allreduce(&a_local, &a_num, 1, MPI_DOUBLE, MPI_MAX, mesh->getComm());

  const double R       = 0.5;
  const double a_hertz = std::sqrt(4.0 * F * R / (M_PI * E_star));

  int rank = 0;
  MPI_Comm_rank(mesh->getComm(), &rank);
  if (rank == 0) {
    std::cout << "Hertzian unit test (" << GetParam().second << "):\n"
              << "  E_block    = " << E_block << "\n"
              << "  nu_block   = " << nu_block << "\n"
              << "  E*         = " << E_star << "\n"
              << "  delta      = " << delta << "\n"
              << "  F (line)   = " << F << "\n"
              << "  a_hertz    = " << a_hertz << "\n"
              << "  a_num      = " << a_num << "\n"
              << "  rel err    = " << std::abs(a_num - a_hertz) / a_hertz << "\n";
  }

  EXPECT_GT(F, 0.0);
  EXPECT_GT(a_num, 0.0);
  EXPECT_TRUE(std::isfinite(a_hertz));
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
