// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

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
#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/EnergyMortar.hpp"

// static void enable_fpe() {
//   // trap on invalid ops (NaN), divide-by-zero, and overflow
//   feclearexcept(FE_ALL_EXCEPT);
//   feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

// }

TEST(H1EnergyMortarTotalDerivativeCheck, GtildeFDvsAD)
{
  tribol::RealT x1[3] = {0.0, 1.0, 2.0};
  tribol::RealT y1[3] = {0.0, 0.0, 0.1};
  tribol::RealT x2[3] = {0.2, 0.9, 1.6};
  tribol::RealT y2[3] = {0.55, 0.52, 0.62};

  tribol::IndexT conn1[4] = {1, 0, 2, 1};
  tribol::IndexT conn2[4] = {0, 1, 1, 2};

  tribol::MeshData mesh1(0, 2, 3, conn1, tribol::LINEAR_EDGE, x1, y1, nullptr, tribol::MemorySpace::Host);
  tribol::MeshData mesh2(1, 2, 3, conn2, tribol::LINEAR_EDGE, x2, y2, nullptr, tribol::MemorySpace::Host);
  mesh1.setReferencePosition(x1, y1, nullptr);
  mesh2.setReferencePosition(x2, y2, nullptr);

  tribol::ContactParams params;
  params.del = 0.1;
  params.k = 1.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.normal_mode = tribol::EnergyMortarNormalMode::H1_NODAL_NORMAL;
  params.projection_smoothing = false;
  params.residual_gap = 0.03;

  tribol::EnergyMortarCalculator evaluator(params);
  tribol::InterfacePair pair(0, 0);
  auto base = evaluator.compute_h1_total_derivatives(pair, mesh1.getView(), mesh2.getView());
  const int ndof = 2 * (base.num_mesh1_nodes + base.num_mesh2_nodes);

  std::vector<tribol::RealT> x1_orig(x1, x1 + 3);
  std::vector<tribol::RealT> y1_orig(y1, y1 + 3);
  std::vector<tribol::RealT> x2_orig(x2, x2 + 3);
  std::vector<tribol::RealT> y2_orig(y2, y2 + 3);

  auto eval_from_dofs = [&](const std::vector<double>& du) {
    auto x1_pert = x1_orig;
    auto y1_pert = y1_orig;
    auto x2_pert = x2_orig;
    auto y2_pert = y2_orig;
    std::size_t idx = 0;
    for (int i = 0; i < base.num_mesh1_nodes; ++i) {
      x1_pert[static_cast<std::size_t>(base.mesh1_nodes[static_cast<std::size_t>(i)])] += du[idx++];
    }
    for (int i = 0; i < base.num_mesh1_nodes; ++i) {
      y1_pert[static_cast<std::size_t>(base.mesh1_nodes[static_cast<std::size_t>(i)])] += du[idx++];
    }
    for (int i = 0; i < base.num_mesh2_nodes; ++i) {
      x2_pert[static_cast<std::size_t>(base.mesh2_nodes[static_cast<std::size_t>(i)])] += du[idx++];
    }
    for (int i = 0; i < base.num_mesh2_nodes; ++i) {
      y2_pert[static_cast<std::size_t>(base.mesh2_nodes[static_cast<std::size_t>(i)])] += du[idx++];
    }
    mesh1.setPosition(x1_pert.data(), y1_pert.data(), nullptr);
    mesh2.setPosition(x2_pert.data(), y2_pert.data(), nullptr);
    return evaluator.compute_h1_total_derivatives(pair, mesh1.getView(), mesh2.getView(), false);
  };

  const double eps_grad = 1.0e-7;
  for (int j = 0; j < ndof; ++j) {
    const auto j_idx = static_cast<std::size_t>(j);
    std::vector<double> du(static_cast<std::size_t>(ndof), 0.0);
    du[j_idx] = eps_grad;
    auto plus = eval_from_dofs(du);
    du[j_idx] = -eps_grad;
    auto minus = eval_from_dofs(du);

    EXPECT_NEAR(base.dg1_dx[j_idx], (plus.g_tilde[0] - minus.g_tilde[0]) / (2.0 * eps_grad), 1.0e-6);
    EXPECT_NEAR(base.dg2_dx[j_idx], (plus.g_tilde[1] - minus.g_tilde[1]) / (2.0 * eps_grad), 1.0e-6);
    EXPECT_NEAR(base.dA1_dx[j_idx], (plus.area[0] - minus.area[0]) / (2.0 * eps_grad), 1.0e-6);
    EXPECT_NEAR(base.dA2_dx[j_idx], (plus.area[1] - minus.area[1]) / (2.0 * eps_grad), 1.0e-6);
  }

  const double eps_hess = 1.0e-6;
  for (int col = 0; col < ndof; ++col) {
    const auto col_idx = static_cast<std::size_t>(col);
    std::vector<double> du(static_cast<std::size_t>(ndof), 0.0);
    du[col_idx] = eps_hess;
    auto plus = eval_from_dofs(du);
    du[col_idx] = -eps_hess;
    auto minus = eval_from_dofs(du);
    for (int row = 0; row < ndof; ++row) {
      const auto row_idx = static_cast<std::size_t>(row);
      const auto idx = static_cast<std::size_t>(row * ndof + col);
      EXPECT_NEAR(base.d2g1_dx2[idx], (plus.dg1_dx[row_idx] - minus.dg1_dx[row_idx]) / (2.0 * eps_hess), 1.0e-4);
      EXPECT_NEAR(base.d2g2_dx2[idx], (plus.dg2_dx[row_idx] - minus.dg2_dx[row_idx]) / (2.0 * eps_hess), 1.0e-4);
      EXPECT_NEAR(base.d2A1_dx2[idx], (plus.dA1_dx[row_idx] - minus.dA1_dx[row_idx]) / (2.0 * eps_hess), 1.0e-4);
      EXPECT_NEAR(base.d2A2_dx2[idx], (plus.dA2_dx[row_idx] - minus.dA2_dx[row_idx]) / (2.0 * eps_hess), 1.0e-4);
    }
  }

  mesh1.setPosition(x1_orig.data(), y1_orig.data(), nullptr);
  mesh2.setPosition(x2_orig.data(), y2_orig.data(), nullptr);
}

TEST(EnergyMortarResidualGapCheck, AssembledGapIsShiftedByArea)
{
  tribol::RealT x1[2] = {0.0, 1.0};
  tribol::RealT y1[2] = {0.0, 0.0};
  tribol::IndexT conn1[2] = {1, 0};
  tribol::MeshData mesh1(0, 1, 2, conn1, tribol::LINEAR_EDGE, x1, y1, nullptr, tribol::MemorySpace::Host);

  tribol::RealT x2[2] = {0.2, 0.8};
  tribol::RealT y2[2] = {0.1, 0.1};
  tribol::IndexT conn2[2] = {0, 1};
  tribol::MeshData mesh2(1, 1, 2, conn2, tribol::LINEAR_EDGE, x2, y2, nullptr, tribol::MemorySpace::Host);

  tribol::ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.penalty_mode = tribol::EnergyMortarPenaltyMode::NODAL_GAP;

  double g0[2] = {0.0, 0.0};
  double area0[2] = {0.0, 0.0};
  tribol::EnergyMortarCalculator evaluator_without_residual(params);
  evaluator_without_residual.compute_gtilde_and_area(tribol::InterfacePair(0, 0), mesh1.getView(), mesh2.getView(), g0,
                                                     area0);

  params.residual_gap = 0.15;
  double g_residual[2] = {0.0, 0.0};
  double area_residual[2] = {0.0, 0.0};
  tribol::EnergyMortarCalculator evaluator_with_residual(params);
  evaluator_with_residual.compute_gtilde_and_area(tribol::InterfacePair(0, 0), mesh1.getView(), mesh2.getView(),
                                                  g_residual, area_residual);

  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(area_residual[i], area0[i], 1.0e-14);
    EXPECT_NEAR(g_residual[i], g0[i] - params.residual_gap * area0[i], 1.0e-14);
  }
}

TEST(EnergyMortarResidualGapCheck, QuadraturePointOpenGapBecomesActive)
{
  tribol::RealT x1[2] = {0.0, 1.0};
  tribol::RealT y1[2] = {0.0, 0.0};
  tribol::IndexT conn1[2] = {1, 0};
  tribol::MeshData mesh1(0, 1, 2, conn1, tribol::LINEAR_EDGE, x1, y1, nullptr, tribol::MemorySpace::Host);

  tribol::RealT x2[2] = {0.2, 0.8};
  tribol::RealT y2[2] = {0.1, 0.1};
  tribol::IndexT conn2[2] = {0, 1};
  tribol::MeshData mesh2(1, 1, 2, conn2, tribol::LINEAR_EDGE, x2, y2, nullptr, tribol::MemorySpace::Host);

  tribol::ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.penalty_mode = tribol::EnergyMortarPenaltyMode::QUADRATURE_POINT_GAP;

  tribol::EnergyMortarCalculator evaluator_without_residual(params);
  const auto inactive = evaluator_without_residual.compute_quadrature_point_penalty_data(
      tribol::InterfacePair(0, 0), mesh1.getView(), mesh2.getView());
  EXPECT_EQ(inactive.energy, 0.0);

  params.residual_gap = 0.15;
  tribol::EnergyMortarCalculator evaluator_with_residual(params);
  const auto active = evaluator_with_residual.compute_quadrature_point_penalty_data(tribol::InterfacePair(0, 0),
                                                                                    mesh1.getView(), mesh2.getView());
  EXPECT_GT(active.energy, 0.0);
}

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
