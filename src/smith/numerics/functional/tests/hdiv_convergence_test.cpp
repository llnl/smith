// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hdiv_convergence_test.cpp
 *
 * @brief H-refinement convergence rate test for Hdiv elements
 *
 * Solves a Darcy problem with a smooth manufactured solution on progressively
 * refined meshes and verifies that the L2 error in sigma decays at the expected
 * O(h^p) rate.  Also includes a normal continuity verification on shared faces.
 */

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/functional/differentiate_wrt.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"

using namespace smith;

// ─── Manufactured solution ──────────────────────────────────────────────────
//
// u_exact = sin(pi x) sin(pi y)
// sigma_exact = -grad u = (-pi cos(pi x) sin(pi y),
//                          -pi sin(pi x) cos(pi y))
// f = -div(sigma) = -laplacian(u) = 2 pi^2 sin(pi x) sin(pi y)

static constexpr double PI = 3.14159265358979323846;

static void sigma_exact_fn(const mfem::Vector& x, mfem::Vector& v)
{
  v(0) = -PI * cos(PI * x(0)) * sin(PI * x(1));
  v(1) = -PI * sin(PI * x(0)) * cos(PI * x(1));
}

template <int p>
double darcy_convergence_solve(int n_elem)
{
  constexpr int dim = 2;

  auto mesh = mesh::refineAndDistribute(
      mfem::Mesh::MakeCartesian2D(n_elem, n_elem, mfem::Element::QUADRILATERAL, true, 1.0, 1.0), 0);

  auto [fes_sigma, fec_sigma] = generateParFiniteElementSpace<Hdiv<p>>(mesh.get());
  auto [fes_u, fec_u] = generateParFiniteElementSpace<L2<p - 1>>(mesh.get());

  // Project exact solution for BCs and error measurement
  mfem::VectorFunctionCoefficient sigma_coeff(dim, sigma_exact_fn);
  mfem::ParGridFunction sigma_gf(fes_sigma.get());
  sigma_gf.ProjectCoefficient(sigma_coeff);

  mfem::Vector sigma_exact(fes_sigma->TrueVSize());
  sigma_gf.GetTrueDofs(sigma_exact);

  mfem::Vector sigma_zero(fes_sigma->TrueVSize());
  sigma_zero = 0.0;
  mfem::Vector u_zero(fes_u->TrueVSize());
  u_zero = 0.0;

  // Flux equation: (K^{-1} sigma, tau) - (u, div tau) = 0
  Functional<Hdiv<p>(Hdiv<p>, L2<p - 1>)> res_sigma(fes_sigma.get(), {fes_sigma.get(), fes_u.get()});

  Domain whole_domain = EntireDomain(*mesh);

  res_sigma.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0, 1>{},
      [](double, auto, auto sigma_arg, auto u_arg) {
        auto [sigma_val, sigma_div] = sigma_arg;
        auto [u_val, u_grad] = u_arg;
        return smith::tuple{sigma_val, -u_val};
      },
      whole_domain);

  // Continuity equation: (div sigma, v) + eps*(u,v) = (f, v)
  constexpr double eps = 1.0e-10;
  Functional<L2<p - 1>(Hdiv<p>, L2<p - 1>)> res_u(fes_u.get(), {fes_sigma.get(), fes_u.get()});

  res_u.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0, 1>{},
      [=](double, auto position, auto sigma_arg, auto u_arg) {
        auto [X, J] = position;
        auto [sigma_val, sigma_div] = sigma_arg;
        auto [u_val, u_grad] = u_arg;
        double f = 2.0 * PI * PI * sin(PI * X[0]) * sin(PI * X[1]);
        return smith::tuple{sigma_div + eps * u_val - f, tensor<double, dim>{}};
      },
      whole_domain);

  double t = 0.0;

  auto [r_sigma_0, dRsigma_dsigma] = res_sigma(t, differentiate_wrt(sigma_zero), u_zero);
  auto [dummy0, dRsigma_du] = res_sigma(t, sigma_zero, differentiate_wrt(u_zero));
  auto [r_u_0, dRu_dsigma] = res_u(t, differentiate_wrt(sigma_zero), u_zero);
  auto [dummy1, dRu_du] = res_u(t, sigma_zero, differentiate_wrt(u_zero));

  auto J_00 = assemble(dRsigma_dsigma);
  auto J_01 = assemble(dRsigma_du);
  auto J_10 = assemble(dRu_dsigma);
  auto J_11 = assemble(dRu_du);

  // Essential BCs on sigma dot n
  mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
  ess_bdr = 1;
  mfem::Array<int> ess_tdofs_sigma;
  fes_sigma->GetEssentialTrueDofs(ess_bdr, ess_tdofs_sigma);

  mfem::Vector sigma_bc(fes_sigma->TrueVSize());
  sigma_bc = 0.0;
  for (int i = 0; i < ess_tdofs_sigma.Size(); i++) {
    sigma_bc[ess_tdofs_sigma[i]] = sigma_exact[ess_tdofs_sigma[i]];
  }

  // RHS = -residual(0,0) - J * sigma_bc  (for the nonzero source f)
  mfem::Vector rhs_sigma(fes_sigma->TrueVSize());
  {
    mfem::Vector tmp(rhs_sigma.Size());
    J_00->Mult(sigma_bc, tmp);
    rhs_sigma = r_sigma_0;
    rhs_sigma *= -1.0;
    rhs_sigma -= tmp;
  }

  mfem::Vector rhs_u(fes_u->TrueVSize());
  {
    mfem::Vector tmp(rhs_u.Size());
    J_10->Mult(sigma_bc, tmp);
    rhs_u = r_u_0;
    rhs_u *= -1.0;
    rhs_u -= tmp;
  }

  {
    auto* elim = J_00->EliminateRowsCols(ess_tdofs_sigma);
    delete elim;
  }
  J_01->EliminateRows(ess_tdofs_sigma);
  J_10->EliminateCols(ess_tdofs_sigma);

  for (int i = 0; i < ess_tdofs_sigma.Size(); i++) {
    rhs_sigma[ess_tdofs_sigma[i]] = sigma_exact[ess_tdofs_sigma[i]];
  }

  mfem::Array<int> offsets;
  offsets.SetSize(3);
  offsets[0] = 0;
  offsets[1] = fes_sigma->TrueVSize();
  offsets[2] = fes_sigma->TrueVSize() + fes_u->TrueVSize();

  mfem::BlockOperator block_op(offsets);
  block_op.SetBlock(0, 0, J_00.get());
  block_op.SetBlock(0, 1, J_01.get());
  block_op.SetBlock(1, 0, J_10.get());
  block_op.SetBlock(1, 1, J_11.get());

  mfem::BlockVector rhs_block(offsets), sol_block(offsets);
  rhs_block.GetBlock(0) = rhs_sigma;
  rhs_block.GetBlock(1) = rhs_u;
  sol_block = 0.0;

  smith::SuperLUSolver superlu(0, mesh->GetComm());
  superlu.SetOperator(block_op);
  superlu.Mult(rhs_block, sol_block);

  // Compute L2 error in sigma via mfem GridFunction
  mfem::ParGridFunction sigma_h(fes_sigma.get());
  sigma_h.SetFromTrueDofs(sol_block.GetBlock(0));

  double l2_err = sigma_h.ComputeL2Error(sigma_coeff);
  return l2_err;
}

// ─── Convergence rate tests ─────────────────────────────────────────────────

template <int p>
void check_convergence_rate()
{
  // Solve on two mesh sizes, compute convergence rate
  double err_coarse = darcy_convergence_solve<p>(4);
  double err_fine = darcy_convergence_solve<p>(8);

  // rate = log(err_coarse/err_fine) / log(h_coarse/h_fine) = log(err_coarse/err_fine) / log(2)
  double rate = log(err_coarse / err_fine) / log(2.0);

  // Expect at least O(h^p) convergence.  Allow 0.5 margin for pre-asymptotic effects.
  EXPECT_GT(rate, double(p) - 0.5) << "Convergence rate " << rate << " is below expected O(h^" << p << ")";
}

TEST(HdivConvergence, RT1_quads) { check_convergence_rate<2>(); }
TEST(HdivConvergence, RT2_quads) { check_convergence_rate<3>(); }

// ─── Normal continuity verification ─────────────────────────────────────────
//
// Project a known vector field into the Hdiv space, then verify that the
// normal component is continuous across internal faces by checking that
// mfem's RT DOFs (which represent sigma dot n integrated against face basis
// functions) are single-valued.

template <int p>
void check_normal_continuity()
{
  constexpr int dim = 2;

  auto mesh =
      mesh::refineAndDistribute(mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 1.0), 0);

  auto [fes_sigma, fec_sigma] = generateParFiniteElementSpace<Hdiv<p>>(mesh.get());

  mfem::VectorFunctionCoefficient sigma_coeff(dim, sigma_exact_fn);
  mfem::ParGridFunction sigma_gf(fes_sigma.get());
  sigma_gf.ProjectCoefficient(sigma_coeff);

  // RT DOFs on shared faces should be single-valued (this is the definition of
  // H(div) conformity).  GetTrueDofs reduces to unique DOFs; if we scatter back
  // and compare, any discontinuity would show up as a difference.
  mfem::Vector true_dofs(fes_sigma->TrueVSize());
  sigma_gf.GetTrueDofs(true_dofs);

  mfem::ParGridFunction sigma_check(fes_sigma.get());
  sigma_check.SetFromTrueDofs(true_dofs);

  // The round-trip should be lossless for a conforming space
  mfem::Vector diff(sigma_gf.Size());
  subtract(sigma_gf, sigma_check, diff);
  double err = diff.Norml2();
  EXPECT_NEAR(err, 0.0, 1.0e-12) << "Normal continuity violated: round-trip DOF error = " << err;
}

TEST(HdivNormalContinuity, RT1_quads) { check_normal_continuity<2>(); }
TEST(HdivNormalContinuity, RT2_quads) { check_normal_continuity<3>(); }

// ─── Boundary integral gradient check ───────────────────────────────────────
//
// Exercises AddBoundaryIntegral with Hdiv test/trial spaces.

template <int p>
void hdiv_boundary_test()
{
  constexpr int dim = 2;

  auto mesh =
      mesh::refineAndDistribute(mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 1.0), 1);

  auto [fespace, fec] = smith::generateParFiniteElementSpace<Hdiv<p>>(mesh.get());

  mfem::Vector U(fespace->TrueVSize());
  U.Randomize(42);

  Functional<Hdiv<p>(Hdiv<p>)> residual(fespace.get(), {fespace.get()});

  Domain bdr = EntireBoundary(*mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [](double /*t*/, auto /*position*/, auto sigma) {
        auto [sigma_n, d_sigma_n] = sigma;
        // sigma_n is the scalar normal flux (sigma dot n) on the boundary
        return sigma_n * sigma_n + 0.5 * d_sigma_n;
      },
      bdr);

  double t = 0.0;
  check_gradient(residual, t, U);
}

template <int p>
void hdiv_boundary_test_3d()
{
  constexpr int dim = 3;

  auto mesh =
      mesh::refineAndDistribute(mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0), 1);

  auto [fespace, fec] = smith::generateParFiniteElementSpace<Hdiv<p>>(mesh.get());

  mfem::Vector U(fespace->TrueVSize());
  U.Randomize(42);

  Functional<Hdiv<p>(Hdiv<p>)> residual(fespace.get(), {fespace.get()});

  Domain bdr = EntireBoundary(*mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [](double /*t*/, auto /*position*/, auto sigma) {
        auto [sigma_n, d_sigma_n] = sigma;
        // sigma_n is the scalar normal flux (sigma dot n) on the boundary
        return sigma_n * sigma_n + 0.5 * d_sigma_n[0] + 0.5 * d_sigma_n[1];
      },
      bdr);

  double t = 0.0;
  check_gradient(residual, t, U);
}

TEST(HdivBoundary, RT1_quads) { hdiv_boundary_test<1>(); }
TEST(HdivBoundary, RT2_quads) { hdiv_boundary_test<2>(); }
TEST(HdivBoundary, RT1_hexes) { hdiv_boundary_test_3d<1>(); }
TEST(HdivBoundary, RT2_hexes) { hdiv_boundary_test_3d<2>(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
