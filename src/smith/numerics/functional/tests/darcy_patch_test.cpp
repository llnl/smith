// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file darcy_patch_test.cpp
 *
 * @brief Patch test for the mixed Hdiv × L2 (Darcy flow) formulation
 *
 * Mixed weak form: find (σ, u) in Hdiv × L2 such that
 *   (K⁻¹ σ, τ)  − (u, div τ)  = 0    ∀ τ ∈ Hdiv   [flux equation]
 *   (div σ,  v)                = (f, v) ∀ v ∈ L2    [continuity]
 *
 * Exact affine solution (exactly representable in Hdiv<2> × L2<1>):
 *   u_exact(x,y) = x + 2y          (linear pressure)
 *   σ_exact      = (−1, −2)        (constant flux = −K ∇u with K = I)
 *   f            = div σ = 0
 *
 * Essential BCs:  σ·n = σ_exact·n  on entire ∂Ω  (enforced via mfem DOF elimination)
 * Null-space fix: add ε (u,v) to equation 2 so the system is non-singular.
 *                 ε = 1e-10 causes O(ε) error in u, negligible error in σ.
 *
 * The assembled block system is solved with smith::SuperLUSolver.
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

using namespace smith;

// ─── helpers ─────────────────────────────────────────────────────────────────

static void sigma_exact_fn(const mfem::Vector& /*x*/, mfem::Vector& v)
{
  v(0) = -1.0;
  v(1) = -2.0;
}

static double u_exact_fn(const mfem::Vector& x) { return x(0) + 2.0 * x(1); }

// ─── main solve ──────────────────────────────────────────────────────────────

/**
 * @brief Assemble and solve the Darcy block system, return relative σ error.
 *
 * @tparam p  smith Hdiv order  (Hdiv<p> ↔ RT_FECollection(p-1))
 *            Minimum p=2 so that σ_exact (constant) and u_exact (linear)
 *            are exactly representable in Hdiv<p> × L2<p-1>.
 */
template <int p>
double darcy_solve()
{
  constexpr int dim = 2;

  // 1. Mesh: 4×4 Cartesian quad mesh on [0,1]²
  auto mesh =
      mesh::refineAndDistribute(mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 1.0), 0);

  // 2. FE spaces
  //    Hdiv<p>  ↔ RT_FECollection(p-1)  — flux space
  //    L2<p-1>  ↔ L2_FECollection(p-1) — pressure space
  auto [fes_sigma, fec_sigma] = generateParFiniteElementSpace<Hdiv<p>>(mesh.get());
  auto [fes_u, fec_u] = generateParFiniteElementSpace<L2<p - 1>>(mesh.get());

  // 3. Project exact solution
  mfem::VectorFunctionCoefficient sigma_coeff(dim, sigma_exact_fn);
  mfem::ParGridFunction sigma_gf(fes_sigma.get());
  sigma_gf.ProjectCoefficient(sigma_coeff);

  mfem::FunctionCoefficient u_coeff(u_exact_fn);
  mfem::ParGridFunction u_gf(fes_u.get());
  u_gf.ProjectCoefficient(u_coeff);

  mfem::Vector sigma_exact(fes_sigma->TrueVSize());
  sigma_gf.GetTrueDofs(sigma_exact);
  mfem::Vector u_exact(fes_u->TrueVSize());
  u_gf.GetTrueDofs(u_exact);

  // 4. Working vectors (linearization point = zero)
  mfem::Vector sigma_zero(fes_sigma->TrueVSize());
  sigma_zero = 0.0;
  mfem::Vector u_zero(fes_u->TrueVSize());
  u_zero = 0.0;

  // 5. Build Functionals
  //
  // Flux equation (Hdiv test, trials = {Hdiv, L2}):
  //   R_σ(σ,u;τ) = ∫ (K⁻¹ σ)·τ dx − ∫ u div(τ) dx
  //   source = K⁻¹ σ  (dual to τ),  flux = −u  (dual to div τ)
  Functional<Hdiv<p>(Hdiv<p>, L2<p - 1>)> res_sigma(fes_sigma.get(), {fes_sigma.get(), fes_u.get()});

  Domain whole_domain = EntireDomain(*mesh);

  res_sigma.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0, 1>{},
      [](double, auto, auto sigma_arg, auto u_arg) {
        auto [sigma_val, sigma_div] = sigma_arg;
        auto [u_val, u_grad] = u_arg;
        return smith::tuple{sigma_val, -u_val};  // K = I
      },
      whole_domain);

  // Continuity equation (L2 test, trials = {Hdiv, L2}):
  //   R_u(σ,u;v) = ∫ div(σ) v dx + ε ∫ u v dx − ∫ f v dx,  f = 0
  //   source = div(σ) + ε u  (dual to v),  flux = 0  (dual to grad v)
  //   Small ε removes the pressure null-space so SuperLU can invert.
  constexpr double eps = 1.0e-10;
  Functional<L2<p - 1>(Hdiv<p>, L2<p - 1>)> res_u(fes_u.get(), {fes_sigma.get(), fes_u.get()});

  res_u.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0, 1>{},
      [=](double, auto, auto sigma_arg, auto u_arg) {
        auto [sigma_val, sigma_div] = sigma_arg;
        auto [u_val, u_grad] = u_arg;
        // source = div(σ) + ε u,  flux = 0
        return smith::tuple{sigma_div + eps * u_val, tensor<double, dim>{}};
      },
      whole_domain);

  // 6. Assemble Jacobian blocks
  double t = 0.0;

  auto [r_sigma_0, dRsigma_dsigma] = res_sigma(t, differentiate_wrt(sigma_zero), u_zero);
  auto [dummy0, dRsigma_du] = res_sigma(t, sigma_zero, differentiate_wrt(u_zero));
  auto [r_u_0, dRu_dsigma] = res_u(t, differentiate_wrt(sigma_zero), u_zero);
  auto [dummy1, dRu_du] = res_u(t, sigma_zero, differentiate_wrt(u_zero));

  auto J_00 = assemble(dRsigma_dsigma);  //  K⁻¹ mass matrix (Hdiv × Hdiv)
  auto J_01 = assemble(dRsigma_du);      // −B^T               (Hdiv × L2)
  auto J_10 = assemble(dRu_dsigma);      //  B = divergence     (L2 × Hdiv)
  auto J_11 = assemble(dRu_du);          //  ε M_L2             (L2 × L2)

  // 7. Essential BCs: fix σ·n on entire boundary
  mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
  ess_bdr = 1;
  mfem::Array<int> ess_tdofs_sigma;
  fes_sigma->GetEssentialTrueDofs(ess_bdr, ess_tdofs_sigma);

  // Build BC vector (exact σ at boundary DOFs, 0 elsewhere)
  mfem::Vector sigma_bc(fes_sigma->TrueVSize());
  sigma_bc = 0.0;
  for (int i = 0; i < ess_tdofs_sigma.Size(); i++) {
    sigma_bc[ess_tdofs_sigma[i]] = sigma_exact[ess_tdofs_sigma[i]];
  }

  // RHS from linearization at (0,0) is zero (linear, no source).
  // Modify RHS for the known BC values: rhs -= J_col_bc * sigma_bc
  mfem::Vector rhs_sigma(fes_sigma->TrueVSize());
  rhs_sigma = 0.0;
  {
    mfem::Vector tmp(rhs_sigma.Size());
    J_00->Mult(sigma_bc, tmp);
    rhs_sigma -= tmp;  // contribution of BC values through J_00
  }

  mfem::Vector rhs_u(fes_u->TrueVSize());
  rhs_u = 0.0;
  {
    mfem::Vector tmp(rhs_u.Size());
    J_10->Mult(sigma_bc, tmp);
    rhs_u -= tmp;  // contribution of BC values through J_10
  }

  // Eliminate BC rows / cols from the system
  {
    auto* elim = J_00->EliminateRowsCols(ess_tdofs_sigma);
    delete elim;
  }
  J_01->EliminateRows(ess_tdofs_sigma);
  J_10->EliminateCols(ess_tdofs_sigma);

  // Enforce BC values in the σ rows of the RHS
  for (int i = 0; i < ess_tdofs_sigma.Size(); i++) {
    rhs_sigma[ess_tdofs_sigma[i]] = sigma_exact[ess_tdofs_sigma[i]];
  }

  // 8. Assemble block operator and block RHS
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

  // 9. Solve with SuperLU
  smith::SuperLUSolver superlu(0 /*print_level*/, mesh->GetComm());
  superlu.SetOperator(block_op);
  superlu.Mult(rhs_block, sol_block);

  // 10. Compute relative σ error
  mfem::Vector sigma_err(sol_block.GetBlock(0));
  sigma_err -= sigma_exact;
  double err = sigma_err.Norml2();
  double norm_ref = sigma_exact.Norml2();
  return err / (norm_ref > 0.0 ? norm_ref : 1.0);
}

// ─── 3D Darcy solve ──────────────────────────────────────────────────────────

static void sigma_exact_3d_fn(const mfem::Vector& /*x*/, mfem::Vector& v)
{
  v(0) = -1.0;
  v(1) = -2.0;
  v(2) = -3.0;
}

static double u_exact_3d_fn(const mfem::Vector& x) { return x(0) + 2.0 * x(1) + 3.0 * x(2); }

template <int p>
double darcy_solve_3d()
{
  constexpr int dim = 3;

  auto mesh =
      mesh::refineAndDistribute(mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0), 0);

  auto [fes_sigma, fec_sigma] = generateParFiniteElementSpace<Hdiv<p>>(mesh.get());
  auto [fes_u, fec_u] = generateParFiniteElementSpace<L2<p - 1>>(mesh.get());

  mfem::VectorFunctionCoefficient sigma_coeff(dim, sigma_exact_3d_fn);
  mfem::ParGridFunction sigma_gf(fes_sigma.get());
  sigma_gf.ProjectCoefficient(sigma_coeff);

  mfem::FunctionCoefficient u_coeff(u_exact_3d_fn);
  mfem::ParGridFunction u_gf(fes_u.get());
  u_gf.ProjectCoefficient(u_coeff);

  mfem::Vector sigma_exact(fes_sigma->TrueVSize());
  sigma_gf.GetTrueDofs(sigma_exact);
  mfem::Vector u_exact(fes_u->TrueVSize());
  u_gf.GetTrueDofs(u_exact);

  mfem::Vector sigma_zero(fes_sigma->TrueVSize());
  sigma_zero = 0.0;
  mfem::Vector u_zero(fes_u->TrueVSize());
  u_zero = 0.0;

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

  constexpr double eps = 1.0e-10;
  Functional<L2<p - 1>(Hdiv<p>, L2<p - 1>)> res_u(fes_u.get(), {fes_sigma.get(), fes_u.get()});

  res_u.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0, 1>{},
      [=](double, auto, auto sigma_arg, auto u_arg) {
        auto [sigma_val, sigma_div] = sigma_arg;
        auto [u_val, u_grad] = u_arg;
        return smith::tuple{sigma_div + eps * u_val, tensor<double, dim>{}};
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

  mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
  ess_bdr = 1;
  mfem::Array<int> ess_tdofs_sigma;
  fes_sigma->GetEssentialTrueDofs(ess_bdr, ess_tdofs_sigma);

  mfem::Vector sigma_bc(fes_sigma->TrueVSize());
  sigma_bc = 0.0;
  for (int i = 0; i < ess_tdofs_sigma.Size(); i++) {
    sigma_bc[ess_tdofs_sigma[i]] = sigma_exact[ess_tdofs_sigma[i]];
  }

  mfem::Vector rhs_sigma(fes_sigma->TrueVSize());
  rhs_sigma = 0.0;
  {
    mfem::Vector tmp(rhs_sigma.Size());
    J_00->Mult(sigma_bc, tmp);
    rhs_sigma -= tmp;
  }

  mfem::Vector rhs_u(fes_u->TrueVSize());
  rhs_u = 0.0;
  {
    mfem::Vector tmp(rhs_u.Size());
    J_10->Mult(sigma_bc, tmp);
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

  mfem::Vector sigma_err(sol_block.GetBlock(0));
  sigma_err -= sigma_exact;
  double err = sigma_err.Norml2();
  double norm_ref = sigma_exact.Norml2();
  return err / (norm_ref > 0.0 ? norm_ref : 1.0);
}

// ─── tests ───────────────────────────────────────────────────────────────────

// 2D tests
TEST(DarcyPatch, RT1_DG1_2D) { EXPECT_LT(darcy_solve<2>(), 1.0e-8); }
TEST(DarcyPatch, RT2_DG2_2D) { EXPECT_LT(darcy_solve<3>(), 1.0e-8); }

// 3D tests
TEST(DarcyPatch, RT1_DG1_3D) { EXPECT_LT(darcy_solve_3d<2>(), 1.0e-8); }
TEST(DarcyPatch, RT2_DG2_3D) { EXPECT_LT(darcy_solve_3d<3>(), 1.0e-8); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
