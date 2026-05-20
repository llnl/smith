// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"
#include "mfem/fem/dfem/doperator.hpp"
#include "mfem/fem/dfem/backends/local_qf/prelude.hpp"
#include "smith/infrastructure/application_manager.hpp"

// =====================================================================
// dFEM Bug Regression Tests
// =====================================================================
//
// Each TEST below reproduces a specific bug in the dFEM cached derivative
// machinery on the `dfem-multiple-outputs` branch (commit 1fa7888706).
// Every test below FAILS on the unpatched branch.  Bug numbers reference
// dfem_bugs.md.  Each test comment names the root cause and the
// format-patch(es) in mfem_patches/ needed to make it pass.
//
// Patches (in mfem_patches/):
//   0001 — Fix broken include paths and guard Enzyme templates (Bugs 5,6)
//   0002 — Fix DerivativeOperator height and AddBoundaryIntegrator (Bugs 4,7)
//   0003 — Fix SparseMatrix leak in DerivativeAssemble (Bug 3)
//   0004 — Fix nfields loop bounds to use ctx.unionfds.size() (Bug 2)
//   0005 — Fix cache indexing in DerivativeSetup/Apply/Assemble (Bug 1,12)
//   0006 — Fix cached MultTranspose per-input contraction (Bug 13)
//
// Tests and their patch dependencies:
//   Bug12_JacobianMismatch             — needs 0005
//   Bug12_MultipleFields               — needs 0004 + 0005
//   Bug12_NonlinearJacobianAndTranspose — needs 0005 + 0006
//   Bug1_2_BoundaryIntegrator          — UNFIXED (Bugs 8+9, no patch)
//   Bug13_SingleInputTranspose         — needs 0005 + 0006
//   Bug13_ValueOnlyTranspose           — needs 0005 + 0006
//   Bug10_GlobalQFBackendIncompatible  — UNFIXED (compile-time trait check)
// =====================================================================

namespace dfem_bugs_test {

/**
 * @brief Simple boundary qfunction: integral(u * v) on the boundary
 */
struct BoundaryMassQf {
  static constexpr int dim = 2;
  SMITH_HOST_DEVICE inline void operator()(
      const double& u,
      const double& w,
      double& out_v) const
  {
    out_v = u * w;
  }
};

struct DfemBugsFixture : public testing::Test {
  std::unique_ptr<mfem::ParMesh> pmesh;

  void SetUp() override
  {
    auto mesh = mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  }
};

/**
 * @brief Bug 12 (linear case) — Assembled Jacobian from cached
 * `LocalQFBackend` does not match the residual linearisation for
 * `Gradient<U> -> Gradient<U>` vector residuals.
 *
 * Root cause: `derivative_setup.hpp` writes per-QP Jacobian entries into
 * the cache using row-major strides, but `derivative_apply.hpp` and
 * `derivative_assemble.hpp` read them back assuming column-major order.
 * Additionally, the residual write offset in `derivative_apply.hpp` uses a
 * division-based formula that garbles DOF ordering for vector fields
 * (vdim > 1).  These are catalogued as Bug 1 in dfem_bugs.md.
 *
 * The residual here is LINEAR in `u` (`grad_u * (1 + coords(0)) * w`), so
 * the exact Jacobian equals the residual operator.  The test compares
 * three Jacobian-action computations — cached `Mult`, assembled sparse
 * `Mult`, and central finite differences — against each other.
 *
 * On the unpatched `dfem-multiple-outputs` branch the cached and assembled
 * actions disagree with finite differences by O(1)–O(10) because the cache
 * strides are inconsistent.
 *
 * Fix: patch 0005 (cache indexing reconciliation in derivative_setup.hpp,
 * derivative_apply.hpp, and derivative_assemble.hpp).
 */
struct VectorGradientQf {
  static constexpr int dim = 2;
  SMITH_HOST_DEVICE inline void operator()(
      const mfem::future::tensor<double, 2, 2>& grad_u,
      const mfem::future::tensor<double, 2>& coords,
      const double& w,
      mfem::future::tensor<double, 2, 2>& out_grad_v) const
  {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        // Just use coords to ensure it's evaluated, e.g. multiply by (1 + coords(0))
        out_grad_v(i, j) = grad_u(i, j) * w * (1.0 + coords(0));
      }
    }
  }
};

TEST_F(DfemBugsFixture, Bug12_JacobianMismatch)
{
  int dim = pmesh->Dimension();
  mfem::H1_FECollection fec(1, dim);
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim);

  mfem::future::FieldDescriptor u_fd(0, &fes);
  mfem::future::FieldDescriptor v_fd(1, &fes);
  
  // Create coordinate field space and descriptor
  mfem::ParMesh* mesh_ptr = pmesh.get();
  mesh_ptr->EnsureNodes(); // CRITICAL FIX for segfault
  mfem::ParFiniteElementSpace* nodes_fes = dynamic_cast<mfem::ParFiniteElementSpace*>(mesh_ptr->GetNodes()->FESpace());
  mfem::future::FieldDescriptor coords_fd(2, nodes_fes);

  mfem::future::DifferentiableOperator dop({u_fd, coords_fd}, {v_fd}, *pmesh);

  const mfem::IntegrationRule* ir = &mfem::IntRules.Get(pmesh->GetTypicalElementGeometry(), 2);

  mfem::Array<int> attrs(1);
  attrs[0] = 1;

  VectorGradientQf qf;
  dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Gradient<0>{}, mfem::future::Value<2>{}, mfem::future::Weight{}},
      mfem::future::tuple{mfem::future::Gradient<1>{}},
      *ir,
      attrs,
      std::integer_sequence<size_t, 0>{});

  mfem::Vector u(fes.GetTrueVSize());
  u.UseDevice(true);
  u.Randomize(1);

  mfem::MultiVector X(2);
  X.MakeRef(0, u);
  X.MakeRef(1, *mesh_ptr->GetNodes());

  // 1. Get Jacobian action via cached Mult
  auto deriv_op = dop.GetDerivative(0, X);
  
  mfem::Vector u_tangent(fes.GetTrueVSize());
  u_tangent.UseDevice(true);
  u_tangent.Randomize(2);
  
  mfem::Vector jac_action(fes.GetTrueVSize());
  jac_action.UseDevice(true);
  jac_action = 0.0;
  deriv_op->Mult(u_tangent, jac_action);

  // 2. Get Jacobian action via Assemble
  mfem::SparseMatrix* local_jac = nullptr;
  deriv_op->Assemble(local_jac);
  ASSERT_NE(local_jac, nullptr);

  mfem::Vector assembled_jac_action(fes.GetTrueVSize());
  assembled_jac_action.UseDevice(false);
  u_tangent.HostRead();
  local_jac->Mult(u_tangent, assembled_jac_action);

  // Also check cached Mult explicitly (if it's different from FD)
  mfem::Vector cached_mult_action(fes.GetTrueVSize());
  cached_mult_action.UseDevice(true);
  cached_mult_action = 0.0;
  deriv_op->Mult(u_tangent, cached_mult_action);

  // 3. Finite Difference of residual
  constexpr double eps = 1.0e-7;
  mfem::Vector u_plus(u);
  u_plus.Add(eps, u_tangent);
  mfem::Vector u_minus(u);
  u_minus.Add(-eps, u_tangent);
  
  mfem::MultiVector X_plus(2);
  X_plus.MakeRef(0, u_plus);
  X_plus.MakeRef(1, *mesh_ptr->GetNodes());
  
  mfem::MultiVector X_minus(2);
  X_minus.MakeRef(0, u_minus);
  X_minus.MakeRef(1, *mesh_ptr->GetNodes());
  
  mfem::Vector res_plus(fes.GetTrueVSize());
  mfem::Vector res_minus(fes.GetTrueVSize());
  res_plus.UseDevice(true);
  res_minus.UseDevice(true);
  mfem::MultiVector Y_plus(1);
  Y_plus.MakeRef(0, res_plus);
  mfem::MultiVector Y_minus(1);
  Y_minus.MakeRef(0, res_minus);
  
  dop.Mult(X_plus, Y_plus);
  dop.Mult(X_minus, Y_minus);
  
  mfem::Vector fd_jac_action(fes.GetTrueVSize());
  fd_jac_action = res_plus;
  fd_jac_action -= res_minus;
  fd_jac_action *= 0.5 / eps;

  jac_action.HostRead();
  assembled_jac_action.HostRead();
  fd_jac_action.HostRead();
  cached_mult_action.HostRead();

  std::cout << "Jac (cached Mult) action norm: " << cached_mult_action.Norml2() << std::endl;
  std::cout << "Jac (assembled) action norm: " << assembled_jac_action.Norml2() << std::endl;
  std::cout << "FD action norm: " << fd_jac_action.Norml2() << std::endl;

  mfem::Vector diff(fes.GetTrueVSize());
  
  diff = cached_mult_action;
  diff -= fd_jac_action;
  std::cout << "||Jac(cached Mult) - FD||: " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-5);

  diff = assembled_jac_action;
  diff -= fd_jac_action;
  std::cout << "||Jac(assembled) - FD||: " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-5);
  
  delete local_jac;
}

/**
 * @brief Bug 12 (multi-field variant) — exercises the multiple-inputs path
 * of the cached `LocalQFBackend` Jacobian assembly with two scalar H1
 * input fields.
 *
 * Root causes (two bugs interact):
 *  1. Cache indexing (Bug 1): row-major vs column-major mismatch between
 *     derivative_setup.hpp and derivative_apply.hpp / derivative_assemble.hpp
 *     corrupts the per-QP Jacobian read-back.
 *  2. Loop bounds (Bug 2): four loops in derivative_action.hpp and one in
 *     derivative_assemble.hpp iterate `uf < nfields` (compile-time template
 *     max) instead of `uf < ctx.unionfds.size()` (runtime count).  When
 *     runtime < compile-time max this walks over uninitialised entries;
 *     when searching for a field whose union index >= nfields the search
 *     fails and `test_field_idx` stays -1, causing `ctx.unionfds[-1]`
 *     access (undefined behaviour / crash).
 *
 * Residual: `u1 * u2 * w -> Value<V>` with `derivative_ids = {0, 1}`.
 * Differentiates w.r.t. `u1` and compares assembled Jacobian, cached
 * `Mult`, and central finite differences.
 *
 * On the unpatched branch both bugs contribute to incorrect results.
 *
 * Fix: patch 0004 (nfields loop bounds) + patch 0005 (cache indexing).
 */
struct MultiFieldQf {
  static constexpr int dim = 2;
  SMITH_HOST_DEVICE inline void operator()(
      const double& u1,
      const double& u2,
      const double& w,
      double& out_v) const
  {
    out_v = u1 * u2 * w;
  }
};

TEST_F(DfemBugsFixture, Bug12_MultipleFields)
{
  auto mesh = mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh pmesh2x1(MPI_COMM_WORLD, mesh);

  mfem::H1_FECollection fec(1, pmesh2x1.Dimension());
  mfem::ParFiniteElementSpace fes(&pmesh2x1, &fec);

  mfem::future::FieldDescriptor u1_fd(0, &fes);
  mfem::future::FieldDescriptor u2_fd(1, &fes);
  mfem::future::FieldDescriptor v_fd(2, &fes);

  mfem::future::DifferentiableOperator dop({u1_fd, u2_fd}, {v_fd}, pmesh2x1);

  const mfem::IntegrationRule* ir = &mfem::IntRules.Get(pmesh2x1.GetTypicalElementGeometry(), 2);
  mfem::Array<int> attrs(1);
  attrs[0] = 1;

  MultiFieldQf qf;
  dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Value<0>{}, mfem::future::Value<1>{}, mfem::future::Weight{}},
      mfem::future::tuple{mfem::future::Value<2>{}},
      *ir,
      attrs,
      std::integer_sequence<size_t, 0, 1>{}); // Differentiate w.r.t u1 and u2

  mfem::Vector u1(fes.GetTrueVSize());
  mfem::Vector u2(fes.GetTrueVSize());
  u1.UseDevice(true); u2.UseDevice(true);
  u1 = 1.0; u2 = 2.0;

  mfem::MultiVector X(2);
  X.MakeRef(0, u1);
  X.MakeRef(1, u2);

  // Check Jacobian w.r.t u1
  auto deriv_op = dop.GetDerivative(0, X);
  mfem::SparseMatrix* local_jac = nullptr;
  deriv_op->Assemble(local_jac);
  ASSERT_NE(local_jac, nullptr);
  
  mfem::Vector u1_tangent(fes.GetTrueVSize());
  u1_tangent.UseDevice(true);
  u1_tangent = 1.0;
  
  mfem::Vector jac_action(fes.GetTrueVSize());
  jac_action.UseDevice(true);
  local_jac->Mult(u1_tangent, jac_action);
  
  // Also check cached Mult
  mfem::Vector cached_mult_action(fes.GetTrueVSize());
  cached_mult_action.UseDevice(true);
  cached_mult_action = 0.0;
  deriv_op->Mult(u1_tangent, cached_mult_action);

  // Finite Difference
  constexpr double eps = 1.0e-7;
  mfem::Vector u1_plus(u1);
  u1_plus.Add(eps, u1_tangent);
  mfem::Vector u1_minus(u1);
  u1_minus.Add(-eps, u1_tangent);
  
  mfem::MultiVector X_plus(2);
  X_plus.MakeRef(0, u1_plus);
  X_plus.MakeRef(1, u2);
  
  mfem::MultiVector X_minus(2);
  X_minus.MakeRef(0, u1_minus);
  X_minus.MakeRef(1, u2);
  
  mfem::Vector res_plus(fes.GetTrueVSize());
  mfem::Vector res_minus(fes.GetTrueVSize());
  res_plus.UseDevice(true);
  res_minus.UseDevice(true);
  mfem::MultiVector Y_plus(1);
  Y_plus.MakeRef(0, res_plus);
  mfem::MultiVector Y_minus(1);
  Y_minus.MakeRef(0, res_minus);
  
  dop.Mult(X_plus, Y_plus);
  dop.Mult(X_minus, Y_minus);
  
  mfem::Vector fd_action(fes.GetTrueVSize());
  fd_action = res_plus;
  fd_action -= res_minus;
  fd_action *= 0.5 / eps;

  jac_action.HostRead();
  cached_mult_action.HostRead();
  fd_action.HostRead();

  std::cout << "Multi-field FD action sum: " << fd_action.Sum() << std::endl;
  std::cout << "Multi-field cached Mult action sum: " << cached_mult_action.Sum() << std::endl;
  std::cout << "Multi-field assembled Jac action sum: " << jac_action.Sum() << std::endl;
  
  EXPECT_NEAR(jac_action.Sum(), fd_action.Sum(), 1.0e-5);
  EXPECT_NEAR(cached_mult_action.Sum(), fd_action.Sum(), 1.0e-5);

  delete local_jac;
}

/**
 * @brief Bugs 1, 12, and 13 (nonlinear case) — Cached Jacobian forward
 * action AND cached `MultTranspose` are both wrong for a genuinely
 * nonlinear `Gradient<U> -> Gradient<U>` residual with multiple dependent
 * inputs on the same field.
 *
 * Root causes (two distinct bugs):
 *  Bug 1/12 (forward path): row-major vs column-major cache stride
 *    mismatch between derivative_setup.hpp and derivative_apply.hpp /
 *    derivative_assemble.hpp.  The per-QP Jacobian entries are written in
 *    one order and read back in another, producing O(1)–O(10) errors in
 *    the forward cached Mult and assembled Mult relative to FD.
 *  Bug 13 (transpose path): derivative_apply_transpose.hpp attempted to
 *    contract all dependent inputs simultaneously into a single output
 *    buffer.  When multiple dependent inputs (Value<0> and Gradient<0>)
 *    map to the same trial field, the per-input Jacobian slices have
 *    different trial operator dimensions and must be contracted
 *    independently then accumulated.  The original code mixed these
 *    slices, producing ~O(1) error in MultTranspose vs assembled J^T and
 *    breaking the adjoint identity <J*t, v> == <t, J^T*v>.
 *
 * Residual: `(1 + u(0) + 0.5*u(1)) * grad_u * w` on a dim=2 H1 vector
 * field.  Both Value<0> and Gradient<0> are dependent inputs.  The test
 * checks five things: cached Mult vs FD, assembled vs FD, MultTranspose
 * vs assembled J^T, and the adjoint identity.
 *
 * On the unpatched branch all five checks fail.
 *
 * Fix: patch 0005 (cache indexing) + patch 0006 (per-input transpose
 * contraction in derivative_apply_transpose.hpp).
 */
struct NonlinearVectorQf {
  static constexpr int dim = 2;
  SMITH_HOST_DEVICE inline void operator()(
      const mfem::future::tensor<double, 2>& u,
      const mfem::future::tensor<double, 2, 2>& grad_u,
      const double& w,
      mfem::future::tensor<double, 2, 2>& out_grad_v) const
  {
    const double scale = (1.0 + u(0) + 0.5 * u(1)) * w;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        out_grad_v(i, j) = scale * grad_u(i, j);
      }
    }
  }
};

TEST_F(DfemBugsFixture, Bug12_NonlinearJacobianAndTranspose)
{
  int dim = pmesh->Dimension();
  mfem::H1_FECollection fec(1, dim);
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim);

  mfem::future::FieldDescriptor u_fd(0, &fes);
  mfem::future::FieldDescriptor v_fd(1, &fes);

  mfem::future::DifferentiableOperator dop({u_fd}, {v_fd}, *pmesh);

  const mfem::IntegrationRule* ir = &mfem::IntRules.Get(pmesh->GetTypicalElementGeometry(), 2);
  mfem::Array<int> attrs(1);
  attrs[0] = 1;

  NonlinearVectorQf qf;
  dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Value<0>{}, mfem::future::Gradient<0>{}, mfem::future::Weight{}},
      mfem::future::tuple{mfem::future::Gradient<1>{}},
      *ir,
      attrs,
      std::integer_sequence<size_t, 0>{});

  const int N = fes.GetTrueVSize();
  mfem::Vector u(N);
  u.UseDevice(true);
  u.Randomize(7);

  mfem::MultiVector X(1);
  X.MakeRef(0, u);

  auto deriv_op = dop.GetDerivative(0, X);

  mfem::Vector u_tangent(N);
  u_tangent.UseDevice(true);
  u_tangent.Randomize(13);

  // 1. Cached Mult action
  mfem::Vector cached_mult(N);
  cached_mult.UseDevice(true);
  cached_mult = 0.0;
  deriv_op->Mult(u_tangent, cached_mult);

  // 2. Assembled action
  mfem::SparseMatrix* J = nullptr;
  deriv_op->Assemble(J);
  ASSERT_NE(J, nullptr);

  mfem::Vector assembled_action(N);
  assembled_action.UseDevice(false);
  u_tangent.HostRead();
  J->Mult(u_tangent, assembled_action);

  // 3. Finite-difference action (truth)
  constexpr double eps = 1.0e-7;
  mfem::Vector u_plus(u), u_minus(u);
  u_plus.Add(eps, u_tangent);
  u_minus.Add(-eps, u_tangent);

  mfem::MultiVector Xp(1), Xm(1);
  Xp.MakeRef(0, u_plus);
  Xm.MakeRef(0, u_minus);

  mfem::Vector res_p(N), res_m(N);
  res_p.UseDevice(true); res_m.UseDevice(true);
  mfem::MultiVector Yp(1), Ym(1);
  Yp.MakeRef(0, res_p);
  Ym.MakeRef(0, res_m);
  dop.Mult(Xp, Yp);
  dop.Mult(Xm, Ym);

  mfem::Vector fd_action(N);
  fd_action = res_p;
  fd_action -= res_m;
  fd_action *= 0.5 / eps;

  cached_mult.HostRead();
  assembled_action.HostRead();
  fd_action.HostRead();

  mfem::Vector diff(N);
  diff = cached_mult; diff -= fd_action;
  std::cout << "||Jac(cached Mult) - FD|| = " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-5);

  diff = assembled_action; diff -= fd_action;
  std::cout << "||Jac(assembled) - FD|| = " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-5);

  // 4. MultTranspose vs assembled^T action
  mfem::Vector v_dir(N);
  v_dir.UseDevice(true);
  v_dir.Randomize(29);

  mfem::Vector mt_action(N);
  mt_action.UseDevice(true);
  mt_action = 0.0;
  deriv_op->MultTranspose(v_dir, mt_action);

  mfem::Vector assembled_mt_action(N);
  assembled_mt_action.UseDevice(false);
  v_dir.HostRead();
  J->MultTranspose(v_dir, assembled_mt_action);

  mt_action.HostRead();
  assembled_mt_action.HostRead();

  diff = mt_action; diff -= assembled_mt_action;
  std::cout << "||MultTranspose - Assembled^T|| = " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-8);

  // 5. Adjoint identity: <Jac*tangent, v> == <tangent, Jac^T*v>
  const double lhs = mfem::InnerProduct(v_dir, cached_mult);
  const double rhs = mfem::InnerProduct(u_tangent, mt_action);
  std::cout << "<J*t, v> = " << lhs << "   <t, J^T*v> = " << rhs << std::endl;
  EXPECT_NEAR(lhs, rhs, 1.0e-10 * std::max(std::abs(lhs), 1.0));

  delete J;
}

/**
 * @brief Bugs 8 and 9 — `LocalQFBackend` boundary integration is not
 * entity-aware, and `DifferentiableOperator::Mult` is hardcoded to volume
 * entities.
 *
 * Root causes (two bugs, both UNFIXED):
 *  Bug 8: `LocalQFImpl::Action` is templated on `qfunc_t / inputs_t /
 *    outputs_t` but NOT on `entity_t`.  Every entity-dependent operation
 *    is hardcoded to `Entity::Element`:
 *      - `GetDofToQuad<Entity::Element>` builds volume DofToQuad maps
 *        paired with a boundary IntegrationRule (inconsistent dof counts).
 *      - `get_restriction<Entity::Element>` returns volume element
 *        restriction, scattering boundary contributions into volume DOF
 *        slots.
 *      - sum-factorisation predicate uses volume geometry instead of face
 *        geometry.
 *    The dispatch drops `entity_t` at `MakeAction`, constructing the
 *    `Entity::Element`-hardcoded Action regardless of whether the
 *    integrator was registered via AddDomainIntegrator or
 *    AddBoundaryIntegrator.
 *  Bug 9: `DifferentiableOperator::Mult` calls
 *    `restriction<Entity::Element>` and
 *    `restriction_transpose<Entity::Element>` for ALL integrators,
 *    regardless of entity kind.  Even if Action were entity-aware, the
 *    surrounding Mult plumbing would still use the wrong topology for
 *    boundary callbacks.
 *
 * Test: boundary mass form `u * v * w` on the bottom edge (attr 1) of the
 * unit square.  With u=1 the residual sum should be 1.0 (the edge length).
 * Observed: ~0.789, because volume shape functions and restrictions are
 * used instead of boundary ones.  Also checks derivative via FD (also
 * wrong for the same reason).
 *
 * Status: FAILS on unpatched branch AND with all current patches applied.
 * No patch fixes this — requires threading entity_t through the entire
 * LocalQFBackend template chain and grouping callbacks by entity kind in
 * DifferentiableOperator::Mult.
 */
TEST_F(DfemBugsFixture, Bug1_2_BoundaryIntegrator)
{
  mfem::H1_FECollection fec(1, pmesh->Dimension());
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec);

  mfem::future::FieldDescriptor u_fd(0, &fes);
  mfem::future::FieldDescriptor v_fd(1, &fes);

  mfem::future::DifferentiableOperator dop({u_fd}, {v_fd}, *pmesh);

  mfem::Array<int> bdr_attrs;
  bdr_attrs.Append(1); // Bottom edge

  const mfem::IntegrationRule* ir = &mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2);

  BoundaryMassQf qf;
  dop.AddBoundaryIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Value<0>{}, mfem::future::Weight{}},
      mfem::future::tuple{mfem::future::Value<1>{}},
      *ir,
      bdr_attrs,
      std::integer_sequence<size_t, 0>{});

  mfem::Vector u(fes.GetTrueVSize());
  u.UseDevice(true);
  u = 1.0;

  mfem::MultiVector X(1);
  X.MakeRef(0, u);

  mfem::Vector res(fes.GetTrueVSize());
  res.UseDevice(true);
  res = 0.0;
  mfem::MultiVector Y(1);
  Y.MakeRef(0, res);

  dop.Mult(X, Y);

  res.HostRead();
  double sum = res.Sum();
  std::cout << "Residual sum: " << sum << std::endl;
  EXPECT_NEAR(sum, 1.0, 1.0e-6);

  auto deriv_op = dop.GetDerivative(0, X);
  mfem::Vector jac_action(fes.GetTrueVSize());
  jac_action.UseDevice(true);
  jac_action = 0.0;
  
  mfem::Vector u_tangent(fes.GetTrueVSize());
  u_tangent.UseDevice(true);
  u_tangent.Randomize(123);
  
  deriv_op->Mult(u_tangent, jac_action);
  
  constexpr double eps = 1.0e-7;
  mfem::Vector u_plus(u);
  u_plus.Add(eps, u_tangent);
  mfem::Vector u_minus(u);
  u_minus.Add(-eps, u_tangent);
  
  mfem::MultiVector X_plus(1);
  X_plus.MakeRef(0, u_plus);
  mfem::MultiVector X_minus(1);
  X_minus.MakeRef(0, u_minus);
  
  mfem::Vector res_plus(fes.GetTrueVSize());
  mfem::Vector res_minus(fes.GetTrueVSize());
  res_plus.UseDevice(true);
  res_minus.UseDevice(true);
  mfem::MultiVector Y_plus(1);
  Y_plus.MakeRef(0, res_plus);
  mfem::MultiVector Y_minus(1);
  Y_minus.MakeRef(0, res_minus);
  
  dop.Mult(X_plus, Y_plus);
  dop.Mult(X_minus, Y_minus);
  
  mfem::Vector fd_jac_action(fes.GetTrueVSize());
  fd_jac_action = res_plus;
  fd_jac_action -= res_minus;
  fd_jac_action *= 0.5 / eps;
  
  jac_action.HostRead();
  fd_jac_action.HostRead();
  
  for (int i = 0; i < fes.GetTrueVSize(); ++i) {
    EXPECT_NEAR(jac_action(i), fd_jac_action(i), 1.0e-5);
  }
}

/**
 * @brief Bugs 1 and 13 (single-dependent-input variant) — Cached forward
 * Mult and MultTranspose are both wrong when only Gradient<0> is the
 * dependent input (no Value<0>).
 *
 * Root causes:
 *  Bug 1 (forward): cache stride mismatch between derivative_setup.hpp
 *    (row-major write) and derivative_apply.hpp / derivative_assemble.hpp
 *    (column-major read).  Corrupts the per-QP Jacobian for vector fields.
 *  Bug 13 (transpose): derivative_apply_transpose.hpp uses incorrect
 *    contraction strides inherited from the same cache-ordering mismatch,
 *    producing O(0.1) error in MultTranspose vs assembled J^T and breaking
 *    the adjoint identity.
 *
 * This test isolates the single-dependent-input path from the
 * multi-dependent-input path tested in Bug12_NonlinearJacobianAndTranspose.
 * The qfunction is nonlinear: `(1 + grad_u(0,0) + 0.5*grad_u(1,1)) *
 * grad_u * w * (1 + coords(0))` with only Gradient<0> dependent.
 *
 * On the unpatched branch: forward Mult, assembled, MultTranspose, and
 * adjoint identity all fail.
 *
 * Fix: patch 0005 (cache indexing) + patch 0006 (per-input transpose
 * contraction).
 */
struct NonlinearGradOnlyQf {
  static constexpr int dim = 2;
  SMITH_HOST_DEVICE inline void operator()(
      const mfem::future::tensor<double, 2, 2>& grad_u,
      const mfem::future::tensor<double, 2>& coords,
      const double& w,
      mfem::future::tensor<double, 2, 2>& out_grad_v) const
  {
    const double scale = (1.0 + grad_u(0, 0) + 0.5 * grad_u(1, 1)) * w;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        out_grad_v(i, j) = scale * grad_u(i, j) * (1.0 + coords(0));
      }
    }
  }
};

TEST_F(DfemBugsFixture, Bug13_SingleInputTranspose)
{
  int dim = pmesh->Dimension();
  mfem::H1_FECollection fec(1, dim);
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim);

  pmesh->EnsureNodes();
  auto* nodes_fes = dynamic_cast<mfem::ParFiniteElementSpace*>(
      pmesh->GetNodes()->FESpace());

  mfem::future::FieldDescriptor u_fd(0, &fes);
  mfem::future::FieldDescriptor coords_fd(1, nodes_fes);
  mfem::future::FieldDescriptor v_fd(2, &fes);

  mfem::future::DifferentiableOperator dop({u_fd, coords_fd}, {v_fd}, *pmesh);

  const mfem::IntegrationRule* ir =
      &mfem::IntRules.Get(pmesh->GetTypicalElementGeometry(), 2);
  mfem::Array<int> attrs(1);
  attrs[0] = 1;

  NonlinearGradOnlyQf qf;
  dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Gradient<0>{}, mfem::future::Value<1>{},
                           mfem::future::Weight{}},
      mfem::future::tuple{mfem::future::Gradient<2>{}},
      *ir, attrs,
      std::integer_sequence<size_t, 0>{});

  const int N = fes.GetTrueVSize();
  mfem::Vector u(N);
  u.UseDevice(true);
  u.Randomize(7);

  mfem::MultiVector X(2);
  X.MakeRef(0, u);
  X.MakeRef(1, *pmesh->GetNodes());

  auto deriv_op = dop.GetDerivative(0, X);

  mfem::Vector u_tangent(N);
  u_tangent.UseDevice(true);
  u_tangent.Randomize(13);

  // 1. Forward cached Mult
  mfem::Vector cached_mult(N);
  cached_mult.UseDevice(true);
  cached_mult = 0.0;
  deriv_op->Mult(u_tangent, cached_mult);

  // 2. Assembled action
  mfem::SparseMatrix* J = nullptr;
  deriv_op->Assemble(J);
  ASSERT_NE(J, nullptr);

  mfem::Vector assembled_action(N);
  assembled_action.UseDevice(false);
  u_tangent.HostRead();
  J->Mult(u_tangent, assembled_action);

  // 3. FD truth
  constexpr double eps = 1.0e-7;
  mfem::Vector u_plus(u), u_minus(u);
  u_plus.Add(eps, u_tangent);
  u_minus.Add(-eps, u_tangent);

  mfem::MultiVector Xp(2), Xm(2);
  Xp.MakeRef(0, u_plus);  Xp.MakeRef(1, *pmesh->GetNodes());
  Xm.MakeRef(0, u_minus); Xm.MakeRef(1, *pmesh->GetNodes());

  mfem::Vector res_p(N), res_m(N);
  res_p.UseDevice(true); res_m.UseDevice(true);
  mfem::MultiVector Yp(1), Ym(1);
  Yp.MakeRef(0, res_p);
  Ym.MakeRef(0, res_m);
  dop.Mult(Xp, Yp);
  dop.Mult(Xm, Ym);

  mfem::Vector fd_action(N);
  fd_action = res_p;
  fd_action -= res_m;
  fd_action *= 0.5 / eps;

  cached_mult.HostRead();
  assembled_action.HostRead();
  fd_action.HostRead();

  mfem::Vector diff(N);
  diff = cached_mult; diff -= fd_action;
  std::cout << "[SingleInput] ||Mult - FD|| = " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-5);

  diff = assembled_action; diff -= fd_action;
  std::cout << "[SingleInput] ||Assembled - FD|| = " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-5);

  // 4. MultTranspose vs assembled^T
  mfem::Vector v_dir(N);
  v_dir.UseDevice(true);
  v_dir.Randomize(29);

  mfem::Vector mt_action(N);
  mt_action.UseDevice(true);
  mt_action = 0.0;
  deriv_op->MultTranspose(v_dir, mt_action);

  mfem::Vector assembled_mt(N);
  assembled_mt.UseDevice(false);
  v_dir.HostRead();
  J->MultTranspose(v_dir, assembled_mt);

  mt_action.HostRead();
  assembled_mt.HostRead();

  diff = mt_action; diff -= assembled_mt;
  std::cout << "[SingleInput] ||MultTranspose - Assembled^T|| = "
            << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-8);

  // 5. Adjoint identity
  const double lhs = mfem::InnerProduct(v_dir, cached_mult);
  const double rhs = mfem::InnerProduct(u_tangent, mt_action);
  std::cout << "[SingleInput] <J*t, v> = " << lhs
            << "   <t, J^T*v> = " << rhs << std::endl;
  EXPECT_NEAR(lhs, rhs, 1.0e-10 * std::max(std::abs(lhs), 1.0));

  delete J;
}

/**
 * @brief Bugs 1 and 13 (Value-only variant) — Cached forward Mult and
 * MultTranspose are wrong when only Value<0> is the dependent input and
 * the Gradient comes from a separate non-differentiated field.
 *
 * Root causes: identical to Bug13_SingleInputTranspose (cache stride
 * mismatch in the forward path + incorrect transpose contraction), but
 * exercises a different Jacobian cache shape: Value has op_dim=1 vs
 * Gradient's op_dim=dim, so the per-QP Jacobian slice has fewer entries
 * and different stride arithmetic.
 *
 * The qfunction is `(1 + u(0) + 0.5*u(1)) * grad_u * w` where u (field 0)
 * is the Value input (dependent) and grad_u comes from field 1 (not
 * differentiated).  Both fields carry the same data, but only field 0
 * appears in derivative_ids.
 *
 * On the unpatched branch: forward Mult, assembled, MultTranspose, and
 * adjoint identity all fail.
 *
 * Fix: patch 0005 (cache indexing) + patch 0006 (per-input transpose
 * contraction).
 */
struct NonlinearValueOnlyQf {
  static constexpr int dim = 2;
  SMITH_HOST_DEVICE inline void operator()(
      const mfem::future::tensor<double, 2>& u,
      const mfem::future::tensor<double, 2, 2>& grad_u,
      const double& w,
      mfem::future::tensor<double, 2, 2>& out_grad_v) const
  {
    const double scale = (1.0 + u(0) + 0.5 * u(1)) * w;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        out_grad_v(i, j) = scale * grad_u(i, j);
      }
    }
  }
};

TEST_F(DfemBugsFixture, Bug13_ValueOnlyTranspose)
{
  int dim = pmesh->Dimension();
  mfem::H1_FECollection fec(1, dim);
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim);

  // Use two DIFFERENT field ids: id 0 for Value, id 1 for Gradient
  // Only differentiate w.r.t. id 0 (Value part)
  mfem::future::FieldDescriptor u_val_fd(0, &fes);  // Value input (dependent)
  mfem::future::FieldDescriptor u_grad_fd(1, &fes);  // Gradient input (NOT dependent)
  mfem::future::FieldDescriptor v_fd(2, &fes);

  mfem::future::DifferentiableOperator dop({u_val_fd, u_grad_fd}, {v_fd}, *pmesh);

  const mfem::IntegrationRule* ir =
      &mfem::IntRules.Get(pmesh->GetTypicalElementGeometry(), 2);
  mfem::Array<int> attrs(1);
  attrs[0] = 1;

  NonlinearValueOnlyQf qf;
  dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Value<0>{}, mfem::future::Gradient<1>{},
                           mfem::future::Weight{}},
      mfem::future::tuple{mfem::future::Gradient<2>{}},
      *ir, attrs,
      std::integer_sequence<size_t, 0>{});  // Only differentiate w.r.t. field 0

  const int N = fes.GetTrueVSize();
  mfem::Vector u(N);
  u.UseDevice(true);
  u.Randomize(7);

  mfem::MultiVector X(2);
  X.MakeRef(0, u);
  X.MakeRef(1, u);  // Same data, but field id 1 is not differentiated

  auto deriv_op = dop.GetDerivative(0, X);

  mfem::Vector u_tangent(N);
  u_tangent.UseDevice(true);
  u_tangent.Randomize(13);

  // Forward cached Mult
  mfem::Vector cached_mult(N);
  cached_mult.UseDevice(true);
  cached_mult = 0.0;
  deriv_op->Mult(u_tangent, cached_mult);

  // Assembled
  mfem::SparseMatrix* J = nullptr;
  deriv_op->Assemble(J);
  ASSERT_NE(J, nullptr);

  mfem::Vector assembled_action(N);
  assembled_action.UseDevice(false);
  u_tangent.HostRead();
  J->Mult(u_tangent, assembled_action);

  // FD
  constexpr double eps = 1.0e-7;
  mfem::Vector u_plus(u), u_minus(u);
  u_plus.Add(eps, u_tangent);
  u_minus.Add(-eps, u_tangent);

  mfem::MultiVector Xp(2), Xm(2);
  Xp.MakeRef(0, u_plus);  Xp.MakeRef(1, u);
  Xm.MakeRef(0, u_minus); Xm.MakeRef(1, u);

  mfem::Vector res_p(N), res_m(N);
  res_p.UseDevice(true); res_m.UseDevice(true);
  mfem::MultiVector Yp(1), Ym(1);
  Yp.MakeRef(0, res_p);
  Ym.MakeRef(0, res_m);
  dop.Mult(Xp, Yp);
  dop.Mult(Xm, Ym);

  mfem::Vector fd_action(N);
  fd_action = res_p;
  fd_action -= res_m;
  fd_action *= 0.5 / eps;

  cached_mult.HostRead();
  assembled_action.HostRead();
  fd_action.HostRead();

  mfem::Vector diff(N);
  diff = cached_mult; diff -= fd_action;
  std::cout << "[ValueOnly] ||Mult - FD|| = " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-5);

  diff = assembled_action; diff -= fd_action;
  std::cout << "[ValueOnly] ||Assembled - FD|| = " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-5);

  // MultTranspose vs assembled^T
  mfem::Vector v_dir(N);
  v_dir.UseDevice(true);
  v_dir.Randomize(29);

  mfem::Vector mt_action(N);
  mt_action.UseDevice(true);
  mt_action = 0.0;
  deriv_op->MultTranspose(v_dir, mt_action);

  mfem::Vector assembled_mt(N);
  assembled_mt.UseDevice(false);
  v_dir.HostRead();
  J->MultTranspose(v_dir, assembled_mt);

  mt_action.HostRead();
  assembled_mt.HostRead();

  diff = mt_action; diff -= assembled_mt;
  std::cout << "[ValueOnly] ||MultTranspose - Assembled^T|| = "
            << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-8);

  const double lhs = mfem::InnerProduct(v_dir, cached_mult);
  const double rhs = mfem::InnerProduct(u_tangent, mt_action);
  std::cout << "[ValueOnly] <J*t, v> = " << lhs
            << "   <t, J^T*v> = " << rhs << std::endl;
  EXPECT_NEAR(lhs, rhs, 1.0e-10 * std::max(std::abs(lhs), 1.0));

  delete J;
}

/**
 * @brief Bug 10 — `GlobalQFBackend` is not a drop-in replacement for
 * the per-QP qfunction style used by `LocalQFBackend`.
 *
 * Root cause: `GlobalQFImpl::Action::operator()` enforces a compile-time
 * `static_assert` requiring
 * `detail::supports_tensor_array_qfunc<qfunc_t, inputs_t, outputs_t>`.
 * This trait (in `fem/dfem/backends/util.hpp`) checks that every qfunction
 * parameter is a `tensor_array<scalar_t, Dims...>` — a batched type
 * spanning all quadrature points simultaneously.  Per-QP qfunctions use
 * bare `double`, `tensor<double, ...>`, or mutable-reference output
 * parameters, none of which satisfy `is_tensor_array`.
 *
 * Symptom: switching from `LocalQFBackend` to `GlobalQFBackend` on any
 * per-QP qfunction causes a compile-time `static_assert` failure.
 * `GlobalQFBackend` therefore cannot serve as a fallback for boundary
 * integrals or any other use case where `LocalQFBackend` is deficient.
 *
 * This test verifies the trait returns `false` for representative per-QP
 * qfunctions (scalar and vector-gradient), confirming the incompatibility
 * at compile time.
 *
 * Fix: UNFIXED.  Either add a per-QP adapter to `GlobalQFBackend` or
 * document that the two backends require different qfunction conventions.
 */
TEST_F(DfemBugsFixture, Bug10_GlobalQFBackendIncompatible)
{
  using scalar_inputs_t = mfem::future::tuple<mfem::future::Value<0>,
                                               mfem::future::Weight>;
  using scalar_outputs_t = mfem::future::tuple<mfem::future::Value<1>>;

  constexpr bool scalar_compatible =
      mfem::future::detail::supports_tensor_array_qfunc<
          BoundaryMassQf, scalar_inputs_t, scalar_outputs_t>::value;
  EXPECT_FALSE(scalar_compatible)
      << "Scalar per-QP qfunction should NOT pass the tensor_array trait";

  using vector_inputs_t = mfem::future::tuple<
      mfem::future::Gradient<0>, mfem::future::Value<2>, mfem::future::Weight>;
  using vector_outputs_t = mfem::future::tuple<mfem::future::Gradient<1>>;

  constexpr bool vector_compatible =
      mfem::future::detail::supports_tensor_array_qfunc<
          VectorGradientQf, vector_inputs_t, vector_outputs_t>::value;
  EXPECT_FALSE(vector_compatible)
      << "Vector-gradient per-QP qfunction should NOT pass the tensor_array trait";
}

} // namespace dfem_bugs_test

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager app(argc, argv);
  return RUN_ALL_TESTS();
}
