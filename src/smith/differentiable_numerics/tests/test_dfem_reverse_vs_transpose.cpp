// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// =====================================================================
// Verify that smith's reverse-mode dFEM operator (built via
// AddReverseGradientIntegrator in dfem_reverse_derivative.hpp) computes
// the same J^T*v as the forward DerivativeOperator's MultTranspose and
// the assembled SparseMatrix's MultTranspose.
//
// Math: given residual R(u) registered on a forward DifferentiableOperator
// `fwd` with one output field, J = ∂R/∂u at the current primal state.
//
//   forward path:    J^T * v   ==   fwd_deriv->MultTranspose(v)
//                                ==   J_assembled->MultTranspose(v)
//   reverse path:    J^T * v   ==   rev->Mult({u, v}, {y})  where y is the
//                                   ∂/∂u of the synthesized scalar density
//                                   <v, R(u)>.
//
// The three should agree to roundoff.
// =====================================================================

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "mfem/fem/dfem/backends/local_qf/prelude.hpp"
#include "mfem/fem/dfem/doperator.hpp"
#include "smith/differentiable_numerics/dfem_reverse_derivative.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace dfem_reverse_vs_transpose_test {

// Asymmetric nonlinear residual on a vector H1 field (dim=2).
// R(u)_{Gradient<V>} = (1 + u(0) + 0.5*u(1)) * grad_u * w
// Both Value<0> and Gradient<0> are dependent inputs; J = ∂R/∂u is non-symmetric.
struct NonlinearVectorQf {
  static constexpr int dim = 2;
  SMITH_HOST_DEVICE inline void operator()(const mfem::future::tensor<double, 2>& u,
                                           const mfem::future::tensor<double, 2, 2>& grad_u, const double& w,
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

struct DfemReverseVsTransposeFixture : public testing::Test {
  std::unique_ptr<mfem::ParMesh> pmesh;

  void SetUp() override
  {
    auto mesh = mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    pmesh     = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  }
};

TEST_F(DfemReverseVsTransposeFixture, ReverseModeMatchesAssembledTranspose)
{
  const int dim = pmesh->Dimension();
  mfem::H1_FECollection      fec(1, dim);
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim);

  constexpr int U_ID = 0;
  constexpr int V_ID = 1;

  mfem::future::FieldDescriptor u_fd(U_ID, &fes);
  mfem::future::FieldDescriptor v_fd(V_ID, &fes);

  // ---- forward operator: R: u -> v ----
  mfem::future::DifferentiableOperator fwd({u_fd}, {v_fd}, *pmesh);

  const mfem::IntegrationRule* ir = &mfem::IntRules.Get(pmesh->GetTypicalElementGeometry(), 2);
  mfem::Array<int>             attrs(1);
  attrs[0] = 1;

  NonlinearVectorQf qf;
  fwd.AddDomainIntegrator<mfem::future::LocalQFBackend>(
      qf,
      mfem::future::tuple{mfem::future::Value<U_ID>{}, mfem::future::Gradient<U_ID>{}, mfem::future::Weight{}},
      mfem::future::tuple{mfem::future::Gradient<V_ID>{}}, *ir, attrs, std::integer_sequence<size_t, U_ID>{});

  // primal state
  const int   N = fes.GetTrueVSize();
  mfem::Vector u(N);
  u.UseDevice(true);
  u.Randomize(7);

  // seed v (dual to residual space, same finite element space here)
  mfem::Vector v_seed(N);
  v_seed.UseDevice(true);
  v_seed.Randomize(29);

  mfem::MultiVector X_fwd(1);
  X_fwd.MakeRef(0, u);

  // ---- forward derivative + assembled Jacobian ----
  auto              fwd_deriv = fwd.GetDerivative(0, X_fwd);
  mfem::SparseMatrix* J       = nullptr;
  fwd_deriv->Assemble(J);
  ASSERT_NE(J, nullptr);

  // J^T * v from the assembled matrix (reference truth)
  mfem::Vector assembled_jt_v(N);
  assembled_jt_v.UseDevice(false);
  v_seed.HostRead();
  J->MultTranspose(v_seed, assembled_jt_v);

  // J^T * v from the forward DerivativeOperator's MultTranspose
  mfem::Vector fwd_mt(N);
  fwd_mt.UseDevice(true);
  fwd_mt = 0.0;
  fwd_deriv->MultTranspose(v_seed, fwd_mt);
  fwd_mt.HostRead();

  // ---- reverse-mode operator built via smith's factory ----
  // Inputs: u (field 0), v_seed (field 1).  Output: ∂/∂u of <v, R(u)> = J^T*v (field 0).
  mfem::future::DifferentiableOperator rev({u_fd, v_fd}, {u_fd}, *pmesh);
  mfem::future::reverse::AddReverseGradientIntegrator<V_ID, U_ID>(
      rev, qf,
      mfem::future::tuple{mfem::future::Value<U_ID>{}, mfem::future::Gradient<U_ID>{}, mfem::future::Weight{}},
      mfem::future::tuple{mfem::future::Gradient<V_ID>{}}, *ir, attrs,
      std::integer_sequence<std::size_t, U_ID>{});

  mfem::MultiVector X_rev(2);
  X_rev.MakeRef(0, u);
  X_rev.MakeRef(1, v_seed);

  mfem::Vector rev_out(N);
  rev_out.UseDevice(true);
  rev_out = 0.0;
  mfem::MultiVector Y_rev(1);
  Y_rev.MakeRef(0, rev_out);

  rev.Mult(X_rev, Y_rev);
  rev_out.HostRead();

  // ---- comparisons ----
  mfem::Vector diff(N);
  const double ref_norm = std::max(assembled_jt_v.Norml2(), 1.0);

  diff = fwd_mt;
  diff -= assembled_jt_v;
  std::cout << "||fwd.MultTranspose(v) - J^T*v|| = " << diff.Norml2() << "  (ref ||J^T*v||=" << assembled_jt_v.Norml2()
            << ")" << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-10 * ref_norm);

  diff = rev_out;
  diff -= assembled_jt_v;
  std::cout << "||rev.Mult({u,v}) - J^T*v|| = " << diff.Norml2() << std::endl;
  EXPECT_NEAR(diff.Norml2(), 0.0, 1.0e-10 * ref_norm);

  // adjoint identity as a third independent witness
  mfem::Vector u_tangent(N);
  u_tangent.UseDevice(true);
  u_tangent.Randomize(13);
  mfem::Vector j_t(N);
  j_t.UseDevice(true);
  j_t = 0.0;
  fwd_deriv->Mult(u_tangent, j_t);
  j_t.HostRead();
  u_tangent.HostRead();

  const double lhs = mfem::InnerProduct(v_seed, j_t);
  const double rhs = mfem::InnerProduct(u_tangent, rev_out);
  std::cout << "<J*t, v> = " << lhs << "   <t, rev->Mult>= " << rhs << std::endl;
  EXPECT_NEAR(lhs, rhs, 1.0e-10 * std::max(std::abs(lhs), 1.0));

  delete J;
}

}  // namespace dfem_reverse_vs_transpose_test

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv, MPI_COMM_WORLD, true, smith::ExecutionSpace::CPU);
  return RUN_ALL_TESTS();
}
