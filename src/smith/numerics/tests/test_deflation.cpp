// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/numerics/deflation.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/numerics/bsr_operator.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace {

struct ProblemSetup {
  std::unique_ptr<mfem::ParMesh> pmesh;
  std::unique_ptr<mfem::H1_FECollection> fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes;
  std::unique_ptr<mfem::HypreParMatrix> A;
};

ProblemSetup makeCube(mfem::Ordering::Type ordering, double translate = 0.0, int order = 1)
{
  ProblemSetup s;
  auto serial_mesh = mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0);
  if (translate != 0.0) {
    for (int v = 0; v < serial_mesh.GetNV(); ++v) {
      double* coord = serial_mesh.GetVertex(v);
      coord[0] += translate;
      coord[1] += translate;
      coord[2] += translate;
    }
  }
  s.pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
  constexpr int dim = 3;
  s.fec = std::make_unique<mfem::H1_FECollection>(order, dim);
  s.fes = std::make_unique<mfem::ParFiniteElementSpace>(s.pmesh.get(), s.fec.get(), dim, ordering);

  mfem::ConstantCoefficient one(1.0), mass(0.1);
  mfem::ParBilinearForm a(s.fes.get());
  a.AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(one));
  a.AddDomainIntegrator(new mfem::VectorMassIntegrator(mass));
  a.Assemble();
  a.Finalize();
  s.A.reset(a.ParallelAssemble());
  return s;
}

// Build v = Wα where α is set GLOBALLY (so all ranks agree). Uses each rank's own α slice
// to multiply its own W_local_ columns; result lives only on the rank's local tdofs since
// W has disjoint supports across ranks.
std::pair<mfem::Vector, mfem::Vector> makeAffine(const smith::DeflationPreconditioner& defl, int vec_tsize)
{
  mfem::Vector alpha(defl.numGlobalColumns());
  for (int i = 0; i < alpha.Size(); ++i) {
    alpha(i) = 0.37 * (i + 1) - 0.05 * i * i;
  }
  mfem::Vector v(vec_tsize);
  v = 0.0;
  const auto& Wcols = defl.localColumns();
  // each rank's local α slice begins at my_rank * numLocalColumns
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int offset = my_rank * defl.numLocalColumns();
  for (int j = 0; j < defl.numLocalColumns(); ++j) {
    v.Add(alpha(offset + j), Wcols[static_cast<size_t>(j)]);
  }
  return {alpha, v};
}

void symmetrize(mfem::DenseMatrix& matrix)
{
  for (int j = 0; j < matrix.Width(); ++j) {
    for (int i = 0; i < j; ++i) {
      const double s = 0.5 * (matrix(i, j) + matrix(j, i));
      matrix(i, j) = matrix(j, i) = s;
    }
  }
}

}  // namespace

TEST(Deflation, AffinePatchPureCoarse_byNODES)
{
  auto s = makeCube(mfem::Ordering::byNODES);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.SetOperator(*s.A);

  auto [alpha, v] = makeAffine(defl, s.fes->GetTrueVSize());
  mfem::Vector r(v.Size());
  s.A->Mult(v, r);
  r *= -1.0;

  mfem::Vector z0(v.Size());
  defl.coarseSolve(r, z0);

  mfem::Vector diff(z0);
  diff -= v;
  double err = diff.Norml2(), ref = v.Norml2();
  // global norm via the ParGridFunction route would be better; on 1 rank Norml2 == global.
  EXPECT_LT(err / ref, 1.0e-10);
}

TEST(Deflation, AffinePatchPureCoarse_byVDIM)
{
  auto s = makeCube(mfem::Ordering::byVDIM);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.SetOperator(*s.A);

  auto [alpha, v] = makeAffine(defl, s.fes->GetTrueVSize());
  mfem::Vector r(v.Size());
  s.A->Mult(v, r);
  r *= -1.0;

  mfem::Vector z0(v.Size());
  defl.coarseSolve(r, z0);

  mfem::Vector diff(z0);
  diff -= v;
  EXPECT_LT(diff.Norml2() / v.Norml2(), 1.0e-10);
}

// Pure-coarse manufactured test for the QUADRATIC basis: build v = W * alpha for a random alpha
// with the quadratic-order W, form RHS r = -A * v, and check defl.coarseSolve(r) ≈ v.
// If u_exact lies in range(W_quadratic), the coarse-solve recovers it exactly (to round-off).
// Catches W/WtAW/coarse-solve bugs that manifest only at quadratic order across ranks.
TEST(Deflation, QuadraticPatchPureCoarse_byVDIM)
{
  auto s = makeCube(mfem::Ordering::byVDIM);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.setDeflationOrder(smith::DeflationOrder::Quadratic);
  defl.SetOperator(*s.A);
  EXPECT_TRUE(defl.coarseIsSPD());

  auto [alpha, v] = makeAffine(defl, s.fes->GetTrueVSize());
  mfem::Vector r(v.Size());
  s.A->Mult(v, r);
  r *= -1.0;

  mfem::Vector z0(v.Size());
  defl.coarseSolve(r, z0);

  mfem::Vector diff(z0);
  diff -= v;
  double err_sq_local = diff * diff, ref_sq_local = v * v;
  double err_sq = 0.0, ref_sq = 0.0;
  MPI_Allreduce(&err_sq_local, &err_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ref_sq_local, &ref_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_LT(std::sqrt(err_sq / std::max(ref_sq, 1e-30)), 1.0e-9);
}

TEST(Deflation, QuadraticCoarseSolveDropsNumericalNullspace)
{
  auto s = makeCube(mfem::Ordering::byVDIM);
  *s.A *= -1.0;
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.setDeflationOrder(smith::DeflationOrder::Quadratic);
  const char* old_tol = std::getenv("SMITH_DEFLATION_RANK_TOL");
  const std::string old_tol_value = old_tol ? old_tol : "";
  setenv("SMITH_DEFLATION_RANK_TOL", "2.0", 1);
  defl.SetOperator(*s.A);
  if (old_tol) {
    setenv("SMITH_DEFLATION_RANK_TOL", old_tol_value.c_str(), 1);
  } else {
    unsetenv("SMITH_DEFLATION_RANK_TOL");
  }

  mfem::DenseMatrix M(defl.coarseMatrix());
  symmetrize(M);
  mfem::DenseMatrixEigensystem eig(M);
  eig.Eval();
  const mfem::Vector& evals = eig.Eigenvalues();
  const mfem::DenseMatrix& evecs = eig.Eigenvectors();

  int expected_rank = 0;
  int dropped = -1;
  for (int i = 0; i < evals.Size(); ++i) {
    if (evals(i) > 0.0 || evals(i) < -defl.coarseSpectralTolerance()) {
      ++expected_rank;
    } else if (dropped < 0) {
      dropped = i;
    }
  }
  EXPECT_EQ(defl.coarseSpectralRank(), expected_rank);
  ASSERT_GE(dropped, 0);
  ASSERT_LT(defl.coarseSpectralRank(), defl.numGlobalColumns());
  EXPECT_TRUE(defl.coarseIsSPD());

  (void)evecs;
}

// Quadratic basis, PCG: rhs lies in range(W_quadratic), so the deflation coarse correction
// should solve it exactly on the first iteration. Hard companion to the pure-coarse test —
// catches preconditioner-side bugs (smoother interference, masking) that pure-coarse misses.
TEST(Deflation, QuadraticPatch_CG_OneIter_byVDIM)
{
  auto s = makeCube(mfem::Ordering::byVDIM);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.setDeflationOrder(smith::DeflationOrder::Quadratic);
  defl.SetOperator(*s.A);
  auto [alpha, v] = makeAffine(defl, s.fes->GetTrueVSize());

  mfem::Vector b(v.Size());
  s.A->Mult(v, b);

  mfem::CGSolver cg(MPI_COMM_WORLD);
  cg.SetOperator(*s.A);
  cg.SetPreconditioner(defl);
  cg.SetRelTol(1e-12);
  cg.SetAbsTol(1e-14);
  cg.SetMaxIter(50);
  cg.SetPrintLevel(0);
  cg.iterative_mode = false;

  mfem::Vector x(v.Size());
  x = 0.0;
  cg.Mult(b, x);

  std::cout << "[QuadraticOneIter] CG iters=" << cg.GetNumIterations() << "\n";
  EXPECT_TRUE(cg.GetConverged());
  EXPECT_LE(cg.GetNumIterations(), 1);
}

// Order-2 FES: mid-edge / mid-face tdofs are shared across partition boundaries, so the
// off-diagonal blocks of W^T A W carry more information than at order 1. Stresses the
// cross-rank assembly + masking interplay that the shallow_arch_buckling test (p=2) exercises.
TEST(Deflation, QuadraticPatchPureCoarse_byVDIM_Order2)
{
  auto s = makeCube(mfem::Ordering::byVDIM, /*translate=*/0.0, /*order=*/2);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.setDeflationOrder(smith::DeflationOrder::Quadratic);
  defl.SetOperator(*s.A);

  auto [alpha, v] = makeAffine(defl, s.fes->GetTrueVSize());
  mfem::Vector r(v.Size());
  s.A->Mult(v, r);
  r *= -1.0;

  mfem::Vector z0(v.Size());
  defl.coarseSolve(r, z0);

  mfem::Vector diff(z0);
  diff -= v;
  double err_sq_local = diff * diff, ref_sq_local = v * v;
  double err_sq = 0.0, ref_sq = 0.0;
  MPI_Allreduce(&err_sq_local, &err_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ref_sq_local, &ref_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_LT(std::sqrt(err_sq / std::max(ref_sq, 1e-30)), 1.0e-9);
}

TEST(Deflation, QuadraticPatch_CG_OneIter_byVDIM_Order2)
{
  auto s = makeCube(mfem::Ordering::byVDIM, /*translate=*/0.0, /*order=*/2);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.setDeflationOrder(smith::DeflationOrder::Quadratic);
  defl.SetOperator(*s.A);
  auto [alpha, v] = makeAffine(defl, s.fes->GetTrueVSize());

  mfem::Vector b(v.Size());
  s.A->Mult(v, b);

  mfem::CGSolver cg(MPI_COMM_WORLD);
  cg.SetOperator(*s.A);
  cg.SetPreconditioner(defl);
  cg.SetRelTol(1e-12);
  cg.SetAbsTol(1e-14);
  cg.SetMaxIter(50);
  cg.SetPrintLevel(0);
  cg.iterative_mode = false;

  mfem::Vector x(v.Size());
  x = 0.0;
  cg.Mult(b, x);

  std::cout << "[QuadraticOneIterOrder2] CG iters=" << cg.GetNumIterations() << "\n";
  EXPECT_TRUE(cg.GetConverged());
  EXPECT_LE(cg.GetNumIterations(), 1);
}

TEST(Deflation, AffinePatch_CG_OneIter_byVDIM)
{
  auto s = makeCube(mfem::Ordering::byVDIM);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.SetOperator(*s.A);
  auto [alpha, v] = makeAffine(defl, s.fes->GetTrueVSize());

  mfem::Vector b(v.Size());
  s.A->Mult(v, b);

  mfem::CGSolver cg(MPI_COMM_WORLD);
  cg.SetOperator(*s.A);
  cg.SetPreconditioner(defl);
  cg.SetRelTol(1e-12);
  cg.SetAbsTol(1e-14);
  cg.SetMaxIter(50);
  cg.SetPrintLevel(0);
  cg.iterative_mode = false;

  mfem::Vector x(v.Size());
  x = 0.0;
  cg.Mult(b, x);

  std::cout << "[OneIter] CG iters=" << cg.GetNumIterations() << "\n";
  EXPECT_TRUE(cg.GetConverged());
  EXPECT_LE(cg.GetNumIterations(), 1);
}

// After basis construction (centered), each linear-mode column should have mean zero on its
// component's active (= unmasked) tdofs. Verified by dotting against the constant mode.
TEST(Deflation, CenteredLinearModesHaveZeroMean)
{
  auto s = makeCube(mfem::Ordering::byVDIM, /*translate=*/100.0);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.SetOperator(*s.A);

  constexpr int dim = 3;
  const auto& cols = defl.localColumns();
  for (int c = 0; c < dim; ++c) {
    const auto& const_col = cols[static_cast<size_t>(c * (dim + 1))];
    double n_active = const_col.Sum();
    if (n_active <= 0.0) continue;
    for (int k = 1; k <= dim; ++k) {
      const auto& lin_col = cols[static_cast<size_t>(c * (dim + 1) + k)];
      double mean = lin_col.Sum() / n_active;
      EXPECT_LT(std::abs(mean), 1e-12) << "component " << c << " linear k=" << k;
    }
  }
}

// Translated mesh: span of W is unchanged so the pure-coarse identity on affine RHS must
// still hold to machine precision. Sanity that centering didn't break span.
TEST(Deflation, AffinePatchPureCoarse_TranslatedMesh)
{
  auto s = makeCube(mfem::Ordering::byVDIM, /*translate=*/100.0);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.SetOperator(*s.A);

  auto [alpha, v] = makeAffine(defl, s.fes->GetTrueVSize());
  mfem::Vector r(v.Size());
  s.A->Mult(v, r);
  r *= -1.0;

  mfem::Vector z0(v.Size());
  defl.coarseSolve(r, z0);

  mfem::Vector diff(z0);
  diff -= v;
  EXPECT_LT(diff.Norml2() / v.Norml2(), 1.0e-10);
}

// TR hooks round-trip. WtAW is rank-deficient on this small cube (some ranks own too few
// vertices for the per-rank affine basis to be independent), so alpha_got needn't equal
// alpha_expected — but applying W to it must still reconstruct v.
TEST(Deflation, TrustRegionHooks_RoundTrip)
{
  auto s = makeCube(mfem::Ordering::byVDIM);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.SetOperator(*s.A);

  auto [alpha_expected, v] = makeAffine(defl, s.fes->GetTrueVSize());

  // applyW with this rank's owned slice of alpha reconstructs v exactly (W has disjoint
  // supports across ranks, so v is fully captured by the rank-local W slice).
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  mfem::Vector alpha_local(alpha_expected.GetData() + my_rank * defl.numLocalColumns(), defl.numLocalColumns());
  mfem::Vector v_reconstructed(v.Size());
  v_reconstructed = 0.0;
  defl.applyW(alpha_local, v_reconstructed);
  mfem::Vector vdiff(v_reconstructed);
  vdiff -= v;
  EXPECT_LT(vdiff.Norml2() / v.Norml2(), 1.0e-12);

  // Round-trip through solveCoarse: c = W^T A v, alpha_got = WtAW^{-1} c, then check
  // W * alpha_got reconstructs v. This is what trust-region actually needs.
  mfem::Vector Av(v.Size());
  s.A->Mult(v, Av);
  mfem::Vector c_local;
  defl.applyWtranspose(Av, c_local);
  EXPECT_EQ(c_local.Size(), defl.numLocalColumns());

  mfem::Vector alpha_got;
  defl.solveCoarse(c_local, alpha_got);
  ASSERT_EQ(alpha_got.Size(), alpha_expected.Size());

  mfem::Vector alpha_got_local(alpha_got.GetData() + my_rank * defl.numLocalColumns(), defl.numLocalColumns());
  mfem::Vector v_from_solve(v.Size());
  v_from_solve = 0.0;
  defl.applyW(alpha_got_local, v_from_solve);
  mfem::Vector vd2(v_from_solve);
  vd2 -= v;
  EXPECT_LT(vd2.Norml2() / v.Norml2(), 1.0e-9);
}

// Leftmost eigenpair: must equal the smallest retained eigenvalue of WtAW. Near-null modes
// below the rank-revealing tolerance are dropped and should not be exposed to trust-region.
TEST(Deflation, CoarseLeftmostEigenpair)
{
  auto s = makeCube(mfem::Ordering::byVDIM);
  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.SetOperator(*s.A);

  const double lam = defl.coarseLeftmostEigenvalue();

  mfem::DenseMatrix M(defl.coarseMatrix());
  symmetrize(M);
  mfem::DenseMatrixEigensystem eig(M);
  eig.Eval();
  const mfem::Vector& evs = eig.Eigenvalues();
  double evmin = std::numeric_limits<double>::infinity();
  for (int i = 0; i < evs.Size(); ++i) {
    if (evs(i) > 0.0 || evs(i) < -defl.coarseSpectralTolerance()) evmin = std::min(evmin, evs(i));
  }
  ASSERT_TRUE(std::isfinite(evmin));
  EXPECT_NEAR(lam, evmin, 1e-10 * std::max(1.0, std::abs(evmin)));

  // d = W * v: must be nonzero (so it's a usable direction in the TR subspace).
  mfem::Vector d;
  defl.coarseLeftmostDirection(d);
  EXPECT_EQ(d.Size(), s.fes->GetTrueVSize());
  double dnorm_sq_local = d * d, dnorm_sq = 0.0;
  MPI_Allreduce(&dnorm_sq_local, &dnorm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_GT(dnorm_sq, 1.0e-30);
}

TEST(Deflation, IndefiniteCoarseUsesSmootherOnlyAndReportsNegativeDirection)
{
  auto s = makeCube(mfem::Ordering::byVDIM);
  *s.A *= -1.0;

  smith::DeflationPreconditioner defl(*s.fes, false);
  defl.SetOperator(*s.A);
  EXPECT_FALSE(defl.coarseIsSPD());
  EXPECT_LT(defl.coarseLeftmostEigenvalue(), 0.0);

  mfem::Vector rhs(s.fes->GetTrueVSize());
  rhs = 1.0;
  mfem::Vector z(rhs.Size());
  defl.Mult(rhs, z);
  EXPECT_EQ(z.Size(), rhs.Size());

  mfem::Vector d;
  defl.coarseLeftmostDirection(d);
  double norm2_local = d * d;
  double norm2 = 0.0;
  MPI_Allreduce(&norm2_local, &norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_GT(norm2, 1.0e-30);
}

namespace {

// Build a 3D cantilever beam: x ∈ [0, Lx], y ∈ [0, Ly], z ∈ [0, Lz].
// Bdr attr 1 == fixed (x=0 face), 2 == tip-load (x=Lx face), 3 == free elsewhere.
mfem::Mesh makeBeamMesh(int nx, int ny, int nz, double Lx, double Ly, double Lz)
{
  auto serial_mesh = mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  for (int i = 0; i < serial_mesh.GetNBE(); ++i) {
    auto* be = serial_mesh.GetBdrElement(i);
    mfem::Array<int> verts;
    be->GetVertices(verts);
    double xc = 0.0;
    for (int v = 0; v < verts.Size(); ++v) xc += serial_mesh.GetVertex(verts[v])[0];
    xc /= verts.Size();
    if (xc < 1e-8) {
      be->SetAttribute(1);
    } else if (xc > Lx - 1e-8) {
      be->SetAttribute(2);
    } else {
      be->SetAttribute(3);
    }
  }
  serial_mesh.SetAttributes();
  return serial_mesh;
}

}  // namespace

// Multi-rank beam test. Uses the equation_solver factory to wire the Deflation
// preconditioner through Preconditioner::Deflation. Compares CG iteration counts under
// Jacobi, BoomerAMG, and Deflation. Runs across however many MPI ranks the test was
// launched with.
TEST(Deflation, CantileverBeam_PreconditionerComparison)
{
  int my_rank, n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  // === TIMING BEGIN === Slenderer beam (Lx/Ly = 16) ~98k dofs (3 * 193 * 13 * 13 = 97851).
  constexpr int nx = 192, ny = 12, nz = 12;
  constexpr double Lx = 16.0, Ly = 1.0, Lz = 1.0;
  // === TIMING END ===
  auto serial_mesh = makeBeamMesh(nx, ny, nz, Lx, Ly, Lz);
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);

  constexpr int order = 1;
  constexpr int dim = 3;
  mfem::H1_FECollection fec(order, dim);
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim, mfem::Ordering::byVDIM);

  mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  mfem::Array<int> ess_tdofs;
  fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

  mfem::ConstantCoefficient lambda(1.0), mu(1.0);
  mfem::ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda, mu));
  a.Assemble();

  mfem::Vector traction(dim);
  traction = 0.0;
  traction(2) = -0.01;
  mfem::VectorConstantCoefficient trac_coef(traction);
  mfem::Array<int> tip_bdr(pmesh->bdr_attributes.Max());
  tip_bdr = 0;
  tip_bdr[1] = 1;
  mfem::ParLinearForm b(&fes);
  b.AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(trac_coef), tip_bdr);
  b.Assemble();

  mfem::ParGridFunction x_gf(&fes);
  x_gf = 0.0;
  mfem::HypreParMatrix A;
  mfem::Vector X, B;
  a.FormLinearSystem(ess_tdofs, x_gf, b, A, X, B);

  auto runCG = [&](smith::Preconditioner pc, const char* label, smith::CoarseMode dmode = smith::CoarseMode::Additive,
                   bool use_bsr = false) -> int {
    smith::LinearSolverOptions opts;
    opts.linear_solver = smith::LinearSolver::CG;
    opts.preconditioner = pc;
    opts.relative_tol = 0.0;
    opts.absolute_tol = 1e-10;
    opts.max_iterations = 10000;
    opts.print_level = 0;
    opts.deflation_fes = (pc == smith::Preconditioner::Deflation) ? &fes : nullptr;
    opts.use_bsr_spmv = use_bsr;

    auto [lin_solver, prec] = smith::buildLinearSolverAndPreconditioner(opts, MPI_COMM_WORLD);

    smith::DeflationPreconditioner* dp = nullptr;
    if (pc == smith::Preconditioner::Deflation) {
      dp = dynamic_cast<smith::DeflationPreconditioner*>(prec.get());
      EXPECT_NE(dp, nullptr);
      if (dp) {
        dp->setEssentialTrueDofs(ess_tdofs);
        dp->setCoarseMode(dmode);
      }
    }

    auto* iter = dynamic_cast<mfem::IterativeSolver*>(lin_solver.get());
    EXPECT_NE(iter, nullptr);
    if (!iter) return -1;

    std::unique_ptr<smith::BSROperator> bsr_op;
    mfem::Operator* op = &A;
    if (use_bsr) {
      bsr_op = std::make_unique<smith::BSROperator>(&A, 3);
      op = bsr_op.get();
    }

    // === TIMING BEGIN ===
    MPI_Barrier(MPI_COMM_WORLD);
    double t_setop = MPI_Wtime();
    // === TIMING END ===
    iter->SetOperator(*op);
    // === TIMING BEGIN ===
    MPI_Barrier(MPI_COMM_WORLD);
    t_setop = MPI_Wtime() - t_setop;
    // === TIMING END ===

    mfem::Vector Xs(X.Size());
    Xs = 0.0;
    // === TIMING BEGIN ===
    MPI_Barrier(MPI_COMM_WORLD);
    double t_solve = MPI_Wtime();
    // === TIMING END ===
    iter->Mult(B, Xs);
    // === TIMING BEGIN ===
    MPI_Barrier(MPI_COMM_WORLD);
    t_solve = MPI_Wtime() - t_solve;
    // === TIMING END ===

    // True (unpreconditioned) residual ||B - A*X|| for apples-to-apples comparison across
    // different preconditioners (mfem CG's abs_tol is on the M-norm, not the 2-norm).
    mfem::Vector resid(B.Size());
    A.Mult(Xs, resid);
    resid -= B;
    resid.Neg();
    double rnorm_sq_local = resid * resid, rnorm_sq = 0.0;
    MPI_Allreduce(&rnorm_sq_local, &rnorm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double true_rnorm = std::sqrt(rnorm_sq);

    int n_ess_local = ess_tdofs.Size(), n_ess_global = 0;
    MPI_Allreduce(&n_ess_local, &n_ess_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (my_rank == 0) {
      std::cout << "[Beam " << label << "] ranks=" << n_ranks << " dofs=" << fes.GlobalTrueVSize()
                << " ess_tdofs=" << n_ess_global << " iters=" << iter->GetNumIterations()
                << " converged=" << iter->GetConverged() << " ||r||="
                << true_rnorm
                // === TIMING BEGIN ===
                << " t_setop=" << t_setop << "s t_solve=" << t_solve << "s"
                << " t_per_iter=" << (iter->GetNumIterations() > 0 ? t_solve / iter->GetNumIterations() : 0.0)
                << "s"
                // === TIMING END ===
                << "\n";
      // === TIMING BEGIN === deflation internal breakdown
      if (dp) {
        std::cout << "  [Deflation breakdown] setop: matvec=" << dp->setopMatvecTime()
                  << "s factor=" << dp->setopFactorTime() << "s smoother_setup=" << dp->setopSmootherTime() << "s"
                  << " | mult(" << dp->multCalls() << " calls): total=" << dp->multTotalTime()
                  << "s smoother=" << dp->multSmootherTime() << "s coarse=" << dp->multCoarseTime() << "s\n";
      }
      // === TIMING END ===
    }
    EXPECT_TRUE(iter->GetConverged()) << label;
    return iter->GetNumIterations();
  };

  int iters_jac = runCG(smith::Preconditioner::HypreJacobi, "Jacobi");
  int iters_amg = runCG(smith::Preconditioner::HypreAMG, "HypreAMG");
  int iters_def_add = runCG(smith::Preconditioner::Deflation, "Deflation_Add", smith::CoarseMode::Additive);
  int iters_def_add_bsr =
      runCG(smith::Preconditioner::Deflation, "Deflation_Add_BSR", smith::CoarseMode::Additive, true);
  int iters_def_loc = runCG(smith::Preconditioner::Deflation, "Deflation_AddLocal", smith::CoarseMode::AdditiveLocal);
  int iters_def_sch = runCG(smith::Preconditioner::Deflation, "Deflation_Schwarz", smith::CoarseMode::AdditiveSchwarz);
  int iters_def_mul = runCG(smith::Preconditioner::Deflation, "Deflation_Mult", smith::CoarseMode::Multiplicative);

  if (my_rank == 0) {
    std::cout << "[Beam summary] ranks=" << n_ranks << " Jacobi=" << iters_jac << " AMG=" << iters_amg
              << " Def_Add=" << iters_def_add << " Def_Add_BSR=" << iters_def_add_bsr
              << " Def_AddLocal=" << iters_def_loc << " Def_Schwarz=" << iters_def_sch << " Def_Mult=" << iters_def_mul
              << "\n";
  }
  EXPECT_LT(iters_def_add, iters_jac) << "deflation should beat plain Jacobi";
  EXPECT_LE(std::abs(iters_def_add_bsr - iters_def_add), 2)
      << "BSR solver should keep the deflation iteration count essentially unchanged";
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
