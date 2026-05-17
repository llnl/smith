// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/numerics/deflation.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace {

struct ProblemSetup {
  std::unique_ptr<mfem::ParMesh> pmesh;
  std::unique_ptr<mfem::H1_FECollection> fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes;
  std::unique_ptr<mfem::HypreParMatrix> A;
};

ProblemSetup makeCube(mfem::Ordering::Type ordering)
{
  ProblemSetup s;
  auto serial_mesh = mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0);
  s.pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
  constexpr int order = 1;
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

  auto runCG = [&](smith::Preconditioner pc, const char* label) -> int {
    smith::LinearSolverOptions opts;
    opts.linear_solver = smith::LinearSolver::CG;
    opts.preconditioner = pc;
    opts.relative_tol = 0.0;
    opts.absolute_tol = 1e-10;
    opts.max_iterations = 10000;
    opts.print_level = 0;
    opts.deflation_fes = (pc == smith::Preconditioner::Deflation) ? &fes : nullptr;

    auto [lin_solver, prec] = smith::buildLinearSolverAndPreconditioner(opts, MPI_COMM_WORLD);

    smith::DeflationPreconditioner* dp = nullptr;
    if (pc == smith::Preconditioner::Deflation) {
      dp = dynamic_cast<smith::DeflationPreconditioner*>(prec.get());
      EXPECT_NE(dp, nullptr);
      if (dp) dp->setEssentialTrueDofs(ess_tdofs);
    }

    auto* iter = dynamic_cast<mfem::IterativeSolver*>(lin_solver.get());
    EXPECT_NE(iter, nullptr);
    if (!iter) return -1;

    // === TIMING BEGIN ===
    MPI_Barrier(MPI_COMM_WORLD);
    double t_setop = MPI_Wtime();
    // === TIMING END ===
    iter->SetOperator(A);
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
                << " ess_tdofs=" << n_ess_global
                << " iters=" << iter->GetNumIterations() << " converged=" << iter->GetConverged()
                << " ||r||=" << true_rnorm
                // === TIMING BEGIN ===
                << " t_setop=" << t_setop << "s t_solve=" << t_solve << "s"
                << " t_per_iter=" << (iter->GetNumIterations() > 0 ? t_solve / iter->GetNumIterations() : 0.0) << "s"
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
  int iters_def = runCG(smith::Preconditioner::Deflation, "Deflation");

  if (my_rank == 0) {
    std::cout << "[Beam summary] ranks=" << n_ranks << " Jacobi=" << iters_jac << " AMG=" << iters_amg
              << " Deflation=" << iters_def << "\n";
  }
  EXPECT_LT(iters_def, iters_jac) << "deflation should beat plain Jacobi";
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
