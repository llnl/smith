// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/numerics/batched_matvec.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace {

struct Setup {
  std::unique_ptr<mfem::ParMesh> pmesh;
  std::unique_ptr<mfem::H1_FECollection> fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes;
  std::unique_ptr<mfem::HypreParMatrix> A;
};

Setup makeCube()
{
  Setup s;
  auto serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0);
  s.pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial);
  constexpr int order = 1, dim = 3;
  s.fec = std::make_unique<mfem::H1_FECollection>(order, dim);
  s.fes = std::make_unique<mfem::ParFiniteElementSpace>(s.pmesh.get(), s.fec.get(), dim, mfem::Ordering::byVDIM);
  mfem::ConstantCoefficient one(1.0), mass(0.1);
  mfem::ParBilinearForm a(s.fes.get());
  a.AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(one));
  a.AddDomainIntegrator(new mfem::VectorMassIntegrator(mass));
  a.Assemble();
  a.Finalize();
  s.A.reset(a.ParallelAssemble());
  return s;
}

void fillRandom(mfem::DenseMatrix& M, int seed)
{
  // Deterministic pseudo-random pattern; per-rank phase shift via seed.
  for (int j = 0; j < M.Width(); ++j) {
    for (int i = 0; i < M.Height(); ++i) {
      M(i, j) = std::sin(0.13 * (i + 1) + 0.27 * (j + 1) + 0.91 * seed);
    }
  }
}

double globalNorm(const mfem::DenseMatrix& M)
{
  double local = 0.0;
  for (int j = 0; j < M.Width(); ++j) {
    for (int i = 0; i < M.Height(); ++i) local += M(i, j) * M(i, j);
  }
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return std::sqrt(global);
}

}  // namespace

// Loop strategy is the correctness reference — it must match the obvious per-column
// A.Mult to bit precision (it literally is that).
TEST(BatchedMatvec, LoopMatchesSequentialApply)
{
  auto s = makeCube();
  const int n = s.fes->GetTrueVSize();
  const int k = 7;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  mfem::DenseMatrix X(n, k), Y_batched(n, k), Y_ref(n, k);
  fillRandom(X, rank);

  smith::batchedMatvec(*s.A, X, Y_batched, smith::BatchedMatvecStrategy::Loop);

  mfem::Vector xj(n), yj(n);
  for (int j = 0; j < k; ++j) {
    for (int i = 0; i < n; ++i) xj(i) = X(i, j);
    s.A->Mult(xj, yj);
    for (int i = 0; i < n; ++i) Y_ref(i, j) = yj(i);
  }

  mfem::DenseMatrix diff(Y_batched);
  diff -= Y_ref;
  double err = globalNorm(diff);
  double ref = globalNorm(Y_ref);
  EXPECT_LT(err, 1e-14 * std::max(ref, 1.0));
}

TEST(BatchedMatvec, PackedDenseMatchesLoop)
{
  auto s = makeCube();
  const int n = s.fes->GetTrueVSize();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (int k : {1, 3, 12}) {
    mfem::DenseMatrix X(n, k), Y_loop(n, k), Y_pack(n, k);
    fillRandom(X, rank);

    smith::batchedMatvec(*s.A, X, Y_loop, smith::BatchedMatvecStrategy::Loop);
    smith::batchedMatvec(*s.A, X, Y_pack, smith::BatchedMatvecStrategy::PackedDense);

    mfem::DenseMatrix diff(Y_pack);
    diff -= Y_loop;
    double err = globalNorm(diff);
    double ref = globalNorm(Y_loop);
    EXPECT_LT(err, 1e-12 * std::max(ref, 1.0)) << "k=" << k;
  }
}

// W-shaped input: each column nonzero only on its owning rank.
// Mirrors the deflation SetOperator usage pattern.
TEST(BatchedMatvec, PackedDenseColumnDistributed)
{
  auto s = makeCube();
  const int n = s.fes->GetTrueVSize();
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  constexpr int cols_per_rank = 12;
  const int k = cols_per_rank * nproc;
  mfem::DenseMatrix X(n, k), Y_loop(n, k), Y_pack(n, k);
  X = 0.0;
  // Fill only the columns owned by this rank, in this rank's local rows.
  for (int j = 0; j < cols_per_rank; ++j) {
    int global_col = rank * cols_per_rank + j;
    for (int i = 0; i < n; ++i) {
      X(i, global_col) = std::sin(0.11 * (i + 1) + 0.37 * (global_col + 1));
    }
  }
  smith::batchedMatvec(*s.A, X, Y_loop, smith::BatchedMatvecStrategy::Loop);
  smith::batchedMatvec(*s.A, X, Y_pack, smith::BatchedMatvecStrategy::PackedDense);

  mfem::DenseMatrix diff(Y_pack);
  diff -= Y_loop;
  EXPECT_LT(globalNorm(diff), 1e-12 * std::max(globalNorm(Y_loop), 1.0));
}

// Timing benchmark — mirrors the deflation SetOperator workload: A is a stiff
// elasticity operator on a slender beam, X is W-shaped (12 cols/rank, nonzero only
// in owner rank's local rows). Reports Loop vs PackedDense wall time.
TEST(BatchedMatvec, BeamTiming)
{
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  constexpr int nx = 192, ny = 12, nz = 12;
  constexpr double Lx = 16.0, Ly = 1.0, Lz = 1.0;
  auto serial_mesh = mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);

  constexpr int order = 1, dim = 3;
  mfem::H1_FECollection fec(order, dim);
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim, mfem::Ordering::byVDIM);

  mfem::ConstantCoefficient lambda(1.0), mu(1.0);
  mfem::ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda, mu));
  a.Assemble();
  a.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> A(a.ParallelAssemble());

  const int n = fes.GetTrueVSize();
  constexpr int cols_per_rank = 12;
  const int k = cols_per_rank * nproc;
  mfem::DenseMatrix X(n, k), Y(n, k);
  X = 0.0;
  for (int j = 0; j < cols_per_rank; ++j) {
    int gc = rank * cols_per_rank + j;
    for (int i = 0; i < n; ++i) X(i, gc) = std::sin(0.11 * (i + 1) + 0.37 * (gc + 1));
  }

  constexpr int nreps = 5;
  // warm-up + time Loop
  smith::batchedMatvec(*A, X, Y, smith::BatchedMatvecStrategy::Loop);
  MPI_Barrier(MPI_COMM_WORLD);
  double t_loop = MPI_Wtime();
  for (int r = 0; r < nreps; ++r) smith::batchedMatvec(*A, X, Y, smith::BatchedMatvecStrategy::Loop);
  MPI_Barrier(MPI_COMM_WORLD);
  t_loop = (MPI_Wtime() - t_loop) / nreps;

  smith::batchedMatvec(*A, X, Y, smith::BatchedMatvecStrategy::PackedDense);
  MPI_Barrier(MPI_COMM_WORLD);
  double t_pack = MPI_Wtime();
  for (int r = 0; r < nreps; ++r) smith::batchedMatvec(*A, X, Y, smith::BatchedMatvecStrategy::PackedDense);
  MPI_Barrier(MPI_COMM_WORLD);
  t_pack = (MPI_Wtime() - t_pack) / nreps;

  if (rank == 0) {
    std::cout << "[BeamTiming] ranks=" << nproc << " dofs=" << fes.GlobalTrueVSize() << " k=" << k
              << " Loop=" << t_loop * 1000 << "ms PackedDense=" << t_pack * 1000
              << "ms speedup=" << t_loop / t_pack << "x\n";
  }
}

// Reference W^T A W via the current Loop-style assembly (per-col matvec + dot + Allreduce).
mfem::DenseMatrix referenceWtAW(const mfem::HypreParMatrix& A, const mfem::DenseMatrix& W_local, int mpr)
{
  int my_rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const int m = mpr * nproc;
  const int n = W_local.Height();
  const int my_off = my_rank * mpr;

  mfem::DenseMatrix WtAW(m, m);
  WtAW = 0.0;
  mfem::Vector Wq(n), AWq(n), Wi(n);
  for (int q = 0; q < m; ++q) {
    int owner = q / mpr;
    int qloc = q % mpr;
    Wq = 0.0;
    if (owner == my_rank) {
      for (int i = 0; i < n; ++i) Wq(i) = W_local(i, qloc);
    }
    const_cast<mfem::HypreParMatrix&>(A).Mult(Wq, AWq);
    for (int i = 0; i < mpr; ++i) {
      for (int r = 0; r < n; ++r) Wi(r) = W_local(r, i);
      WtAW(my_off + i, q) = Wi * AWq;
    }
  }
  mfem::DenseMatrix out(m, m);
  MPI_Allreduce(WtAW.Data(), out.Data(), m * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

TEST(BatchedMatvec, TripleProductMatchesReference)
{
  auto s = makeCube();
  const int n = s.fes->GetTrueVSize();
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  for (int mpr : {1, 3, 12}) {
    mfem::DenseMatrix W_local(n, mpr);
    for (int j = 0; j < mpr; ++j) {
      for (int i = 0; i < n; ++i) W_local(i, j) = std::sin(0.13 * (i + 1) + 0.27 * (j + 1) + 0.91 * rank);
    }

    mfem::DenseMatrix WtAW_tp, WtAW_ref;
    smith::assembleWtAW(*s.A, W_local, mpr, WtAW_tp);
    WtAW_ref = referenceWtAW(*s.A, W_local, mpr);

    mfem::DenseMatrix diff(WtAW_tp);
    diff -= WtAW_ref;
    double err = 0.0, ref = 0.0;
    for (int j = 0; j < WtAW_tp.Width(); ++j)
      for (int i = 0; i < WtAW_tp.Height(); ++i) {
        err += diff(i, j) * diff(i, j);
        ref += WtAW_ref(i, j) * WtAW_ref(i, j);
      }
    EXPECT_LT(std::sqrt(err), 1e-12 * std::max(std::sqrt(ref), 1.0)) << "mpr=" << mpr;
  }
}

// Correctness on a realistic elasticity matrix (slender beam, ~98k dofs).
// Exercises the multi-neighbor halo + offd block paths that the small-cube tests
// don't reach.
TEST(BatchedMatvec, TripleProductElasticityBeam)
{
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  auto serial_mesh = mfem::Mesh::MakeCartesian3D(48, 6, 6, mfem::Element::HEXAHEDRON, 8.0, 1.0, 1.0);
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);

  constexpr int order = 1, dim = 3;
  mfem::H1_FECollection fec(order, dim);
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim, mfem::Ordering::byVDIM);

  mfem::ConstantCoefficient lambda(1.0), mu(1.0);
  mfem::ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda, mu));
  a.Assemble();
  a.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> A(a.ParallelAssemble());

  const int n = fes.GetTrueVSize();
  constexpr int mpr = 12;
  mfem::DenseMatrix W_local(n, mpr);
  for (int j = 0; j < mpr; ++j) {
    for (int i = 0; i < n; ++i) W_local(i, j) = std::sin(0.11 * (i + 1) + 0.37 * (j + 1) + 0.5 * rank);
  }

  mfem::DenseMatrix WtAW_tp, WtAW_ref;
  smith::assembleWtAW(*A, W_local, mpr, WtAW_tp);
  WtAW_ref = referenceWtAW(*A, W_local, mpr);

  mfem::DenseMatrix diff(WtAW_tp);
  diff -= WtAW_ref;
  double err = 0.0, ref = 0.0;
  for (int j = 0; j < WtAW_tp.Width(); ++j)
    for (int i = 0; i < WtAW_tp.Height(); ++i) {
      err += diff(i, j) * diff(i, j);
      ref += WtAW_ref(i, j) * WtAW_ref(i, j);
    }
  EXPECT_LT(std::sqrt(err), 1e-10 * std::max(std::sqrt(ref), 1.0));
}

TEST(BatchedMatvec, BeamTimingTripleProduct)
{
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  constexpr int nx = 192, ny = 12, nz = 12;
  constexpr double Lx = 16.0, Ly = 1.0, Lz = 1.0;
  auto serial_mesh = mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);

  constexpr int order = 1, dim = 3;
  mfem::H1_FECollection fec(order, dim);
  mfem::ParFiniteElementSpace fes(pmesh.get(), &fec, dim, mfem::Ordering::byVDIM);

  mfem::ConstantCoefficient lambda(1.0), mu(1.0);
  mfem::ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda, mu));
  a.Assemble();
  a.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> A(a.ParallelAssemble());

  const int n = fes.GetTrueVSize();
  constexpr int mpr = 12;
  mfem::DenseMatrix W_local(n, mpr);
  for (int j = 0; j < mpr; ++j) {
    for (int i = 0; i < n; ++i) W_local(i, j) = std::sin(0.11 * (i + 1) + 0.37 * (j + 1) + 0.5 * rank);
  }

  // For Loop / PackedDense, build the W-shaped column-distributed dense X (n × m).
  const int m = mpr * nproc;
  mfem::DenseMatrix X(n, m), Y(n, m);
  X = 0.0;
  for (int j = 0; j < mpr; ++j) {
    for (int i = 0; i < n; ++i) X(i, rank * mpr + j) = W_local(i, j);
  }

  constexpr int nreps = 5;

  // Warm up + time Loop.
  smith::batchedMatvec(*A, X, Y, smith::BatchedMatvecStrategy::Loop);
  MPI_Barrier(MPI_COMM_WORLD);
  double t_loop = MPI_Wtime();
  for (int r = 0; r < nreps; ++r) smith::batchedMatvec(*A, X, Y, smith::BatchedMatvecStrategy::Loop);
  MPI_Barrier(MPI_COMM_WORLD);
  t_loop = (MPI_Wtime() - t_loop) / nreps;

  smith::batchedMatvec(*A, X, Y, smith::BatchedMatvecStrategy::PackedDense);
  MPI_Barrier(MPI_COMM_WORLD);
  double t_pack = MPI_Wtime();
  for (int r = 0; r < nreps; ++r) smith::batchedMatvec(*A, X, Y, smith::BatchedMatvecStrategy::PackedDense);
  MPI_Barrier(MPI_COMM_WORLD);
  t_pack = (MPI_Wtime() - t_pack) / nreps;

  mfem::DenseMatrix WtAW;
  smith::AssembleWtAWTimings tp_tm;
  smith::assembleWtAW(*A, W_local, mpr, WtAW);  // warm up
  MPI_Barrier(MPI_COMM_WORLD);
  double t_tp = MPI_Wtime();
  for (int r = 0; r < nreps; ++r) smith::assembleWtAW(*A, W_local, mpr, WtAW, &tp_tm);
  MPI_Barrier(MPI_COMM_WORLD);
  t_tp = (MPI_Wtime() - t_tp) / nreps;

  if (rank == 0) {
    std::cout << "[BeamTiming3] ranks=" << nproc << " dofs=" << fes.GlobalTrueVSize() << " m=" << m
              << " Loop=" << t_loop * 1000 << "ms PackedDense=" << t_pack * 1000
              << "ms TripleProduct=" << t_tp * 1000 << "ms"
              << " (TP vs Loop=" << t_loop / t_tp << "x, TP vs Pack=" << t_pack / t_tp << "x)\n";
    std::cout << "  [TP breakdown avg/iter] halo=" << tp_tm.halo / nreps * 1000
              << "ms diag=" << tp_tm.diag / nreps * 1000 << "ms offd=" << tp_tm.offd / nreps * 1000
              << "ms allreduce=" << tp_tm.allreduce / nreps * 1000 << "ms\n";
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
