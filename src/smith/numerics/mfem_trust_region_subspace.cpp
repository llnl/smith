// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/trust_region_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "smith/infrastructure/profiling.hpp"

namespace smith {

namespace {

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start)
{
  return std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - start).count();
}

TrustRegionSubspaceTimings& mutableTrustRegionSubspaceTimings()
{
  static TrustRegionSubspaceTimings timings;
  return timings;
}

}  // namespace

void resetTrustRegionSubspaceTimings()
{
  mutableTrustRegionSubspaceTimings() = TrustRegionSubspaceTimings {};
}

TrustRegionSubspaceTimings trustRegionSubspaceTimings()
{
  return mutableTrustRegionSubspaceTimings();
}

int globalSize(const mfem::Vector& parallel_v, const MPI_Comm& comm)
{
  int local_size = parallel_v.Size();
  int global_size;
  MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
  return global_size;
}

double innerProduct(const mfem::Vector& a, const mfem::Vector& b, const MPI_Comm& comm)
{
  return mfem::InnerProduct(comm, a, b);
}

std::pair<std::vector<const mfem::Vector*>, std::vector<const mfem::Vector*>> removeDependentDirections(
    std::vector<const mfem::Vector*> directions, std::vector<const mfem::Vector*> A_directions)
{
  SMITH_MARK_FUNCTION;
  std::vector<double> norms;
  size_t num_dirs = directions.size();

  for (size_t i = 0; i < num_dirs; ++i) {
    norms.push_back(std::sqrt(mfem::InnerProduct(MPI_COMM_WORLD, *directions[i], *directions[i])));
  }

  std::vector<std::pair<const mfem::Vector*, size_t>> kepts;
  for (size_t i = 0; i < num_dirs; ++i) {
    bool keepi = true;
    if (norms[i] == 0) keepi = false;
    for (auto&& kept_and_j : kepts) {
      size_t j = kept_and_j.second;
      double dot_ij = mfem::InnerProduct(MPI_COMM_WORLD, *directions[i], *kept_and_j.first);
      if (dot_ij > 0.999 * norms[i] * norms[j]) {
        keepi = false;
      }
    }
    if (keepi) {
      kepts.emplace_back(std::make_pair(directions[i], i));
    }
  }

  std::vector<const mfem::Vector*> directions_new;
  std::vector<const mfem::Vector*> A_directions_new;

  for (auto kept_and_j : kepts) {
    directions_new.push_back(directions[kept_and_j.second]);
    A_directions_new.push_back(A_directions[kept_and_j.second]);
  }

  return std::make_pair(directions_new, A_directions_new);
}

#ifdef MFEM_USE_LAPACK

TrustRegionSubspaceResult solveSubspaceProblem(const std::vector<const mfem::Vector*>& directions,
                                               const std::vector<const mfem::Vector*>& A_directions,
                                               const mfem::Vector& b, double delta, int num_leftmost)
{
  return solveSubspaceProblemMfem(directions, A_directions, b, delta, num_leftmost);
}

namespace {

double dot(const mfem::Vector& a, const mfem::Vector& b)
{
  return a * b;
}

double norm(const mfem::Vector& x)
{
  return x.Norml2();
}

mfem::Vector operator+(const mfem::Vector& x, double value)
{
  mfem::Vector out(x);
  for (int i = 0; i < out.Size(); ++i) {
    out[i] += value;
  }
  return out;
}

mfem::Vector pointwiseMultiply(const mfem::Vector& a, const mfem::Vector& b)
{
  mfem::Vector out(a.Size());
  for (int i = 0; i < a.Size(); ++i) {
    out[i] = a[i] * b[i];
  }
  return out;
}

mfem::Vector pointwiseDivide(const mfem::Vector& a, const mfem::Vector& b)
{
  mfem::Vector out(a.Size());
  for (int i = 0; i < a.Size(); ++i) {
    out[i] = a[i] / b[i];
  }
  return out;
}

double sumAbs(const mfem::Vector& x)
{
  double total = 0.0;
  for (int i = 0; i < x.Size(); ++i) {
    total += std::abs(x[i]);
  }
  return total;
}

double sum(const mfem::Vector& x)
{
  double total = 0.0;
  for (int i = 0; i < x.Size(); ++i) {
    total += x[i];
  }
  return total;
}

void symmetrize(mfem::DenseMatrix& A)
{
  MFEM_VERIFY(A.Height() == A.Width(), "symmetrize requires square matrix");
  for (int i = 0; i < A.Height(); ++i) {
    for (int j = 0; j < i; ++j) {
      const double value = 0.5 * (A(i, j) + A(j, i));
      A(i, j) = value;
      A(j, i) = value;
    }
  }
}

struct SubspaceProjections {
  mfem::DenseMatrix sAs;
  mfem::DenseMatrix ss;
  mfem::Vector sb;
};

SubspaceProjections denseSubspaceProjections(const std::vector<const mfem::Vector*>& states,
                                             const std::vector<const mfem::Vector*>& Astates, const mfem::Vector& b)
{
  MFEM_VERIFY(states.size() == Astates.size(),
              "Search directions and their linear operator result must have same number of columns");
  MFEM_VERIFY(!states.empty(), "Subspace projections require at least one direction.");

  const int n = static_cast<int>(states.size());
  const int vector_size = states[0]->Size();
  for (int j = 0; j < n; ++j) {
    MFEM_VERIFY(states[size_t(j)]->Size() == vector_size, "Subspace direction sizes differ.");
    MFEM_VERIFY(Astates[size_t(j)]->Size() == vector_size, "Subspace Hessian-vector sizes differ.");
  }
  MFEM_VERIFY(b.Size() == vector_size, "Subspace right-hand-side size differs.");

  const int triangular_size = n * (n + 1) / 2;
  const auto triangular_index = [n](int i, int j) {
    return i * n - (i * (i - 1)) / 2 + (j - i);
  };
  const int sAs_offset = 0;
  const int ss_offset = triangular_size;
  const int sb_offset = 2 * triangular_size;
  const int buffer_size = 2 * triangular_size + n;
  std::vector<mfem::real_t> local(size_t(buffer_size), 0.0);
  std::vector<mfem::real_t> global(size_t(buffer_size), 0.0);

  for (int k = 0; k < vector_size; ++k) {
    const double b_k = b[k];
    for (int i = 0; i < n; ++i) {
      const double s_i = (*states[size_t(i)])[k];
      local[size_t(sb_offset + i)] += s_i * b_k;
      for (int j = i; j < n; ++j) {
        const size_t ij = size_t(triangular_index(i, j));
        local[size_t(sAs_offset) + ij] += s_i * (*Astates[size_t(j)])[k];
        local[size_t(ss_offset) + ij] += s_i * (*states[size_t(j)])[k];
      }
    }
  }

  MPI_Allreduce(local.data(), global.data(), buffer_size, MFEM_MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

  SubspaceProjections projections{mfem::DenseMatrix(n), mfem::DenseMatrix(n), mfem::Vector(n)};
  for (int i = 0; i < n; ++i) {
    projections.sb[i] = global[size_t(sb_offset + i)];
    for (int j = i; j < n; ++j) {
      const size_t ij = size_t(triangular_index(i, j));
      projections.sAs(i, j) = global[size_t(sAs_offset) + ij];
      projections.sAs(j, i) = projections.sAs(i, j);
      projections.ss(i, j) = global[size_t(ss_offset) + ij];
      projections.ss(j, i) = projections.ss(i, j);
    }
  }

  return projections;
}

mfem::Vector solveDense(const mfem::DenseMatrix& A, const mfem::Vector& b)
{
  mfem::DenseMatrix A_copy(A);
  mfem::DenseMatrixInverse inv(A_copy);
  mfem::Vector x(b.Size());
  inv.Mult(b, x);
  return x;
}

double quadraticEnergy(const mfem::DenseMatrix& A, const mfem::Vector& b, const mfem::Vector& x)
{
  mfem::Vector Ax(x.Size());
  A.Mult(x, Ax);
  return 0.5 * dot(x, Ax) - dot(x, b);
}

double pnormSquared(const mfem::Vector& bvv, const mfem::Vector& sig)
{
  return sum(pointwiseDivide(bvv, pointwiseMultiply(sig, sig)));
}

double qnormSquared(const mfem::Vector& bvv, const mfem::Vector& sig)
{
  mfem::Vector sig_sq = pointwiseMultiply(sig, sig);
  mfem::Vector sig_cu = pointwiseMultiply(sig_sq, sig);
  return sum(pointwiseDivide(bvv, sig_cu));
}

mfem::Vector matrixColumn(const mfem::DenseMatrix& A, int j)
{
  mfem::Vector col(A.Height());
  for (int i = 0; i < A.Height(); ++i) {
    col[i] = A(i, j);
  }
  return col;
}

mfem::DenseMatrix columnsToMatrix(const std::vector<mfem::Vector>& cols)
{
  mfem::DenseMatrix A(cols.empty() ? 0 : cols[0].Size(), static_cast<int>(cols.size()));
  for (int j = 0; j < A.Width(); ++j) {
    for (int i = 0; i < A.Height(); ++i) {
      A(i, j) = cols[size_t(j)][i];
    }
  }
  return A;
}

std::tuple<mfem::Vector, std::vector<mfem::Vector>, std::vector<double>, bool> exactTrustRegionSolve(
    mfem::DenseMatrix A, const mfem::Vector& b, double delta, int num_leftmost)
{
  auto dense_solve_start = Clock::now();
  if (A.Height() != A.Width()) {
    throw PetscException("Exact trust region solver requires square matrices");
  }
  if (A.Height() != b.Size()) {
    throw PetscException("The right hand size for exact trust region solve must be consistent with the input matrix size");
  }

  mfem::Vector sigs;
  mfem::DenseMatrix V;
  auto eig_start = Clock::now();
  A.Eigensystem(sigs, V);
  mutableTrustRegionSubspaceTimings().dense_eigensystem_seconds += secondsSince(eig_start);

  std::vector<mfem::Vector> leftmosts;
  std::vector<double> minsigs;
  const int num_leftmost_possible = std::min(num_leftmost, sigs.Size());
  for (int i = 0; i < num_leftmost_possible; ++i) {
    leftmosts.emplace_back(matrixColumn(V, i));
    minsigs.emplace_back(sigs[i]);
  }

  const mfem::Vector leftMost = matrixColumn(V, 0);
  const double minSig = sigs[0];

  mfem::Vector bv(sigs.Size());
  for (int i = 0; i < sigs.Size(); ++i) {
    const mfem::Vector vi = matrixColumn(V, i);
    bv[i] = dot(vi, b);
  }

  mfem::Vector bvOverSigs = pointwiseDivide(bv, sigs);
  const double sigScale = sumAbs(sigs) / sigs.Size();
  const double eps = 1e-12 * sigScale;

  if ((minSig >= eps) && (norm(bvOverSigs) <= delta)) {
    mutableTrustRegionSubspaceTimings().dense_trust_solve_seconds += secondsSince(dense_solve_start);
    return std::make_tuple(solveDense(A, b), leftmosts, minsigs, true);
  }

  double lam = minSig < eps ? -minSig + eps : 0.0;
  mfem::Vector sigsPlusLam = sigs + lam;
  bvOverSigs = pointwiseDivide(bv, sigsPlusLam);

  if ((minSig < eps) && (norm(bvOverSigs) < delta)) {
    mfem::Vector p(b.Size());
    p = 0.0;
    for (int i = 0; i < b.Size(); ++i) {
      const mfem::Vector vi = matrixColumn(V, i);
      p.Add(bv[i], vi);
    }

    const double pz = dot(p, leftMost);
    const double pp = dot(p, p);
    const double ddmpp = std::max(delta * delta - pp, 0.0);

    const double tau1 = -pz + std::sqrt(pz * pz + ddmpp);
    const double tau2 = -pz - std::sqrt(pz * pz + ddmpp);

    mfem::Vector x1(p);
    mfem::Vector x2(p);
    x1.Add(tau1, leftMost);
    x2.Add(tau2, leftMost);

    const double e1 = quadraticEnergy(A, b, x1);
    const double e2 = quadraticEnergy(A, b, x2);

    mutableTrustRegionSubspaceTimings().dense_trust_solve_seconds += secondsSince(dense_solve_start);
    return std::make_tuple(e1 < e2 ? x1 : x2, leftmosts, minsigs, true);
  }

  const mfem::Vector bvbv = pointwiseMultiply(bv, bv);
  sigsPlusLam = sigs + lam;

  double pNormSq = pnormSquared(bvbv, sigsPlusLam);
  double pNorm = std::sqrt(pNormSq);
  double bError = (pNorm - delta) / delta;

  size_t iters = 0;
  constexpr size_t maxIters = 30;
  while ((std::abs(bError) > 1e-9) && (iters++ < maxIters)) {
    const double qNormSq = qnormSquared(bvbv, sigsPlusLam);
    lam += (pNormSq / qNormSq) * bError;
    sigsPlusLam = sigs + lam;
    pNormSq = pnormSquared(bvbv, sigsPlusLam);
    pNorm = std::sqrt(pNormSq);
    bError = (pNorm - delta) / delta;
  }

  const bool success = iters < maxIters;

  bvOverSigs = pointwiseDivide(bv, sigsPlusLam);

  mfem::Vector x(b.Size());
  x = 0.0;
  for (int i = 0; i < b.Size(); ++i) {
    const mfem::Vector vi = matrixColumn(V, i);
    x.Add(bvOverSigs[i], vi);
  }

  const double e1 = quadraticEnergy(A, b, x);
  mfem::Vector neg_x(x);
  neg_x *= -1.0;
  const double e2 = quadraticEnergy(A, b, neg_x);

  x *= (e2 < e1 ? -delta : delta) / norm(x);

  mutableTrustRegionSubspaceTimings().dense_trust_solve_seconds += secondsSince(dense_solve_start);
  return std::make_tuple(x, leftmosts, minsigs, success);
}

mfem::DenseMatrix orthonormalBasisTransform(const mfem::DenseMatrix& gram, double& trace_mag)
{
  mfem::DenseMatrix gram_copy(gram);
  mfem::Vector evals;
  mfem::DenseMatrix evecs;
  gram_copy.Eigensystem(evals, evecs);

  trace_mag = 0.0;
  for (int i = 0; i < evals.Size(); ++i) {
    trace_mag += std::abs(evals[i]);
  }

  std::vector<mfem::Vector> kept_columns;
  for (int i = 0; i < evals.Size(); ++i) {
    if (evals[i] > 1e-9 * trace_mag) {
      mfem::Vector col = matrixColumn(evecs, i);
      col /= std::sqrt(evals[i]);
      kept_columns.emplace_back(std::move(col));
    }
  }

  return columnsToMatrix(kept_columns);
}

mfem::DenseMatrix tripleProduct(const mfem::DenseMatrix& L, const mfem::DenseMatrix& A, const mfem::DenseMatrix& R)
{
  mfem::DenseMatrix tmp(A.Height(), R.Width());
  mfem::Mult(A, R, tmp);
  mfem::DenseMatrix out(L.Width(), R.Width());
  mfem::MultAtB(L, tmp, out);
  return out;
}

mfem::Vector projectWithTranspose(const mfem::DenseMatrix& A, const mfem::Vector& x)
{
  mfem::Vector out(A.Width());
  A.MultTranspose(x, out);
  return out;
}

mfem::Vector combineDirections(const std::vector<const mfem::Vector*>& states, const mfem::Vector& coeffs)
{
  mfem::Vector out(*states[0]);
  out = 0.0;
  for (int i = 0; i < coeffs.Size(); ++i) {
    out.Add(coeffs[i], *states[size_t(i)]);
  }
  return out;
}

}  // namespace

TrustRegionSubspaceResult solveSubspaceProblemMfem(const std::vector<const mfem::Vector*>& states,
                                                   const std::vector<const mfem::Vector*>& Astates,
                                                   const mfem::Vector& b, double delta, int num_leftmost)
{
  SMITH_MARK_FUNCTION;
  auto& timings = mutableTrustRegionSubspaceTimings();
  ++timings.num_solves;
  timings.total_input_dim += states.size();
  timings.max_input_dim = std::max(timings.max_input_dim, states.size());

  auto project_A_start = Clock::now();
  SubspaceProjections projections = denseSubspaceProjections(states, Astates, b);
  mfem::DenseMatrix& sAs = projections.sAs;
  timings.project_A_seconds += secondsSince(project_A_start);
  symmetrize(sAs);

  for (int i = 0; i < sAs.Height(); ++i) {
    for (int j = 0; j < sAs.Width(); ++j) {
      if (std::isnan(sAs(i, j))) {
        throw PetscException("States in subspace solve contain NaNs.");
      }
    }
  }

  auto project_gram_start = Clock::now();
  mfem::DenseMatrix& ss = projections.ss;
  timings.project_gram_seconds += secondsSince(project_gram_start);
  symmetrize(ss);

  double trace_mag = 0.0;
  auto basis_start = Clock::now();
  mfem::DenseMatrix T = orthonormalBasisTransform(ss, trace_mag);
  timings.basis_seconds += secondsSince(basis_start);
  if (T.Width() == 0) {
    throw PetscException("No independent directions in MFEM subspace solve.");
  }
  timings.total_reduced_dim += static_cast<size_t>(T.Width());
  timings.max_reduced_dim = std::max(timings.max_reduced_dim, static_cast<size_t>(T.Width()));

  auto reduced_A_start = Clock::now();
  mfem::DenseMatrix pAp = tripleProduct(T, sAs, T);
  timings.reduced_A_seconds += secondsSince(reduced_A_start);
  symmetrize(pAp);

  auto project_b_start = Clock::now();
  const mfem::Vector& sb = projections.sb;
  timings.project_b_seconds += secondsSince(project_b_start);
  const mfem::Vector pb = projectWithTranspose(T, sb);

  auto [reduced_x, leftvecs, leftvals, success] = exactTrustRegionSolve(pAp, pb, delta, num_leftmost);
  (void)success;
  const double energy = quadraticEnergy(pAp, pb, reduced_x);

  auto reconstruct_solution_start = Clock::now();
  mfem::Vector coeffs(T.Height());
  T.Mult(reduced_x, coeffs);
  mfem::Vector sol = combineDirections(states, coeffs);
  timings.reconstruct_solution_seconds += secondsSince(reconstruct_solution_start);

  auto reconstruct_leftmost_start = Clock::now();
  std::vector<std::shared_ptr<mfem::Vector>> leftmosts;
  for (const auto& leftvec : leftvecs) {
    mfem::Vector left_coeffs(T.Height());
    T.Mult(leftvec, left_coeffs);
    leftmosts.emplace_back(std::make_shared<mfem::Vector>(combineDirections(states, left_coeffs)));
  }
  timings.reconstruct_leftmost_seconds += secondsSince(reconstruct_leftmost_start);

  return std::make_tuple(sol, leftmosts, leftvals, energy);
}

#else

TrustRegionSubspaceResult solveSubspaceProblem(const std::vector<const mfem::Vector*>& directions,
                                               const std::vector<const mfem::Vector*>& A_directions,
                                               const mfem::Vector& b, double delta, int num_leftmost)
{
#if defined(SMITH_USE_SLEPC) && defined(SMITH_TRUST_REGION_USE_PETSC_SUBSPACE)
  return solveSubspaceProblemPetsc(directions, A_directions, b, delta, num_leftmost);
#else
  throw PetscException("MFEM trust-region subspace solve requires MFEM LAPACK support.");
  return std::make_tuple(b, std::vector<std::shared_ptr<mfem::Vector>> {}, std::vector<double> {}, 0.0);
#endif
}

TrustRegionSubspaceResult solveSubspaceProblemMfem(const std::vector<const mfem::Vector*>&,
                                                   const std::vector<const mfem::Vector*>&, const mfem::Vector& b,
                                                   double, int)
{
  throw PetscException("MFEM trust-region subspace solve requires MFEM LAPACK support.");
  return std::make_tuple(b, std::vector<std::shared_ptr<mfem::Vector>> {}, std::vector<double> {}, 0.0);
}

#endif  // MFEM_USE_LAPACK

}  // namespace smith
