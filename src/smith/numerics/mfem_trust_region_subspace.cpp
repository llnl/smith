// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/trust_region_solver.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "smith/infrastructure/profiling.hpp"

namespace smith {

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

#ifdef MFEM_USE_LAPACK

TrustRegionSubspaceResult solveSubspaceProblem(const std::vector<const mfem::Vector*>& directions,
                                               const std::vector<const mfem::Vector*>& A_directions,
                                               const mfem::Vector& b, double delta, int num_leftmost)
{
  return solveSubspaceProblemMfem(directions, A_directions, b, delta, num_leftmost);
}

namespace {

double dot(const mfem::Vector& a, const mfem::Vector& b) { return a * b; }

double norm(const mfem::Vector& x) { return x.Norml2(); }

double sumAbs(const mfem::Vector& x)
{
  double total = 0.0;
  for (int i = 0; i < x.Size(); ++i) {
    total += std::abs(x[i]);
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

void checkProjectionInputs(const std::vector<const mfem::Vector*>& states,
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
}

SubspaceProjections globalSubspaceProjectionFromLocalInnerProducts(const std::vector<const mfem::Vector*>& states,
                                                                   const std::vector<const mfem::Vector*>& Astates,
                                                                   const mfem::Vector& b)
{
  const int n = static_cast<int>(states.size());
  const int triangular_size = n * (n + 1) / 2;
  const auto triangular_index = [n](int i, int j) { return i * n - (i * (i - 1)) / 2 + (j - i); };
  const int sAs_offset = 0;
  const int ss_offset = triangular_size;
  const int sb_offset = 2 * triangular_size;
  const int buffer_size = 2 * triangular_size + n;
  std::vector<mfem::real_t> local_projection_entries(size_t(buffer_size), 0.0);
  std::vector<mfem::real_t> global_projection_entries(size_t(buffer_size), 0.0);

  for (int i = 0; i < n; ++i) {
    local_projection_entries[size_t(sb_offset + i)] = mfem::InnerProduct(*states[size_t(i)], b);
    for (int j = i; j < n; ++j) {
      const size_t ij = size_t(triangular_index(i, j));
      local_projection_entries[size_t(sAs_offset) + ij] = mfem::InnerProduct(*states[size_t(i)], *Astates[size_t(j)]);
      local_projection_entries[size_t(ss_offset) + ij] = mfem::InnerProduct(*states[size_t(i)], *states[size_t(j)]);
    }
  }

  MPI_Allreduce(local_projection_entries.data(), global_projection_entries.data(), buffer_size, MFEM_MPI_REAL_T,
                MPI_SUM, MPI_COMM_WORLD);

  SubspaceProjections projections{mfem::DenseMatrix(n), mfem::DenseMatrix(n), mfem::Vector(n)};
  for (int i = 0; i < n; ++i) {
    projections.sb[i] = global_projection_entries[size_t(sb_offset + i)];
    for (int j = i; j < n; ++j) {
      const size_t ij = size_t(triangular_index(i, j));
      projections.sAs(i, j) = global_projection_entries[size_t(sAs_offset) + ij];
      projections.sAs(j, i) = projections.sAs(i, j);
      projections.ss(i, j) = global_projection_entries[size_t(ss_offset) + ij];
      projections.ss(j, i) = projections.ss(i, j);
    }
  }

  return projections;
}

SubspaceProjections projectSubspaceGlobally(const std::vector<const mfem::Vector*>& states,
                                            const std::vector<const mfem::Vector*>& Astates, const mfem::Vector& b)
{
  checkProjectionInputs(states, Astates, b);
  return globalSubspaceProjectionFromLocalInnerProducts(states, Astates, b);
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
  double total = 0.0;
  for (int i = 0; i < bvv.Size(); ++i) {
    total += bvv[i] / (sig[i] * sig[i]);
  }
  return total;
}

double qnormSquared(const mfem::Vector& bvv, const mfem::Vector& sig)
{
  double total = 0.0;
  for (int i = 0; i < bvv.Size(); ++i) {
    total += bvv[i] / (sig[i] * sig[i] * sig[i]);
  }
  return total;
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

/**
 * @brief Solves the exact trust region subproblem:
 *        min 1/2 x^T A x - b^T x, subject to ||x|| <= delta.
 *
 * Implements a variant of the Moore-Sorensen algorithm:
 * 1. Computes the eigensystem of A.
 * 2. Checks if the unconstrained minimum lies strictly inside the trust region.
 * 3. Checks for the "hard case" where the minimum eigenvalue is near zero or negative,
 *    and the Newton step points outside the trust region, requiring a shift along the leftmost eigenvector.
 * 4. Otherwise, performs a Newton iteration on the secular equation (1/||p(\lambda)|| - 1/delta = 0)
 *    to find the optimal Lagrange multiplier \lambda.
 *
 * @param A The reduced Hessian matrix (square).
 * @param b The reduced gradient vector.
 * @param delta The trust region radius.
 * @param num_leftmost The number of leftmost eigenvectors/values to return.
 * @return A tuple containing:
 *         - The optimal solution vector.
 *         - A list of the leftmost eigenvectors.
 *         - A list of the corresponding leftmost eigenvalues.
 *         - A boolean indicating success.
 */
std::tuple<mfem::Vector, std::vector<mfem::Vector>, std::vector<double>, bool> exactTrustRegionSolve(
    mfem::DenseMatrix A, const mfem::Vector& b, double delta, int num_leftmost)
{
  if (A.Height() != A.Width()) {
    throw TrustRegionException("Exact trust region solver requires square matrices");
  }
  if (A.Height() != b.Size()) {
    throw TrustRegionException(
        "The right hand size for exact trust region solve must be consistent with the input matrix size");
  }

  mfem::Vector workspace(b.Size() * b.Size() + 8 * b.Size());
  int offset = 0;
  auto alloc_vector = [&](int size) {
    mfem::Vector v(workspace.GetData() + offset, size);
    offset += size;
    return v;
  };

  mfem::Vector sigs = alloc_vector(b.Size());
  mfem::DenseMatrix V(workspace.GetData() + offset, b.Size(), b.Size());
  offset += b.Size() * b.Size();

  A.Eigensystem(sigs, V);
  std::vector<mfem::Vector> leftmosts;
  std::vector<double> minsigs;
  const int num_leftmost_possible = std::min(num_leftmost, sigs.Size());
  for (int i = 0; i < num_leftmost_possible; ++i) {
    leftmosts.emplace_back(matrixColumn(V, i));
    minsigs.emplace_back(sigs[i]);
  }

  const mfem::Vector leftMost = matrixColumn(V, 0);
  const double minSig = sigs[0];

  mfem::Vector bv = alloc_vector(sigs.Size());
  for (int i = 0; i < sigs.Size(); ++i) {
    const mfem::Vector vi = matrixColumn(V, i);
    bv[i] = dot(vi, b);
  }

  mfem::Vector bvOverSigs = alloc_vector(sigs.Size());
  for (int i = 0; i < sigs.Size(); ++i) bvOverSigs[i] = bv[i] / sigs[i];
  const double sigScale = sumAbs(sigs) / sigs.Size();
  const double eps = 1e-12 * sigScale;

  if ((minSig >= eps) && (norm(bvOverSigs) <= delta)) {
    return std::make_tuple(solveDense(A, b), leftmosts, minsigs, true);
  }

  double lam = minSig < eps ? -minSig + eps : 0.0;
  mfem::Vector sigsPlusLam = alloc_vector(sigs.Size());
  for (int i = 0; i < sigs.Size(); ++i) sigsPlusLam[i] = sigs[i] + lam;
  for (int i = 0; i < sigs.Size(); ++i) bvOverSigs[i] = bv[i] / sigsPlusLam[i];

  if ((minSig < eps) && (norm(bvOverSigs) < delta)) {
    mfem::Vector p = alloc_vector(b.Size());
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

    mfem::Vector x1 = alloc_vector(p.Size());
    x1 = p;
    mfem::Vector x2 = alloc_vector(p.Size());
    x2 = p;
    x1.Add(tau1, leftMost);
    x2.Add(tau2, leftMost);

    const double e1 = quadraticEnergy(A, b, x1);
    const double e2 = quadraticEnergy(A, b, x2);

    return std::make_tuple(e1 < e2 ? x1 : x2, leftmosts, minsigs, true);
  }

  mfem::Vector bvbv = alloc_vector(bv.Size());
  for (int i = 0; i < bv.Size(); ++i) bvbv[i] = bv[i] * bv[i];
  for (int i = 0; i < sigs.Size(); ++i) sigsPlusLam[i] = sigs[i] + lam;

  double pNormSq = pnormSquared(bvbv, sigsPlusLam);
  double pNorm = std::sqrt(pNormSq);
  double bError = (pNorm - delta) / delta;

  size_t iters = 0;
  constexpr size_t maxIters = 30;
  while ((std::abs(bError) > 1e-9) && (iters++ < maxIters)) {
    const double qNormSq = qnormSquared(bvbv, sigsPlusLam);
    lam += (pNormSq / qNormSq) * bError;
    for (int i = 0; i < sigs.Size(); ++i) sigsPlusLam[i] = sigs[i] + lam;
    pNormSq = pnormSquared(bvbv, sigsPlusLam);
    pNorm = std::sqrt(pNormSq);
    bError = (pNorm - delta) / delta;
  }

  const bool success = iters < maxIters;

  for (int i = 0; i < sigs.Size(); ++i) bvOverSigs[i] = bv[i] / sigsPlusLam[i];

  mfem::Vector x = alloc_vector(b.Size());
  x = 0.0;
  for (int i = 0; i < b.Size(); ++i) {
    const mfem::Vector vi = matrixColumn(V, i);
    x.Add(bvOverSigs[i], vi);
  }

  const double e1 = quadraticEnergy(A, b, x);
  mfem::Vector neg_x = alloc_vector(x.Size());
  neg_x = x;
  neg_x *= -1.0;
  const double e2 = quadraticEnergy(A, b, neg_x);

  x *= (e2 < e1 ? -delta : delta) / norm(x);

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
  SubspaceProjections projections = projectSubspaceGlobally(states, Astates, b);
  mfem::DenseMatrix& sAs = projections.sAs;
  symmetrize(sAs);

  for (int i = 0; i < sAs.Height(); ++i) {
    for (int j = 0; j < sAs.Width(); ++j) {
      if (std::isnan(sAs(i, j))) {
        throw TrustRegionException("States in subspace solve contain NaNs.");
      }
    }
  }

  mfem::DenseMatrix& ss = projections.ss;
  symmetrize(ss);

  double trace_mag = 0.0;
  mfem::DenseMatrix T = orthonormalBasisTransform(ss, trace_mag);
  if (trace_mag == 0.0) {
    mfem::Vector sol(*states[0]);
    sol = 0.0;
    return std::make_tuple(sol, std::vector<std::shared_ptr<mfem::Vector>>{}, std::vector<double>{}, 0.0);
  }
  if (T.Width() == 0) {
    throw TrustRegionException("No independent directions in MFEM subspace solve.");
  }
  mfem::DenseMatrix pAp = tripleProduct(T, sAs, T);
  symmetrize(pAp);

  const mfem::Vector& sb = projections.sb;
  const mfem::Vector pb = projectWithTranspose(T, sb);

  auto [reduced_x, leftvecs, leftvals, success] = exactTrustRegionSolve(pAp, pb, delta, num_leftmost);
  (void)success;
  const double energy = quadraticEnergy(pAp, pb, reduced_x);

  mfem::Vector coeffs(T.Height());
  T.Mult(reduced_x, coeffs);
  mfem::Vector sol = combineDirections(states, coeffs);
  std::vector<std::shared_ptr<mfem::Vector>> leftmosts;
  for (const auto& leftvec : leftvecs) {
    mfem::Vector left_coeffs(T.Height());
    T.Mult(leftvec, left_coeffs);
    leftmosts.emplace_back(std::make_shared<mfem::Vector>(combineDirections(states, left_coeffs)));
  }
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
  throw TrustRegionException("Trust-region subspace solve requires MFEM LAPACK support.");
  return std::make_tuple(b, std::vector<std::shared_ptr<mfem::Vector>>{}, std::vector<double>{}, 0.0);
#endif
}

/// @brief report unavailable MFEM subspace solve when MFEM was built without LAPACK.
TrustRegionSubspaceResult solveSubspaceProblemMfem(const std::vector<const mfem::Vector*>&,
                                                   const std::vector<const mfem::Vector*>&, const mfem::Vector& b,
                                                   double, int)
{
  throw TrustRegionException("MFEM trust-region subspace solve requires MFEM LAPACK support.");
  return std::make_tuple(b, std::vector<std::shared_ptr<mfem::Vector>>{}, std::vector<double>{}, 0.0);
}

#endif  // MFEM_USE_LAPACK

}  // namespace smith
