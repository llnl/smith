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

std::vector<const mfem::Vector*> toPointers(const std::vector<std::shared_ptr<mfem::Vector>>& vectors)
{
  std::vector<const mfem::Vector*> ptrs;
  ptrs.reserve(vectors.size());
  for (const auto& vector : vectors) {
    ptrs.push_back(vector.get());
  }
  return ptrs;
}

std::vector<mfem::Vector> prepareExactTrustRegionLeftmosts(CachedTrustRegionSubspaceProblem& prepared, int num_leftmost)
{
  prepared.eigenvalues.SetSize(prepared.projected_rhs.Size());
  prepared.eigenvectors.SetSize(prepared.projected_hessian.Height(), prepared.projected_hessian.Width());

  mfem::DenseMatrix projected_hessian_copy(prepared.projected_hessian);
  projected_hessian_copy.Eigensystem(prepared.eigenvalues, prepared.eigenvectors);

  prepared.eigen_rhs.SetSize(prepared.eigenvalues.Size());
  for (int i = 0; i < prepared.eigenvalues.Size(); ++i) {
    const mfem::Vector vi = matrixColumn(prepared.eigenvectors, i);
    prepared.eigen_rhs[i] = dot(vi, prepared.projected_rhs);
  }

  std::vector<mfem::Vector> reduced_leftmosts;
  const int num_leftmost_possible = std::min(num_leftmost, prepared.eigenvalues.Size());
  reduced_leftmosts.reserve(static_cast<size_t>(num_leftmost_possible));
  prepared.leftvals.clear();
  prepared.leftvals.reserve(static_cast<size_t>(num_leftmost_possible));
  for (int i = 0; i < num_leftmost_possible; ++i) {
    reduced_leftmosts.emplace_back(matrixColumn(prepared.eigenvectors, i));
    prepared.leftvals.emplace_back(prepared.eigenvalues[i]);
  }
  return reduced_leftmosts;
}

std::pair<mfem::Vector, bool> solvePreparedExactTrustRegionProblem(const CachedTrustRegionSubspaceProblem& prepared,
                                                                   double delta)
{
  const mfem::DenseMatrix& A = prepared.projected_hessian;
  const mfem::Vector& b = prepared.projected_rhs;
  const mfem::Vector& sigs = prepared.eigenvalues;
  const mfem::DenseMatrix& V = prepared.eigenvectors;
  const mfem::Vector& bv = prepared.eigen_rhs;

  mfem::Vector workspace(6 * b.Size());
  int offset = 0;
  auto alloc_vector = [&](int size) {
    mfem::Vector v(workspace.GetData() + offset, size);
    offset += size;
    return v;
  };

  mfem::Vector bvOverSigs = alloc_vector(sigs.Size());
  for (int i = 0; i < sigs.Size(); ++i) {
    bvOverSigs[i] = bv[i] / sigs[i];
  }
  const double sigScale = sumAbs(sigs) / sigs.Size();
  const double eps = 1e-12 * sigScale;
  const mfem::Vector leftMost = matrixColumn(V, 0);
  const double minSig = sigs[0];

  if ((minSig >= eps) && (norm(bvOverSigs) <= delta)) {
    mfem::Vector x = alloc_vector(b.Size());
    x = 0.0;
    for (int i = 0; i < b.Size(); ++i) {
      const mfem::Vector vi = matrixColumn(V, i);
      x.Add(bvOverSigs[i], vi);
    }
    return std::make_pair(x, true);
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

    return std::make_pair(e1 < e2 ? x1 : x2, true);
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

  return std::make_pair(x, success);
}

}  // namespace

/// @brief prepares reduced trust-region subspace data reusable across trust-region radius updates
CachedTrustRegionSubspaceProblem prepareSubspaceProblem(const std::vector<const mfem::Vector*>& directions,
                                                        const std::vector<const mfem::Vector*>& A_directions,
                                                        const mfem::Vector& b, int num_leftmost)
{
  SMITH_MARK_FUNCTION;
  CachedTrustRegionSubspaceProblem prepared;
  prepared.zero_solution = b;
  prepared.zero_solution = 0.0;

  SubspaceProjections projections = projectSubspaceGlobally(directions, A_directions, b);
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
    return prepared;
  }
  if (T.Width() == 0) {
    throw TrustRegionException("No independent directions in MFEM subspace solve.");
  }
  prepared.projected_hessian = tripleProduct(T, sAs, T);
  symmetrize(prepared.projected_hessian);

  const mfem::Vector& sb = projections.sb;
  prepared.projected_rhs = projectWithTranspose(T, sb);

  for (int j = 0; j < T.Width(); ++j) {
    prepared.basis.emplace_back(std::make_shared<mfem::Vector>(combineDirections(directions, matrixColumn(T, j))));
  }
  const auto reduced_leftmosts = prepareExactTrustRegionLeftmosts(prepared, num_leftmost);
  const auto basis_ptrs = toPointers(prepared.basis);
  prepared.leftmosts.clear();
  prepared.leftmosts.reserve(reduced_leftmosts.size());
  for (const auto& leftvec : reduced_leftmosts) {
    prepared.leftmosts.emplace_back(std::make_shared<mfem::Vector>(combineDirections(basis_ptrs, leftvec)));
  }

  return prepared;
}

/// @brief solves cached reduced trust-region problem for given trust-region radius
TrustRegionSubspaceResult solvePreparedSubspaceProblem(const CachedTrustRegionSubspaceProblem& prepared, double delta)
{
  SMITH_MARK_FUNCTION;
  if (prepared.basis.empty()) {
    mfem::Vector sol(prepared.zero_solution);
    sol = 0.0;
    return std::make_tuple(sol, prepared.leftmosts, prepared.leftvals, 0.0);
  }

  auto [reduced_x, success] = solvePreparedExactTrustRegionProblem(prepared, delta);
  const double energy = quadraticEnergy(prepared.projected_hessian, prepared.projected_rhs, reduced_x);

  const auto basis_ptrs = toPointers(prepared.basis);
  mfem::Vector sol = combineDirections(basis_ptrs, reduced_x);
  return std::make_tuple(sol, prepared.leftmosts, prepared.leftvals, energy);
}

TrustRegionSubspaceResult solveSubspaceProblem(const std::vector<const mfem::Vector*>& directions,
                                               const std::vector<const mfem::Vector*>& A_directions,
                                               const mfem::Vector& b, double delta, int num_leftmost)
{
  return solvePreparedSubspaceProblem(prepareSubspaceProblem(directions, A_directions, b, num_leftmost), delta);
}

TrustRegionSubspaceResult solveSubspaceProblemMfem(const std::vector<const mfem::Vector*>& directions,
                                                   const std::vector<const mfem::Vector*>& A_directions,
                                                   const mfem::Vector& b, double delta, int num_leftmost)
{
  return solvePreparedSubspaceProblem(prepareSubspaceProblem(directions, A_directions, b, num_leftmost), delta);
}

#else

TrustRegionSubspaceResult solveSubspaceProblem(const std::vector<const mfem::Vector*>& directions,
                                               const std::vector<const mfem::Vector*>& A_directions,
                                               const mfem::Vector& b, double delta, int num_leftmost)
{
  throw TrustRegionException("Trust-region subspace solve requires MFEM LAPACK support.");
  return std::make_tuple(b, std::vector<std::shared_ptr<mfem::Vector>>{}, std::vector<double>{}, 0.0);
}

CachedTrustRegionSubspaceProblem prepareSubspaceProblem(const std::vector<const mfem::Vector*>& directions,
                                                        const std::vector<const mfem::Vector*>& A_directions,
                                                        const mfem::Vector& b, int)
{
  throw TrustRegionException("Trust-region subspace solve requires MFEM LAPACK support.");
  CachedTrustRegionSubspaceProblem prepared;
  prepared.zero_solution = b;
  return prepared;
}

/// @brief report unavailable MFEM subspace solve when MFEM was built without LAPACK.
TrustRegionSubspaceResult solveSubspaceProblemMfem(const std::vector<const mfem::Vector*>&,
                                                   const std::vector<const mfem::Vector*>&, const mfem::Vector& b,
                                                   double, int)
{
  throw TrustRegionException("MFEM trust-region subspace solve requires MFEM LAPACK support.");
  return std::make_tuple(b, std::vector<std::shared_ptr<mfem::Vector>>{}, std::vector<double>{}, 0.0);
}

TrustRegionSubspaceResult solvePreparedSubspaceProblem(const CachedTrustRegionSubspaceProblem& prepared, double)
{
  throw TrustRegionException("Trust-region subspace solve requires MFEM LAPACK support.");
  return std::make_tuple(prepared.zero_solution, std::vector<std::shared_ptr<mfem::Vector>>{}, std::vector<double>{},
                         0.0);
}

#endif  // MFEM_USE_LAPACK

}  // namespace smith
