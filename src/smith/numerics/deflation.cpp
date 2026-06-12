// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/deflation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "mpi.h"

#include "_hypre_parcsr_mv.h"

#include "smith/numerics/batched_matvec.hpp"
#include "smith/numerics/bsr_operator.hpp"

namespace smith {

namespace {

bool factorCholesky(const mfem::DenseMatrix& matrix, mfem::DenseMatrix& factors_storage, mfem::CholeskyFactors& factors)
{
  factors_storage = matrix;
  factors.data = factors_storage.Data();
  return factors.Factor(matrix.Height());
}

void choleskySolve(const mfem::CholeskyFactors& factors, const mfem::DenseMatrixInverse& fallback_lu, bool use_cholesky,
                   int size, const mfem::Vector& rhs, mfem::Vector& x)
{
  if (!use_cholesky) {
    fallback_lu.Mult(rhs, x);
    return;
  }
  x.SetSize(size);
  for (int i = 0; i < size; ++i) x(i) = rhs(i);
  factors.Solve(size, 1, x.GetData());
}

bool envFlagEnabled(const char* name)
{
  const char* value = std::getenv(name);
  if (!value) return false;
  return value[0] != '\0' && value[0] != '0';
}

double envDoubleOrDefault(const char* name, double default_value)
{
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') return default_value;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  return (end != value && std::isfinite(parsed) && parsed >= 0.0) ? parsed : default_value;
}

/// Recursive coordinate bisection of element centroids into `s` pieces: split the index
/// range along the longest bounding-box axis at the proportional median. Deterministic
/// (stable ordering, no randomness) so every run produces the same partition.
void recursiveBisect(const std::vector<std::array<double, 3>>& centroids, std::vector<int>& idx, int begin, int end,
                     int s, int& next_piece, std::vector<int>& piece_of_element)
{
  if (s <= 1 || end - begin <= 1) {
    const int p = next_piece++;
    for (int i = begin; i < end; ++i) piece_of_element[static_cast<size_t>(idx[static_cast<size_t>(i)])] = p;
    return;
  }
  double lo[3] = {1e300, 1e300, 1e300};
  double hi[3] = {-1e300, -1e300, -1e300};
  for (int i = begin; i < end; ++i) {
    const auto& c = centroids[static_cast<size_t>(idx[static_cast<size_t>(i)])];
    for (int d = 0; d < 3; ++d) {
      lo[d] = std::min(lo[d], c[static_cast<size_t>(d)]);
      hi[d] = std::max(hi[d], c[static_cast<size_t>(d)]);
    }
  }
  int axis = 0;
  for (int d = 1; d < 3; ++d) {
    if (hi[d] - lo[d] > hi[axis] - lo[axis]) axis = d;
  }
  std::stable_sort(idx.begin() + begin, idx.begin() + end, [&](int a, int b) {
    return centroids[static_cast<size_t>(a)][static_cast<size_t>(axis)] <
           centroids[static_cast<size_t>(b)][static_cast<size_t>(axis)];
  });
  const int sl = s / 2;
  const int sr = s - sl;
  const int mid = begin + static_cast<int>((static_cast<long long>(end - begin) * sl) / s);
  recursiveBisect(centroids, idx, begin, mid, sl, next_piece, piece_of_element);
  recursiveBisect(centroids, idx, mid, end, sr, next_piece, piece_of_element);
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

DeflationIndefiniteCoarseException::DeflationIndefiniteCoarseException(double eigenvalue, const mfem::Vector& direction)
    : std::runtime_error("Deflation coarse operator is indefinite"), eigenvalue_(eigenvalue), direction_(direction)
{
}

int lanczosExtremeEigenpairs(int n, int k, const std::function<void(const double*, double*)>& apply, bool largest,
                             std::vector<double>& evals, std::vector<mfem::Vector>& evecs, int max_iters, double tol)
{
  evals.clear();
  evecs.clear();
  if (n <= 0 || k <= 0) return 0;
  k = std::min(k, n);
  const int maxit = std::min(n, (max_iters > 0) ? max_iters : std::max(100, 20 * k));

  // Deterministic pseudo-random start vector (fixed LCG): identical on every rank, and
  // generically non-orthogonal to the target eigenvectors.
  std::vector<mfem::Vector> V;
  V.reserve(static_cast<size_t>(maxit) + 1);
  {
    mfem::Vector v0(n);
    unsigned long long state = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < n; ++i) {
      state = state * 6364136223846793005ull + 1442695040888963407ull;
      v0(i) = static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0) - 0.5;
    }
    v0 /= std::sqrt(v0 * v0);
    V.emplace_back(std::move(v0));
  }

  std::vector<double> alpha;
  std::vector<double> beta;
  mfem::Vector w(n);
  mfem::DenseMatrix ritz_vectors;  // columns of the last tridiagonal eigensystem
  mfem::Vector ritz_values;
  int steps = 0;
  bool converged = false;

  for (int it = 0; it < maxit; ++it) {
    apply(V[static_cast<size_t>(it)].GetData(), w.GetData());
    const double a = V[static_cast<size_t>(it)] * w;
    alpha.push_back(a);
    w.Add(-a, V[static_cast<size_t>(it)]);
    if (it > 0) w.Add(-beta[static_cast<size_t>(it) - 1], V[static_cast<size_t>(it) - 1]);
    // Full reorthogonalization, two passes for robustness.
    for (int pass = 0; pass < 2; ++pass) {
      for (int j = 0; j <= it; ++j) {
        const double proj = V[static_cast<size_t>(j)] * w;
        w.Add(-proj, V[static_cast<size_t>(j)]);
      }
    }
    const double b = std::sqrt(w * w);
    steps = it + 1;

    // Eigensystem of the current tridiagonal T (tiny: steps x steps).
    mfem::DenseMatrix T(steps, steps);
    T = 0.0;
    for (int i = 0; i < steps; ++i) {
      T(i, i) = alpha[static_cast<size_t>(i)];
      if (i + 1 < steps) T(i, i + 1) = T(i + 1, i) = beta[static_cast<size_t>(i)];
    }
    mfem::DenseMatrixEigensystem eig(T);
    eig.Eval();
    ritz_values = eig.Eigenvalues();    // ascending
    ritz_vectors = eig.Eigenvectors();  // columns

    if (steps >= k) {
      // Residual of Ritz pair (theta, y): ||A y - theta y|| = |b * S(last, idx)|.
      converged = true;
      double scale = 0.0;
      for (int i = 0; i < steps; ++i) scale = std::max(scale, std::abs(ritz_values(i)));
      for (int q = 0; q < k; ++q) {
        const int idx = largest ? steps - 1 - q : q;
        if (std::abs(b * ritz_vectors(steps - 1, idx)) > tol * std::max(scale, 1.0e-300)) {
          converged = false;
          break;
        }
      }
    }
    if (converged || b <= 1.0e-14) break;  // invariant subspace found if b ~ 0

    beta.push_back(b);
    mfem::Vector vnext(w);
    vnext *= 1.0 / b;
    V.emplace_back(std::move(vnext));
  }

  const int kk = std::min(k, steps);
  evals.reserve(static_cast<size_t>(kk));
  evecs.reserve(static_cast<size_t>(kk));
  for (int q = 0; q < kk; ++q) {
    const int idx = largest ? steps - 1 - q : q;
    evals.push_back(ritz_values(idx));
    mfem::Vector y(n);
    y = 0.0;
    for (int j = 0; j < steps; ++j) y.Add(ritz_vectors(j, idx), V[static_cast<size_t>(j)]);
    const double nrm = std::sqrt(y * y);
    if (nrm > 0.0) y *= 1.0 / nrm;
    evecs.emplace_back(std::move(y));
  }
  return kk;
}

DeflationPreconditioner::DeflationPreconditioner(mfem::ParFiniteElementSpace& fes, bool use_smoother,
                                                 mfem::HypreSmoother::Type smoother_type)
    : mfem::Solver(fes.GetTrueVSize()), smoother_type_(smoother_type), use_smoother_(use_smoother)
{
  attachFES(fes);
}

DeflationPreconditioner::DeflationPreconditioner(bool use_smoother, mfem::HypreSmoother::Type smoother_type)
    : mfem::Solver(0), smoother_type_(smoother_type), use_smoother_(use_smoother)
{
  // FES bound later via attachFES(...). SetOperator before attach is an error.
}

void DeflationPreconditioner::attachFES(mfem::ParFiniteElementSpace& fes)
{
  fes_ = &fes;
  height = width = fes.GetTrueVSize();
  dim_ = fes_->GetVDim();
  if (dim_ < 2 || dim_ > 3) {
    throw std::runtime_error("DeflationPreconditioner: vdim must be 2 or 3");
  }
  // Per-component mode counts: 1 (const) + dim (linear) [+ dim*(dim+1)/2 (quadratic)].
  const int per_comp_affine = 1 + dim_;
  const int per_comp_quadratic = per_comp_affine + dim_ * (dim_ + 1) / 2;
  const int per_comp = (order_ == DeflationOrder::Quadratic) ? per_comp_quadratic : per_comp_affine;
  modes_per_rank_ = num_pieces_ * dim_ * per_comp;
  MPI_Comm comm = fes_->GetComm();
  MPI_Comm_rank(comm, &my_rank_);
  MPI_Comm_size(comm, &n_ranks_);
  my_col_offset_ = my_rank_ * modes_per_rank_;
  m_ = n_ranks_ * modes_per_rank_;

  // Element-volume-weighted centroid of this rank's mesh, used to shift monomials.
  rank_centroid_.SetSize(dim_);
  rank_centroid_ = 0.0;
  if (mfem::Mesh* mesh = fes_->GetMesh()) {
    double total_vol = 0.0;
    mfem::Vector elem_center(dim_);
    for (int e = 0; e < mesh->GetNE(); ++e) {
      mfem::ElementTransformation* T = mesh->GetElementTransformation(e);
      const mfem::Geometry::Type geom = mesh->GetElementBaseGeometry(e);
      const mfem::IntegrationPoint& ref_center = mfem::Geometries.GetCenter(geom);
      T->Transform(ref_center, elem_center);
      const double vol = mesh->GetElementVolume(e);
      rank_centroid_.Add(vol, elem_center);
      total_vol += vol;
    }
    if (total_vol > 0.0) rank_centroid_ *= (1.0 / total_vol);
  }

  // Force rebuild on next SetOperator.
  W_local_.clear();
  W_mat_.reset();
  factored_ = false;
  leftmost_valid_ = false;
}

void DeflationPreconditioner::setDeflationOrder(DeflationOrder order)
{
  if (order_ == order) return;
  order_ = order;
  if (fes_) {
    // Re-run sizing + centroid + invalidate basis.
    attachFES(*fes_);
  }
}

void DeflationPreconditioner::setNumPieces(int s)
{
  s = std::max(1, s);
  if (num_pieces_ == s) return;
  num_pieces_ = s;
  if (fes_) {
    attachFES(*fes_);  // re-run sizing + invalidate basis
  }
}

void DeflationPreconditioner::buildBasis()
{
  if (!W_local_.empty()) return;

  const int dim = dim_;
  W_local_.clear();
  W_local_.reserve(static_cast<size_t>(modes_per_rank_));

  // Monomial list (in centered coordinates xi = x - centroid):
  //   degree 0:  1
  //   degree 1:  xi_0, xi_1, [xi_2]
  //   degree 2:  xi_j * xi_k  for 0 <= j <= k < dim
  // Each is replicated `dim_` times, one per spatial component.
  struct Monomial {
    int i = -1;
    int j = -1;
  };  // -1 means "constant" sentinel; otherwise i<=j<dim.
  std::vector<Monomial> monomials;
  monomials.push_back({-1, -1});  // constant
  for (int k = 0; k < dim_; ++k) monomials.push_back({k, -1});
  if (order_ == DeflationOrder::Quadratic) {
    for (int j = 0; j < dim_; ++j) {
      for (int k = j; k < dim_; ++k) monomials.push_back({j, k});
    }
  }

  mfem::Vector centroid = rank_centroid_;
  for (int c = 0; c < dim_; ++c) {
    for (const Monomial& mo : monomials) {
      const Monomial cap = mo;
      mfem::VectorFunctionCoefficient coef(dim_, [c, cap, dim, centroid](const mfem::Vector& p, mfem::Vector& v) {
        v.SetSize(dim);
        v = 0.0;
        double val = 1.0;
        if (cap.i >= 0) val *= (p(cap.i) - centroid(cap.i));
        if (cap.j >= 0) val *= (p(cap.j) - centroid(cap.j));
        v(c) = val;
      });
      mfem::ParGridFunction gf(fes_);
      gf.ProjectCoefficient(coef);
      mfem::Vector col;
      gf.GetTrueDofs(col);
      W_local_.emplace_back(std::move(col));
    }
  }

  // Sub-rank pieces: replicate each rank-wide column into num_pieces_ piece-masked copies
  // (column order piece-major, then component, then monomial — every tdof row still touches
  // exactly one contiguous per_comp block, which packWMatrix's compact layout requires).
  // Shared interface nodes are assigned to the lowest adjacent piece id; per-(piece,comp)
  // MGS below re-centers and orthonormalizes each piece's monomials.
  if (num_pieces_ > 1) {
    mfem::Mesh* mesh = fes_->GetMesh();
    const int ne = mesh->GetNE();
    std::vector<std::array<double, 3>> centroids(static_cast<size_t>(ne), {0.0, 0.0, 0.0});
    mfem::Vector center(dim_);
    for (int e = 0; e < ne; ++e) {
      mfem::ElementTransformation* T = mesh->GetElementTransformation(e);
      T->Transform(mfem::Geometries.GetCenter(mesh->GetElementBaseGeometry(e)), center);
      for (int d = 0; d < dim_; ++d) centroids[static_cast<size_t>(e)][static_cast<size_t>(d)] = center(d);
    }
    std::vector<int> idx(static_cast<size_t>(ne));
    for (int e = 0; e < ne; ++e) idx[static_cast<size_t>(e)] = e;
    std::vector<int> piece_of_element(static_cast<size_t>(ne), 0);
    int next_piece = 0;
    recursiveBisect(centroids, idx, 0, ne, num_pieces_, next_piece, piece_of_element);

    const int n = fes_->GetTrueVSize();
    std::vector<int> piece_of_tdof(static_cast<size_t>(n), std::numeric_limits<int>::max());
    mfem::Array<int> vdofs;
    for (int e = 0; e < ne; ++e) {
      const int p = piece_of_element[static_cast<size_t>(e)];
      fes_->GetElementVDofs(e, vdofs);
      for (int v = 0; v < vdofs.Size(); ++v) {
        const int ldof = (vdofs[v] >= 0) ? vdofs[v] : -1 - vdofs[v];
        const int tdof = fes_->GetLocalTDofNumber(ldof);
        if (tdof >= 0) {
          piece_of_tdof[static_cast<size_t>(tdof)] = std::min(piece_of_tdof[static_cast<size_t>(tdof)], p);
        }
      }
    }
    for (int i = 0; i < n; ++i) {
      if (piece_of_tdof[static_cast<size_t>(i)] == std::numeric_limits<int>::max()) {
        piece_of_tdof[static_cast<size_t>(i)] = 0;
      }
    }

    const int base_modes = dim_ * static_cast<int>(monomials.size());
    std::vector<mfem::Vector> base_cols(std::make_move_iterator(W_local_.begin()),
                                        std::make_move_iterator(W_local_.end()));
    W_local_.clear();
    W_local_.reserve(static_cast<size_t>(num_pieces_ * base_modes));
    for (int p = 0; p < num_pieces_; ++p) {
      for (int j = 0; j < base_modes; ++j) {
        mfem::Vector col(base_cols[static_cast<size_t>(j)]);
        for (int i = 0; i < n; ++i) {
          if (piece_of_tdof[static_cast<size_t>(i)] != p) col(i) = 0.0;
        }
        W_local_.emplace_back(std::move(col));
      }
    }
  }
  applyEssentialDofMask();
  // Per-component modified Gram-Schmidt over the local tdof inner product.
  // Columns belonging to different components have disjoint support, so we only
  // need to orthonormalize within each component's mode block.
  //
  // Two safeguards needed once `Quadratic` monomials enter the basis: monomials
  // of different polynomial degree have very different natural scales (e.g. for
  // this thin-arch geometry the y-domain is ~0.025, so xi_y^2 ~ 1e-4 vs
  // const ~ 1). Without (a) pre-normalization, MGS projection coefficients span
  // many orders and the subtraction loses precision; without (b) a relative
  // collapse threshold, a near-rank-deficient column is renormalized from noise
  // and corrupts W^T A W.
  const int per_comp = static_cast<int>(monomials.size());
  constexpr double rank_drop_tol = 1.0e-12;
  const int n_blocks = modes_per_rank_ / per_comp;  // dim_ blocks per piece
  for (int c = 0; c < n_blocks; ++c) {
    for (int k = 0; k < per_comp; ++k) {
      mfem::Vector& vk = W_local_[static_cast<size_t>(c * per_comp + k)];
      const double init_nrm = std::sqrt(vk * vk);
      if (init_nrm > 0.0) vk *= 1.0 / init_nrm;
      for (int j = 0; j < k; ++j) {
        const mfem::Vector& vj = W_local_[static_cast<size_t>(c * per_comp + j)];
        const double proj = (vk * vj);
        vk.Add(-proj, vj);
      }
      const double nrm2 = (vk * vk);
      if (nrm2 > rank_drop_tol) {
        vk *= 1.0 / std::sqrt(nrm2);
      } else {
        vk = 0.0;  // mode collapsed onto earlier modes; drop it
      }
    }
  }
  packWMatrix();
}

void DeflationPreconditioner::packWMatrix()
{
  const int n = fes_->GetTrueVSize();
  // Build CSR row-by-row. Each row touches up to modes_per_rank_ cols; in practice exactly
  // (dim_+1) cols are nonzero (one component's modes).
  int* I = new int[static_cast<size_t>(n) + 1];
  std::vector<int> J;
  std::vector<double> Vals;
  J.reserve(static_cast<size_t>(n) * static_cast<size_t>(dim_ + 1));
  Vals.reserve(static_cast<size_t>(n) * static_cast<size_t>(dim_ + 1));
  for (int i = 0; i < n; ++i) {
    I[i] = static_cast<int>(J.size());
    for (int j = 0; j < modes_per_rank_; ++j) {
      double v = W_local_[static_cast<size_t>(j)](i);
      if (v != 0.0) {
        J.push_back(j);
        Vals.push_back(v);
      }
    }
  }
  I[n] = static_cast<int>(J.size());
  int* Jp = new int[J.size()];
  double* Ap = new double[Vals.size()];
  std::copy(J.begin(), J.end(), Jp);
  std::copy(Vals.begin(), Vals.end(), Ap);
  W_mat_.reset(new mfem::SparseMatrix(I, Jp, Ap, n, modes_per_rank_, /*ownij*/ true, /*owna*/ true,
                                      /*issorted*/ true));
  W_dense_.SetSize(n, modes_per_rank_);
  for (int j = 0; j < modes_per_rank_; ++j) {
    for (int i = 0; i < n; ++i) W_dense_(i, j) = W_local_[static_cast<size_t>(j)](i);
  }

  // Compact per-row layout: each tdof row is nonzero only inside its (piece, component)
  // contiguous monomial block [b*per_comp, (b+1)*per_comp). Fully-masked rows store zeros
  // (block 0).
  const int per_comp = modes_per_rank_ / (dim_ * num_pieces_);
  w_compact_.assign(static_cast<size_t>(n) * static_cast<size_t>(per_comp), 0.0);
  w_row_comp_.assign(static_cast<size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    int comp = 0;
    for (int j = 0; j < modes_per_rank_; ++j) {
      if (W_local_[static_cast<size_t>(j)](i) != 0.0) {
        comp = j / per_comp;
        break;
      }
    }
    w_row_comp_[static_cast<size_t>(i)] = comp;
    for (int k = 0; k < per_comp; ++k) {
      w_compact_[static_cast<size_t>(i) * static_cast<size_t>(per_comp) + static_cast<size_t>(k)] =
          W_local_[static_cast<size_t>(comp * per_comp + k)](i);
    }
  }
}

void DeflationPreconditioner::wTransposeCompact(const mfem::Vector& r, mfem::Vector& rhs) const
{
  const int n = fes_->GetTrueVSize();
  const int pc = modes_per_rank_ / (dim_ * num_pieces_);
  rhs.SetSize(modes_per_rank_);
  rhs = 0.0;
  double* out = rhs.GetData();
  const double* rd = r.GetData();
  for (int i = 0; i < n; ++i) {
    const double ri = rd[i];
    const double* wrow = &w_compact_[static_cast<size_t>(i) * static_cast<size_t>(pc)];
    double* dst = out + w_row_comp_[static_cast<size_t>(i)] * pc;
    for (int k = 0; k < pc; ++k) dst[k] += wrow[k] * ri;
  }
}

void DeflationPreconditioner::wAddMultCompact(const double* alpha_local, mfem::Vector& y, double scale) const
{
  const int n = fes_->GetTrueVSize();
  const int pc = modes_per_rank_ / (dim_ * num_pieces_);
  double* yd = y.GetData();
  for (int i = 0; i < n; ++i) {
    const double* wrow = &w_compact_[static_cast<size_t>(i) * static_cast<size_t>(pc)];
    const double* ab = alpha_local + w_row_comp_[static_cast<size_t>(i)] * pc;
    double d = 0.0;
    for (int k = 0; k < pc; ++k) d += wrow[k] * ab[k];
    yd[i] += scale * d;
  }
}

bool DeflationPreconditioner::skylineFactorWtAW() const
{
  const int m = m_;
  sky_first_.assign(static_cast<size_t>(m), 0);
  sky_ptr_.assign(static_cast<size_t>(m) + 1, 0);
  // Profile from the lower triangle's first structural nonzero per row. Non-neighbor rank
  // blocks are exact zeros (assembled as allreduced sums of zeros), so this is the
  // rank-coupling block sparsity; Cholesky fill stays inside the row profile.
  for (int i = 0; i < m; ++i) {
    int first = i;
    for (int j = 0; j < i; ++j) {
      if (WtAW_(i, j) != 0.0) {
        first = j;
        break;
      }
    }
    sky_first_[static_cast<size_t>(i)] = first;
    sky_ptr_[static_cast<size_t>(i) + 1] = sky_ptr_[static_cast<size_t>(i)] + (i - first + 1);
  }
  sky_L_.assign(static_cast<size_t>(sky_ptr_[static_cast<size_t>(m)]), 0.0);

  // Row-oriented (up-looking) Cholesky-Crout over the profile:
  //   L(i,j) = (A(i,j) - sum_k L(i,k) L(j,k)) / L(j,j),  k in [max(first_i, first_j), j)
  //   L(i,i) = sqrt(A(i,i) - sum_k L(i,k)^2)
  for (int i = 0; i < m; ++i) {
    const int fi = sky_first_[static_cast<size_t>(i)];
    double* Li = &sky_L_[static_cast<size_t>(sky_ptr_[static_cast<size_t>(i)])] - fi;  // Li[j] = L(i,j)
    for (int j = fi; j < i; ++j) {
      const int fj = sky_first_[static_cast<size_t>(j)];
      const double* Lj = &sky_L_[static_cast<size_t>(sky_ptr_[static_cast<size_t>(j)])] - fj;
      double a = 0.0;
      for (int k = std::max(fi, fj); k < j; ++k) a += Li[k] * Lj[k];
      Li[j] = (WtAW_(i, j) - a) / Lj[j];
    }
    double a = 0.0;
    for (int k = fi; k < i; ++k) a += Li[k] * Li[k];
    const double pivot = WtAW_(i, i) - a;
    if (!(pivot > 0.0)) return false;
    Li[i] = std::sqrt(pivot);
  }
  return true;
}

void DeflationPreconditioner::skylineSolve(const mfem::Vector& rhs, mfem::Vector& x) const
{
  const int m = m_;
  x.SetSize(m);
  for (int i = 0; i < m; ++i) x(i) = rhs(i);
  double* xd = x.GetData();
  // forward: L y = b
  for (int i = 0; i < m; ++i) {
    const int fi = sky_first_[static_cast<size_t>(i)];
    const double* Li = &sky_L_[static_cast<size_t>(sky_ptr_[static_cast<size_t>(i)])] - fi;
    double a = 0.0;
    for (int j = fi; j < i; ++j) a += Li[j] * xd[j];
    xd[i] = (xd[i] - a) / Li[i];
  }
  // back: L^T x = y
  for (int i = m - 1; i >= 0; --i) {
    const int fi = sky_first_[static_cast<size_t>(i)];
    const double* Li = &sky_L_[static_cast<size_t>(sky_ptr_[static_cast<size_t>(i)])] - fi;
    const double xi = (xd[i] /= Li[i]);
    for (int j = fi; j < i; ++j) xd[j] -= Li[j] * xi;
  }
}

void DeflationPreconditioner::SetOperator(const mfem::Operator& op)
{
  if (!fes_) throw std::runtime_error("DeflationPreconditioner::SetOperator: attachFES not called");

  const mfem::HypreParMatrix* hyp = dynamic_cast<const mfem::HypreParMatrix*>(&op);
  const BSROperator* bsr_op = nullptr;
  if (!hyp) {
    bsr_op = dynamic_cast<const BSROperator*>(&op);
    if (bsr_op) {
      hyp = bsr_op->GetHypreMatrix();
    }
  }

  if (!hyp) {
    throw std::runtime_error("DeflationPreconditioner::SetOperator requires HypreParMatrix or BSROperator");
  }
  A_ = hyp;
  op_for_mult_ = &op;
  height = op.Height();
  width = op.Width();

  if (W_local_.empty()) buildBasis();

  // === TIMING BEGIN ===
  double t0 = MPI_Wtime();
  ++setop_calls_;
  // === TIMING END ===
  if (bsr_op && bsr_op->Enabled()) {
    assembleWtAW(*bsr_op, W_dense_, modes_per_rank_, WtAW_, &assemble_timings_);
  } else {
    assembleWtAW(*A_, W_dense_, modes_per_rank_, WtAW_, &assemble_timings_);
  }
  // === TIMING BEGIN ===
  double t1 = MPI_Wtime();
  setop_matvec_time_ += t1 - t0;
  // === TIMING END ===

  if (envFlagEnabled("SMITH_DEFLATION_VERIFY")) {
    verifyWtAWAssembly();
  }

  WtAW_uses_cholesky_ = skylineFactorWtAW();
  if (!WtAW_uses_cholesky_) {
    ++wtaw_cholesky_failures_;
  }
  coarse_sparse_eig_mode_ = m_ > static_cast<int>(envDoubleOrDefault("SMITH_DEFLATION_DENSE_EIG_MAX", 512.0));
  if (!coarse_sparse_eig_mode_) {
    computeCoarseSpectralData();
  } else {
    // Scale-out path: no replicated dense eigendecomposition. SPD follows from the skyline
    // Cholesky (all pivots positive); leftmost eigenpairs come from Lanczos on demand; the
    // rank-revealing pseudo-inverse fallback is unavailable, so an indefinite coarse matrix
    // runs smoother-only in Mult (its negative-curvature role is served by the Lanczos
    // leftmost directions routed to the trust region).
    coarse_is_spd_ = WtAW_uses_cholesky_;
    coarse_spectral_rank_ = WtAW_uses_cholesky_ ? m_ : 0;
    coarse_spectral_tol_ = 0.0;
    coarse_evals_kept_.SetSize(0);
    coarse_evecs_kept_.SetSize(m_, 0);
  }

  // Diagonal block (my_offset, my_offset) used by the AdditiveLocal mode. Only the owner
  // contributes to its own (p,p) block during assembleWtAW, so post-Allreduce this is
  // already exactly W_p^T A_diag|_p W_p.
  WtAW_pp_.SetSize(modes_per_rank_, modes_per_rank_);
  for (int j = 0; j < modes_per_rank_; ++j) {
    for (int i = 0; i < modes_per_rank_; ++i) {
      WtAW_pp_(i, j) = WtAW_(my_col_offset_ + i, my_col_offset_ + j);
    }
  }
  for (int j = 0; j < WtAW_pp_.Height(); ++j) {
    if (WtAW_pp_(j, j) == 0.0) WtAW_pp_(j, j) = 1.0;
  }
  WtAW_pp_uses_cholesky_ = factorCholesky(WtAW_pp_, WtAW_pp_cholesky_, WtAW_pp_cholesky_factors_);
  if (!WtAW_pp_uses_cholesky_) {
    ++wtaw_pp_cholesky_failures_;
    WtAW_pp_lu_inv_.SetOperator(WtAW_pp_);
  }

  // Cache neighbor list + per-neighbor WtAW blocks for AdditiveSchwarz mode.
  // Neighbor graph = A's halo neighbor graph (= WtAW's structural off-diag pattern).
  {
    auto* parA = static_cast<hypre_ParCSRMatrix*>(const_cast<mfem::HypreParMatrix&>(*A_));
    hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(parA);
    if (!comm_pkg) {
      hypre_MatvecCommPkgCreate(parA);
      comm_pkg = hypre_ParCSRMatrixCommPkg(parA);
    }
    const int n_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
    const int n_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
    schwarz_send_neighbors_.clear();
    schwarz_send_neighbors_.reserve(static_cast<size_t>(n_sends));
    for (int s = 0; s < n_sends; ++s) {
      schwarz_send_neighbors_.push_back(hypre_ParCSRCommPkgSendProc(comm_pkg, s));
    }
    schwarz_neighbors_.clear();
    schwarz_neighbor_blocks_.clear();
    schwarz_neighbors_.reserve(static_cast<size_t>(n_recvs));
    schwarz_neighbor_blocks_.reserve(static_cast<size_t>(n_recvs));
    for (int r = 0; r < n_recvs; ++r) {
      int s_proc = hypre_ParCSRCommPkgRecvProc(comm_pkg, r);
      int s_off = s_proc * modes_per_rank_;
      mfem::DenseMatrix block(modes_per_rank_, modes_per_rank_);
      for (int j = 0; j < modes_per_rank_; ++j) {
        for (int i = 0; i < modes_per_rank_; ++i) {
          block(i, j) = WtAW_(my_col_offset_ + i, s_off + j);
        }
      }
      schwarz_neighbors_.push_back(s_proc);
      schwarz_neighbor_blocks_.emplace_back(std::move(block));
    }
  }

  factored_ = true;
  leftmost_valid_ = false;
  // === TIMING BEGIN ===
  double t2 = MPI_Wtime();
  setop_factor_time_ += t2 - t1;
  // === TIMING END ===

  switch (smoother_variant_) {
    case DeflationSmoother::BlockJacobi:
      buildBlockJacobi();
      smoother_.reset();
      break;
    case DeflationSmoother::PointJacobi:
      buildPointJacobi(bsr_op);
      smoother_.reset();
      break;
    case DeflationSmoother::Hypre:
      smoother_ = std::make_unique<mfem::HypreSmoother>();
      smoother_->SetType(smoother_type_);
      smoother_->SetOperator(const_cast<mfem::HypreParMatrix&>(*A_));
      break;
  }
  // === TIMING BEGIN ===
  setop_smoother_time_ += MPI_Wtime() - t2;
  // === TIMING END ===
}

void DeflationPreconditioner::buildBlockJacobi()
{
  const int b = fes_->GetVDim();
  const int n = A_->Height();
  if (b < 1 || n % b != 0) {
    throw std::runtime_error("DeflationPreconditioner: block-Jacobi requires vdim-aligned true dofs");
  }
  auto* parA = static_cast<hypre_ParCSRMatrix*>(const_cast<mfem::HypreParMatrix&>(*A_));
  hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag(parA);
  HYPRE_Int* diag_i = hypre_CSRMatrixI(diag);
  HYPRE_Int* diag_j = hypre_CSRMatrixJ(diag);
  HYPRE_Real* diag_a = hypre_CSRMatrixData(diag);

  const int nb = n / b;
  block_jacobi_inv_.assign(static_cast<size_t>(nb) * static_cast<size_t>(b * b), 0.0);
  mfem::DenseMatrix block(b, b);
  for (int br = 0; br < nb; ++br) {
    // Gather the b x b diagonal block. BC-eliminated dofs appear as a zeroed row/col with
    // 1.0 on the diagonal, which decouples them inside the block; the LU inverse below is
    // then exactly identity on those components and the inverse of the active sub-block.
    block = 0.0;
    for (int i = 0; i < b; ++i) {
      const int row = br * b + i;
      for (HYPRE_Int idx = diag_i[row]; idx < diag_i[row + 1]; ++idx) {
        const int col = static_cast<int>(diag_j[idx]);
        if (col >= br * b && col < (br + 1) * b) {
          block(i, col - br * b) = diag_a[idx];
        }
      }
    }
    block.Invert();
    double* out = &block_jacobi_inv_[static_cast<size_t>(br) * static_cast<size_t>(b * b)];
    for (int i = 0; i < b; ++i) {
      for (int j = 0; j < b; ++j) out[i * b + j] = block(i, j);
    }
  }
}

void DeflationPreconditioner::buildPointJacobi(const BSROperator* bsr_op)
{
  const int n = A_->Height();
  point_diag_.assign(static_cast<size_t>(n), 1.0);
  if (bsr_op && bsr_op->Enabled()) {
    const BSRMatrix& diag = bsr_op->DiagBSR();
    const int b = diag.b;
    for (int br = 0; br < diag.nb_rows; ++br) {
      for (int m = diag.I[static_cast<size_t>(br)]; m < diag.I[static_cast<size_t>(br) + 1]; ++m) {
        if (diag.J[static_cast<size_t>(m)] == br) {
          const double* block = &diag.data[static_cast<size_t>(m) * static_cast<size_t>(b * b)];
          for (int i = 0; i < b; ++i) {
            point_diag_[static_cast<size_t>(br * b + i)] = block[i * b + i];
          }
          break;
        }
      }
    }
  } else {
    auto* parA = static_cast<hypre_ParCSRMatrix*>(const_cast<mfem::HypreParMatrix&>(*A_));
    hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag(parA);
    HYPRE_Int* diag_i = hypre_CSRMatrixI(diag);
    HYPRE_Int* diag_j = hypre_CSRMatrixJ(diag);
    HYPRE_Real* diag_a = hypre_CSRMatrixData(diag);
    for (int row = 0; row < n; ++row) {
      for (HYPRE_Int idx = diag_i[row]; idx < diag_i[row + 1]; ++idx) {
        if (static_cast<int>(diag_j[idx]) == row) {
          point_diag_[static_cast<size_t>(row)] = diag_a[idx];
          break;
        }
      }
    }
  }
}

void DeflationPreconditioner::applySmoother(const mfem::Vector& r, mfem::Vector& z) const
{
  if (smoother_variant_ == DeflationSmoother::PointJacobi && !point_diag_.empty()) {
    const int n = static_cast<int>(r.Size());
    const double* rd = r.GetData();
    double* zd = z.GetData();
    // divide (not multiply by a stored inverse) to match hypre Jacobi bit-for-bit
    for (int i = 0; i < n; ++i) zd[i] = rd[i] / point_diag_[static_cast<size_t>(i)];
  } else if (smoother_variant_ == DeflationSmoother::BlockJacobi && !block_jacobi_inv_.empty()) {
    const int b = fes_->GetVDim();
    const int nb = static_cast<int>(r.Size()) / b;
    const double* rd = r.GetData();
    double* zd = z.GetData();
    for (int br = 0; br < nb; ++br) {
      const double* inv = &block_jacobi_inv_[static_cast<size_t>(br) * static_cast<size_t>(b * b)];
      const double* rb = &rd[br * b];
      double* zb = &zd[br * b];
      for (int i = 0; i < b; ++i) {
        double acc = 0.0;
        for (int j = 0; j < b; ++j) acc += inv[i * b + j] * rb[j];
        zb[i] = acc;
      }
    }
  } else {
    smoother_->Mult(r, z);
  }
}

void DeflationPreconditioner::coarseSolve(const mfem::Vector& r, mfem::Vector& z0) const
{
  z0.SetSize(fes_->GetTrueVSize());
  z0 = 0.0;
  addScaledCoarseCorrection(r, z0, -1.0);
}

void DeflationPreconditioner::addCoarseCorrection(const mfem::Vector& r, mfem::Vector& y) const
{
  addScaledCoarseCorrection(r, y, 1.0);
}

void DeflationPreconditioner::addScaledCoarseCorrection(const mfem::Vector& r, mfem::Vector& y, double scale) const
{
  if (!factored_) throw std::runtime_error("DeflationPreconditioner::addCoarseCorrection: call SetOperator first");

  ++cc_calls_;
  double tcc0 = MPI_Wtime();
  mfem::Vector rhs_local(modes_per_rank_);
  W_mat_->MultTranspose(r, rhs_local);
  double tcc1 = MPI_Wtime();
  cc_wt_time_ += tcc1 - tcc0;

  if (coarse_mode_ == CoarseMode::AdditiveLocal) {
    mfem::Vector alpha_local(modes_per_rank_);
    choleskySolve(WtAW_pp_cholesky_factors_, WtAW_pp_lu_inv_, WtAW_pp_uses_cholesky_, modes_per_rank_, rhs_local,
                  alpha_local);
    double tcc2 = MPI_Wtime();
    cc_solve_time_ += tcc2 - tcc1;
    W_mat_->AddMult(alpha_local, y, scale);
    cc_w_time_ += MPI_Wtime() - tcc2;
    return;
  }

  if (coarse_mode_ == CoarseMode::AdditiveSchwarz) {
    // K=1 multi-step block-Jacobi-Schwarz. Replaces Allgather with one neighbor-only round.
    //   u = WtAW_pp^{-1} c
    //   exchange u with neighbors → u_s
    //   alpha = u - WtAW_pp^{-1} Σ_s WtAW_{p,s} u_s
    mfem::Vector u_local(modes_per_rank_);
    double tsolv0 = MPI_Wtime();
    choleskySolve(WtAW_pp_cholesky_factors_, WtAW_pp_lu_inv_, WtAW_pp_uses_cholesky_, modes_per_rank_, rhs_local,
                  u_local);
    cc_solve_time_ += MPI_Wtime() - tsolv0;
    double texch0 = MPI_Wtime();

    MPI_Comm comm = fes_->GetComm();
    const int n_nbr = static_cast<int>(schwarz_neighbors_.size());
    std::vector<std::vector<double>> recv_bufs(static_cast<size_t>(n_nbr),
                                               std::vector<double>(static_cast<size_t>(modes_per_rank_)));
    std::vector<MPI_Request> reqs;
    reqs.reserve(static_cast<size_t>(2 * n_nbr));
    for (int n = 0; n < n_nbr; ++n) {
      MPI_Request req;
      MPI_Irecv(recv_bufs[static_cast<size_t>(n)].data(), modes_per_rank_, MPI_DOUBLE,
                schwarz_neighbors_[static_cast<size_t>(n)], 31, comm, &req);
      reqs.push_back(req);
    }
    // Send to the ranks that reference our halo (hypre send-procs); receiving from
    // recv-procs only pairs correctly when each of our recv neighbors lists us as a
    // send target, which an asymmetric comm package does not guarantee.
    for (int neighbor : schwarz_send_neighbors_) {
      MPI_Request req;
      MPI_Isend(u_local.GetData(), modes_per_rank_, MPI_DOUBLE, neighbor, 31, comm, &req);
      reqs.push_back(req);
    }
    if (!reqs.empty()) {
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }
    cc_allgather_time_ += MPI_Wtime() - texch0;

    double tsolv1 = MPI_Wtime();
    mfem::Vector corr(modes_per_rank_);
    corr = 0.0;
    mfem::Vector tmp(modes_per_rank_);
    for (int n = 0; n < n_nbr; ++n) {
      mfem::Vector u_s(recv_bufs[static_cast<size_t>(n)].data(), modes_per_rank_);
      schwarz_neighbor_blocks_[static_cast<size_t>(n)].Mult(u_s, tmp);
      corr += tmp;
    }
    mfem::Vector delta(modes_per_rank_);
    choleskySolve(WtAW_pp_cholesky_factors_, WtAW_pp_lu_inv_, WtAW_pp_uses_cholesky_, modes_per_rank_, corr, delta);

    mfem::Vector alpha_local(u_local);
    alpha_local -= delta;
    cc_solve_time_ += MPI_Wtime() - tsolv1;
    double twa0 = MPI_Wtime();
    W_mat_->AddMult(alpha_local, y, scale);
    cc_w_time_ += MPI_Wtime() - twa0;
    return;
  }

  double tag0 = MPI_Wtime();
  mfem::Vector rhs_global(m_);
  MPI_Allgather(rhs_local.GetData(), modes_per_rank_, MPI_DOUBLE, rhs_global.GetData(), modes_per_rank_, MPI_DOUBLE,
                fes_->GetComm());
  cc_allgather_time_ += MPI_Wtime() - tag0;

  double tsolv2 = MPI_Wtime();
  mfem::Vector alpha(m_);
  if (WtAW_uses_cholesky_) {
    skylineSolve(rhs_global, alpha);
  } else {
    spectralCoarseSolve(rhs_global, alpha);
  }
  cc_solve_time_ += MPI_Wtime() - tsolv2;

  double twa1 = MPI_Wtime();
  mfem::Vector alpha_local(alpha.GetData() + my_col_offset_, modes_per_rank_);
  W_mat_->AddMult(alpha_local, y, scale);
  cc_w_time_ += MPI_Wtime() - twa1;
}

void DeflationPreconditioner::setEssentialTrueDofs(const mfem::Array<int>& ess_tdofs)
{
  ess_tdofs_ = ess_tdofs;
  // Force a clean rebuild: essential-dof masking changes the active tdof set,
  // so the per-component MGS must be redone over the new active set rather
  // than patched in place.
  W_local_.clear();
  W_mat_.reset();
  factored_ = false;
  leftmost_valid_ = false;
}

void DeflationPreconditioner::applyEssentialDofMask()
{
  for (auto& col : W_local_) {
    for (int k = 0; k < ess_tdofs_.Size(); ++k) {
      col(ess_tdofs_[k]) = 0.0;
    }
  }
}

void DeflationPreconditioner::applyW(const mfem::Vector& alpha_local, mfem::Vector& y, double scale) const
{
  if (!W_mat_) throw std::runtime_error("DeflationPreconditioner::applyW: basis not built");
  if (y.Size() != fes_->GetTrueVSize()) y.SetSize(fes_->GetTrueVSize());
  W_mat_->AddMult(alpha_local, y, scale);
}

void DeflationPreconditioner::applyWtranspose(const mfem::Vector& r, mfem::Vector& alpha_local) const
{
  if (!W_mat_) throw std::runtime_error("DeflationPreconditioner::applyWtranspose: basis not built");
  alpha_local.SetSize(modes_per_rank_);
  W_mat_->MultTranspose(r, alpha_local);
}

void DeflationPreconditioner::solveCoarse(const mfem::Vector& c_local, mfem::Vector& alpha_global) const
{
  if (!factored_) throw std::runtime_error("DeflationPreconditioner::solveCoarse: SetOperator not called");
  alpha_global.SetSize(m_);
  mfem::Vector rhs_global(m_);
  MPI_Allgather(c_local.GetData(), modes_per_rank_, MPI_DOUBLE, rhs_global.GetData(), modes_per_rank_, MPI_DOUBLE,
                fes_->GetComm());
  if (WtAW_uses_cholesky_) {
    skylineSolve(rhs_global, alpha_global);
  } else {
    spectralCoarseSolve(rhs_global, alpha_global);
  }
}

void DeflationPreconditioner::computeCoarseSpectralData()
{
  mfem::DenseMatrix sym(WtAW_);
  symmetrize(sym);
  mfem::DenseMatrixEigensystem eig(sym);
  eig.Eval();
  const mfem::Vector& evals = eig.Eigenvalues();
  const mfem::DenseMatrix& evecs = eig.Eigenvectors();

  double max_abs_eval = 0.0;
  for (int i = 0; i < evals.Size(); ++i) max_abs_eval = std::max(max_abs_eval, std::abs(evals(i)));

  const double rank_tol = envDoubleOrDefault("SMITH_DEFLATION_RANK_TOL", 1.0e-5);
  coarse_spectral_tol_ = rank_tol * max_abs_eval;
  coarse_spectral_rank_ = 0;
  coarse_is_spd_ = true;
  for (int i = 0; i < evals.Size(); ++i) {
    if (evals(i) > 0.0 || evals(i) < -coarse_spectral_tol_) {
      if (evals(i) < -coarse_spectral_tol_) coarse_is_spd_ = false;
      ++coarse_spectral_rank_;
    }
  }

  coarse_evals_kept_.SetSize(coarse_spectral_rank_);
  coarse_evecs_kept_.SetSize(m_, coarse_spectral_rank_);
  int kept = 0;
  for (int j = 0; j < evals.Size(); ++j) {
    if (evals(j) <= 0.0 && evals(j) >= -coarse_spectral_tol_) continue;
    coarse_evals_kept_(kept) = evals(j);
    for (int i = 0; i < m_; ++i) coarse_evecs_kept_(i, kept) = evecs(i, j);
    ++kept;
  }
}

void DeflationPreconditioner::lanczosLeftmost(int k, std::vector<double>& evals,
                                              std::vector<mfem::Vector>& evecs) const
{
  if (WtAW_uses_cholesky_) {
    // SPD: shift-invert about 0 — the largest eigenvalues of A^{-1} (applied via the skyline
    // factor) are the smallest of A; eigenvectors coincide.
    auto apply = [this](const double* x, double* y) {
      mfem::Vector xv(const_cast<double*>(x), m_);
      mfem::Vector yv(y, m_);
      skylineSolve(xv, yv);
    };
    std::vector<double> mu;
    const int conv = lanczosExtremeEigenpairs(m_, k, apply, /*largest=*/true, mu, evecs);
    evals.resize(static_cast<size_t>(conv));
    for (int i = 0; i < conv; ++i) evals[static_cast<size_t>(i)] = 1.0 / mu[static_cast<size_t>(i)];
  } else {
    // Indefinite: plain Lanczos on sym(WtAW) for the algebraically smallest eigenpairs.
    auto apply = [this](const double* x, double* y) {
      for (int i = 0; i < m_; ++i) {
        double acc = 0.0;
        for (int j = 0; j < m_; ++j) acc += 0.5 * (WtAW_(i, j) + WtAW_(j, i)) * x[j];
        y[i] = acc;
      }
    };
    lanczosExtremeEigenpairs(m_, k, apply, /*largest=*/false, evals, evecs);
  }
}

void DeflationPreconditioner::spectralCoarseSolve(const mfem::Vector& rhs, mfem::Vector& alpha) const
{
  if (coarse_sparse_eig_mode_) {
    throw std::runtime_error(
        "DeflationPreconditioner: rank-revealing spectral coarse solve requires the dense "
        "eigendecomposition (m <= SMITH_DEFLATION_DENSE_EIG_MAX)");
  }
  alpha.SetSize(m_);
  alpha = 0.0;
  if (coarse_spectral_rank_ == 0) return;

  cc_spectral_coeff_.SetSize(coarse_spectral_rank_);
  mfem::Vector& coeff = cc_spectral_coeff_;
  coarse_evecs_kept_.MultTranspose(rhs, coeff);
  for (int i = 0; i < coarse_spectral_rank_; ++i) coeff(i) /= coarse_evals_kept_(i);
  coarse_evecs_kept_.Mult(coeff, alpha);
}

void DeflationPreconditioner::ensureLeftmostComputed() const
{
  if (leftmost_valid_) return;
  if (!factored_) throw std::runtime_error("DeflationPreconditioner::leftmost: SetOperator not called");
  if (coarse_sparse_eig_mode_) {
    std::vector<double> evals;
    std::vector<mfem::Vector> evecs;
    lanczosLeftmost(1, evals, evecs);
    if (!evals.empty()) {
      leftmost_eval_ = evals[0];
      leftmost_evec_ = evecs[0];
    } else {
      leftmost_eval_ = 0.0;
      leftmost_evec_.SetSize(m_);
      leftmost_evec_ = 0.0;
    }
    leftmost_valid_ = true;
    return;
  }
  if (coarse_spectral_rank_ == 0) {
    leftmost_eval_ = 0.0;
    leftmost_evec_.SetSize(m_);
    leftmost_evec_ = 0.0;
    leftmost_valid_ = true;
    return;
  }
  int imin = 0;
  for (int i = 1; i < coarse_spectral_rank_; ++i) {
    if (coarse_evals_kept_(i) < coarse_evals_kept_(imin)) imin = i;
  }
  leftmost_eval_ = coarse_evals_kept_(imin);
  leftmost_evec_.SetSize(m_);
  for (int i = 0; i < m_; ++i) leftmost_evec_(i) = coarse_evecs_kept_(i, imin);
  leftmost_valid_ = true;
}

double DeflationPreconditioner::coarseLeftmostEigenvalue() const
{
  ensureLeftmostComputed();
  return leftmost_eval_;
}

void DeflationPreconditioner::coarseLeftmostDirection(mfem::Vector& d) const
{
  ensureLeftmostComputed();
  d.SetSize(fes_->GetTrueVSize());
  d = 0.0;
  mfem::Vector v_local(leftmost_evec_.GetData() + my_col_offset_, modes_per_rank_);
  W_mat_->AddMult(v_local, d, 1.0);
}

void DeflationPreconditioner::coarseSmallestDirections(int k, std::vector<mfem::Vector>& dirs,
                                                       std::vector<double>& evals_out) const
{
  dirs.clear();
  evals_out.clear();
  if (!factored_ || k <= 0) return;
  if (coarse_sparse_eig_mode_) {
    std::vector<mfem::Vector> evecs;
    lanczosLeftmost(k, evals_out, evecs);
    const int n = fes_->GetTrueVSize();
    dirs.reserve(evecs.size());
    for (const mfem::Vector& v : evecs) {
      mfem::Vector d(n);
      d = 0.0;
      mfem::Vector v_local(const_cast<double*>(v.GetData()) + my_col_offset_, modes_per_rank_);
      W_mat_->AddMult(v_local, d, 1.0);
      dirs.emplace_back(std::move(d));
    }
    return;
  }
  std::vector<int> order(static_cast<size_t>(coarse_spectral_rank_));
  for (int i = 0; i < coarse_spectral_rank_; ++i) order[static_cast<size_t>(i)] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return coarse_evals_kept_(a) < coarse_evals_kept_(b); });
  const int n = fes_->GetTrueVSize();
  const int kk = std::min(k, coarse_spectral_rank_);
  dirs.reserve(static_cast<size_t>(kk));
  evals_out.reserve(static_cast<size_t>(kk));
  for (int q = 0; q < kk; ++q) {
    const int idx = order[static_cast<size_t>(q)];
    evals_out.push_back(coarse_evals_kept_(idx));
    mfem::Vector d(n);
    d = 0.0;
    mfem::Vector v_local(modes_per_rank_);
    for (int i = 0; i < modes_per_rank_; ++i) v_local(i) = coarse_evecs_kept_(my_col_offset_ + i, idx);
    W_mat_->AddMult(v_local, d, 1.0);
    dirs.emplace_back(std::move(d));
  }
}

void DeflationPreconditioner::verifyWtAWAssembly() const
{
  if (!A_ || !W_mat_) return;

  MPI_Comm comm = fes_->GetComm();
  const int n = fes_->GetTrueVSize();

  mfem::DenseMatrix WtAW_ref(m_, m_);
  WtAW_ref = 0.0;

  mfem::Vector x(n);
  mfem::Vector ax(n);
  mfem::Vector local_block_col(modes_per_rank_);

  for (int owner = 0; owner < n_ranks_; ++owner) {
    const int owner_offset = owner * modes_per_rank_;
    for (int j = 0; j < modes_per_rank_; ++j) {
      x.SetSize(n);
      if (my_rank_ == owner) {
        for (int i = 0; i < n; ++i) x(i) = W_dense_(i, j);
      } else {
        x = 0.0;
      }

      A_->Mult(x, ax);
      W_mat_->MultTranspose(ax, local_block_col);

      for (int i = 0; i < modes_per_rank_; ++i) {
        WtAW_ref(my_col_offset_ + i, owner_offset + j) = local_block_col(i);
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, WtAW_ref.Data(), m_ * m_, MPI_DOUBLE, MPI_SUM, comm);

  double max_abs = 0.0;
  double max_rel = 0.0;
  int max_i = 0;
  int max_j = 0;
  double max_fast = 0.0;
  double max_ref = 0.0;
  double max_sym_fast = 0.0;
  double max_sym_ref = 0.0;

  mfem::DenseMatrix block_max(n_ranks_, n_ranks_);
  block_max = 0.0;

  for (int j = 0; j < m_; ++j) {
    for (int i = 0; i < m_; ++i) {
      const double fast = WtAW_(i, j);
      const double ref = WtAW_ref(i, j);
      const double diff = std::abs(fast - ref);
      const double denom = std::max({1.0, std::abs(fast), std::abs(ref)});
      const double rel = diff / denom;
      block_max(i / modes_per_rank_, j / modes_per_rank_) =
          std::max(block_max(i / modes_per_rank_, j / modes_per_rank_), diff);
      if (diff > max_abs) {
        max_abs = diff;
        max_rel = rel;
        max_i = i;
        max_j = j;
        max_fast = fast;
        max_ref = ref;
      }
      max_sym_fast = std::max(max_sym_fast, std::abs(WtAW_(i, j) - WtAW_(j, i)));
      max_sym_ref = std::max(max_sym_ref, std::abs(WtAW_ref(i, j) - WtAW_ref(j, i)));
    }
  }

  mfem::DenseMatrix sym(WtAW_);
  for (int j = 0; j < m_; ++j) {
    for (int i = 0; i < j; ++i) {
      const double s = 0.5 * (sym(i, j) + sym(j, i));
      sym(i, j) = sym(j, i) = s;
    }
  }
  mfem::DenseMatrixEigensystem eig(sym);
  eig.Eval();
  const mfem::Vector& evals = eig.Eigenvalues();
  double eval_min = std::numeric_limits<double>::infinity();
  double eval_max = -std::numeric_limits<double>::infinity();
  int n_negative = 0;
  for (int i = 0; i < evals.Size(); ++i) {
    eval_min = std::min(eval_min, evals(i));
    eval_max = std::max(eval_max, evals(i));
    if (evals(i) < 0.0) ++n_negative;
  }

  std::vector<double> diag_block_min(static_cast<size_t>(n_ranks_), 0.0);
  std::vector<double> diag_block_max(static_cast<size_t>(n_ranks_), 0.0);
  for (int b = 0; b < n_ranks_; ++b) {
    mfem::DenseMatrix block(modes_per_rank_, modes_per_rank_);
    const int off = b * modes_per_rank_;
    for (int j = 0; j < modes_per_rank_; ++j) {
      for (int i = 0; i < modes_per_rank_; ++i) {
        block(i, j) = 0.5 * (WtAW_(off + i, off + j) + WtAW_(off + j, off + i));
      }
    }
    mfem::DenseMatrixEigensystem block_eig(block);
    block_eig.Eval();
    const mfem::Vector& block_evals = block_eig.Eigenvalues();
    double bmin = std::numeric_limits<double>::infinity();
    double bmax = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < block_evals.Size(); ++i) {
      bmin = std::min(bmin, block_evals(i));
      bmax = std::max(bmax, block_evals(i));
    }
    diag_block_min[static_cast<size_t>(b)] = bmin;
    diag_block_max[static_cast<size_t>(b)] = bmax;
  }

  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  if (rank != 0) return;

  const int row_rank = max_i / modes_per_rank_;
  const int col_rank = max_j / modes_per_rank_;
  const int row_mode = max_i % modes_per_rank_;
  const int col_mode = max_j % modes_per_rank_;

  mfem::out << "\n========= Deflation WtAW verifier =========\n";
  mfem::out << "  max |fast-ref| = " << std::setprecision(17) << max_abs << ", rel = " << max_rel << "\n";
  mfem::out << "  at (" << max_i << ", " << max_j << ") block (" << row_rank << ", " << col_rank << ") mode ("
            << row_mode << ", " << col_mode << ")\n";
  mfem::out << "  fast = " << max_fast << ", ref = " << max_ref << "\n";
  mfem::out << "  symmetry max |fast-fast^T| = " << max_sym_fast << "\n";
  mfem::out << "  symmetry max |ref-ref^T| = " << max_sym_ref << "\n";
  mfem::out << "  sym(WtAW) eigen min/max = " << eval_min << " / " << eval_max << ", negative count = " << n_negative
            << "\n";
  mfem::out << "  diagonal-block eigen min/max:\n";
  for (int b = 0; b < n_ranks_; ++b) {
    mfem::out << "    block " << b << ": " << diag_block_min[static_cast<size_t>(b)] << " / "
              << diag_block_max[static_cast<size_t>(b)] << "\n";
  }
  mfem::out << "  block max |fast-ref|:\n";
  for (int bi = 0; bi < n_ranks_; ++bi) {
    mfem::out << "    ";
    for (int bj = 0; bj < n_ranks_; ++bj) {
      mfem::out << std::setw(12) << std::setprecision(4) << std::scientific << block_max(bi, bj);
    }
    mfem::out << "\n";
  }
  mfem::out << "===========================================\n\n";
}

void DeflationPreconditioner::Mult(const mfem::Vector& r, mfem::Vector& z) const
{
  // === TIMING BEGIN ===
  double t0 = MPI_Wtime();
  // === TIMING END ===
  const int n = fes_->GetTrueVSize();
  z.SetSize(n);
  const bool have_smoother = smoother_ || (smoother_variant_ == DeflationSmoother::BlockJacobi && !block_jacobi_inv_.empty()) ||
                             (smoother_variant_ == DeflationSmoother::PointJacobi && !point_diag_.empty());
  const bool do_smoother = use_smoother_ && have_smoother;

  // Hot path: Additive coarse correction with the rhs Allgather overlapped with the (local)
  // smoother apply. Arithmetic is identical to the generic path below — the compact W kernels
  // and the smoother produce bitwise-identical results; only the comm is nonblocking.
  if (coarse_is_spd_ && coarse_mode_ == CoarseMode::Additive && factored_) {
    ++cc_calls_;
    double tw0 = MPI_Wtime();
    wTransposeCompact(r, cc_rhs_local_);
    double tw1 = MPI_Wtime();
    cc_wt_time_ += tw1 - tw0;

    cc_rhs_global_.SetSize(m_);
    MPI_Request gather_req;
    MPI_Iallgather(cc_rhs_local_.GetData(), modes_per_rank_, MPI_DOUBLE, cc_rhs_global_.GetData(), modes_per_rank_,
                   MPI_DOUBLE, fes_->GetComm(), &gather_req);

    if (do_smoother) {
      applySmoother(r, z);
    } else {
      z = 0.0;
    }
    double ts = MPI_Wtime();
    mult_smoother_time_ += ts - tw1;

    MPI_Wait(&gather_req, MPI_STATUS_IGNORE);
    double ta = MPI_Wtime();
    cc_allgather_time_ += ta - ts;

    if (WtAW_uses_cholesky_) {
      skylineSolve(cc_rhs_global_, cc_alpha_);
    } else {
      spectralCoarseSolve(cc_rhs_global_, cc_alpha_);
    }
    double tsv = MPI_Wtime();
    cc_solve_time_ += tsv - ta;

    wAddMultCompact(cc_alpha_.GetData() + my_col_offset_, z, 1.0);
    double tend = MPI_Wtime();
    cc_w_time_ += tend - tsv;
    mult_coarse_time_ += (tw1 - tw0) + (tend - ts);
    mult_total_time_ += tend - t0;
    ++mult_calls_;
    return;
  }

  if (do_smoother) {
    applySmoother(r, z);
  } else {
    z = 0.0;
  }
  // Smoother-only fallback when the retained coarse op is materially indefinite. Tiny signed
  // eigenvalues below the rank-revealing tolerance were dropped in the spectral solve.
  if (!coarse_is_spd_) {
    mult_total_time_ += MPI_Wtime() - t0;
    mult_smoother_time_ += MPI_Wtime() - t0;
    ++mult_calls_;
    return;
  }
  // === TIMING BEGIN ===
  double t1 = MPI_Wtime();
  mult_smoother_time_ += t1 - t0;
  // === TIMING END ===

  if (coarse_mode_ == CoarseMode::Multiplicative) {
    // Symmetric multiplicative V-cycle. Overwrite the additive `z = smoother(r)` from above:
    //   z = Π r                       (coarse pre)
    //   z += M^{-1} (r - A z)         (smoother in the middle)
    //   z += Π (r - A z)              (coarse post)
    if (mult_tmp_.Size() != n) mult_tmp_.SetSize(n);
    z = 0.0;
    addCoarseCorrection(r, z);  // z = Π r
    op_for_mult_->Mult(z, mult_tmp_);
    mfem::Vector r_mid(n);
    for (int i = 0; i < n; ++i) r_mid(i) = r(i) - mult_tmp_(i);
    mfem::Vector s_mid(n);
    applySmoother(r_mid, s_mid);
    for (int i = 0; i < n; ++i) z(i) += s_mid(i);
    op_for_mult_->Mult(z, mult_tmp_);
    mfem::Vector r_post(n);
    for (int i = 0; i < n; ++i) r_post(i) = r(i) - mult_tmp_(i);
    addCoarseCorrection(r_post, z);
  } else {
    addCoarseCorrection(r, z);
  }

  // === TIMING BEGIN ===
  double t2 = MPI_Wtime();
  mult_coarse_time_ += t2 - t1;
  mult_total_time_ += t2 - t0;
  ++mult_calls_;
  // === TIMING END ===
}

void DeflationPreconditioner::printTimingSummary(MPI_Comm comm) const
{
  int rank = 0, nproc = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);

  auto rmax = [comm](double v) {
    double out = 0.0;
    MPI_Reduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    return out;
  };
  auto isum = [comm](long long v) {
    long long out = 0;
    MPI_Reduce(&v, &out, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    return out;
  };

  double mt_total = rmax(mult_total_time_);
  double mt_smoo = rmax(mult_smoother_time_);
  double mt_coarse = rmax(mult_coarse_time_);
  double so_mv = rmax(setop_matvec_time_);
  double so_fac = rmax(setop_factor_time_);
  double so_smo = rmax(setop_smoother_time_);
  double a_halo = rmax(assemble_timings_.halo);
  double a_diag = rmax(assemble_timings_.diag);
  double a_offd = rmax(assemble_timings_.offd);
  double a_ar = rmax(assemble_timings_.allreduce);
  double c_wt = rmax(cc_wt_time_);
  double c_ag = rmax(cc_allgather_time_);
  double c_sv = rmax(cc_solve_time_);
  double c_w = rmax(cc_w_time_);
  long long n_mult = isum(static_cast<long long>(mult_calls_));
  long long n_cc = isum(static_cast<long long>(cc_calls_));
  long long n_so = isum(static_cast<long long>(setop_calls_));
  long long n_wtaw_chol_fail = isum(static_cast<long long>(wtaw_cholesky_failures_));
  long long n_wtaw_pp_chol_fail = isum(static_cast<long long>(wtaw_pp_cholesky_failures_));
  long long n_wtaw_reg = isum(static_cast<long long>(wtaw_regularized_));
  long long n_wtaw_null = isum(wtaw_null_filtered_);

  if (rank != 0) return;
  auto pf = [](const char* label, double t) { mfem::out << "    " << label << ": " << t << " s\n"; };
  mfem::out << "\n========= DeflationPreconditioner timing (max across " << nproc << " ranks) =========\n";
  mfem::out << "  SetOperator (" << n_so / std::max<long long>(nproc, 1) << " calls/rank)\n";
  pf("assembleWtAW (total)", so_mv);
  pf("  halo (pack+isend/irecv+wait)", a_halo);
  pf("  diag SpMV+gemm", a_diag);
  pf("  offd SpMV+gemm", a_offd);
  pf("  Allreduce(m*m)", a_ar);
  pf("factor (Cholesky/LU)", so_fac);
  mfem::out << "    WtAW Cholesky failures: " << n_wtaw_chol_fail / std::max<long long>(nproc, 1) << " / rank\n";
  mfem::out << "    WtAW_pp Cholesky failures: " << n_wtaw_pp_chol_fail / std::max<long long>(nproc, 1) << " / rank\n";
  mfem::out << "    WtAW A-null cols filtered: " << n_wtaw_null / std::max<long long>(nproc, 1) << " / rank\n";
  (void)n_wtaw_reg;
  pf("smoother setup", so_smo);
  mfem::out << "  Mult (" << n_mult / std::max<long long>(nproc, 1) << " calls/rank, "
            << n_cc / std::max<long long>(nproc, 1) << " coarse-corrections/rank)\n";
  pf("total", mt_total);
  pf("  smoother", mt_smoo);
  pf("  coarse", mt_coarse);
  pf("    W^T * r", c_wt);
  pf("    Allgather / nbr-exchange", c_ag);
  pf("    factor-solve", c_sv);
  pf("    W * alpha", c_w);
  mfem::out << "===================================================================\n\n";
}

}  // namespace smith
