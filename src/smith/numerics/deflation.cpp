// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/deflation.hpp"

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

}  // namespace

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
  modes_per_rank_ = (dim_ + 1) * dim_;  // 12 in 3D, 6 in 2D
  MPI_Comm comm = fes_->GetComm();
  MPI_Comm_rank(comm, &my_rank_);
  MPI_Comm_size(comm, &n_ranks_);
  my_col_offset_ = my_rank_ * modes_per_rank_;
  m_ = n_ranks_ * modes_per_rank_;
  // Force rebuild on next SetOperator.
  W_local_.clear();
  W_mat_.reset();
  factored_ = false;
  leftmost_valid_ = false;
}

void DeflationPreconditioner::buildBasis()
{
  if (!W_local_.empty()) return;

  const int dim = dim_;
  W_local_.clear();
  W_local_.reserve(static_cast<size_t>(modes_per_rank_));

  for (int c = 0; c < dim_; ++c) {
    for (int mode = 0; mode < dim_ + 1; ++mode) {
      const int m = mode;
      mfem::VectorFunctionCoefficient coef(dim_, [c, m, dim](const mfem::Vector& p, mfem::Vector& v) {
        v.SetSize(dim);
        v = 0.0;
        v(c) = (m == 0) ? 1.0 : p(m - 1);
      });
      mfem::ParGridFunction gf(fes_);
      gf.ProjectCoefficient(coef);
      mfem::Vector col;
      gf.GetTrueDofs(col);
      W_local_.emplace_back(std::move(col));
    }
  }
  applyEssentialDofMask();
  centerLinearModes();
  packWMatrix();
}

void DeflationPreconditioner::centerLinearModes()
{
  // For each component c, subtract per-rank mean of the linear-in-X_k mode from itself, using
  // the constant mode (= 1 on c's tdofs) as the "indicator" for the active tdof set. This
  // keeps span(constant, linear) unchanged while improving the conditioning of W^T A W when
  // the mesh is far from the origin or the local subdomain bounding box is large.
  for (int c = 0; c < dim_; ++c) {
    const int const_idx = c * (dim_ + 1);
    mfem::Vector& const_col = W_local_[static_cast<size_t>(const_idx)];
    double n_active = const_col.Sum();
    if (n_active <= 0.0) continue;  // no owned/active dofs for this component on this rank
    for (int k = 1; k <= dim_; ++k) {
      mfem::Vector& lin_col = W_local_[static_cast<size_t>(const_idx + k)];
      double mean = lin_col.Sum() / n_active;
      // lin_col -= mean * const_col
      lin_col.Add(-mean, const_col);
    }
  }
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
}

void DeflationPreconditioner::SetOperator(const mfem::Operator& op)
{
  if (!fes_) throw std::runtime_error("DeflationPreconditioner::SetOperator: attachFES not called");

  const mfem::HypreParMatrix* hyp = dynamic_cast<const mfem::HypreParMatrix*>(&op);
  if (!hyp) {
    if (const auto* bsr_op = dynamic_cast<const BSROperator*>(&op)) {
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
  // === TIMING END ===
  // W^T A W via the block-sparse triple product: one halo exchange of W (m/p cols per rank),
  // local diag DGEMM, per-neighbor offd SpMV, single MPI_Allreduce of the m × m result.
  // See deflation.md Phase 5b for the math.
  assembleWtAW(*A_, W_dense_, modes_per_rank_, WtAW_);
  // === TIMING BEGIN ===
  double t1 = MPI_Wtime();
  setop_matvec_time_ += t1 - t0;
  // === TIMING END ===

  WtAW_uses_cholesky_ = factorCholesky(WtAW_, WtAW_cholesky_, WtAW_cholesky_factors_);
  if (!WtAW_uses_cholesky_) {
    WtAW_lu_inv_.SetOperator(WtAW_);
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
  WtAW_pp_uses_cholesky_ = factorCholesky(WtAW_pp_, WtAW_pp_cholesky_, WtAW_pp_cholesky_factors_);
  if (!WtAW_pp_uses_cholesky_) {
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

  smoother_ = std::make_unique<mfem::HypreSmoother>();
  smoother_->SetType(smoother_type_);
  smoother_->SetOperator(const_cast<mfem::HypreParMatrix&>(*A_));
  // === TIMING BEGIN ===
  setop_smoother_time_ += MPI_Wtime() - t2;
  // === TIMING END ===
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

  mfem::Vector rhs_local(modes_per_rank_);
  W_mat_->MultTranspose(r, rhs_local);

  if (coarse_mode_ == CoarseMode::AdditiveLocal) {
    // Skip the Allgather — solve only against the diagonal block of WtAW (per-rank only).
    mfem::Vector alpha_local(modes_per_rank_);
    choleskySolve(WtAW_pp_cholesky_factors_, WtAW_pp_lu_inv_, WtAW_pp_uses_cholesky_, modes_per_rank_, rhs_local,
                  alpha_local);
    W_mat_->AddMult(alpha_local, y, scale);
    return;
  }

  if (coarse_mode_ == CoarseMode::AdditiveSchwarz) {
    // K=1 multi-step block-Jacobi-Schwarz. Replaces Allgather with one neighbor-only round.
    //   u = WtAW_pp^{-1} c
    //   exchange u with neighbors → u_s
    //   alpha = u - WtAW_pp^{-1} Σ_s WtAW_{p,s} u_s
    mfem::Vector u_local(modes_per_rank_);
    choleskySolve(WtAW_pp_cholesky_factors_, WtAW_pp_lu_inv_, WtAW_pp_uses_cholesky_, modes_per_rank_, rhs_local,
                  u_local);

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
    for (int n = 0; n < n_nbr; ++n) {
      MPI_Request req;
      MPI_Isend(u_local.GetData(), modes_per_rank_, MPI_DOUBLE, schwarz_neighbors_[static_cast<size_t>(n)], 31, comm,
                &req);
      reqs.push_back(req);
    }
    if (!reqs.empty()) {
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }

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
    W_mat_->AddMult(alpha_local, y, scale);
    return;
  }

  mfem::Vector rhs_global(m_);
  MPI_Allgather(rhs_local.GetData(), modes_per_rank_, MPI_DOUBLE, rhs_global.GetData(), modes_per_rank_, MPI_DOUBLE,
                fes_->GetComm());

  mfem::Vector alpha(m_);
  choleskySolve(WtAW_cholesky_factors_, WtAW_lu_inv_, WtAW_uses_cholesky_, m_, rhs_global, alpha);

  mfem::Vector alpha_local(alpha.GetData() + my_col_offset_, modes_per_rank_);
  W_mat_->AddMult(alpha_local, y, scale);
}

void DeflationPreconditioner::setEssentialTrueDofs(const mfem::Array<int>& ess_tdofs)
{
  ess_tdofs_ = ess_tdofs;
  applyEssentialDofMask();
  if (!W_local_.empty()) {
    centerLinearModes();
    if (W_mat_) packWMatrix();
  }
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
  choleskySolve(WtAW_cholesky_factors_, WtAW_lu_inv_, WtAW_uses_cholesky_, m_, rhs_global, alpha_global);
}

void DeflationPreconditioner::ensureLeftmostComputed() const
{
  if (leftmost_valid_) return;
  if (!factored_) throw std::runtime_error("DeflationPreconditioner::leftmost: SetOperator not called");
  // Symmetric eigen-decomposition of the dense m×m W^T A W. m = 12·P (3D) ⇒ tiny for P ≤ few k.
  // Replicated on every rank; result is consistent without further communication.
  mfem::DenseMatrix sym(WtAW_);
  for (int j = 0; j < m_; ++j) {
    for (int i = 0; i < j; ++i) {
      double s = 0.5 * (sym(i, j) + sym(j, i));
      sym(i, j) = sym(j, i) = s;
    }
  }
  mfem::DenseMatrixEigensystem eig(sym);
  eig.Eval();
  const mfem::Vector& evals = eig.Eigenvalues();
  const mfem::DenseMatrix& evecs = eig.Eigenvectors();
  int imin = 0;
  for (int i = 1; i < m_; ++i) {
    if (evals(i) < evals(imin)) imin = i;
  }
  leftmost_eval_ = evals(imin);
  leftmost_evec_.SetSize(m_);
  for (int i = 0; i < m_; ++i) leftmost_evec_(i) = evecs(i, imin);
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

void DeflationPreconditioner::Mult(const mfem::Vector& r, mfem::Vector& z) const
{
  // === TIMING BEGIN ===
  double t0 = MPI_Wtime();
  // === TIMING END ===
  const int n = fes_->GetTrueVSize();
  z.SetSize(n);
  if (use_smoother_ && smoother_) {
    smoother_->Mult(r, z);
  } else {
    z = 0.0;
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
    smoother_->Mult(r_mid, s_mid);
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

}  // namespace smith
