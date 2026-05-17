// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/deflation.hpp"

#include <stdexcept>

#include "mpi.h"

namespace smith {

DeflationPreconditioner::DeflationPreconditioner(mfem::ParFiniteElementSpace& fes,
                                                 bool use_smoother,
                                                 mfem::HypreSmoother::Type smoother_type)
    : mfem::Solver(fes.GetTrueVSize()), fes_(fes), smoother_type_(smoother_type), use_smoother_(use_smoother)
{
  dim_ = fes_.GetVDim();
  if (dim_ < 2 || dim_ > 3) {
    throw std::runtime_error("DeflationPreconditioner: vdim must be 2 or 3");
  }
  modes_per_rank_ = (dim_ + 1) * dim_;  // 12 in 3D, 6 in 2D

  MPI_Comm comm = fes_.GetComm();
  MPI_Comm_rank(comm, &my_rank_);
  MPI_Comm_size(comm, &n_ranks_);
  my_col_offset_ = my_rank_ * modes_per_rank_;
  m_ = n_ranks_ * modes_per_rank_;
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
      mfem::ParGridFunction gf(&fes_);
      gf.ProjectCoefficient(coef);
      mfem::Vector col;
      gf.GetTrueDofs(col);
      W_local_.emplace_back(std::move(col));
    }
  }
  applyEssentialDofMask();
  packWMatrix();
}

void DeflationPreconditioner::packWMatrix()
{
  const int n = fes_.GetTrueVSize();
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
}

void DeflationPreconditioner::SetOperator(const mfem::Operator& op)
{
  const auto* hyp = dynamic_cast<const mfem::HypreParMatrix*>(&op);
  if (!hyp) {
    throw std::runtime_error("DeflationPreconditioner::SetOperator requires HypreParMatrix");
  }
  A_ = hyp;
  height = op.Height();
  width = op.Width();

  if (W_local_.empty()) buildBasis();

  // Parallel WtAW assembly. For each global column q in [0, m):
  //   - rank owning q fills its local slot with W_local_[q_local]; other ranks fill zeros.
  //   - parallel matvec: AWq_full = A * Wq_full (distributed).
  //   - this rank then computes block_pq(i, j) = local_dot(W_local_[i], AWq_full|_p) for all
  //     local i, where p is this rank, j = q - q_owner*modes_per_rank_.
  // After loop, each rank has filled rows [my_offset, my_offset+modes_per_rank_) of WtAW.
  // Allreduce(SUM) sums them all in (other rows are zero on this rank, so SUM == gather).
  const int local_tsize = fes_.GetTrueVSize();
  mfem::Vector Wq(local_tsize), AWq(local_tsize);

  mfem::DenseMatrix local_block(m_, m_);
  local_block = 0.0;

  // === TIMING BEGIN ===
  double t0 = MPI_Wtime();
  // === TIMING END ===
  for (int q = 0; q < m_; ++q) {
    int owner = q / modes_per_rank_;
    int q_in_owner = q % modes_per_rank_;
    Wq = 0.0;
    if (owner == my_rank_) {
      Wq = W_local_[static_cast<size_t>(q_in_owner)];
    }
    A_->Mult(Wq, AWq);
    for (int i = 0; i < modes_per_rank_; ++i) {
      local_block(my_col_offset_ + i, q) = mfem::InnerProduct(W_local_[static_cast<size_t>(i)], AWq);
    }
  }

  WtAW_.SetSize(m_, m_);
  MPI_Allreduce(local_block.Data(), WtAW_.Data(), m_ * m_, MPI_DOUBLE, MPI_SUM, fes_.GetComm());
  // === TIMING BEGIN ===
  double t1 = MPI_Wtime();
  setop_matvec_time_ += t1 - t0;
  // === TIMING END ===

  WtAW_inv_.SetOperator(WtAW_);
  factored_ = true;
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
  z0.SetSize(fes_.GetTrueVSize());
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

  mfem::Vector rhs_global(m_);
  MPI_Allgather(rhs_local.GetData(), modes_per_rank_, MPI_DOUBLE, rhs_global.GetData(), modes_per_rank_, MPI_DOUBLE,
                fes_.GetComm());

  mfem::Vector alpha(m_);
  WtAW_inv_.Mult(rhs_global, alpha);

  mfem::Vector alpha_local(alpha.GetData() + my_col_offset_, modes_per_rank_);
  W_mat_->AddMult(alpha_local, y, scale);
}

void DeflationPreconditioner::setEssentialTrueDofs(const mfem::Array<int>& ess_tdofs)
{
  ess_tdofs_ = ess_tdofs;
  applyEssentialDofMask();
  factored_ = false;
}

void DeflationPreconditioner::applyEssentialDofMask()
{
  for (auto& col : W_local_) {
    for (int k = 0; k < ess_tdofs_.Size(); ++k) {
      col(ess_tdofs_[k]) = 0.0;
    }
  }
  if (W_mat_) packWMatrix();
}

void DeflationPreconditioner::Mult(const mfem::Vector& r, mfem::Vector& z) const
{
  // === TIMING BEGIN ===
  double t0 = MPI_Wtime();
  // === TIMING END ===
  z.SetSize(fes_.GetTrueVSize());
  if (use_smoother_ && smoother_) {
    smoother_->Mult(r, z);
  } else {
    z = 0.0;
  }
  // === TIMING BEGIN ===
  double t1 = MPI_Wtime();
  mult_smoother_time_ += t1 - t0;
  // === TIMING END ===
  addCoarseCorrection(r, z);
  // === TIMING BEGIN ===
  double t2 = MPI_Wtime();
  mult_coarse_time_ += t2 - t1;
  mult_total_time_ += t2 - t0;
  ++mult_calls_;
  // === TIMING END ===
}

}  // namespace smith
