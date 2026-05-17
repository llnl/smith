// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file deflation.hpp
 *
 * @brief Custom deflation preconditioner built from per-rank affine basis vectors.
 *
 * See deflation.md for the mathematical writeup. For a vector H1 displacement field
 * partitioned across MPI ranks, W has columns spanning per-rank affine deformations
 * (constant + linear-in-x/y/z, per component). 3D: 12 columns/rank. Columns have
 * disjoint supports across ranks so W is block-diagonal and W^T A W is small/dense.
 *
 * Mult applies the additive two-level preconditioner M^{-1} r = M_J^{-1} r + W (W^T A W)^{-1} W^T r,
 * where M_J is a HypreSmoother (Jacobi by default).
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"

namespace smith {

class DeflationPreconditioner : public mfem::Solver {
 public:
  /**
   * @param fes vector ParFiniteElementSpace (vdim == mesh dim).
   * @param use_smoother  Enable the second-level smoother.
   * @param smoother_type Hypre smoother type for the second-level smoother.
   */
  explicit DeflationPreconditioner(mfem::ParFiniteElementSpace& fes,
                                   bool use_smoother = true,
                                   mfem::HypreSmoother::Type smoother_type = mfem::HypreSmoother::Jacobi);

  /// factor W^T A W; also forwards to the inner smoother. A must be HypreParMatrix.
  void SetOperator(const mfem::Operator& op) override;

  /// z0 = W (W^T A W)^{-1} (-W^T r)
  void coarseSolve(const mfem::Vector& r, mfem::Vector& z0) const;

  /// y += W (W^T A W)^{-1} W^T r   (adds the coarse correction, does NOT negate)
  void addCoarseCorrection(const mfem::Vector& r, mfem::Vector& y) const;

  /// Additive two-level Mult:  z = Jacobi(r) + W (W^T A W)^{-1} W^T r
  void Mult(const mfem::Vector& r, mfem::Vector& z) const override;

  int numLocalColumns() const { return static_cast<int>(W_local_.size()); }
  int numGlobalColumns() const { return m_; }
  const std::vector<mfem::Vector>& localColumns() const { return W_local_; }
  const mfem::DenseMatrix& coarseMatrix() const { return WtAW_; }

  // === TIMING BEGIN === (remove block when no longer needed)
  double setopMatvecTime() const { return setop_matvec_time_; }
  double setopFactorTime() const { return setop_factor_time_; }
  double setopSmootherTime() const { return setop_smoother_time_; }
  double multTotalTime() const { return mult_total_time_; }
  double multCoarseTime() const { return mult_coarse_time_; }
  double multSmootherTime() const { return mult_smoother_time_; }
  int multCalls() const { return mult_calls_; }
  void resetTimers() const
  {
    setop_matvec_time_ = setop_factor_time_ = setop_smoother_time_ = 0.0;
    mult_total_time_ = mult_coarse_time_ = mult_smoother_time_ = 0.0;
    mult_calls_ = 0;
  }
  // === TIMING END ===

  /// zero out the rows of W on the given essential true dofs.
  /// Necessary when A came from FormLinearSystem (identity rows on essential dofs);
  /// without it W^T A W includes constrained dofs in the coarse space.
  void setEssentialTrueDofs(const mfem::Array<int>& ess_tdofs);

 private:
  void buildBasis();
  void applyEssentialDofMask();
  void packWMatrix();
  void addScaledCoarseCorrection(const mfem::Vector& r, mfem::Vector& y, double scale) const;

  mfem::ParFiniteElementSpace& fes_;
  int dim_ = 0;
  int modes_per_rank_ = 0;
  int m_ = 0;
  int my_rank_ = 0;
  int n_ranks_ = 1;
  int my_col_offset_ = 0;

  std::vector<mfem::Vector> W_local_;
  // packed N_local x modes_per_rank sparse matrix, columns = W_local_[i]. Per-row nnz is 4 (3D)
  // or 3 (2D) since each col is nonzero only on its component's tdofs.
  std::unique_ptr<mfem::SparseMatrix> W_mat_;

  mutable mfem::DenseMatrix WtAW_;
  mutable mfem::DenseMatrixInverse WtAW_inv_;
  mutable bool factored_ = false;

  const mfem::HypreParMatrix* A_ = nullptr;

  std::unique_ptr<mfem::HypreSmoother> smoother_;
  mfem::HypreSmoother::Type smoother_type_;
  bool use_smoother_ = true;
  mfem::Array<int> ess_tdofs_;

  // === TIMING BEGIN ===
  mutable double setop_matvec_time_ = 0.0;
  mutable double setop_factor_time_ = 0.0;
  mutable double setop_smoother_time_ = 0.0;
  mutable double mult_total_time_ = 0.0;
  mutable double mult_coarse_time_ = 0.0;
  mutable double mult_smoother_time_ = 0.0;
  mutable int mult_calls_ = 0;
  // === TIMING END ===
};

}  // namespace smith
