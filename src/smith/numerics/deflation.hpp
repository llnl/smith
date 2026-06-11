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
#include "mpi.h"

#include "smith/numerics/batched_matvec.hpp"

namespace smith {

class DeflationIndefiniteCoarseException : public std::runtime_error {
 public:
  DeflationIndefiniteCoarseException(double eigenvalue, const mfem::Vector& direction);

  double eigenvalue() const { return eigenvalue_; }
  const mfem::Vector& direction() const { return direction_; }

 private:
  double eigenvalue_ = 0.0;
  mfem::Vector direction_;
};

/// Polynomial order of the per-rank deflation basis.
enum class DeflationOrder
{
  /// Constant + linear per component. mpr = dim*(dim+1)  (6 in 2D, 12 in 3D).
  Affine,
  /// Constant + linear + quadratic per component, all centered on the rank
  /// element-volume centroid and per-component Gram–Schmidt orthonormalized.
  /// mpr = dim*(1 + dim + dim*(dim+1)/2)  (12 in 2D, 30 in 3D).
  Quadratic,
};

/// How the coarse correction is applied inside `Mult`.
enum class CoarseMode
{
  /// z = M_J^{-1} r + W (WtAW)^{-1} W^T r   (default — global Galerkin coarse solve via Allgather)
  Additive,
  /// z = M_J^{-1} r + W (WtAW_pp)^{-1} W^T r
  /// Uses only the diagonal block of WtAW (= W_p^T A_diag|_p W_p). No MPI comm in the coarse step.
  AdditiveLocal,
  /// K=1 multi-step block-Jacobi-Schwarz approximation of (WtAW)^{-1}. One neighbor-only
  /// exchange of mpr doubles per neighbor (replaces the Allgather). Symmetric by construction.
  ///   u = WtAW_pp^{-1} c_local
  ///   exchange u with halo neighbors → u_s
  ///   alpha = u - WtAW_pp^{-1} Σ_s WtAW_{p,s} u_s
  AdditiveSchwarz,
  /// Symmetric multiplicative V-cycle:
  ///   z = Π r;                              # coarse pre
  ///   z += M_J^{-1} (r - A z);              # smoother
  ///   z += Π (r - A z);                     # coarse post   (with Π = W (WtAW)^{-1} W^T)
  /// Costs 2 extra parallel A·z matvecs per Mult vs Additive. Symmetric by construction.
  Multiplicative,
};

class BSROperator;

/// Which first-level smoother `Mult` applies.
enum class DeflationSmoother
{
  /// mfem::HypreSmoother (scalar Jacobi by default). Needs a hypre matrix handle.
  Hypre,
  /// z_i = r_i / a_ii with the diagonal read from the BSR blocks (hypre fallback).
  /// Same arithmetic as Hypre Jacobi (weight 1) — exists so the apply path works
  /// without a hypre matrix once assembly goes direct-to-BSR.
  PointJacobi,
  /// Exact vdim x vdim diagonal-block inverses (block-Jacobi).
  BlockJacobi,
};

class DeflationPreconditioner : public mfem::Solver {
 public:
  /**
   * @param fes vector ParFiniteElementSpace (vdim == mesh dim).
   * @param use_smoother  Enable the second-level smoother.
   * @param smoother_type Hypre smoother type for the second-level smoother.
   */
  explicit DeflationPreconditioner(mfem::ParFiniteElementSpace& fes, bool use_smoother = true,
                                   mfem::HypreSmoother::Type smoother_type = mfem::HypreSmoother::Jacobi);

  /// Deferred-FES constructor. The FES must be supplied via `attachFES(...)` before the first
  /// `SetOperator`. Lets framework code (SolidMechanics) construct the preconditioner before
  /// it knows the displacement FES.
  explicit DeflationPreconditioner(bool use_smoother = true,
                                   mfem::HypreSmoother::Type smoother_type = mfem::HypreSmoother::Jacobi);

  /// Bind the FES post-construction. Required when the deferred-FES constructor was used.
  /// Subsequent `SetOperator` calls build the basis against this FES.
  void attachFES(mfem::ParFiniteElementSpace& fes);

  /// Select the first-level smoother variant. Takes effect at the next SetOperator.
  void setSmootherVariant(DeflationSmoother variant) { smoother_variant_ = variant; }

  bool hasFES() const { return fes_ != nullptr; }

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
  int myColumnOffset() const { return my_col_offset_; }
  const std::vector<mfem::Vector>& localColumns() const { return W_local_; }
  const mfem::DenseMatrix& coarseMatrix() const { return WtAW_; }

  // -------------------- Hooks for trust-region integration --------------------

  /// y_local += scale * W * alpha_local, where alpha_local is this rank's mpr-slice.
  /// Only this rank's owned tdofs receive nonzero contributions.
  void applyW(const mfem::Vector& alpha_local, mfem::Vector& y, double scale = 1.0) const;

  /// alpha_local = W^T * r  (size mpr, this rank's owned slice).
  void applyWtranspose(const mfem::Vector& r, mfem::Vector& alpha_local) const;

  /// Given mpr-size local rhs `c_local = W^T r |_p`, solve W^T A W * alpha = c_global and
  /// return the full global m-vector `alpha`. Allgather + rank-revealing spectral solve.
  void solveCoarse(const mfem::Vector& c_local, mfem::Vector& alpha_global) const;

  /// True iff the retained coarse spectrum is positive. Tiny signed eigenvalues below the
  /// rank-revealing tolerance are dropped and do not make the preconditioner indefinite.
  bool coarseIsSPD() const { return coarse_is_spd_; }

  int coarseSpectralRank() const { return coarse_spectral_rank_; }
  double coarseSpectralTolerance() const { return coarse_spectral_tol_; }

  /// Smallest eigenvalue of W^T A W. Computed lazily on first call after SetOperator.
  double coarseLeftmostEigenvalue() const;

  /// Builds a coarse negative-curvature direction d = W * v where v is the unit eigenvector
  /// of W^T A W associated with the smallest eigenvalue. Output `d` is sized to the local
  /// tdof count and contains only this rank's W-slice contribution.
  void coarseLeftmostDirection(mfem::Vector& d) const;

  /// Fill `dirs` with up to `k` directions d_i = W * v_i, where v_i are the unit
  /// eigenvectors of sym(W^T A W) associated with the `k` smallest eigenvalues
  /// (ascending). `evals` receives the matching eigenvalues. Sized to local tdofs.
  void coarseSmallestDirections(int k, std::vector<mfem::Vector>& dirs, std::vector<double>& evals) const;

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
    assemble_timings_ = AssembleWtAWTimings{};
    cc_wt_time_ = cc_allgather_time_ = cc_solve_time_ = cc_w_time_ = 0.0;
    cc_calls_ = 0;
    setop_calls_ = 0;
    wtaw_cholesky_failures_ = wtaw_pp_cholesky_failures_ = 0;
    wtaw_regularized_ = 0;
  }
  void printTimingSummary(MPI_Comm comm) const;
  // === TIMING END ===

  /// zero out the rows of W on the given essential true dofs.
  /// Necessary when A came from FormLinearSystem (identity rows on essential dofs);
  /// without it W^T A W includes constrained dofs in the coarse space.
  void setEssentialTrueDofs(const mfem::Array<int>& ess_tdofs);

  /// Switch between coarse-correction modes. Can be called any time after SetOperator.
  void setCoarseMode(CoarseMode m) { coarse_mode_ = m; }
  CoarseMode coarseMode() const { return coarse_mode_; }

  /// Polynomial order of the deflation basis. Must be called before the first
  /// `SetOperator`; later calls force a basis rebuild.
  void setDeflationOrder(DeflationOrder order);
  DeflationOrder deflationOrder() const { return order_; }

 private:
  void buildBasis();
  void applyEssentialDofMask();
  void packWMatrix();
  void addScaledCoarseCorrection(const mfem::Vector& r, mfem::Vector& y, double scale) const;
  /// rhs = W^T r via the compact layout (bitwise == W_mat_->MultTranspose).
  void wTransposeCompact(const mfem::Vector& r, mfem::Vector& rhs) const;
  /// y += scale * W * alpha_local via the compact layout (bitwise == W_mat_->AddMult).
  void wAddMultCompact(const double* alpha_local, mfem::Vector& y, double scale) const;
  void computeCoarseSpectralData();
  void spectralCoarseSolve(const mfem::Vector& rhs, mfem::Vector& alpha) const;
  void ensureLeftmostComputed() const;
  void verifyWtAWAssembly() const;

  mfem::ParFiniteElementSpace* fes_ = nullptr;
  int dim_ = 0;
  int modes_per_rank_ = 0;
  DeflationOrder order_ = DeflationOrder::Affine;
  /// Element-volume centroid of this rank's mesh; used to center linear and
  /// quadratic monomials for conditioning of W^T A W.
  mfem::Vector rank_centroid_;
  int m_ = 0;
  int my_rank_ = 0;
  int n_ranks_ = 1;
  int my_col_offset_ = 0;

  std::vector<mfem::Vector> W_local_;
  // packed N_local x modes_per_rank sparse matrix, columns = W_local_[i]. Per-row nnz is 4 (3D)
  // or 3 (2D) since each col is nonzero only on its component's tdofs. Used in the per-iter
  // coarse correction (W_mat_ MultTranspose + AddMult).
  std::unique_ptr<mfem::SparseMatrix> W_mat_;
  // Compact per-row view of the same matrix for the hot Additive coarse apply: row i can only
  // touch its component's contiguous mode block, so w_compact_[i*per_comp + k] holds those
  // per_comp = modes_per_rank_/dim_ weights and w_row_comp_[i] the block index. Kernels using
  // it accumulate in the same row-ascending order as the CSR Mult/MultTranspose (structural
  // zeros contribute exact zeros), so results are bitwise-identical to W_mat_.
  std::vector<double> w_compact_;
  std::vector<int> w_row_comp_;
  // Dense packed view of the same columns. Used by `assembleWtAW` in SetOperator.
  mfem::DenseMatrix W_dense_;

  mutable mfem::DenseMatrix WtAW_;
  mutable mfem::DenseMatrix WtAW_cholesky_;
  mutable mfem::CholeskyFactors WtAW_cholesky_factors_;
  mutable mfem::DenseMatrixInverse WtAW_lu_inv_;
  mutable bool WtAW_uses_cholesky_ = false;
  mutable mfem::DenseMatrix coarse_evecs_kept_;
  mutable mfem::Vector coarse_evals_kept_;
  mutable int coarse_spectral_rank_ = 0;
  mutable double coarse_spectral_tol_ = 0.0;
  mutable bool coarse_is_spd_ = true;
  mutable mfem::DenseMatrix WtAW_pp_;  // diagonal block (p, p) — for AdditiveLocal mode.
  mutable mfem::DenseMatrix WtAW_pp_cholesky_;
  mutable mfem::CholeskyFactors WtAW_pp_cholesky_factors_;
  mutable mfem::DenseMatrixInverse WtAW_pp_lu_inv_;
  mutable bool WtAW_pp_uses_cholesky_ = false;
  mutable bool factored_ = false;
  // Leftmost eigenpair cache (recomputed on each SetOperator via invalidation).
  mutable mfem::Vector leftmost_evec_;  // length m_
  mutable double leftmost_eval_ = 0.0;
  mutable bool leftmost_valid_ = false;
  CoarseMode coarse_mode_ = CoarseMode::Additive;
  mutable mfem::Vector mult_tmp_;  // buffer for A·z in Multiplicative mode.

  // Hot-path scratch for the per-CG-iteration coarse apply (avoids per-call allocation).
  mutable mfem::Vector cc_rhs_local_;
  mutable mfem::Vector cc_rhs_global_;
  mutable mfem::Vector cc_alpha_;
  mutable mfem::Vector cc_spectral_coeff_;

  // AdditiveSchwarz: cached neighbor list + off-diag WtAW blocks (mpr × mpr each).
  mutable std::vector<int> schwarz_neighbors_;
  /// Ranks that need our coarse coefficients (hypre send-procs). May differ from
  /// `schwarz_neighbors_` (recv-procs) when the halo graph is asymmetric.
  mutable std::vector<int> schwarz_send_neighbors_;
  mutable std::vector<mfem::DenseMatrix> schwarz_neighbor_blocks_;

  /// Build the inverted vdim x vdim diagonal blocks for the block-Jacobi smoother.
  void buildBlockJacobi();

  /// Extract the scalar matrix diagonal for the PointJacobi smoother. Reads the BSR
  /// diagonal blocks when available (the direct-BSR-assembly path), hypre otherwise.
  void buildPointJacobi(const BSROperator* bsr_op);

  /// Apply the configured smoother (block-Jacobi or HypreSmoother): z = M^{-1} r.
  void applySmoother(const mfem::Vector& r, mfem::Vector& z) const;

  const mfem::HypreParMatrix* A_ = nullptr;
  const mfem::Operator* op_for_mult_ = nullptr;

  std::unique_ptr<mfem::HypreSmoother> smoother_;
  mfem::HypreSmoother::Type smoother_type_;
  bool use_smoother_ = true;
  /// First-level smoother variant. For BlockJacobi: BC-eliminated dofs (zeroed row/col,
  /// 1 on the diagonal) decouple inside their block, so the block inverse handles them
  /// exactly. PointJacobi reproduces Hypre Jacobi without needing a hypre handle.
  DeflationSmoother smoother_variant_ = DeflationSmoother::Hypre;
  /// Inverted diagonal blocks, row-major b x b per block (size = (n/b) * b * b).
  mutable std::vector<double> block_jacobi_inv_;
  /// Scalar matrix diagonal for the PointJacobi variant.
  mutable std::vector<double> point_diag_;
  mfem::Array<int> ess_tdofs_;

  // === TIMING BEGIN ===
  mutable double setop_matvec_time_ = 0.0;
  mutable double setop_factor_time_ = 0.0;
  mutable double setop_smoother_time_ = 0.0;
  mutable double mult_total_time_ = 0.0;
  mutable double mult_coarse_time_ = 0.0;
  mutable double mult_smoother_time_ = 0.0;
  mutable int mult_calls_ = 0;
  mutable int setop_calls_ = 0;
  mutable AssembleWtAWTimings assemble_timings_{};
  mutable double cc_wt_time_ = 0.0;
  mutable double cc_allgather_time_ = 0.0;
  mutable double cc_solve_time_ = 0.0;
  mutable double cc_w_time_ = 0.0;
  mutable int cc_calls_ = 0;
  mutable int wtaw_cholesky_failures_ = 0;
  mutable int wtaw_pp_cholesky_failures_ = 0;
  mutable int wtaw_regularized_ = 0;
  mutable long long wtaw_null_filtered_ = 0;
  // === TIMING END ===
};

}  // namespace smith
