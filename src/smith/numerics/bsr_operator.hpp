#pragma once

#include "_hypre_parcsr_mv.h"
#include "mfem.hpp"
#include <vector>

namespace smith {

/// BSR Matrix structure for a local CSR block
struct BSRMatrix {
  int nb_rows = 0;
  int b = 0;
  std::vector<int> I;
  std::vector<int> J;
  std::vector<double> data;
};

/**
 * @brief Multi-RHS BSR SpMM: Y += alpha * A * X for k right-hand sides in one matrix sweep.
 *
 * X and Y are row-major with k contiguous values per scalar dof (X row i = k values of
 * dof i), matching the packed-halo layouts used by the deflation/batched-matvec paths.
 */
void bsrSpMMAdd(const BSRMatrix& A, const double* X, double* Y, int k, double alpha = 1.0);

/**
 * @brief Operator wrapper that intercepts a HypreParMatrix, converts its local
 * diagonal and off-diagonal blocks to a Block Sparse Row (BSR) format, and
 * performs an optimized SpMV during Mult.
 *
 * This targets 2x2/3x3 blocking (2D/3D elasticity byVDIM) to alleviate memory
 * bandwidth limits in the standard CSR SpMV kernel. Unsupported layouts fall
 * back to the wrapped HypreParMatrix.
 */
class BSROperator : public mfem::Operator {
 public:
  /**
   * @brief Construct a new BSROperator
   *
   * @param A The original HypreParMatrix to wrap.
   * @param block_size The block size (e.g., 3 for 3D elasticity).
   */
  BSROperator(mfem::HypreParMatrix* A, int block_size = 3);

  ~BSROperator() override;

  /// Perform the matrix-vector multiplication y = A * x using the BSR representation
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

  /// Perform the matrix-vector multiplication y += A * x using the BSR representation
  void AddMult(const mfem::Vector& x, mfem::Vector& y, const double a = 1.0) const override;

  /// y = A^T x. Supported when the operator is declared symmetric (y = A x); otherwise falls
  /// back to the wrapped hypre matrix, which is only valid when its values are current.
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

  /// Declare A = A^T so MultTranspose can use the BSR Mult (no hypre dependency).
  void SetSymmetric(bool symmetric) { symmetric_ = symmetric; }
  /// @overload
  bool IsSymmetric() const { return symmetric_; }

  /**
   * @brief Multi-RHS multiply: ys[j] = A * xs[j] for all j with a single packed halo
   * exchange and one sweep over the matrix (each loaded block is applied to every RHS).
   * Falls back to per-vector hypre Mult when the BSR layout checks failed.
   */
  void MultBatch(const std::vector<const mfem::Vector*>& xs, const std::vector<mfem::Vector*>& ys) const;

  /// Retrieve the underlying HypreParMatrix
  mfem::HypreParMatrix* GetHypreMatrix() const { return A_; }

  /// True when the matrix satisfied the current prototype's block-layout checks.
  bool Enabled() const { return enabled_; }

  /// Block size of the BSR representation
  int BlockSize() const { return block_size_; }

  /// Local diagonal block of the BSR representation (valid when Enabled())
  const BSRMatrix& DiagBSR() const { return diag_bsr_; }

  /// Local off-diagonal (halo-coupled) block of the BSR representation (valid when Enabled())
  const BSRMatrix& OffdBSR() const { return offd_bsr_; }

  /// Mutable value access for in-place refresh by the direct-BSR assembly path
  std::vector<double>& MutableDiagData() { return diag_bsr_.data; }
  /// @overload
  std::vector<double>& MutableOffdData() { return offd_bsr_.data; }

 private:
  /// Convert a HYPRE CSR matrix to the internal BSR representation
  static BSRMatrix convertCSRToBSR(hypre_CSRMatrix* csr, int b);

  template <int b>
  static void bsrSpMVAdd(const BSRMatrix& A, const double* x, double* y, double alpha);

  static void bsrSpMVAdd(const BSRMatrix& A, const double* x, double* y, double alpha);

  mfem::HypreParMatrix* A_ = nullptr;
  int block_size_ = 3;
  bool enabled_ = false;
  bool symmetric_ = false;

  BSRMatrix diag_bsr_;
  BSRMatrix offd_bsr_;

  // Cached comm pkg parameters for the HYPRE matrix
  hypre_ParCSRCommPkg* comm_pkg_ = nullptr;
  int num_sends_ = 0;

  // Buffers for MPI halo exchange
  mutable std::vector<double> send_buf_;
  mutable std::vector<double> recv_buf_;
};

}  // namespace smith
