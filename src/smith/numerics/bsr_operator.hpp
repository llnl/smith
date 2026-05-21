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

  /// Retrieve the underlying HypreParMatrix
  mfem::HypreParMatrix* GetHypreMatrix() const { return A_; }

  /// True when the matrix satisfied the current prototype's block-layout checks.
  bool Enabled() const { return enabled_; }

private:
  /// Convert a HYPRE CSR matrix to the internal BSR representation
  static BSRMatrix convertCSRToBSR(hypre_CSRMatrix* csr, int b);

  template <int b>
  static void bsrSpMVAdd(const BSRMatrix& A, const double* x, double* y, double alpha);

  static void bsrSpMVAdd(const BSRMatrix& A, const double* x, double* y, double alpha);

  mfem::HypreParMatrix* A_ = nullptr;
  int block_size_ = 3;
  bool enabled_ = false;

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
