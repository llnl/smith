#include "smith/numerics/bsr_operator.hpp"

#include <algorithm>
#include <map>

namespace smith {

BSROperator::BSROperator(mfem::HypreParMatrix* A, int block_size)
    : mfem::Operator(A ? A->Height() : 0, A ? A->Width() : 0), A_(A), block_size_(block_size)
{
  if (!A_ || (block_size_ != 2 && block_size_ != 3)) return;

  auto* hypre_A = static_cast<hypre_ParCSRMatrix*>(*A_);
  hypre_CSRMatrix* A_diag = hypre_ParCSRMatrixDiag(hypre_A);
  hypre_CSRMatrix* A_offd = hypre_ParCSRMatrixOffd(hypre_A);

  const int diag_rows = hypre_CSRMatrixNumRows(A_diag);
  const int diag_cols = hypre_CSRMatrixNumCols(A_diag);
  const int offd_rows = hypre_CSRMatrixNumRows(A_offd);
  const int offd_cols = hypre_CSRMatrixNumCols(A_offd);
  if (diag_rows % block_size_ != 0 || diag_cols % block_size_ != 0 || offd_rows % block_size_ != 0 ||
      offd_cols % block_size_ != 0) {
    return;
  }

  diag_bsr_ = convertCSRToBSR(A_diag, block_size_);
  offd_bsr_ = convertCSRToBSR(A_offd, block_size_);

  comm_pkg_ = hypre_ParCSRMatrixCommPkg(hypre_A);
  if (!comm_pkg_) {
    hypre_MatvecCommPkgCreate(hypre_A);
    comm_pkg_ = hypre_ParCSRMatrixCommPkg(hypre_A);
  }
  if (!comm_pkg_) return;

  num_sends_ = hypre_ParCSRCommPkgNumSends(comm_pkg_);

  const int send_size = hypre_ParCSRCommPkgSendMapStart(comm_pkg_, num_sends_);
  send_buf_.resize(static_cast<size_t>(send_size));
  recv_buf_.resize(static_cast<size_t>(offd_cols));
  enabled_ = true;
}

BSROperator::~BSROperator() = default;

BSRMatrix BSROperator::convertCSRToBSR(hypre_CSRMatrix* csr, int b)
{
  const int n_rows = hypre_CSRMatrixNumRows(csr);
  const int n_cols = hypre_CSRMatrixNumCols(csr);
  int* csr_i = hypre_CSRMatrixI(csr);
  int* csr_j = hypre_CSRMatrixJ(csr);
  double* csr_data = hypre_CSRMatrixData(csr);

  BSRMatrix bsr;
  bsr.nb_rows = n_rows / b;
  bsr.b = b;
  bsr.I.reserve(static_cast<size_t>(bsr.nb_rows) + 1);
  bsr.I.push_back(0);

  std::map<int, int> block_to_index;
  for (int br = 0; br < bsr.nb_rows; ++br) {
    block_to_index.clear();
    for (int bi = 0; bi < b; ++bi) {
      const int row = br * b + bi;
      for (int k = csr_i[row]; k < csr_i[row + 1]; ++k) {
        const int col = csr_j[k];
        if (col >= 0 && col < n_cols) block_to_index.emplace(col / b, -1);
      }
    }

    for (auto& entry : block_to_index) {
      entry.second = static_cast<int>(bsr.J.size());
      bsr.J.push_back(entry.first);
      bsr.data.resize(bsr.data.size() + static_cast<size_t>(b * b), 0.0);
    }
    bsr.I.push_back(static_cast<int>(bsr.J.size()));

    for (int bi = 0; bi < b; ++bi) {
      const int row = br * b + bi;
      for (int k = csr_i[row]; k < csr_i[row + 1]; ++k) {
        const int col = csr_j[k];
        const int bj = col / b;
        const int local_j = col % b;
        const auto it = block_to_index.find(bj);
        if (it != block_to_index.end()) {
          const int block_index = it->second;
          bsr.data[static_cast<size_t>(block_index * b * b + bi * b + local_j)] = csr_data[k];
        }
      }
    }
  }
  return bsr;
}

template <int b>
void BSROperator::bsrSpMVAdd(const BSRMatrix& A, const double* x, double* y, double alpha)
{
  const int* I = A.I.data();
  const int* J = A.J.data();
  const double* data = A.data.data();

  for (int br = 0; br < A.nb_rows; ++br) {
    double accum[b] = {};
    for (int m = I[br]; m < I[br + 1]; ++m) {
      const int bc = J[m];
      const double* block = &data[static_cast<size_t>(m * b * b)];
      const double* x_block = &x[bc * b];
      for (int i = 0; i < b; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < b; ++j) {
          row_sum += block[i * b + j] * x_block[j];
        }
        accum[i] += row_sum;
      }
    }
    for (int i = 0; i < b; ++i) {
      y[br * b + i] += alpha * accum[i];
    }
  }
}

void BSROperator::bsrSpMVAdd(const BSRMatrix& A, const double* x, double* y, double alpha)
{
  if (A.b == 2) {
    bsrSpMVAdd<2>(A, x, y, alpha);
  } else if (A.b == 3) {
    bsrSpMVAdd<3>(A, x, y, alpha);
  }
}

void BSROperator::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
  y = 0.0;
  AddMult(x, y, 1.0);
}

void BSROperator::AddMult(const mfem::Vector& x, mfem::Vector& y, const double a) const
{
  if (!enabled_) {
    if (A_) A_->AddMult(x, y, a);
    return;
  }

  const double* x_data = x.HostRead();
  double* y_data = y.HostReadWrite();

  int index = 0;
  for (int i = 0; i < num_sends_; i++) {
    const int start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_, i);
    const int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg_, i + 1);
    for (int j = start; j < end; j++) {
      send_buf_[static_cast<size_t>(index++)] = x_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_, j)];
    }
  }

  hypre_ParCSRCommHandle* comm_handle = nullptr;
  if (!send_buf_.empty() || !recv_buf_.empty()) {
    comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg_, send_buf_.data(), recv_buf_.data());
  }

  bsrSpMVAdd(diag_bsr_, x_data, y_data, a);

  if (comm_handle) hypre_ParCSRCommHandleDestroy(comm_handle);

  if (offd_bsr_.nb_rows > 0 && !recv_buf_.empty()) {
    bsrSpMVAdd(offd_bsr_, recv_buf_.data(), y_data, a);
  }
}

}  // namespace smith
