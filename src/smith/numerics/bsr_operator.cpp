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

namespace {

template <int b>
void bsrSpMMAddImpl(const BSRMatrix& A, const double* X, double* Y, int k, double alpha)
{
  const int* I = A.I.data();
  const int* J = A.J.data();
  const double* data = A.data.data();

  for (int br = 0; br < A.nb_rows; ++br) {
    double* y_block = &Y[static_cast<size_t>(br * b) * static_cast<size_t>(k)];
    for (int m = I[br]; m < I[br + 1]; ++m) {
      const double* block = &data[static_cast<size_t>(m) * static_cast<size_t>(b * b)];
      const double* x_block = &X[static_cast<size_t>(J[m] * b) * static_cast<size_t>(k)];
      for (int i = 0; i < b; ++i) {
        for (int j = 0; j < b; ++j) {
          const double a_ij = alpha * block[i * b + j];
          const double* x_row = &x_block[j * k];
          double* y_row = &y_block[i * k];
          for (int c = 0; c < k; ++c) {
            y_row[c] += a_ij * x_row[c];
          }
        }
      }
    }
  }
}

}  // namespace

void bsrSpMMAdd(const BSRMatrix& A, const double* X, double* Y, int k, double alpha)
{
  if (A.b == 2) {
    bsrSpMMAddImpl<2>(A, X, Y, k, alpha);
  } else if (A.b == 3) {
    bsrSpMMAddImpl<3>(A, X, Y, k, alpha);
  }
}

void BSROperator::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
  y = 0.0;
  AddMult(x, y, 1.0);
}

void BSROperator::MultTranspose(const mfem::Vector& x, mfem::Vector& y) const
{
  if (symmetric_) {
    Mult(x, y);
    return;
  }
  MFEM_VERIFY(A_, "BSROperator::MultTranspose: no hypre fallback available");
  A_->MultTranspose(x, y);
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

void BSROperator::MultBatch(const std::vector<const mfem::Vector*>& xs, const std::vector<mfem::Vector*>& ys) const
{
  MFEM_VERIFY(xs.size() == ys.size(), "BSROperator::MultBatch input/output size mismatch");
  const int k = static_cast<int>(xs.size());
  if (k == 0) return;

  if (!enabled_) {
    for (int j = 0; j < k; ++j) {
      A_->Mult(*xs[static_cast<size_t>(j)], *ys[static_cast<size_t>(j)]);
    }
    return;
  }

  const int n = Height();
  const size_t ku = static_cast<size_t>(k);

  // Pack the RHS row-major (k contiguous values per dof) for the SpMM kernel.
  std::vector<double> X(static_cast<size_t>(n) * ku);
  for (int j = 0; j < k; ++j) {
    const double* xj = xs[static_cast<size_t>(j)]->HostRead();
    for (int i = 0; i < n; ++i) {
      X[static_cast<size_t>(i) * ku + static_cast<size_t>(j)] = xj[i];
    }
  }

  MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg_);
  const int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg_);

  std::vector<double> recv(recv_buf_.size() * ku, 0.0);
  std::vector<MPI_Request> reqs;
  reqs.reserve(static_cast<size_t>(num_sends_ + num_recvs));

  for (int r = 0; r < num_recvs; ++r) {
    int peer = hypre_ParCSRCommPkgRecvProc(comm_pkg_, r);
    int start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg_, r);
    int len = hypre_ParCSRCommPkgRecvVecStart(comm_pkg_, r + 1) - start;
    MPI_Request req;
    MPI_Irecv(recv.data() + static_cast<size_t>(start) * ku, len * k, MPI_DOUBLE, peer, 29, comm, &req);
    reqs.push_back(req);
  }

  std::vector<std::vector<double>> send_bufs(static_cast<size_t>(num_sends_));
  for (int s = 0; s < num_sends_; ++s) {
    int peer = hypre_ParCSRCommPkgSendProc(comm_pkg_, s);
    int start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_, s);
    int len = hypre_ParCSRCommPkgSendMapStart(comm_pkg_, s + 1) - start;
    auto& sb = send_bufs[static_cast<size_t>(s)];
    sb.resize(static_cast<size_t>(len) * ku);
    for (int i = 0; i < len; ++i) {
      const size_t src = static_cast<size_t>(hypre_ParCSRCommPkgSendMapElmt(comm_pkg_, start + i)) * ku;
      std::copy_n(&X[src], ku, &sb[static_cast<size_t>(i) * ku]);
    }
    MPI_Request req;
    MPI_Isend(sb.data(), len * k, MPI_DOUBLE, peer, 29, comm, &req);
    reqs.push_back(req);
  }

  // Local diag SpMM overlaps with the halo exchange.
  std::vector<double> Y(static_cast<size_t>(n) * ku, 0.0);
  bsrSpMMAdd(diag_bsr_, X.data(), Y.data(), k);

  if (!reqs.empty()) {
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  }

  if (offd_bsr_.nb_rows > 0 && !recv.empty()) {
    bsrSpMMAdd(offd_bsr_, recv.data(), Y.data(), k);
  }

  for (int j = 0; j < k; ++j) {
    double* yj = ys[static_cast<size_t>(j)]->HostWrite();
    for (int i = 0; i < n; ++i) {
      yj[i] = Y[static_cast<size_t>(i) * ku + static_cast<size_t>(j)];
    }
  }
}

}  // namespace smith
