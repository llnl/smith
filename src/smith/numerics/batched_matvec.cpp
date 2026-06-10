// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/batched_matvec.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "_hypre_parcsr_mv.h"

namespace smith {

namespace {

bool envFlagEnabled(const char* name)
{
  const char* value = std::getenv(name);
  if (!value) return false;
  return value[0] != '\0' && value[0] != '0';
}

void batchedMatvecLoop(const mfem::HypreParMatrix& A, const mfem::DenseMatrix& X, mfem::DenseMatrix& Y)
{
  const int n = X.Height();
  const int k = X.Width();
  Y.SetSize(n, k);
  mfem::Vector xj(n), yj(n);
  for (int j = 0; j < k; ++j) {
    // Copy into a parallel-vector-shaped temporary; HypreParMatrix::Mult assumes
    // the same local layout on every rank.
    for (int i = 0; i < n; ++i) xj(i) = X(i, j);
    const_cast<mfem::HypreParMatrix&>(A).Mult(xj, yj);
    for (int i = 0; i < n; ++i) Y(i, j) = yj(i);
  }
}

// Single packed halo exchange + local SpMVs. Sends/receives k values per halo entry
// in one MPI message per neighbor (vs k messages in the Loop strategy).
//
// Layout: send_buf to neighbor s is row-major (per-element, k contiguous), recv_buf
// is row-major (per offd col, k contiguous). Local SpMV inner loop iterates k.
void batchedMatvecPackedDense(const mfem::HypreParMatrix& A, const mfem::DenseMatrix& X, mfem::DenseMatrix& Y)
{
  hypre_ParCSRMatrix* parA = static_cast<hypre_ParCSRMatrix*>(const_cast<mfem::HypreParMatrix&>(A));
  hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag(parA);
  hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd(parA);

  hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(parA);
  if (!comm_pkg) {
    hypre_MatvecCommPkgCreate(parA);
    comm_pkg = hypre_ParCSRMatrixCommPkg(parA);
  }
  MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg);

  const int n = X.Height();
  const int k = X.Width();
  Y.SetSize(n, k);
  Y = 0.0;

  const int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  const int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  const int num_cols_offd = hypre_CSRMatrixNumCols(offd);

  std::vector<double> recv_buf(static_cast<size_t>(num_cols_offd) * static_cast<size_t>(k), 0.0);
  std::vector<MPI_Request> reqs;
  reqs.reserve(static_cast<size_t>(num_sends + num_recvs));

  // Post receives — one per neighbor, length = len_r * k.
  for (int r = 0; r < num_recvs; ++r) {
    int peer = hypre_ParCSRCommPkgRecvProc(comm_pkg, r);
    int start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, r);
    int end = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, r + 1);
    int len = end - start;
    MPI_Request req;
    MPI_Irecv(recv_buf.data() + static_cast<size_t>(start) * static_cast<size_t>(k), len * k, MPI_DOUBLE, peer, 17,
              comm, &req);
    reqs.push_back(req);
  }

  // Pack and send — one buffer per neighbor, row-major (per element, k cols contiguous).
  std::vector<std::vector<double>> send_bufs(static_cast<size_t>(num_sends));
  for (int s = 0; s < num_sends; ++s) {
    int peer = hypre_ParCSRCommPkgSendProc(comm_pkg, s);
    int start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, s);
    int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, s + 1);
    int len = end - start;
    send_bufs[static_cast<size_t>(s)].resize(static_cast<size_t>(len) * static_cast<size_t>(k));
    double* sb = send_bufs[static_cast<size_t>(s)].data();
    for (int i = 0; i < len; ++i) {
      int local_idx = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, start + i);
      for (int j = 0; j < k; ++j) {
        sb[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)] = X(local_idx, j);
      }
    }
    MPI_Request req;
    MPI_Isend(sb, len * k, MPI_DOUBLE, peer, 17, comm, &req);
    reqs.push_back(req);
  }

  // Local diag-block SpMV overlap with comm: Y += A_diag * X.
  HYPRE_Int* diag_i = hypre_CSRMatrixI(diag);
  HYPRE_Int* diag_j = hypre_CSRMatrixJ(diag);
  HYPRE_Real* diag_a = hypre_CSRMatrixData(diag);
  for (int row = 0; row < n; ++row) {
    for (HYPRE_Int idx = diag_i[row]; idx < diag_i[row + 1]; ++idx) {
      const int col = static_cast<int>(diag_j[idx]);
      const double val = diag_a[idx];
      for (int j = 0; j < k; ++j) {
        Y(row, j) += val * X(col, j);
      }
    }
  }

  if (!reqs.empty()) {
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  }

  // Off-diag block SpMV using packed halo data: Y += A_offd * recv_buf.
  HYPRE_Int* offd_i = hypre_CSRMatrixI(offd);
  HYPRE_Int* offd_j = hypre_CSRMatrixJ(offd);
  HYPRE_Real* offd_a = hypre_CSRMatrixData(offd);
  for (int row = 0; row < n; ++row) {
    for (HYPRE_Int idx = offd_i[row]; idx < offd_i[row + 1]; ++idx) {
      const int col_offd = static_cast<int>(offd_j[idx]);
      const double val = offd_a[idx];
      const double* rb = recv_buf.data() + static_cast<size_t>(col_offd) * static_cast<size_t>(k);
      for (int j = 0; j < k; ++j) {
        Y(row, j) += val * rb[j];
      }
    }
  }
}

}  // namespace

void assembleWtAW(const mfem::HypreParMatrix& A, const mfem::DenseMatrix& W_local, int modes_per_rank,
                  mfem::DenseMatrix& WtAW, AssembleWtAWTimings* timings)
{
  hypre_ParCSRMatrix* parA = static_cast<hypre_ParCSRMatrix*>(const_cast<mfem::HypreParMatrix&>(A));
  hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag(parA);
  hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd(parA);

  hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(parA);
  if (!comm_pkg) {
    hypre_MatvecCommPkgCreate(parA);
    comm_pkg = hypre_ParCSRMatrixCommPkg(parA);
  }
  MPI_Comm comm = hypre_ParCSRCommPkgComm(comm_pkg);

  int my_rank = 0, nproc = 1;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nproc);

  const int n = W_local.Height();
  const int mpr = modes_per_rank;
  const int m = mpr * nproc;
  const int my_offset = my_rank * mpr;

  if (W_local.Width() != mpr) {
    throw std::runtime_error("assembleWtAW: W_local.Width() != modes_per_rank");
  }
  if (n != A.Height()) {
    throw std::runtime_error("assembleWtAW: W_local.Height() != A.Height()");
  }

  WtAW.SetSize(m, m);
  WtAW = 0.0;

  const int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  const int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  const int num_cols_offd = hypre_CSRMatrixNumCols(offd);

  if (envFlagEnabled("SMITH_DEFLATION_VERIFY")) {
    std::vector<int> recv_procs(static_cast<size_t>(num_recvs));
    for (int r = 0; r < num_recvs; ++r) recv_procs[static_cast<size_t>(r)] = hypre_ParCSRCommPkgRecvProc(comm_pkg, r);
    std::vector<int> sorted_recv_procs = recv_procs;
    std::sort(sorted_recv_procs.begin(), sorted_recv_procs.end());
    const bool has_duplicate_recv =
        std::adjacent_find(sorted_recv_procs.begin(), sorted_recv_procs.end()) != sorted_recv_procs.end();

    std::ostringstream os;
    os << "[rank " << my_rank << "] assembleWtAW comm: sends=" << num_sends << ", recvs=" << num_recvs
       << ", offd_cols=" << num_cols_offd << ", duplicate_recv_proc=" << has_duplicate_recv << "\n";
    for (int r = 0; r < num_recvs; ++r) {
      os << "  recv[" << r << "] peer=" << hypre_ParCSRCommPkgRecvProc(comm_pkg, r) << " range=["
         << hypre_ParCSRCommPkgRecvVecStart(comm_pkg, r) << ", " << hypre_ParCSRCommPkgRecvVecStart(comm_pkg, r + 1)
         << ")\n";
    }
    mfem::out << os.str();
  }

  // Halo exchange of W_local (one packed message per neighbor, mpr values per halo dof).
  double t_halo_start = MPI_Wtime();
  std::vector<double> recv_buf(static_cast<size_t>(num_cols_offd) * static_cast<size_t>(mpr), 0.0);
  std::vector<MPI_Request> reqs;
  reqs.reserve(static_cast<size_t>(num_sends + num_recvs));

  for (int r = 0; r < num_recvs; ++r) {
    int peer = hypre_ParCSRCommPkgRecvProc(comm_pkg, r);
    int start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, r);
    int end = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, r + 1);
    int len = end - start;
    MPI_Request req;
    MPI_Irecv(recv_buf.data() + static_cast<size_t>(start) * static_cast<size_t>(mpr), len * mpr, MPI_DOUBLE, peer, 23,
              comm, &req);
    reqs.push_back(req);
  }

  std::vector<std::vector<double>> send_bufs(static_cast<size_t>(num_sends));
  for (int s = 0; s < num_sends; ++s) {
    int peer = hypre_ParCSRCommPkgSendProc(comm_pkg, s);
    int start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, s);
    int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, s + 1);
    int len = end - start;
    send_bufs[static_cast<size_t>(s)].resize(static_cast<size_t>(len) * static_cast<size_t>(mpr));
    double* sb = send_bufs[static_cast<size_t>(s)].data();
    for (int i = 0; i < len; ++i) {
      int local_idx = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, start + i);
      for (int j = 0; j < mpr; ++j) {
        sb[static_cast<size_t>(i) * static_cast<size_t>(mpr) + static_cast<size_t>(j)] = W_local(local_idx, j);
      }
    }
    MPI_Request req;
    MPI_Isend(sb, len * mpr, MPI_DOUBLE, peer, 23, comm, &req);
    reqs.push_back(req);
  }

  // Diagonal block (p, p) = W_p^T · A_diag · W_p — overlap with comm.
  double t_diag_start = MPI_Wtime();
  HYPRE_Int* diag_i = hypre_CSRMatrixI(diag);
  HYPRE_Int* diag_j = hypre_CSRMatrixJ(diag);
  HYPRE_Real* diag_a = hypre_CSRMatrixData(diag);

  mfem::DenseMatrix AW_diag(n, mpr);
  AW_diag = 0.0;
  for (int row = 0; row < n; ++row) {
    for (HYPRE_Int idx = diag_i[row]; idx < diag_i[row + 1]; ++idx) {
      const int col = static_cast<int>(diag_j[idx]);
      const double val = diag_a[idx];
      for (int j = 0; j < mpr; ++j) {
        AW_diag(row, j) += val * W_local(col, j);
      }
    }
  }
  mfem::DenseMatrix block_pp(mpr, mpr);
  mfem::MultAtB(W_local, AW_diag, block_pp);
  for (int j = 0; j < mpr; ++j) {
    for (int i = 0; i < mpr; ++i) {
      WtAW(my_offset + i, my_offset + j) = block_pp(i, j);
    }
  }

  double t_diag_end = MPI_Wtime();

  // Wait for halo data before touching A_offd.
  if (!reqs.empty()) {
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  }
  double t_halo_end = MPI_Wtime();

  // Off-diag blocks (p, s) for each halo-neighbor s = recv_procs[r].
  double t_offd_start = MPI_Wtime();
  // Per-neighbor AW accumulators: AW_per[r] = A_offd|_p · W_s_halo (restricted to s's halo cols).
  HYPRE_Int* offd_i = hypre_CSRMatrixI(offd);
  HYPRE_Int* offd_j = hypre_CSRMatrixJ(offd);
  HYPRE_Real* offd_a = hypre_CSRMatrixData(offd);

  // owner_of_offd_col[c] = recv-index (into recv_procs) of the rank that owns offd col c.
  std::vector<int> owner_of_offd_col(static_cast<size_t>(num_cols_offd), -1);
  for (int r = 0; r < num_recvs; ++r) {
    int start = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, r);
    int end = hypre_ParCSRCommPkgRecvVecStart(comm_pkg, r + 1);
    for (int c = start; c < end; ++c) owner_of_offd_col[static_cast<size_t>(c)] = r;
  }

  std::vector<mfem::DenseMatrix> AW_per(static_cast<size_t>(num_recvs));
  for (int r = 0; r < num_recvs; ++r) {
    AW_per[static_cast<size_t>(r)].SetSize(n, mpr);
    AW_per[static_cast<size_t>(r)] = 0.0;
  }
  for (int row = 0; row < n; ++row) {
    for (HYPRE_Int idx = offd_i[row]; idx < offd_i[row + 1]; ++idx) {
      const int col = static_cast<int>(offd_j[idx]);
      const int r = owner_of_offd_col[static_cast<size_t>(col)];
      if (r < 0) {
        throw std::runtime_error("assembleWtAW: offd column was not covered by the Hypre recv map");
      }
      const double val = offd_a[idx];
      const double* rb = recv_buf.data() + static_cast<size_t>(col) * static_cast<size_t>(mpr);
      mfem::DenseMatrix& AWr = AW_per[static_cast<size_t>(r)];
      for (int j = 0; j < mpr; ++j) {
        AWr(row, j) += val * rb[j];
      }
    }
  }

  mfem::DenseMatrix block_ps(mpr, mpr);
  for (int r = 0; r < num_recvs; ++r) {
    int peer = hypre_ParCSRCommPkgRecvProc(comm_pkg, r);
    int peer_offset = peer * mpr;
    mfem::MultAtB(W_local, AW_per[static_cast<size_t>(r)], block_ps);
    for (int j = 0; j < mpr; ++j) {
      for (int i = 0; i < mpr; ++i) {
        WtAW(my_offset + i, peer_offset + j) = block_ps(i, j);
      }
    }
  }

  double t_offd_end = MPI_Wtime();

  double t_ar_start = MPI_Wtime();
  MPI_Allreduce(MPI_IN_PLACE, WtAW.Data(), m * m, MPI_DOUBLE, MPI_SUM, comm);
  double t_ar_end = MPI_Wtime();

  if (timings) {
    // Halo time = pack + irecv/isend + the waitall. The diag SpMV overlaps with comm so
    // we attribute only the wait portion to halo. Charge diag separately.
    timings->halo += (t_halo_end - t_halo_start) - (t_diag_end - t_diag_start);
    timings->diag += t_diag_end - t_diag_start;
    timings->offd += t_offd_end - t_offd_start;
    timings->allreduce += t_ar_end - t_ar_start;
  }
}

void batchedMatvec(const mfem::HypreParMatrix& A, const mfem::DenseMatrix& X, mfem::DenseMatrix& Y,
                   BatchedMatvecStrategy strategy)
{
  if (X.Height() != A.Height()) {
    throw std::runtime_error("batchedMatvec: X.Height() != A.Height()");
  }
  switch (strategy) {
    case BatchedMatvecStrategy::Loop:
      batchedMatvecLoop(A, X, Y);
      return;
    case BatchedMatvecStrategy::PackedDense:
      batchedMatvecPackedDense(A, X, Y);
      return;
  }
}

}  // namespace smith
