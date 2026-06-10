// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/bsr_direct_assembler.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "_hypre_parcsr_mv.h"

namespace smith {

namespace {

/// position of global column g in the sorted offd column map; -1 when absent
int offdPosition(const std::vector<HYPRE_BigInt>& col_map, HYPRE_BigInt g)
{
  auto it = std::lower_bound(col_map.begin(), col_map.end(), g);
  if (it == col_map.end() || *it != g) return -1;
  return static_cast<int>(it - col_map.begin());
}

/// data index of scalar entry (local row i, block col C, scalar col offset cj) in a BSRMatrix; -1 when absent
long long bsrEntryIndex(const BSRMatrix& M, int i, int C, int cj)
{
  const int b = M.b;
  const int R = i / b;
  const int* begin = M.J.data() + M.I[static_cast<size_t>(R)];
  const int* end = M.J.data() + M.I[static_cast<size_t>(R) + 1];
  const int* it = std::lower_bound(begin, end, C);
  if (it == end || *it != C) return -1;
  const long long m = it - M.J.data();
  return m * b * b + (i % b) * b + cj;
}

}  // namespace

BSRDirectAssembler::BSRDirectAssembler(mfem::ParFiniteElementSpace& fes, const mfem::Array<int>& ess_tdofs,
                                       mfem::HypreParMatrix* A, const std::vector<int>& row_ptr,
                                       const std::vector<int>& col_ind, const std::vector<double>& values)
    : comm_(fes.GetComm())
{
  // own a clone: the caller reassembles/frees its matrix (e.g. the warm-start path), but
  // this structure + comm package must outlive it
  auto* src = static_cast<hypre_ParCSRMatrix*>(*A);
  A_owned_ = std::make_unique<mfem::HypreParMatrix>(hypre_ParCSRMatrixClone(src, 1), true);
  A_ = A_owned_.get();
  b_ = fes.GetVDim();
  bsr_ = std::make_unique<BSROperator>(A_, b_);
  if (!bsr_->Enabled()) {
    throw std::runtime_error("BSRDirectAssembler: BSR layout checks failed for this matrix/block size");
  }
  diag_data_size_ = bsr_->DiagBSR().data.size();
  my_first_tdof_ = fes.GetMyTDofOffset();
  my_tsize_ = fes.GetTrueVSize();

  buildRouting(fes, ess_tdofs, row_ptr, col_ind);
  verify(values);
}

long long BSRDirectAssembler::slotOf(HYPRE_BigInt gI, HYPRE_BigInt gJ) const
{
  const int i = static_cast<int>(gI - my_first_tdof_);
  if (gJ >= my_first_tdof_ && gJ < my_first_tdof_ + my_tsize_) {
    const int j = static_cast<int>(gJ - my_first_tdof_);
    return bsrEntryIndex(bsr_->DiagBSR(), i, j / b_, j % b_);
  }
  const int pos = offdPosition(col_map_offd_, gJ);
  if (pos < 0) return -1;
  const long long idx = bsrEntryIndex(bsr_->OffdBSR(), i, pos / b_, pos % b_);
  return idx < 0 ? -1 : static_cast<long long>(diag_data_size_) + idx;
}

void BSRDirectAssembler::buildRouting(mfem::ParFiniteElementSpace& fes, const mfem::Array<int>& ess_tdofs,
                                      const std::vector<int>& row_ptr, const std::vector<int>& col_ind)
{
  int my_rank = 0, nproc = 1;
  MPI_Comm_rank(comm_, &my_rank);
  MPI_Comm_size(comm_, &nproc);

  auto* parA = static_cast<hypre_ParCSRMatrix*>(*A_);
  hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd(parA);
  const int n_offd = hypre_CSRMatrixNumCols(offd);
  HYPRE_BigInt* col_map = hypre_ParCSRMatrixColMapOffd(parA);
  col_map_offd_.assign(col_map, col_map + n_offd);

  // owned + halo essential-dof flags. Eliminated *neighbor* columns must be dropped too,
  // so exchange ess flags over the matrix halo (same comm pattern as a matvec).
  ess_owned_.assign(static_cast<size_t>(my_tsize_), 0);
  for (int idx = 0; idx < ess_tdofs.Size(); ++idx) {
    ess_owned_[static_cast<size_t>(ess_tdofs[idx])] = 1;
  }
  ess_offd_.assign(static_cast<size_t>(n_offd), 0);
  {
    hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(parA);
    if (!comm_pkg) {
      hypre_MatvecCommPkgCreate(parA);
      comm_pkg = hypre_ParCSRMatrixCommPkg(parA);
    }
    const int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
    const int send_size = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
    std::vector<double> send_flags(static_cast<size_t>(send_size));
    std::vector<double> recv_flags(static_cast<size_t>(n_offd), 0.0);
    for (int t = 0; t < send_size; ++t) {
      send_flags[static_cast<size_t>(t)] =
          static_cast<double>(ess_owned_[static_cast<size_t>(hypre_ParCSRCommPkgSendMapElmt(comm_pkg, t))]);
    }
    hypre_ParCSRCommHandle* handle =
        hypre_ParCSRCommHandleCreate(1, comm_pkg, send_flags.data(), recv_flags.data());
    if (handle) hypre_ParCSRCommHandleDestroy(handle);
    for (int c = 0; c < n_offd; ++c) {
      ess_offd_[static_cast<size_t>(c)] = recv_flags[static_cast<size_t>(c)] != 0.0 ? 1 : 0;
    }
  }

  // global tdof of every local L-dof, and tdof-range owners
  const int n_ldofs = static_cast<int>(row_ptr.size()) - 1;
  std::vector<HYPRE_BigInt> gt(static_cast<size_t>(n_ldofs));
  for (int i = 0; i < n_ldofs; ++i) {
    gt[static_cast<size_t>(i)] = fes.GetGlobalTDofNumber(i);
  }
  std::vector<long long> tdof_starts(static_cast<size_t>(nproc) + 1, 0);
  long long my_start = static_cast<long long>(my_first_tdof_);
  MPI_Allgather(&my_start, 1, MPI_LONG_LONG, tdof_starts.data(), 1, MPI_LONG_LONG, comm_);
  tdof_starts[static_cast<size_t>(nproc)] = static_cast<long long>(fes.GlobalTrueVSize());
  auto owner_of = [&](HYPRE_BigInt g) {
    auto it = std::upper_bound(tdof_starts.begin(), tdof_starts.end(), static_cast<long long>(g));
    return static_cast<int>(it - tdof_starts.begin()) - 1;
  };

  auto ess_of = [&](HYPRE_BigInt g) -> int {
    if (g >= my_first_tdof_ && g < my_first_tdof_ + my_tsize_) {
      return ess_owned_[static_cast<size_t>(g - my_first_tdof_)];
    }
    const int pos = offdPosition(col_map_offd_, g);
    // a column absent from the (eliminated) structure carries no stiffness: treat as dropped
    return pos < 0 ? 1 : ess_offd_[static_cast<size_t>(pos)];
  };

  // route local entries; collect (gI, gJ) pairs destined for other ranks
  const int nnz = row_ptr.back();
  local_dest_.assign(static_cast<size_t>(nnz), -1);
  std::vector<std::vector<long long>> send_pairs(static_cast<size_t>(nproc));
  std::vector<std::vector<int>> send_entries(static_cast<size_t>(nproc));
  for (int i = 0; i < n_ldofs; ++i) {
    const HYPRE_BigInt gI = gt[static_cast<size_t>(i)];
    const int owner = owner_of(gI);
    for (int k = row_ptr[static_cast<size_t>(i)]; k < row_ptr[static_cast<size_t>(i) + 1]; ++k) {
      const HYPRE_BigInt gJ = gt[static_cast<size_t>(col_ind[static_cast<size_t>(k)])];
      if (owner == my_rank) {
        if (ess_owned_[static_cast<size_t>(gI - my_first_tdof_)] || ess_of(gJ)) continue;  // dropped (eliminated)
        local_dest_[static_cast<size_t>(k)] = slotOf(gI, gJ);
        if (local_dest_[static_cast<size_t>(k)] < 0) {
          throw std::runtime_error("BSRDirectAssembler: non-eliminated local entry missing from BSR structure");
        }
      } else {
        send_entries[static_cast<size_t>(owner)].push_back(k);
        send_pairs[static_cast<size_t>(owner)].push_back(static_cast<long long>(gI));
        send_pairs[static_cast<size_t>(owner)].push_back(static_cast<long long>(gJ));
      }
    }
  }

  // exchange (gI, gJ) lists once; receivers resolve each incoming value to a data slot
  std::vector<int> send_counts(static_cast<size_t>(nproc), 0);
  for (int r = 0; r < nproc; ++r) {
    send_counts[static_cast<size_t>(r)] = static_cast<int>(send_entries[static_cast<size_t>(r)].size());
  }
  std::vector<int> recv_counts(static_cast<size_t>(nproc), 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm_);

  std::vector<MPI_Request> reqs;
  std::vector<std::vector<long long>> recv_pairs(static_cast<size_t>(nproc));
  for (int r = 0; r < nproc; ++r) {
    if (recv_counts[static_cast<size_t>(r)] > 0) {
      recv_pairs[static_cast<size_t>(r)].resize(static_cast<size_t>(recv_counts[static_cast<size_t>(r)]) * 2);
      MPI_Request req;
      MPI_Irecv(recv_pairs[static_cast<size_t>(r)].data(), recv_counts[static_cast<size_t>(r)] * 2, MPI_LONG_LONG, r,
                41, comm_, &req);
      reqs.push_back(req);
    }
    if (send_counts[static_cast<size_t>(r)] > 0) {
      MPI_Request req;
      MPI_Isend(send_pairs[static_cast<size_t>(r)].data(), send_counts[static_cast<size_t>(r)] * 2, MPI_LONG_LONG, r,
                41, comm_, &req);
      reqs.push_back(req);
    }
  }
  if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  for (int r = 0; r < nproc; ++r) {
    if (send_counts[static_cast<size_t>(r)] > 0) {
      Peer p;
      p.rank = r;
      p.entries = std::move(send_entries[static_cast<size_t>(r)]);
      p.buf.resize(p.entries.size());
      send_peers_.push_back(std::move(p));
    }
    if (recv_counts[static_cast<size_t>(r)] > 0) {
      Peer p;
      p.rank = r;
      const auto& pairs = recv_pairs[static_cast<size_t>(r)];
      p.recv_slots.resize(static_cast<size_t>(recv_counts[static_cast<size_t>(r)]));
      for (size_t t = 0; t < p.recv_slots.size(); ++t) {
        const HYPRE_BigInt gI = static_cast<HYPRE_BigInt>(pairs[2 * t]);
        const HYPRE_BigInt gJ = static_cast<HYPRE_BigInt>(pairs[2 * t + 1]);
        if (gI < my_first_tdof_ || gI >= my_first_tdof_ + my_tsize_) {
          throw std::runtime_error("BSRDirectAssembler: received a contribution for a row this rank does not own");
        }
        if (ess_owned_[static_cast<size_t>(gI - my_first_tdof_)] || ess_of(gJ)) {
          p.recv_slots[t] = -1;  // eliminated
          continue;
        }
        p.recv_slots[t] = slotOf(gI, gJ);
        if (p.recv_slots[t] < 0) {
          throw std::runtime_error("BSRDirectAssembler: non-eliminated remote entry missing from BSR structure");
        }
      }
      p.buf.resize(p.recv_slots.size());
      recv_peers_.push_back(std::move(p));
    }
  }

  // unit diagonal on eliminated rows (EliminateRowsCols semantics)
  for (int idx = 0; idx < ess_tdofs.Size(); ++idx) {
    const HYPRE_BigInt g = my_first_tdof_ + ess_tdofs[idx];
    const long long slot = slotOf(g, g);
    if (slot < 0) {
      throw std::runtime_error("BSRDirectAssembler: eliminated diagonal entry missing from BSR structure");
    }
    unit_diag_slots_.push_back(slot);
  }
}

void BSRDirectAssembler::update(const std::vector<double>& values)
{
  std::vector<double>& diag_data = bsr_->MutableDiagData();
  std::vector<double>& offd_data = bsr_->MutableOffdData();
  std::fill(diag_data.begin(), diag_data.end(), 0.0);
  std::fill(offd_data.begin(), offd_data.end(), 0.0);

  std::vector<MPI_Request> reqs;
  reqs.reserve(send_peers_.size() + recv_peers_.size());
  for (auto& p : recv_peers_) {
    MPI_Request req;
    MPI_Irecv(p.buf.data(), static_cast<int>(p.buf.size()), MPI_DOUBLE, p.rank, 43, comm_, &req);
    reqs.push_back(req);
  }
  for (auto& p : send_peers_) {
    for (size_t t = 0; t < p.entries.size(); ++t) {
      p.buf[t] = values[static_cast<size_t>(p.entries[t])];
    }
    MPI_Request req;
    MPI_Isend(p.buf.data(), static_cast<int>(p.buf.size()), MPI_DOUBLE, p.rank, 43, comm_, &req);
    reqs.push_back(req);
  }

  auto data_at = [&](long long s) -> double& {
    return s < static_cast<long long>(diag_data_size_) ? diag_data[static_cast<size_t>(s)]
                                                       : offd_data[static_cast<size_t>(s) - diag_data_size_];
  };

  const size_t nnz = local_dest_.size();
  for (size_t k = 0; k < nnz; ++k) {
    const long long d = local_dest_[k];
    if (d >= 0) data_at(d) += values[k];
  }

  if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  for (const auto& p : recv_peers_) {
    for (size_t t = 0; t < p.recv_slots.size(); ++t) {
      const long long s = p.recv_slots[t];
      if (s >= 0) data_at(s) += p.buf[t];
    }
  }

  for (long long s : unit_diag_slots_) {
    data_at(s) = 1.0;
  }
}

void BSRDirectAssembler::verify(const std::vector<double>& values)
{
  const std::vector<double> ref_diag = bsr_->DiagBSR().data;
  const std::vector<double> ref_offd = bsr_->OffdBSR().data;

  update(values);

  double scale = 1.0;
  for (double v : ref_diag) scale = std::max(scale, std::abs(v));
  double max_diff = 0.0;
  size_t max_k = 0;
  bool max_in_diag = true;
  long long n_bad = 0;
  const std::vector<double>& diag_data = bsr_->MutableDiagData();
  const std::vector<double>& offd_data = bsr_->MutableOffdData();
  for (size_t k = 0; k < ref_diag.size(); ++k) {
    const double d = std::abs(diag_data[k] - ref_diag[k]);
    if (d > 1.0e-9 * scale) ++n_bad;
    if (d > max_diff) {
      max_diff = d;
      max_k = k;
      max_in_diag = true;
    }
  }
  for (size_t k = 0; k < ref_offd.size(); ++k) {
    const double d = std::abs(offd_data[k] - ref_offd[k]);
    if (d > 1.0e-9 * scale) ++n_bad;
    if (d > max_diff) {
      max_diff = d;
      max_k = k;
      max_in_diag = false;
    }
  }

  double global[2] = {max_diff, scale};
  MPI_Allreduce(MPI_IN_PLACE, global, 2, MPI_DOUBLE, MPI_MAX, comm_);

  if (global[0] > 1.0e-9 * global[1]) {
    const int bb = b_ * b_;
    const size_t block = max_k / static_cast<size_t>(bb);
    int dbg_row = -1, dbg_col = -1, dbg_ess_row = -1, dbg_ess_col = -1;
    if (max_in_diag) {
      const BSRMatrix& D = bsr_->DiagBSR();
      for (int R = 0; R < D.nb_rows; ++R) {
        if (D.I[static_cast<size_t>(R)] <= static_cast<int>(block) && static_cast<int>(block) < D.I[static_cast<size_t>(R) + 1]) {
          dbg_row = R * b_ + static_cast<int>((max_k % static_cast<size_t>(bb)) / static_cast<size_t>(b_));
          dbg_col = D.J[block] * b_ + static_cast<int>(max_k % static_cast<size_t>(b_));
          dbg_ess_row = ess_owned_[static_cast<size_t>(dbg_row)];
          dbg_ess_col = ess_owned_[static_cast<size_t>(dbg_col)];
          break;
        }
      }
    }
    std::ostringstream os;
    os << "[dbg row " << dbg_row << " col " << dbg_col << " ess(" << dbg_ess_row << "," << dbg_ess_col << ")] ";
    os << "BSRDirectAssembler: routed values disagree with the legacy assembly (max diff " << global[0]
       << ", matrix scale " << global[1] << "); local worst: " << (max_in_diag ? "diag" : "offd") << " data[" << max_k
       << "] block " << block << " sub(" << (max_k % static_cast<size_t>(bb)) / static_cast<size_t>(b_) << ","
       << max_k % static_cast<size_t>(b_) << ") routed "
       << (max_in_diag ? diag_data[max_k] : offd_data[max_k]) << " ref "
       << (max_in_diag ? ref_diag[max_k] : ref_offd[max_k]) << "; " << n_bad << " bad entries of "
       << ref_diag.size() + ref_offd.size();
    throw std::runtime_error(os.str());
  }
}

}  // namespace smith
