// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file bsr_direct_assembler.hpp
 *
 * @brief Direct-to-BSR Jacobian assembly: routes Functional's local L-dof CSR values
 * straight into a true-dof BSR operator, replacing the per-assembly hypre RAP +
 * BC elimination + CSR->BSR conversion with a precomputed value-routing plan.
 *
 * Setup (once) bootstraps structure from one legacy assembly (uneliminated true-dof
 * HypreParMatrix + its eliminated BSR conversion), builds:
 *  - per-L-CSR-entry destinations (BSR diag/offd data index, neighbor send slot, or drop),
 *  - receiver-side slot resolution for contributions to rows this rank owns but other
 *    ranks computed (the communication hypre RAP performs),
 *  - the BC-elimination plan (dropped entries + unit diagonal slots), using an ess-flag
 *    halo exchange so eliminated *neighbor* columns are dropped too.
 * The routed result is verified against the legacy matrix at construction.
 *
 * update() refreshes the BSROperator values in place; no hypre objects are touched.
 * The bootstrap hypre matrix is kept alive for structure-only consumers (comm package).
 */

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"

#include "smith/numerics/bsr_operator.hpp"

namespace smith {

class BSRDirectAssembler {
 public:
  /**
   * @param fes        the (vector H1) space; test == trial.
   * @param ess_tdofs  local true dofs to eliminate (rows/cols zeroed, 1 on diagonal).
   * @param A          legacy-assembled, *eliminated* true-dof matrix at the current state.
   *                   Cloned internally (the caller may free or reassemble it); the clone
   *                   provides structure, first values, and the comm package.
   * @param row_ptr / col_ind / values  Functional's local L-dof CSR at the same state.
   */
  BSRDirectAssembler(mfem::ParFiniteElementSpace& fes, const mfem::Array<int>& ess_tdofs, mfem::HypreParMatrix* A,
                     const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
                     const std::vector<double>& values, const mfem::HypreParMatrix* Ae_reference = nullptr);

  /// Route fresh local CSR values into the BSR operator (zero, route, halo exchange, eliminate).
  void update(const std::vector<double>& values);

  /**
   * @brief y = Ae * x on non-eliminated rows, where Ae holds the eliminated *columns* of the
   * un-eliminated matrix: y_i = sum_{j in ess} A(i,j) x_j for i not in ess; y_i = 0 for i in ess.
   * This is what mfem::EliminateBC needs to correct the RHS for inhomogeneous essential BCs
   * (eliminated *rows* of Ae are irrelevant there — callers overwrite rhs[ess] afterwards).
   */
  void eliminatedColumnsAction(const mfem::Vector& x, mfem::Vector& y) const;

  /// The operator the solver consumes. GetHypreMatrix() returns the stale bootstrap matrix
  /// (structure-only; do not read its values).
  BSROperator& op() { return *bsr_; }

 private:
  void buildRouting(mfem::ParFiniteElementSpace& fes, const mfem::Array<int>& ess_tdofs,
                    const std::vector<int>& row_ptr, const std::vector<int>& col_ind);
  void verify(const std::vector<double>& values);

  /// unified BSR data index: [0, diag_size) -> diag data, [diag_size, ...) -> offd data
  long long slotOf(HYPRE_BigInt gI, HYPRE_BigInt gJ) const;

  std::unique_ptr<mfem::HypreParMatrix> A_owned_;
  mfem::HypreParMatrix* A_ = nullptr;
  std::unique_ptr<BSROperator> bsr_;
  MPI_Comm comm_;
  int b_ = 0;
  HYPRE_BigInt my_first_tdof_ = 0;
  int my_tsize_ = 0;
  size_t diag_data_size_ = 0;

  std::vector<HYPRE_BigInt> col_map_offd_;  ///< sorted global tdofs of offd columns
  std::vector<char> ess_owned_;             ///< per owned tdof
  std::vector<char> ess_offd_;              ///< per offd column (halo-exchanged)

  /// per local CSR entry: >= 0 unified BSR data index; -1 drop / handled by a send list;
  /// <= -2 routes into the Ae store at index (-2 - dest)
  std::vector<long long> local_dest_;
  /// eliminated diagonal slots set to 1.0 after accumulation
  std::vector<long long> unit_diag_slots_;

  /// Ae (eliminated-columns) store. src >= 0: owned tdof index into x; src < 0: offd column
  /// position ~src (halo value of x). Values are partial contributions refreshed by update();
  /// duplicates (same row/col from several ranks) accumulate naturally in the action.
  struct AeEntry {
    int row = 0;
    int src = 0;
  };
  std::vector<AeEntry> ae_entries_;
  std::vector<double> ae_values_;
  bool ae_has_halo_ = false;

  struct Peer {
    int rank = 0;
    std::vector<int> entries;            ///< sender: L-CSR entry indices, in send order
    std::vector<long long> recv_slots;   ///< receiver: unified data index per incoming value (-1 drop)
    std::vector<double> buf;             ///< send or recv value buffer
  };
  std::vector<Peer> send_peers_;
  std::vector<Peer> recv_peers_;
};

}  // namespace smith
