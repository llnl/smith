// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file batched_matvec.hpp
 *
 * @brief Batched (multi-RHS) HypreParMatrix * X.  Y_j = A * X_j for j = 0..k-1.
 *
 * X and Y are local dense column blocks of shape (local_tsize, k). For the deflation
 * preconditioner (see deflation.md Phase 5), X is the local block of the global
 * column-distributed W matrix and the columns are mostly-zero on non-owner ranks.
 *
 * The point is to amortize the m halo exchanges of m sequential `A.Mult(W_q, AW_q)`
 * calls into a single packed exchange.
 *
 * Three strategy variants are exposed for benchmarking:
 *   - Loop        : k sequential `A.Mult` calls (baseline; correctness reference).
 *   - PackedDense : single packed halo exchange + per-element local SpMV.
 *
 * For the deflation use case specifically, `assembleWtAW` below is the higher-level
 * entry point — it returns `W^T A W` directly using a single halo exchange of W
 * itself (volume `m/p` per rank vs `m` for batchedMatvec). See deflation.md Phase 5b.
 */

#pragma once

#include "mfem.hpp"

namespace smith {

enum class BatchedMatvecStrategy
{
  Loop,         // sequential per-column A.Mult — correctness baseline
  PackedDense,  // single packed halo exchange + per-element local SpMV
};

/// Optional fine-grained wall-time accumulators for `assembleWtAW` profiling.
/// Pass `nullptr` to skip.
struct AssembleWtAWTimings {
  double halo = 0.0;       // pack + Isend/Irecv + Waitall
  double diag = 0.0;       // A_diag · W_local + W_local^T · (·)
  double offd = 0.0;       // per-neighbor A_offd · W_s_halo + W_local^T · (·)
  double allreduce = 0.0;  // MPI_Allreduce of the m × m result
};

/**
 * @brief Y = A * X, where X and Y are (local_tsize, k) dense column blocks.
 *
 * Each column j is a parallel vector whose local portion lies in column j of X.
 * Each output column j is the local portion of A * X_j.
 *
 * @param A          parallel matrix (rows/cols sized to fes.GetTrueVSize() locally).
 * @param X          (local_tsize × k) dense block; each col is an input parallel vec.
 * @param Y          (local_tsize × k) dense block; each col gets A * X_col.
 * @param strategy   which implementation to use.
 */
void batchedMatvec(const mfem::HypreParMatrix& A, const mfem::DenseMatrix& X, mfem::DenseMatrix& Y,
                   BatchedMatvecStrategy strategy = BatchedMatvecStrategy::Loop);

/**
 * @brief Assemble W^T A W directly for a column-distributed W (block-diagonal across ranks).
 *
 * Exploits that (W^T A W)(p, q) only requires data from rank p (because W_p has support
 * only on p). Communication: a single halo exchange of W (m/p columns per rank) replaces
 * the m parallel matvecs. Output is replicated on every rank via MPI_Allreduce.
 *
 * @param A                parallel matrix.
 * @param W_local          (n_local × modes_per_rank) — this rank's owned W block.
 * @param modes_per_rank   number of W columns owned by every rank (m / nprocs).
 *                         Must be uniform across ranks.
 * @param WtAW             output (m × m) dense matrix, replicated on all ranks.
 *                         m = modes_per_rank * nprocs. Owner of global col q is q / modes_per_rank.
 */
void assembleWtAW(const mfem::HypreParMatrix& A, const mfem::DenseMatrix& W_local, int modes_per_rank,
                  mfem::DenseMatrix& WtAW, AssembleWtAWTimings* timings = nullptr);

class BSROperator;

/**
 * @brief Same as above, but the local diag/offd SpMM runs on the BSR representation
 * (one block index per b*b entries, contiguous block math). Requires bsr.Enabled().
 */
void assembleWtAW(const BSROperator& bsr, const mfem::DenseMatrix& W_local, int modes_per_rank, mfem::DenseMatrix& WtAW,
                  AssembleWtAWTimings* timings = nullptr);

}  // namespace smith
