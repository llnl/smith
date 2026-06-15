// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file jacobian_assembly.hpp
 *
 * @brief Single owner of a physics module's assembled Jacobian state.
 *
 * Consolidates what used to be five ad-hoc `J_.reset(); J_ = assemble(...);
 * J_e_ = eliminate(*J_)` call sites (forward gradient, warm start, adjoint, ...)
 * behind one capability-based interface, with two assembly modes:
 *
 *  - Hypre:     legacy path. Every assemble() runs the full hypre assembly +
 *               EliminateRowsCols; hypre()/hypreEliminatedEntries() are valid.
 *  - DirectBSR: first assemble() bootstraps a BSRDirectAssembler; later calls
 *               route the Functional local CSR values straight into the BSR
 *               operator (no hypre objects). Requires a symmetric Jacobian for
 *               transpose support; hypre() is unavailable by design.
 *
 * BC handling is uniform: applyBCsToRHS performs the mfem::EliminateBC RHS
 * correction (rhs -= Ae*x; rhs[ess] = x[ess]) in both modes.
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "mfem.hpp"

#include "smith/numerics/bsr_direct_assembler.hpp"

namespace smith {

/// Owns the current Jacobian assembly path and the eliminated solver operator.
class JacobianAssembly {
 public:
  /// Available Jacobian assembly implementations.
  enum class Mode
  {
    /// Legacy hypre assembly and elimination path.
    Hypre,
    /// Direct value routing into a BSR operator after one hypre bootstrap.
    DirectBSR
  };

  /// closure running the full legacy assembly of the *current* state (uneliminated)
  using LegacyAssemble = std::function<std::unique_ptr<mfem::HypreParMatrix>()>;

  /**
   * @param mode       assembly mode (see file docs)
   * @param fes        the (vector H1) test == trial space
   * @param ess_tdofs  local true dofs to eliminate (copied)
   * @param symmetric  declare J = J^T; required by DirectBSR for transpose support
   */
  JacobianAssembly(Mode mode, mfem::ParFiniteElementSpace& fes, const mfem::Array<int>& ess_tdofs, bool symmetric);

  /**
   * @brief Refresh the Jacobian from the current state and return the solver-ready
   * (eliminated) operator.
   *
   * @param legacy             runs the full hypre assembly (always used in Hypre mode;
   *                           once for the DirectBSR bootstrap). It must leave the
   *                           Functional's local CSR refreshed (Gradient::assemble does).
   * @param refresh_local_csr  refreshes the local CSR only (Gradient::assembleLocalCSR);
   *                           used by DirectBSR steady state.
   * @param row_ptr            Functional local CSR row offsets.
   * @param col_ind            Functional local CSR column indices.
   * @param values             Functional local CSR values.
   */
  mfem::Operator& assemble(const LegacyAssemble& legacy, const std::function<void()>& refresh_local_csr,
                           const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
                           const std::vector<double>& values);

  /// the last-assembled eliminated operator
  mfem::Operator& eliminated();

  /// A^T as an operator: same operator when symmetric; lazily transposed hypre otherwise
  mfem::Operator& eliminatedTranspose();

  /// mfem::EliminateBC semantics on the RHS: rhs -= Ae * x; rhs[ess] = x[ess].
  /// `transpose` selects Ae of A^T (identical when symmetric).
  void applyBCsToRHS(const mfem::Vector& x, mfem::Vector& rhs, bool transpose = false) const;

  /// Hypre-mode accessors (throw in DirectBSR mode — no value-bearing hypre escape hatch)
  const mfem::HypreParMatrix& hypre() const;
  /// @overload
  const mfem::HypreParMatrix& hypreEliminatedEntries() const;

  /// Selected assembly mode.
  Mode mode() const { return mode_; }

 private:
  Mode mode_;
  mfem::ParFiniteElementSpace& fes_;
  mfem::Array<int> ess_tdofs_;
  bool symmetric_ = false;

  // Hypre mode state
  std::unique_ptr<mfem::HypreParMatrix> J_;
  std::unique_ptr<mfem::HypreParMatrix> J_e_;
  std::unique_ptr<mfem::HypreParMatrix> JT_;    ///< lazy, invalidated on assemble
  std::unique_ptr<mfem::HypreParMatrix> JT_e_;  ///< Ae of the transpose

  // DirectBSR mode state
  std::unique_ptr<BSRDirectAssembler> assembler_;
  size_t bootstrap_nnz_ = 0;  ///< routing-plan invalidation guard
};

}  // namespace smith
