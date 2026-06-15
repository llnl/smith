// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/jacobian_assembly.hpp"

#include <stdexcept>

namespace smith {

JacobianAssembly::JacobianAssembly(Mode mode, mfem::ParFiniteElementSpace& fes, const mfem::Array<int>& ess_tdofs,
                                   bool symmetric)
    : mode_(mode), fes_(fes), symmetric_(symmetric)
{
  ess_tdofs_ = ess_tdofs;
  if (mode_ == Mode::DirectBSR && !symmetric_) {
    throw std::runtime_error("JacobianAssembly: DirectBSR mode requires a symmetric Jacobian (transpose support)");
  }
}

mfem::Operator& JacobianAssembly::assemble(const LegacyAssemble& legacy, const std::function<void()>& refresh_local_csr,
                                           const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
                                           const std::vector<double>& values)
{
  if (mode_ == Mode::Hypre) {
    J_ = legacy();
    J_e_.reset(J_->EliminateRowsCols(ess_tdofs_));
    JT_.reset();
    JT_e_.reset();
    return *J_;
  }

  if (!assembler_) {
    auto J_boot = legacy();  // refreshes the local CSR as a side effect
    std::unique_ptr<mfem::HypreParMatrix> Je_boot(J_boot->EliminateRowsCols(ess_tdofs_));
    assembler_ =
        std::make_unique<BSRDirectAssembler>(fes_, ess_tdofs_, J_boot.get(), row_ptr, col_ind, values, Je_boot.get());
    assembler_->op().SetSymmetric(symmetric_);
    bootstrap_nnz_ = static_cast<size_t>(row_ptr.back());
  } else {
    refresh_local_csr();
    if (static_cast<size_t>(row_ptr.back()) != bootstrap_nnz_) {
      throw std::runtime_error(
          "JacobianAssembly: local CSR sparsity changed after bootstrap; the routing plan is stale "
          "(integrals/Domains must not be added after the first assemble)");
    }
    assembler_->update(values);
  }
  return assembler_->op();
}

mfem::Operator& JacobianAssembly::eliminated()
{
  if (mode_ == Mode::Hypre) {
    if (!J_) throw std::runtime_error("JacobianAssembly::eliminated: assemble() has not been called");
    return *J_;
  }
  if (!assembler_) throw std::runtime_error("JacobianAssembly::eliminated: assemble() has not been called");
  return assembler_->op();
}

mfem::Operator& JacobianAssembly::eliminatedTranspose()
{
  if (symmetric_) return eliminated();
  if (mode_ != Mode::Hypre) {
    throw std::runtime_error("JacobianAssembly: non-symmetric transpose requires Hypre mode");
  }
  if (!J_) throw std::runtime_error("JacobianAssembly::eliminatedTranspose: assemble() has not been called");
  if (!JT_) {
    // (A_elim)^T == (A^T)_elim for row+col elimination over the same set, so transposing the
    // eliminated matrix matches the legacy transpose-then-eliminate path. Its Ae is built from
    // a fresh transpose of the uneliminated values, which J_ no longer holds — rebuild from Ae:
    // Ae(A^T) = (Ae(A))^T.
    JT_.reset(J_->Transpose());
    JT_e_.reset(J_e_->Transpose());
  }
  return *JT_;
}

void JacobianAssembly::applyBCsToRHS(const mfem::Vector& x, mfem::Vector& rhs, bool transpose) const
{
  mfem::Vector correction(rhs.Size());
  if (mode_ == Mode::Hypre) {
    const mfem::HypreParMatrix* Ae = (transpose && !symmetric_) ? JT_e_.get() : J_e_.get();
    if (!Ae) throw std::runtime_error("JacobianAssembly::applyBCsToRHS: assemble() (and transpose) required first");
    Ae->Mult(x, correction);
  } else {
    assembler_->eliminatedColumnsAction(x, correction);
  }
  rhs -= correction;
  for (int i = 0; i < ess_tdofs_.Size(); ++i) {
    rhs[ess_tdofs_[i]] = x[ess_tdofs_[i]];
  }
}

const mfem::HypreParMatrix& JacobianAssembly::hypre() const
{
  if (mode_ != Mode::Hypre || !J_) {
    throw std::runtime_error("JacobianAssembly::hypre: only available in Hypre mode after assemble()");
  }
  return *J_;
}

const mfem::HypreParMatrix& JacobianAssembly::hypreEliminatedEntries() const
{
  if (mode_ != Mode::Hypre || !J_e_) {
    throw std::runtime_error("JacobianAssembly::hypreEliminatedEntries: only available in Hypre mode after assemble()");
  }
  return *J_e_;
}

}  // namespace smith
