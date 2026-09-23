// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <memory>

#include "mfem.hpp"

namespace smith {

/// @brief Simple wrapper that owns a linear solver and its preconditioner.
///
/// This is used to keep a preconditioner alive when it is referenced by an
/// iterative solver (e.g. GMRES) via SetPreconditioner().
class SolverWithPreconditioner : public mfem::Solver {
 public:
  /// @brief Construct from an owned linear solver and (optional) preconditioner.
  /// @param linear_solver Owned linear solver (must be non-null).
  /// @param preconditioner Owned preconditioner (may be null).
  SolverWithPreconditioner(std::unique_ptr<mfem::Solver> linear_solver, std::unique_ptr<mfem::Solver> preconditioner)
      : linear_solver_(std::move(linear_solver)), preconditioner_(std::move(preconditioner))
  {
    MFEM_VERIFY(linear_solver_ != nullptr, "SolverWithPreconditioner requires a non-null linear solver");
  }

  /// @brief Set the operator on the underlying linear solver.
  /// @param op Operator to be solved/applied.
  void SetOperator(const mfem::Operator& op) override
  {
    height = op.Height();
    width = op.Width();
    linear_solver_->SetOperator(op);
  }

  /// @brief Apply the underlying linear solver.
  /// @param x Input vector.
  /// @param y Output vector.
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    linear_solver_->iterative_mode = iterative_mode;
    linear_solver_->Mult(x, y);
  }

  /// @brief Non-owning access to the underlying linear solver.
  mfem::Solver* linearSolver() const { return linear_solver_.get(); }

  /// @brief Non-owning access to the owned preconditioner (may be null).
  mfem::Solver* preconditioner() const { return preconditioner_.get(); }

 private:
  std::unique_ptr<mfem::Solver> linear_solver_;
  std::unique_ptr<mfem::Solver> preconditioner_;
};

}  // namespace smith
