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
  SolverWithPreconditioner(std::unique_ptr<mfem::Solver> linear_solver, std::unique_ptr<mfem::Solver> preconditioner)
      : linear_solver_(std::move(linear_solver)), preconditioner_(std::move(preconditioner))
  {
    MFEM_VERIFY(linear_solver_ != nullptr, "SolverWithPreconditioner requires a non-null linear solver");
  }

  void SetOperator(const mfem::Operator& op) override
  {
    height = op.Height();
    width = op.Width();
    linear_solver_->SetOperator(op);
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override { linear_solver_->Mult(x, y); }

  mfem::Solver* linearSolver() const { return linear_solver_.get(); }
  mfem::Solver* preconditioner() const { return preconditioner_.get(); }

 private:
  std::unique_ptr<mfem::Solver> linear_solver_;
  std::unique_ptr<mfem::Solver> preconditioner_;
};

}  // namespace smith
