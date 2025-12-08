// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_solver.hpp
 *
 * @brief This file contains the declaration of the DifferentiableSolver interface
 */

#pragma once

#include <memory>
#include <functional>

namespace mfem {
class Solver;
class Vector;
class HypreParMatrix;
}  // namespace mfem

namespace smith {

class EquationSolver;
class BoundaryConditionManager;
class FiniteElementState;
class FiniteElementDual;
class Mesh;
struct NonlinearSolverOptions;
struct LinearSolverOptions;


/// @brief Abstract interface to DifferentiableSolver inteface.  Each dfferenriable solve should provide both its forward solve and an adjoint solve
class DifferentiableSolver {
 public:
  virtual ~DifferentiableSolver() {}

  virtual void completeSetup(const smith::FiniteElementState& u) = 0;

  virtual std::shared_ptr<smith::FiniteElementState> solve(
      const smith::FiniteElementState& u_guess, std::function<mfem::Vector(const smith::FiniteElementState&)> equation,
      std::function<std::unique_ptr<mfem::HypreParMatrix>(const smith::FiniteElementState&)> jacobian) const = 0;

  virtual std::shared_ptr<smith::FiniteElementState> solveAdjoint(
      const smith::FiniteElementDual& u_bar, std::unique_ptr<mfem::HypreParMatrix> jacobian_transposed) const = 0;

  virtual void clearMemory() const {}
};

class LinearDifferentiableSolver : public DifferentiableSolver {
 public:
  LinearDifferentiableSolver(std::unique_ptr<mfem::Solver> s, std::unique_ptr<mfem::Solver> p);

  void completeSetup(const smith::FiniteElementState& u) override;

  std::shared_ptr<smith::FiniteElementState> solve(
      const smith::FiniteElementState& u_guess, std::function<mfem::Vector(const smith::FiniteElementState&)> equation,
      std::function<std::unique_ptr<mfem::HypreParMatrix>(const smith::FiniteElementState&)> jacobian) const override;

  std::shared_ptr<smith::FiniteElementState> solveAdjoint(
      const smith::FiniteElementDual& u_bar, std::unique_ptr<mfem::HypreParMatrix> jacobian_transposed) const override;

  mutable std::unique_ptr<mfem::Solver> mfem_solver;
  mutable std::unique_ptr<mfem::Solver> mfem_preconditioner;
};

class NonlinearDifferentiableSolver : public DifferentiableSolver {
 public:
  NonlinearDifferentiableSolver(std::unique_ptr<EquationSolver> s);

  void completeSetup(const smith::FiniteElementState& u) override;

  std::shared_ptr<smith::FiniteElementState> solve(
      const smith::FiniteElementState& u_guess, std::function<mfem::Vector(const smith::FiniteElementState&)> equation,
      std::function<std::unique_ptr<mfem::HypreParMatrix>(const smith::FiniteElementState&)> jacobian) const override;

  std::shared_ptr<smith::FiniteElementState> solveAdjoint(
      const smith::FiniteElementDual& u_bar, std::unique_ptr<mfem::HypreParMatrix> jacobian_transposed) const override;

  virtual void clearMemory() const override;

  mutable std::unique_ptr<mfem::HypreParMatrix> J_;
  mutable std::unique_ptr<EquationSolver> nonlinear_solver_;
};

/// @brief Create a differentiable linear solver
/// @param linear_opts linear options struct
/// @param mesh mesh
std::shared_ptr<LinearDifferentiableSolver> buildDifferentiableLinearSolve(LinearSolverOptions linear_opts,
                                                                           const smith::Mesh& mesh);

/// @brief Create a differentiable nonlinear solver
/// @param nonlinear_opts nonlinear options struct
/// @param linear_opts linear options struct
/// @param mesh mesh
std::shared_ptr<NonlinearDifferentiableSolver> buildDifferentiableNonlinearSolve(NonlinearSolverOptions nonlinear_opts,
                                                                                 LinearSolverOptions linear_opts,
                                                                                 const smith::Mesh& mesh);

}  // namespace smith