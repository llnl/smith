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

/// @brief Abstract interface to DifferentiableSolver interface.  Each differentiable solve should provide both its
/// forward solve and an adjoint solve
class DifferentiableSolver {
 public:
  /// @brief destructor
  virtual ~DifferentiableSolver() {}

  /// @brief Required for certain solvers/preconditioners, e.g. when multigrid algorithms want a near null-space
  /// For these cases, it should be called before solve
  virtual void completeSetup(const smith::FiniteElementState& u) = 0;

  /// @brief Solve a set of equations with a FiniteElementState as unknown
  /// @param u_guess initial guess for solver
  /// @param equation std::function for equation to be solved
  /// @param jacobian std::function for evaluating the linearized Jacobian about the current solution
  /// @return The solution FiniteElementState
  virtual std::shared_ptr<smith::FiniteElementState> solve(
      const smith::FiniteElementState& u_guess, std::function<mfem::Vector(const smith::FiniteElementState&)> equation,
      std::function<std::unique_ptr<mfem::HypreParMatrix>(const smith::FiniteElementState&)> jacobian) const = 0;

  /// @brief Solve the (linear) adjoint set of equations with a FiniteElementState as unknown
  /// @param u_bar rhs for the solve
  /// @param jacobian_transposed the evaluated linearized adjoint space matrix
  /// @return The adjoint solution field
  virtual std::shared_ptr<smith::FiniteElementState> solveAdjoint(
      const smith::FiniteElementDual& u_bar, std::unique_ptr<mfem::HypreParMatrix> jacobian_transposed) const = 0;

  /// @brief Interface option to clear memory between solves to avoid high-water mark memory usage.
  virtual void clearMemory() const {}
};

/// @brief Implementation of the DifferentiableSolver interface for the special case of linear solves with linear
/// adjoint solves
class LinearDifferentiableSolver : public DifferentiableSolver {
 public:
  /// @brief Construct from a linear solver and linear precondition which may also be used by a nonlinear solver
  LinearDifferentiableSolver(std::unique_ptr<mfem::Solver> s, std::unique_ptr<mfem::Solver> p);

  /// @overload
  void completeSetup(const smith::FiniteElementState& u) override;

  /// @overload
  std::shared_ptr<smith::FiniteElementState> solve(
      const smith::FiniteElementState& u_guess, std::function<mfem::Vector(const smith::FiniteElementState&)> equation,
      std::function<std::unique_ptr<mfem::HypreParMatrix>(const smith::FiniteElementState&)> jacobian) const override;

  /// @overload
  std::shared_ptr<smith::FiniteElementState> solveAdjoint(
      const smith::FiniteElementDual& u_bar, std::unique_ptr<mfem::HypreParMatrix> jacobian_transposed) const override;

  mutable std::unique_ptr<mfem::Solver> mfem_solver;          ///< linear solver
  mutable std::unique_ptr<mfem::Solver> mfem_preconditioner;  ///< optionally used preconditioner
};

/// @brief Implementation of the DifferentiableSolver interface for the special case of nonlinear solves with linear
/// adjoint solves
class NonlinearDifferentiableSolver : public DifferentiableSolver {
 public:
  /// @brief Consruct from a smith nonlinear EquationSolver
  NonlinearDifferentiableSolver(std::unique_ptr<EquationSolver> s);

  /// @overload
  void completeSetup(const smith::FiniteElementState& u) override;

  /// @overload
  std::shared_ptr<smith::FiniteElementState> solve(
      const smith::FiniteElementState& u_guess, std::function<mfem::Vector(const smith::FiniteElementState&)> equation,
      std::function<std::unique_ptr<mfem::HypreParMatrix>(const smith::FiniteElementState&)> jacobian) const override;

  /// @overload
  std::shared_ptr<smith::FiniteElementState> solveAdjoint(
      const smith::FiniteElementDual& u_bar, std::unique_ptr<mfem::HypreParMatrix> jacobian_transposed) const override;

  /// @overload
  void clearMemory() const override;

  mutable std::unique_ptr<mfem::HypreParMatrix> J_;  ///< stored linearized Jacobian matrix for memory reuse
  mutable std::unique_ptr<EquationSolver>
      nonlinear_solver_;  ///< the nonlinear equation solver used for the forward pass
};

/// @brief Abstract interface to DifferentiableBlockSolver interface. Each differentiable block solve should provide
/// both its forward solve and an adjoint solve
class DifferentiableBlockSolver {
 public:
  /// @brief destructor
  virtual ~DifferentiableBlockSolver() {}

  using FieldT = FiniteElementState;                        ///< using
  using FieldPtr = std::shared_ptr<FieldT>;                 ///< using
  using FieldD = FiniteElementDual;                         ///< using
  using DualPtr = std::shared_ptr<FieldD>;                  ///< using
  using MatrixPtr = std::unique_ptr<mfem::HypreParMatrix>;  ///< using

  /// @brief Required for certain solvers/preconditions, e.g. when multigrid algorithms want a near null-space
  /// For these cases, it should be called before solve
  virtual void completeSetup(const std::vector<FieldT>& us) = 0;

  /// @brief Solve a set of equations with a vector of FiniteElementState as unknown
  /// @param u_guesses initial guess for solver
  /// @param residuals std::vector<std::function> for equations to be solved
  /// @param jacobians std::vector<std::vector>> of std::function for evaluating the linearized Jacobians about the
  /// current solution
  /// @return std::vector of solution vectors (FiniteElementState)
  virtual std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses,
      std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)> residuals,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)> jacobians) const = 0;

  /// @brief Solve the (linear) adjoint set of equations with a vector of FiniteElementState as unknown
  /// @param u_bars std::vector of right hand sides (rhs) for the solve
  /// @param jacobian_transposed std::vector<std::vector>> of evaluated linearized adjoint space matrices
  /// @return The adjoint vector of solution field
  virtual std::vector<FieldPtr> solveAdjoint(const std::vector<DualPtr>& u_bars,
                                             std::vector<std::vector<MatrixPtr>>& jacobian_transposed) const = 0;

  /// @brief Interface option to clear memory between solves to avoid high-water mark memory usage.
  virtual void clearMemory() const {}
};

/// @brief Implementation of the DifferentiableBlockSolver interface for the special case of linear solves with linear
/// adjoint solves
class LinearDifferentiableBlockSolver : public DifferentiableBlockSolver {
 public:
  /// @brief Construct from a linear solver and linear block precondition which may be used by the linear solver
  LinearDifferentiableBlockSolver(std::unique_ptr<mfem::Solver> s, std::unique_ptr<mfem::Solver> p);

  /// @overload
  void completeSetup(const std::vector<FieldT>& us) override;

  /// @overload
  std::vector<FieldPtr> solve(
      const std::vector<FieldPtr>& u_guesses,
      std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)> residuals,
      std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)> jacobians) const override;

  /// @overload
  std::vector<FieldPtr> solveAdjoint(const std::vector<DualPtr>& u_bars,
                                     std::vector<std::vector<MatrixPtr>>& jacobian_transposed) const override;

  mutable std::unique_ptr<mfem::Solver> mfem_solver;          ///< stored mfem block solver
  mutable std::unique_ptr<mfem::Solver> mfem_preconditioner;  ///< stored mfem block preconditioner
};

/// @brief Create a differentiable linear solver
/// @param linear_opts linear options struct
/// @param mesh mesh
std::shared_ptr<LinearDifferentiableSolver> buildDifferentiableLinearSolver(LinearSolverOptions linear_opts,
                                                                            const smith::Mesh& mesh);

/// @brief Create a differentiable nonlinear solver
/// @param nonlinear_opts nonlinear options struct
/// @param linear_opts linear options struct
/// @param mesh mesh
std::shared_ptr<NonlinearDifferentiableSolver> buildDifferentiableNonlinearSolver(NonlinearSolverOptions nonlinear_opts,
                                                                                  LinearSolverOptions linear_opts,
                                                                                  const smith::Mesh& mesh);

}  // namespace smith
