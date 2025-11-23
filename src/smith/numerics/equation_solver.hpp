// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file equation_solver.hpp
 *
 * @brief This file contains the declaration of an equation solver wrapper
 */

#pragma once

#include <memory>
#include <optional>
#include <variant>
#include <utility>

#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/input.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/numerics/petsc_solvers.hpp"

namespace smith {

/**
 * @brief This class manages the objects typically required to solve a nonlinear set of equations arising from
 * discretization of a PDE of the form F(x) = 0. Specifically, it has
 *
 *   1. An @a mfem::NewtonSolver containing the nonlinear solution operator
 *   2. An @a mfem::Solver containing a linear solver that is used by the nonlinear solution operator and adjoint
 * solvers
 *   3. An optional @a mfem::Solver containing a preconditioner for the linear solution operator
 *
 * This @a EquationSolver manages these objects together to ensure they all exist when called by their associated
 * physics simulation module.
 *
 * An equation solver can either be constructed by supplying pre-built nonlinear and linear solvers with a
 * preconditioner, or it can be constructed using @a smith::NonlinearSolverOptions and @a smith::LinearSolverOptions
 * structs with the @a smith::mfem_ext::buildEquationSolver factory method.
 */
class EquationSolver {
 public:
  // _equationsolver_constructor_start
  /**
   * Constructs a new nonlinear equation solver
   * @param[in] nonlinear_solver A constructed nonlinear solver
   * @param[in] linear_solver A constructed linear solver to be called by the nonlinear algorithm and adjoint
   * equation solves
   * @param[in] preconditioner An optional constructed precondition to aid the linear solver
   */
  EquationSolver(std::unique_ptr<mfem::NewtonSolver> nonlinear_solver, std::unique_ptr<mfem::Solver> linear_solver,
                 std::unique_ptr<mfem::Solver> preconditioner = nullptr);
  // _equationsolver_constructor_end

  // _build_equationsolver_start
  /**
   * @brief Construct an equation solver object using nonlinear and linear solver option structs.
   *
   * @param nonlinear_opts The options to configure the nonlinear solution scheme
   * @param lin_opts The options to configure the underlying linear solution scheme to be used by the nonlinear solver
   * @param comm The MPI communicator for the supplied nonlinear operators and HypreParVectors
   */
  EquationSolver(NonlinearSolverOptions nonlinear_opts = {}, LinearSolverOptions lin_opts = {},
                 MPI_Comm comm = MPI_COMM_WORLD);
  // _build_equationsolver_end

  /**
   * Updates the solver with the provided operator
   * @param[in] op The operator (nonlinear system of equations) to use, "F" in F(x) = 0
   * @note This operator is required to return an @a mfem::HypreParMatrix from its @a GetGradient method. This is
   * due to the use of Hypre-based linear solvers.
   */
  void setOperator(const mfem::Operator& op);

  /**
   * Solves the system F(x) = 0
   * @param[in,out] x Solution to the system of nonlinear equations
   * @note The input value of @a x will be used as an initial guess for iterative nonlinear solution methods
   */
  void solve(mfem::Vector& x) const;

  /**
   * Returns the underlying solver object
   * @return A non-owning reference to the underlying nonlinear solver
   */
  mfem::NewtonSolver& nonlinearSolver() { return *nonlin_solver_; }

  /**
   * @overload
   */
  const mfem::NewtonSolver& nonlinearSolver() const { return *nonlin_solver_; }

  /**
   * Returns the underlying linear solver object
   * @return A non-owning reference to the underlying linear solver
   */
  mfem::Solver& linearSolver() { return *lin_solver_; }

  /**
   * @overload
   */
  const mfem::Solver& linearSolver() const { return *lin_solver_; }

  /**
   * Returns the underlying preconditioner
   * @return A pointer to the underlying preconditioner
   * @note This may be null if a preconditioner is not given
   */
  mfem::Solver& preconditioner() { return *preconditioner_; }

  /**
   * @overload
   */
  const mfem::Solver& preconditioner() const { return *preconditioner_; }

  /**
   * Input file parameters specific to this class
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

 private:
  /**
   * @brief The optional preconditioner (used for an iterative solver only)
   */
  std::unique_ptr<mfem::Solver> preconditioner_;

  /**
   * @brief The linear solver object, either custom, direct (SuperLU), or iterative
   */
  std::unique_ptr<mfem::Solver> lin_solver_;

  /**
   * @brief The nonlinear solver object
   */
  std::unique_ptr<mfem::NewtonSolver> nonlin_solver_;

  /**
   * @brief Whether the solver (linear solver) has been configured with the nonlinear solver
   * @note This is a workaround as some nonlinear solvers require SetOperator to be called
   * before SetSolver
   */
  bool nonlin_solver_set_solver_called_ = false;
};

/**
 * @brief A wrapper class for using the MFEM SuperLU solver with a HypreParMatrix
 */
class SuperLUSolver : public mfem::Solver {
 public:
  /**
   * @brief Constructs a wrapper over an mfem::SuperLUSolver
   * @param[in] comm The MPI communicator used by the vectors and matrices in the solve
   * @param[in] print_level The verbosity level for the mfem::SuperLUSolver
   */
  SuperLUSolver(int print_level, MPI_Comm comm) : superlu_solver_(comm)
  {
    superlu_solver_.SetColumnPermutation(mfem::superlu::PARMETIS);
    if (print_level == 0) {
      superlu_solver_.SetPrintStatistics(false);
    }
  }

  /**
   * @brief Factor and solve the linear system y = Op^{-1} x using DSuperLU
   *
   * @param input The input RHS vector
   * @param output The output solution vector
   */
  void Mult(const mfem::Vector& input, mfem::Vector& output) const;

  /**
   * @brief Set the underlying matrix operator to use in the solution algorithm
   *
   * @param op The matrix operator to factorize with SuperLU
   * @pre This operator must be an assembled HypreParMatrix or a BlockOperator
   * with all blocks either null or HypreParMatrixs for compatibility with
   * SuperLU
   */
  void SetOperator(const mfem::Operator& op);

 private:
  /**
   * @brief The owner of the SuperLU matrix for the gradient, stored
   * as a member variable for lifetime purposes
   */
  mutable std::unique_ptr<mfem::SuperLURowLocMatrix> superlu_mat_;

  /**
   * @brief The underlying MFEM-based SuperLU solver. It requires a special
   * SuperLU matrix type which we store in this object. This enables compatibility
   * with HypreParMatrix when used as an input.
   */
  mfem::SuperLUSolver superlu_solver_;
};

#ifdef MFEM_USE_STRUMPACK
/**
 * @brief A wrapper class for using the MFEM Strumpack solver with a HypreParMatrix
 */
class StrumpackSolver : public mfem::Solver {
 public:
  /**
   * @brief Constructs a wrapper over an mfem::STRUMPACKSolver
   * @param[in] comm The MPI communicator used by the vectors and matrices in the solve
   * @param[in] print_level The verbosity level for the mfem::STRUMPACKSolver
   */
  StrumpackSolver(int print_level, MPI_Comm comm) : strumpack_solver_(comm)
  {
    strumpack_solver_.SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
    strumpack_solver_.SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);

    if (print_level == 1) {
      strumpack_solver_.SetPrintFactorStatistics(true);
      strumpack_solver_.SetPrintSolveStatistics(true);
    }
  }

  /**
   * @brief Factor and solve the linear system y = Op^{-1} x using Strumpack
   *
   * @param input The input RHS vector
   * @param output The output solution vector
   */
  void Mult(const mfem::Vector& input, mfem::Vector& output) const;

  /**
   * @brief Set the underlying matrix operator to use in the solution algorithm
   *
   * @param op The matrix operator to factorize with Strumpack
   * @pre This operator must be an assembled HypreParMatrix for compatibility with Strumpack
   */
  void SetOperator(const mfem::Operator& op);

 private:
  /**
   * @brief The owner of the Strumpack matrix for the gradient, stored
   * as a member variable for lifetime purposes
   */
  mutable std::unique_ptr<mfem::STRUMPACKRowLocMatrix> strumpack_mat_;

  /**
   * @brief The underlying MFEM-based Strumpack solver. It requires a special
   * Strumpack matrix type which we store in this object. This enables compatibility
   * with HypreParMatrix when used as an input.
   */
  mfem::STRUMPACKSolver strumpack_solver_;
};

#endif

/**
 * @brief Function for building a monolithic parallel Hypre matrix from a block system of smaller Hypre matrices
 *
 * @param block_operator The block system of HypreParMatrices
 * @return The assembled monolithic HypreParMatrix
 *
 * @pre @a block_operator must have assembled HypreParMatrices for its sub-blocks
 */
std::unique_ptr<mfem::HypreParMatrix> buildMonolithicMatrix(const mfem::BlockOperator& block_operator);

/**
 * @brief Build a nonlinear solver using the nonlinear option struct
 *
 * @param nonlinear_opts The options to configure the nonlinear solution scheme
 * @param linear_opts The options to configure the linear solution scheme
 * @param preconditioner A preconditioner to help with either linear or nonlinear solves
 * @param comm The MPI communicator for the supplied nonlinear operators and HypreParVectors
 * @return The constructed nonlinear solver
 */
std::unique_ptr<mfem::NewtonSolver> buildNonlinearSolver(const NonlinearSolverOptions& nonlinear_opts,
                                                         const LinearSolverOptions& linear_opts,
                                                         mfem::Solver& preconditioner, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Build the linear solver and its associated preconditioner given a linear options struct
 *
 * @param linear_opts The options to configure the linear solver and preconditioner
 * @param comm The MPI communicator for the supplied HypreParMatrix and HypreParVectors
 * @return A pair containing the constructed linear solver and preconditioner objects
 */
std::pair<std::unique_ptr<mfem::Solver>, std::unique_ptr<mfem::Solver>> buildLinearSolverAndPreconditioner(
    LinearSolverOptions linear_opts = {}, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Build a preconditioner from the available options
 * @param linear_opts The options to configure the linear solver and preconditioner
 * @param comm The communicator for the underlying operator and HypreParVectors
 * @return A constructed preconditioner based on the input option
 */
std::unique_ptr<mfem::Solver> buildPreconditioner(LinearSolverOptions linear_opts,
                                                  [[maybe_unused]] MPI_Comm comm = MPI_COMM_WORLD);

#ifdef MFEM_USE_AMGX
/**
 * @brief Build an AMGX preconditioner
 *
 * @param options The options used to construct the AMGX preconditioner
 * @param comm The communicator for the underlying operator and HypreParVectors
 * @return The constructed AMGX preconditioner
 */
std::unique_ptr<mfem::AmgXSolver> buildAMGX(const AMGXOptions& options, const MPI_Comm comm);
#endif

}  // namespace smith

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<smith::LinearSolverOptions> {
  /// @brief Returns created object from Inlet container
  smith::LinearSolverOptions operator()(const axom::inlet::Container& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<smith::NonlinearSolverOptions> {
  /// @brief Returns created object from Inlet container
  smith::NonlinearSolverOptions operator()(const axom::inlet::Container& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<smith::EquationSolver> {
  /// @brief Returns created object from Inlet container
  smith::EquationSolver operator()(const axom::inlet::Container& base);
};
