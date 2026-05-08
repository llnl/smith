// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file equation_solver.hpp
 *
 * @brief This file contains the declaration of a trust region subspace solver
 */

#pragma once

#include "smith/smith_config.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mfem.hpp"

namespace smith {

class PetscException : public std::exception {
 public:
  /// constructor
  PetscException(const std::string& message) : msg(message) {}

  /// what is message
  const char* what() const noexcept override { return msg.c_str(); }

 private:
  /// message string
  std::string msg;
};

enum class TrustRegionSubspaceBackend {
  Petsc,
  Mfem
};

using TrustRegionSubspaceResult =
    std::tuple<mfem::Vector, std::vector<std::shared_ptr<mfem::Vector>>, std::vector<double>, double>;

struct TrustRegionSubspaceTimings {
  size_t num_solves = 0;
  size_t total_input_dim = 0;
  size_t total_reduced_dim = 0;
  size_t max_input_dim = 0;
  size_t max_reduced_dim = 0;
  double project_A_seconds = 0.0;
  double project_gram_seconds = 0.0;
  double project_b_seconds = 0.0;
  double basis_seconds = 0.0;
  double reduced_A_seconds = 0.0;
  double dense_eigensystem_seconds = 0.0;
  double dense_trust_solve_seconds = 0.0;
  double reconstruct_solution_seconds = 0.0;
  double reconstruct_leftmost_seconds = 0.0;
};

void resetTrustRegionSubspaceTimings();

TrustRegionSubspaceTimings trustRegionSubspaceTimings();

using DenseCubicTrustRegionResult = std::tuple<mfem::Vector, double>;

/// @brief computes the global size of mfem::Vector
int globalSize(const mfem::Vector& parallel_v, const MPI_Comm& comm);

/// @brief computes the l2 inner product between two mfem::Vector in parallal
double innerProduct(const mfem::Vector& a, const mfem::Vector& b, const MPI_Comm& comm);

/// @brief returns the solution, as well as a list of the N leftmost eigenvectors
/// and their eigenvalues, and the predicted model energy change
TrustRegionSubspaceResult solveSubspaceProblem(
    const std::vector<const mfem::Vector*>& directions, const std::vector<const mfem::Vector*>& A_directions,
    const mfem::Vector& b, double delta, int num_leftmost);

#ifdef SMITH_USE_SLEPC
TrustRegionSubspaceResult solveSubspaceProblemPetsc(
    const std::vector<const mfem::Vector*>& directions, const std::vector<const mfem::Vector*>& A_directions,
    const mfem::Vector& b, double delta, int num_leftmost);
#endif

TrustRegionSubspaceResult solveSubspaceProblemMfem(
    const std::vector<const mfem::Vector*>& directions, const std::vector<const mfem::Vector*>& A_directions,
    const mfem::Vector& b, double delta, int num_leftmost);

/// @brief solves a small dense cubic trust-region model
///   1/2 x^T A x - b^T x + 1/6 sum_k x_k x^T cubic[k] x, ||x|| <= delta.
DenseCubicTrustRegionResult solveDenseCubicTrustRegionProblemMfem(
    const mfem::DenseMatrix& A, const mfem::Vector& b, const std::vector<mfem::DenseMatrix>& cubic, double delta);

TrustRegionSubspaceResult solveCubicSubspaceProblemMfem(
    const std::vector<const mfem::Vector*>& directions, const std::vector<const mfem::Vector*>& A_directions,
    const std::vector<const mfem::Vector*>& previous_A_directions, const mfem::Vector& previous_step,
    const mfem::Vector& b, double delta, int num_leftmost, bool* used_cubic = nullptr);

std::pair<std::vector<const mfem::Vector*>, std::vector<const mfem::Vector*>> removeDependentDirections(
    std::vector<const mfem::Vector*> directions, std::vector<const mfem::Vector*> A_directions);

std::tuple<std::vector<const mfem::Vector*>, std::vector<const mfem::Vector*>, std::vector<const mfem::Vector*>>
removeDependentDirectionTriples(std::vector<const mfem::Vector*> directions,
                                std::vector<const mfem::Vector*> A_directions,
                                std::vector<const mfem::Vector*> previous_A_directions);

}  // namespace smith
