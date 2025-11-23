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

#ifdef SMITH_USE_SLEPC

#include <memory>
#include <optional>
#include <variant>

#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_dual.hpp"

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

/// @brief computes the global size of mfem::Vector
int globalSize(const mfem::Vector& parallel_v, const MPI_Comm& comm);

/// @brief computes the l2 inner product between two mfem::Vector in parallal
double innerProduct(const mfem::Vector& a, const mfem::Vector& b, const MPI_Comm& comm);

/// @brief returns the solution, as well as a list of the N leftmost eigenvectors
/// and their eigenvalues, and the predicted model energy change
std::tuple<mfem::Vector, std::vector<std::shared_ptr<mfem::Vector>>, std::vector<double>, double> solveSubspaceProblem(
    const std::vector<const mfem::Vector*>& directions, const std::vector<const mfem::Vector*>& A_directions,
    const mfem::Vector& b, double delta, int num_leftmost);

std::pair<std::vector<const mfem::Vector*>, std::vector<const mfem::Vector*>> removeDependentDirections(
    std::vector<const mfem::Vector*> directions, std::vector<const mfem::Vector*> A_directions);

}  // namespace smith

#endif  // SMITH_USE_SLEPC
