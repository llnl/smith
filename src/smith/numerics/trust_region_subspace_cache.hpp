// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "smith/numerics/trust_region_solver.hpp"

namespace smith {

/// Cached reduced trust-region subspace problem reusable across trust-region radius updates.
struct TrustRegionSubspaceCache {
  /// @brief prepares reduced trust-region subspace data reusable across trust-region radius updates
  void prepare(const std::vector<const mfem::Vector*>& directions, const std::vector<const mfem::Vector*>& A_directions,
               const mfem::Vector& b, int num_leftmost, MPI_Comm comm = MPI_COMM_WORLD);

  /// @brief solves cached reduced trust-region problem for given trust-region radius
  TrustRegionSubspaceResult solve(double delta) const;

  /// zero vector with correct size/layout for empty-subspace returns
  mfem::Vector zero_solution;
  /// orthonormalized physical-space basis spanning reduced subspace
  std::vector<std::shared_ptr<mfem::Vector>> basis;
  /// reduced Hessian in cached subspace basis
  mfem::DenseMatrix projected_hessian;
  /// reduced right-hand side in cached subspace basis
  mfem::Vector projected_rhs;
  /// eigenvalues of reduced Hessian
  mfem::Vector eigenvalues;
  /// eigenvectors of reduced Hessian
  mfem::DenseMatrix eigenvectors;
  /// reduced right-hand side projected onto reduced Hessian eigenvectors
  mfem::Vector eigen_rhs;
  /// cached leftmost eigenvectors lifted back to physical space
  std::vector<std::shared_ptr<mfem::Vector>> leftmosts;
  /// eigenvalues corresponding to cached leftmost eigenvectors
  std::vector<double> leftvals;
};

}  // namespace smith
