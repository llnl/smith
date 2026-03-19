// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/nonlinear_convergence.hpp"

#include <algorithm>
#include <cmath>

#include "axom/fmt.hpp"
#include "axom/slic.hpp"

namespace smith {

namespace {

double residualNorm(const mfem::Vector& residual, MPI_Comm comm)
{
#ifdef MFEM_USE_MPI
  return mfem::ParNormlp(residual, 2.0, comm);
#else
  static_cast<void>(comm);
  return residual.Norml2();
#endif
}

bool hasPerBlockTolerances(const BlockConvergenceTolerances& tolerances)
{
  return !tolerances.relative_tols.empty() || !tolerances.absolute_tols.empty();
}

}  // namespace

void NonlinearConvergenceContext::reset()
{
  initial_global_norm = -1.0;
  initial_block_norms.clear();
}

std::vector<double> expandPerBlockTolerances(const std::vector<double>& block_tols, size_t num_blocks,
                                             double empty_value, const std::string& tol_name)
{
  if (block_tols.empty()) {
    return std::vector<double>(num_blocks, empty_value);
  }

  SLIC_ERROR_IF(block_tols.size() != num_blocks,
                axom::fmt::format("{} size {} does not match number of residual blocks {}", tol_name, block_tols.size(),
                                  num_blocks));
  return block_tols;
}

std::vector<double> computeResidualBlockNorms(const std::vector<mfem::Vector>& residuals, MPI_Comm comm)
{
  std::vector<double> block_norms(residuals.size(), 0.0);
  for (size_t i = 0; i < residuals.size(); ++i) {
    block_norms[i] = residualNorm(residuals[i], comm);
  }
  return block_norms;
}

std::vector<double> computeResidualBlockNorms(const mfem::Vector& residual, const std::vector<int>& block_offsets,
                                              MPI_Comm comm)
{
  if (block_offsets.empty()) {
    return {residualNorm(residual, comm)};
  }

  SLIC_ERROR_IF(block_offsets.size() < 2, "Block offsets must contain at least two entries");
  SLIC_ERROR_IF(block_offsets.front() != 0, "Block offsets must start at zero");
  SLIC_ERROR_IF(
      block_offsets.back() != residual.Size(),
      axom::fmt::format("Block offsets end at {}, but residual size is {}", block_offsets.back(), residual.Size()));

  size_t num_blocks = block_offsets.size() - 1;
  std::vector<double> local_squared_norms(num_blocks, 0.0);
  for (size_t i = 0; i < num_blocks; ++i) {
    for (int j = block_offsets[i]; j < block_offsets[i + 1]; ++j) {
      local_squared_norms[i] += residual(j) * residual(j);
    }
  }

#ifdef MFEM_USE_MPI
  std::vector<double> global_squared_norms(num_blocks, 0.0);
  MPI_Allreduce(local_squared_norms.data(), global_squared_norms.data(), static_cast<int>(num_blocks), MPI_DOUBLE,
                MPI_SUM, comm);
  local_squared_norms = std::move(global_squared_norms);
#else
  static_cast<void>(comm);
#endif

  for (auto& squared_norm : local_squared_norms) {
    squared_norm = std::sqrt(squared_norm);
  }

  return local_squared_norms;
}

ConvergenceStatus evaluateResidualConvergence(double tolerance_multiplier, double abs_tol, double rel_tol,
                                              const std::vector<double>& block_abs_tols,
                                              const std::vector<double>& block_rel_tols, bool block_path_enabled,
                                              const std::vector<double>& block_norms,
                                              NonlinearConvergenceContext& context)
{
  ConvergenceStatus status;
  status.block_path_enabled = block_path_enabled;
  status.block_norms = block_norms;
  status.block_goals.resize(block_norms.size(), 0.0);

  if (context.initial_block_norms.empty()) {
    context.initial_block_norms.assign(block_norms.size(), -1.0);
  }

  SLIC_ERROR_IF(
      context.initial_block_norms.size() != block_norms.size(),
      axom::fmt::format("Stored initial block residual count {} does not match current residual block count {}",
                        context.initial_block_norms.size(), block_norms.size()));

  double global_norm_squared = 0.0;
  for (size_t i = 0; i < block_norms.size(); ++i) {
    global_norm_squared += block_norms[i] * block_norms[i];
    if (context.initial_block_norms[i] < 0.0) {
      context.initial_block_norms[i] = block_norms[i];
    }
  }

  status.global_norm = std::sqrt(global_norm_squared);
  if (context.initial_global_norm < 0.0) {
    context.initial_global_norm = status.global_norm;
  }
  status.global_goal = std::max(abs_tol, rel_tol * context.initial_global_norm);
  status.global_converged = status.global_norm <= tolerance_multiplier * status.global_goal;

  if (block_path_enabled) {
    status.block_converged = true;
    for (size_t i = 0; i < block_norms.size(); ++i) {
      status.block_goals[i] = std::max(block_abs_tols[i], block_rel_tols[i] * context.initial_block_norms[i]);
      if (block_norms[i] > tolerance_multiplier * status.block_goals[i]) {
        status.block_converged = false;
      }
    }
  }

  status.converged = status.global_converged || (status.block_path_enabled && status.block_converged);
  return status;
}

EquationSolverConvergenceManager::EquationSolverConvergenceManager(MPI_Comm comm, double abs_tol, double rel_tol)
    : comm_(comm), abs_tol_(abs_tol), rel_tol_(rel_tol)
{
}

void EquationSolverConvergenceManager::reset() const { context_.reset(); }

void EquationSolverConvergenceManager::setBlockData(const std::vector<int>& block_offsets,
                                                    BlockConvergenceTolerances block_tolerances) const
{
  block_offsets_ = block_offsets;
  block_tolerances_ = std::move(block_tolerances);
}

bool EquationSolverConvergenceManager::blockPathEnabled() const { return hasPerBlockTolerances(block_tolerances_); }

ConvergenceStatus EquationSolverConvergenceManager::evaluate(double tolerance_multiplier,
                                                             const mfem::Vector& residual) const
{
  auto block_norms = computeResidualBlockNorms(residual, block_offsets_, comm_);
  auto block_abs_tols =
      expandPerBlockTolerances(block_tolerances_.absolute_tols, block_norms.size(), 0.0, "absolute block tolerances");
  auto block_rel_tols =
      expandPerBlockTolerances(block_tolerances_.relative_tols, block_norms.size(), 0.0, "relative block tolerances");

  return evaluateResidualConvergence(tolerance_multiplier, abs_tol_, rel_tol_, block_abs_tols, block_rel_tols,
                                     blockPathEnabled(), block_norms, context_);
}

}  // namespace smith
