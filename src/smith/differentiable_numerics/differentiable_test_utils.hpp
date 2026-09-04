// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_test_utils.hpp
 *
 * @brief Utility functions for testing.
 */

#pragma once

#include "gretl/double_state.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"

namespace smith {

/// @brief testing utility to confirm order of convergence of the finite differences relative to the backprop gradient
inline auto checkGradients(const gretl::State<double>& objectiveState, FieldState& inputState,
                           FiniteElementDual& inputDual, double objectiveBase, gretl::DataStore& dataStore, double eps)
{
  smith::FiniteElementState inputSave(*inputState.get());
  dataStore.reset();
  smith::FiniteElementState& input = *inputState.get();
  smith::FiniteElementState pert(input.space(), input.name() + "_pert");

  int sz = pert.Size();
  for (int i = 0; i < sz; ++i) {
    pert[i] = -1.2 + 2.02 * (double(i) / sz);
    input[i] += eps * pert[i];
  }

  double objectivePlus = objectiveState.get();

  double directionDeriv = 0.0;
  for (int i = 0; i < sz; ++i) {
    directionDeriv += pert[i] * inputDual[i];
  }

  *inputState.get() = inputSave;

  return std::make_pair(directionDeriv, (objectivePlus - objectiveBase) / eps);
}

/// @brief testing utility to confirm order of convergence of the finite differences relative to the backprop gradient
inline auto checkGradients(const gretl::State<double>& objectiveState, gretl::State<double, double>& inputState,
                           double& inputDual, double objectiveBase, gretl::DataStore& dataStore, double eps)
{
  double inputSave = inputState.get();
  dataStore.reset();
  inputState.set(inputSave + eps);
  double objectivePlus = objectiveState.get();
  inputState.set(inputSave);
  return std::make_pair(inputDual, (objectivePlus - objectiveBase) / eps);
}

/// @brief Testing utility function which runs a gretl graph num_fd_steps (with increasingly smaller finite difference
/// steps) to check if the computed graph gradients are converging to the finite differenced gradients at the expected
/// rate
inline double checkGradWrt(const gretl::State<double>& objective, smith::FieldState& input, double eps,
                           size_t num_fd_steps = 4, bool printmore = false)
{
  auto& graph = objective.data_store();

  // reset each time, just to be sure
  graph.reset();

  // re-evaluate the final objective value
  double objectiveBase = objective.get();

  // back-propagate to get sensitivity wrt input states
  gretl::set_as_objective(objective);
  graph.back_prop();

  auto dual_vec = input.get_dual();

  std::vector<double> grad_errors;
  auto [grad, grad_fd] = checkGradients(objective, input, *dual_vec, objectiveBase, graph, eps);
  if (printmore) std::cout << "grad    = " << grad << "\ngrad fd = " << grad_fd << std::endl;
  grad_errors.push_back(std::abs(grad - grad_fd));

  for (size_t step = 1; step < num_fd_steps; ++step) {
    eps /= 2;
    std::tie(grad, grad_fd) = checkGradients(objective, input, *dual_vec, objectiveBase, graph, eps);
    if (printmore) std::cout << "grad    = " << grad << "\ngrad fd = " << grad_fd << std::endl;
    grad_errors.push_back(std::abs(grad - grad_fd));
  }

  for (size_t step = 0; step < num_fd_steps; ++step) {
    std::cout << "grad error " << step << " = " << grad_errors[step] << std::endl;
  }

  if (num_fd_steps >= 2) {
    return std::log2(grad_errors[0] / grad_errors[num_fd_steps - 1]) / static_cast<double>(num_fd_steps - 1);
  }

  return 0;
};

/// @brief Testing utility function which runs a gretl graph num_fd_steps (with increasingly smaller finite difference
/// steps) to check if the computed graph gradients are converging to the finite differenced gradients at the expected
/// rate
inline double checkGradWrt(const gretl::State<double>& objective, gretl::State<double, double>& input, double eps,
                           size_t num_fd_steps = 4, bool printmore = false)
{
  auto& graph = objective.data_store();

  // reset each time, just to be sure
  graph.reset();

  // re-evaluate the final objective value
  double objectiveBase = objective.get();

  // back-propagate to get sensitivity wrt input states
  gretl::set_as_objective(objective);
  graph.back_prop();

  auto dual = input.get_dual();

  std::vector<double> grad_errors;
  auto [grad, grad_fd] = checkGradients(objective, input, dual, objectiveBase, graph, eps);
  grad_errors.push_back(std::abs(grad - grad_fd));

  for (size_t step = 1; step < num_fd_steps; ++step) {
    eps /= 2;
    std::tie(grad, grad_fd) = checkGradients(objective, input, dual, objectiveBase, graph, eps);
    if (printmore) std::cout << "grad    = " << grad << "\ngrad fd = " << grad_fd << std::endl;
    grad_errors.push_back(std::abs(grad - grad_fd));
  }

  for (size_t step = 0; step < num_fd_steps; ++step) {
    std::cout << "grad error " << step << " = " << grad_errors[step] << std::endl;
  }

  if (num_fd_steps >= 2) {
    return std::log2(grad_errors[0] / grad_errors[num_fd_steps - 1]) / static_cast<double>(num_fd_steps - 1);
  }

  return 0;
};

}  // namespace smith
