// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_utils.hpp
 */

#pragma once

#include "gtest/gtest.h"
#include "data_store.hpp"
#include "vector_state.hpp"

namespace gretl {

/// @brief Computes a randon number in a specified range
/// @param x0 The lower bound of the range
/// @param xf The upper bound of the range
/// @return A random double
inline double rand_in_range(double x0, double xf)
{
  return x0 + static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * (xf - x0);
}

/// @brief Performs a gradient check
/// @param objectiveState The double state corresponding to the objective
/// @param inputStates The persistent states in the graph which can be perturbed for finite differencing
/// @param eps Vector of finite difference pertubations (one per input state)
/// @param tol Vector of tolerances for finite difference check (one per input state)
void check_array_gradients(gretl::State<double>& objectiveState, std::vector<gretl::VectorState> inputStates,
                           std::vector<double> eps, std::vector<double> tol)
{
  double objectiveBase = objectiveState.get();
  srand(5);

  size_t num_inputs = inputStates.size();
  gretl_assert(num_inputs == eps.size());
  gretl_assert(num_inputs == tol.size());
  std::vector<std::vector<double> > perturbedInputs(num_inputs);
  std::vector<double> directionalDerivs(num_inputs);

  for (size_t iInput = 0; iInput < num_inputs; ++iInput) {
    auto& inputState = inputStates[iInput];
    auto& perturbedInput = perturbedInputs[iInput];

    auto grad = inputState.get_dual();
    auto pert = inputState.get();
    const size_t S = pert.size();
    for (size_t i = 0; i < S; ++i) {
      pert[i] = rand_in_range(-1.0, 1.0);
    }
    double directionDeriv = 0.0;
    for (size_t i = 0; i < S; ++i) {
      directionDeriv += pert[i] * grad[i];
    }
    directionalDerivs[iInput] = directionDeriv;
    perturbedInput = inputState.get();
    for (size_t i = 0; i < S; ++i) {
      perturbedInput[i] += eps[iInput] * pert[i];
    }
  }

  for (size_t iInput = 0; iInput < num_inputs; ++iInput) {
    auto& inputState = inputStates[iInput];
    auto& perturbedInput = perturbedInputs[iInput];

    auto s0 = inputState.get();
    objectiveState.data_store().reset();
    inputState.set(perturbedInput);
    double objectivePlus = objectiveState.get();
    EXPECT_NEAR(directionalDerivs[iInput], (objectivePlus - objectiveBase) / eps[iInput], tol[iInput]);
    inputState.set(s0);
  }
}

}  // namespace gretl
