// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file data_store_for_testing.hpp
 */

#pragma once

#include "data_store.hpp"
#include "vector_state.hpp"

namespace gretl {

struct DataStoreForTesting : public DataStore {
  DataStoreForTesting(size_t maxStates) : DataStore(maxStates) {}

  // reverse back a single state, updating the duals along the way
  StateBase reverse_state(size_t n) override { return DataStore::reverse_state(n); };
};

inline double rand_in_range(double x0, double xf)
{
  return x0 + static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * (xf - x0);
}

void check_array_gradients(gretl::State<double>& objectiveState, gretl::VectorState& inputState,
                           gretl::DataStore& dataStore, double eps, double tol)
{
  srand(5);

  dataStore.reset();
  double objectiveBase = objectiveState.get();

  auto pert = inputState.get();

  const size_t S = pert.size();

  for (size_t i = 0; i < S; ++i) {
    pert[i] = rand_in_range(-1.0, 1.0);
  }

  auto s0 = inputState.get();
  for (size_t i = 0; i < S; ++i) {
    s0[i] += eps * pert[i];
  }
  inputState.set(s0);

  dataStore.reset();
  double objectivePlus = objectiveState.get();

  auto grad = inputState.get_dual();

  double directionDeriv = 0.0;
  for (size_t i = 0; i < S; ++i) {
    directionDeriv += pert[i] * grad[i];
  }

  EXPECT_NEAR(directionDeriv, (objectivePlus - objectiveBase) / eps, tol);

  for (size_t i = 0; i < S; ++i) {
    s0[i] -= eps * pert[i];
  }
  inputState.set(s0);
}

}  // namespace gretl