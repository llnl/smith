// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <stdio.h>
#include "state.hpp"
#include "gtest/gtest.h"

struct AdaptiveFixture : public ::testing::Test {
  static constexpr size_t costFactor = 1;  // 10;
  // static constexpr size_t S = 200 - costFactor;
  // static constexpr size_t N = 80000 / costFactor; // steps
  static constexpr size_t S = 100;
  static constexpr size_t N = 200;

  size_t cheapCount = 0;
  size_t expensiveCount = 0;

  double advance_cheap(double x)
  {
    cheapCount += costFactor;
    return x * 3.0;
  }

  double advance_expensive(double x)
  {
    ++expensiveCount;
    return x / 3.0 + 2.0;
  }

  gretl::State<double> update_cheap(const gretl::State<double>& a)
  {
    auto b = a.clone({a});

    b.set_eval([&](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
      ++cheapCount;
      output.set(inputs[0].get<double>());
    });

    b.set_vjp([](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
      inputs[0].set_dual(output.get_dual<double>());
    });

    return b.finalize(1.0);
  }

  gretl::State<double> update_expensive(const gretl::State<double>& a)
  {
    auto b = a.clone({a});

    b.set_eval([&](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
      ++expensiveCount;
      output.set(inputs[0].get<double>());
    });

    b.set_vjp([](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
      inputs[0].set_dual(output.get_dual<double>());
    });

    return b.finalize(2.0);
  }
};

TEST_F(AdaptiveFixture, Procedural)
{
  double x0 = 0.0;

  /* for testing */ std::vector<double> states(N + 1);
  /* for testing */ std::vector<double> reverseStates(N + 1);

  gretl::CheckpointManager checkpointManager{.maxNumStates = S};
  std::map<size_t, double> savedCheckpoints;

  savedCheckpoints[0] = x0;
  states[0] = x0;

  std::vector<std::function<double(double)> > advances = {[&](double x) { return advance_cheap(x); },
                                                          [&](double x) { return advance_cheap(x); }};

  double expenseFactor = 2.0;

  bool persistentCheckpoint = true;
  checkpointManager.add_checkpoint_and_get_index_to_remove(0, persistentCheckpoint);
  for (size_t i = 0; i < N; ++i) {
    const auto& xPrev = savedCheckpoints[i];
    int whichAdvance = 0;  // i%2;
    auto x = advances[whichAdvance](xPrev);
    size_t stepToErase =
        checkpointManager.add_checkpoint_and_get_index_to_remove(i + 1, false, whichAdvance ? expenseFactor : 1.0);
    if (checkpointManager.valid_checkpoint_index(stepToErase)) {
      savedCheckpoints.erase(stepToErase);
    }
    savedCheckpoints[i + 1] = x;
    /* for testing */ states[i + 1] = x;
  }

  std::cout << checkpointManager << std::endl;

  for (size_t i_rev = N; i_rev + 1 > 0; --i_rev) {
    for (size_t i = checkpointManager.last_checkpoint_step(); i < i_rev; ++i) {
      const auto& xPrev = savedCheckpoints[i];
      int whichAdvance = 0;  // i%2;
      auto x = advances[whichAdvance](xPrev);
      size_t stepToErase =
          checkpointManager.add_checkpoint_and_get_index_to_remove(i + 1, false, whichAdvance ? expenseFactor : 1.0);
      if (checkpointManager.valid_checkpoint_index(stepToErase)) {
        savedCheckpoints.erase(stepToErase);
      }
      savedCheckpoints[i + 1] = x;
    }

    reverseStates[i_rev] = savedCheckpoints[i_rev];

    checkpointManager.erase_step(i_rev);
    savedCheckpoints.erase(i_rev);
  }

  for (int n = 0; n < N + 1; ++n) {
    ASSERT_EQ(states[n], reverseStates[n]) << n << "\n";
  }

  std::cout << "total cheap count = " << cheapCount << " " << expensiveCount << std::endl;
  // std::cout << "total over cost = " << cheapCount + 12*(expensiveCount) << std::endl;
}

TEST_F(AdaptiveFixture, CountThings)
{
  using State = gretl::State<double>;

  double x0 = 0.0;
  gretl::DataStore dataStore(S);
  State X = dataStore.create_state(x0);

  for (size_t n = 0; n < N; ++n) {
    X = update_cheap(X);
    X = update_cheap(X);
    X = update_cheap(X);
    X = update_expensive(X);
  }
  X = set_as_objective(X);

  dataStore.back_prop();

  std::cout << "counts = " << cheapCount << " " << expensiveCount << std::endl;
}