// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <array>
#include <functional>
#include "gtest/gtest.h"
#include "serac/gretl/checkpoint.hpp"
#include "serac/gretl/state.hpp"
#include "serac/gretl/test_utils.hpp"
#include "serac/gretl/vector_state.hpp"

static constexpr size_t numParams = 4;

using Param = gretl::VectorState;
using State = gretl::VectorState;

State rk4(const State& state0, double time, double dt, std::function<State(State, double)> ode_rate)
{
  double tau = dt / 6.;

  State k1 = ode_rate(state0, time);
  State state1 = state0 + tau * k1;

  State k2 = ode_rate(state0 + (0.5 * dt) * k1, time + 0.5 * dt);
  State state2 = state1 + (2 * tau) * k2;

  State k3 = ode_rate(state0 + (0.5 * dt) * k2, time + 0.5 * dt);
  State state3 = state2 + (2 * tau) * k3;

  State k4 = ode_rate(state0 + dt * k3, time + dt);
  return state3 + tau * k4;
}

State state_rate_equation(const State& state, const Param& params, [[maybe_unused]] double time)
{
  auto newState = state.clone(std::vector<gretl::StateBase>{state, params});

  newState.set_eval([](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    const State::type& s = inputs[0].get<State::type>();
    const Param::type& p = inputs[1].get<Param::type>();

    State::type sNew = s;
    sNew[0] = p[0] * (s[1] - s[0]);
    sNew[1] = s[0] * (p[1] - s[2]) - s[1];
    sNew[2] = s[0] * s[1] - p[2] * s[2];
    output.set(std::move(sNew));
  });

  newState.set_vjp([](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    auto state_ = inputs[0];
    auto params_ = inputs[1];

    const State::type& s = state_.get<State::type>();
    const Param::type& p = params_.get<Param::type>();
    const State::type& sNewBar = output.get_dual<State::type>();

    State::dual_type& sBar = state_.get_dual<State::dual_type, State::type>();
    sBar.resize(s.size());
    sBar[0] += -p[0] * sNewBar[0] + (p[1] - s[2]) * sNewBar[1] + s[1] * sNewBar[2];
    sBar[1] += p[0] * sNewBar[0] - sNewBar[1] + s[0] * sNewBar[2];
    sBar[2] += -s[0] * sNewBar[1] - p[2] * sNewBar[2];

    Param::dual_type& pBar = params_.get_dual<Param::dual_type, Param::type>();
    pBar.resize(p.size());
    pBar[0] += (s[1] - s[0]) * sNewBar[0];
    pBar[1] += s[0] * sNewBar[1];
    pBar[2] -= s[2] * sNewBar[2];
  });

  return newState.finalize();
}

class MeshFixture : public ::testing::Test {
 public:
  void SetUp() { dataStore = std::make_shared<gretl::DataStore>(20); }

  Param::type params_data{1.3, 3.5, 1.1, 0.0};
  State::type state0_data{1.7, 1.1, 0.1};
  State::type state1_data{-1.3, -2.7, 0.4};
  State::type ones_data{1.0, 1.0, 1.0};

  std::shared_ptr<gretl::DataStore> dataStore;

  double totalTime = 0.01;
  size_t N = 20;
  double dt = totalTime / static_cast<double>(N);
};

TEST_F(MeshFixture, NonlinearGraphGradients)
{
  Param params = dataStore->create_state(params_data, gretl::vec::initialize_zero_dual);
  State state0 = dataStore->create_state(state0_data, gretl::vec::initialize_zero_dual);

  State stateRate = state_rate_equation(state0, params, 0.0);
  gretl::State<double> rateNorm = set_as_objective(gretl::inner_product(stateRate, stateRate));

  dataStore->back_prop();

  double constexpr eps = 1e-7;
  check_array_gradients(rateNorm, {state0, params}, {eps, eps}, {40 * eps, 40 * eps});
}

TEST_F(MeshFixture, Dynamics)
{
  Param params = dataStore->create_state(params_data, gretl::vec::initialize_zero_dual);
  State state0 = dataStore->create_state(state0_data, gretl::vec::initialize_zero_dual);

  State state = copy(state0);
  for (size_t i = 0; i < N; ++i) {
    double i_double = static_cast<double>(i);
    state = rk4(state, i_double * dt, dt,
                [params](const State& curState, double time) { return state_rate_equation(curState, params, time); });
  }

  gretl::State<double> stateNorm = set_as_objective(gretl::inner_product(state, state));
  dataStore->back_prop();

  for (size_t i = 0; i < numParams; ++i) {
    std::cout << "param sensitivity = " << params.get_dual()[i] << std::endl;
  }

  double constexpr eps = 1e-7;
  check_array_gradients(stateNorm, {state0, params}, {eps, eps}, {40 * eps, 40 * eps});
}
