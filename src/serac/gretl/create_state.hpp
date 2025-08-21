// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file create_state.hpp
 */

#pragma once

#include "data_store.hpp"
#include "upstream_state.hpp"
#include <functional>

namespace gretl {

/// @brief implementation detail for type-safe way to construct a new state
/// @tparam T new primal type
/// @tparam D new dual type
/// @tparam State0 First upstream state type
/// @tparam ...StatesN Variadic list of upstream state types
/// @tparam ...state_indices integer_sequence for help calling the eval function
/// @param zeroFunc std::function which allocates and zeroes a dual D, given a primal T
/// @param eval std::function which evaluates downstream values, given upstream values
/// @param vjp std::function which plus-equals the jacobian-transposed action on the downstream dual, into the upstream
/// duals
/// @param state0 First upstream state
/// @param ...statesN Variadic list of upstream states
template <typename T, typename D, typename State0, typename... StatesN, int... state_indices>
gretl::State<T, D> create_state_impl(
    const InitializeZeroDual<T, D>& zeroFunc,
    const std::function<T(const typename State0::type&, const typename StatesN::type&...)>& eval,
    const std::function<void(const typename State0::type&, const typename StatesN::type&..., const T&,
                             typename State0::dual_type&, typename StatesN::dual_type&..., const D&)>& vjp,
    std::integer_sequence<int, state_indices...>, State0 state0, StatesN... statesN)
{
  std::vector<gretl::StateBase> state_bases{{state0, statesN...}};

  auto newState = state0.template create_state<T, D>(state_bases, zeroFunc);

  newState.set_eval([eval](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    const T e =
        eval(inputs[0].get<typename State0::type>(), inputs[state_indices + 1].get<typename StatesN::type>()...);
    output.set<T, D>(e);
  });

  newState.set_vjp([vjp](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    vjp(inputs[0].get<typename State0::type>(), inputs[state_indices + 1].get<typename StatesN::type>()...,
        output.get<T>(), inputs[0].get_dual<typename State0::dual_type, typename State0::type>(),
        inputs[state_indices + 1].get_dual<typename StatesN::dual_type, typename StatesN::type>()...,
        output.get_dual<D, T>());
  });

  return newState.finalize();
}

/// @brief create a new state of type T, given forward evals with the type information, and reverse jvps with the type
/// information
/// @tparam State0 First upstream state type
/// @tparam ...StatesN Variadic list of upstream state types
/// @param zeroFunc std::function which allocates and zeroes a dual D, given a primal T
/// @param eval std::function which evaluates downstream values, given upstream values
/// @param vjp std::function which plus-equals the jacobian-transposed action on the downstream dual, into the upstream
/// duals
/// @param state0 First upstream state
/// @param statesN
template <typename T, typename D, typename State0, typename... StatesN>
gretl::State<T, D> create_state(
    const InitializeZeroDual<T, D>& zeroFunc,
    const std::function<T(const typename State0::type&, const typename StatesN::type&...)>& eval,
    const std::function<void(const typename State0::type&, const typename StatesN::type&..., const T&,
                             typename State0::dual_type&, typename StatesN::dual_type&..., const D&)>& vjp,
    State0 state0, StatesN... statesN)
{
  return create_state_impl<T, D>(zeroFunc, eval, vjp, std::make_integer_sequence<int, sizeof...(StatesN)>(), state0,
                                 statesN...);
}

/// @brief implementation detail for type-safe way to clone a new state from an existing state
/// @tparam State0 First upstream state type, also the type of the state to be cloned
/// @tparam ...StatesN Variadic list of upstream state types
/// @tparam ...state_indices integer_sequence for help calling the eval function
/// @param eval std::function which evaluates downstream values, given upstream values
/// @param vjp std::function which plus-equals the jacobian-transposed action on the downstream dual, into the upstream
/// duals
/// @param state0 First upstream state
/// @param ...statesN Variadic list of upstream states

template <typename State0, typename... StatesN, int... state_indices>
gretl::State<typename State0::type, typename State0::dual_type> clone_state_impl(
    const std::function<typename State0::type(const typename State0::type&, const typename StatesN::type&...)>& eval,
    const std::function<void(const typename State0::type&, const typename StatesN::type&...,
                             const typename State0::type&, typename State0::dual_type&, typename StatesN::dual_type&...,
                             const typename State0::dual_type&)>& vjp,
    std::integer_sequence<int, state_indices...>, State0 state0, StatesN... statesN)
{
  using T = typename State0::type;
  using D = typename State0::dual_type;

  std::vector<gretl::StateBase> state_bases{{state0, statesN...}};

  auto newState = state0.clone(state_bases);

  newState.set_eval([eval](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    const T e =
        eval(inputs[0].get<typename State0::type>(), inputs[state_indices + 1].get<typename StatesN::type>()...);
    output.set<T, D>(e);
  });

  newState.set_vjp([vjp](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    vjp(inputs[0].get<typename State0::type>(), inputs[state_indices + 1].get<typename StatesN::type>()...,
        output.get<T>(), inputs[0].get_dual<typename State0::dual_type, typename State0::type>(),
        inputs[state_indices + 1].get_dual<typename StatesN::dual_type, typename StatesN::type>()...,
        output.get_dual<D, T>());
  });

  return newState.finalize();
}

/// @brief clone a new state from state0, given forward evals with the type information, and reverse jvps with the type
/// information
/// @tparam State0 First upstream state type
/// @tparam ...StatesN Variadic list of upstream state types
/// @param eval std::function which evaluates downstream values, given upstream values
/// @param vjp std::function which plus-equals the jacobian-transposed action on the downstream dual, into the upstream
/// duals
/// @param state0 First upstream state
/// @param ...statesN Variadic list of upstream states
template <typename State0, typename... StatesN>
gretl::State<typename State0::type, typename State0::dual_type> clone_state(
    const std::function<typename State0::type(const typename State0::type&, const typename StatesN::type&...)>& eval,
    const std::function<void(const typename State0::type&, const typename StatesN::type&...,
                             const typename State0::type&, typename State0::dual_type&, typename StatesN::dual_type&...,
                             const typename State0::dual_type&)>& vjp,
    State0 state0, StatesN... statesN)
{
  return clone_state_impl(eval, vjp, std::make_integer_sequence<int, sizeof...(StatesN)>(), state0, statesN...);
}

}  // namespace gretl
