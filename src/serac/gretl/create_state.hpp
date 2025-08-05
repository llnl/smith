// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file create_state.hpp
 */

#pragma once

#include "data_store.hpp"
#include <functional>

namespace gretl {

// clang-format off

/// @brief 
/// @tparam T 
/// @tparam D 
/// @tparam ZeroFunc 
/// @tparam State0 
/// @tparam ...StatesN 
/// @tparam ...state_indices 
/// @param zeroFunc 
/// @param eval 
/// @param vjp 
/// @param  
/// @param state0 
/// @param ...statesN 
/// @return 
template <typename T, typename D, typename ZeroFunc, typename State0, typename... StatesN, int... state_indices>
gretl::State<T,D> create_state_impl(const ZeroFunc& zeroFunc,
                                    const std::function<T(const typename State0::type&, const typename StatesN::type&...)>& eval,
                                    const std::function<void(const typename State0::type&, const typename StatesN::type&..., const T&,
                                                             typename State0::dual_type&, typename StatesN::dual_type&..., const D&)>& vjp,
                                    std::integer_sequence<int, state_indices...>,
                                    State0 state0, StatesN... statesN)
{
  std::vector<gretl::StateBase> state_bases { { state0, statesN... } };

  auto newState = state0. template create_state<T,D>(state_bases, zeroFunc);

  newState.set_eval([eval](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    const T e = eval(inputs[0].get<typename State0::type, typename State0::dual_type>(),
                     inputs[state_indices+1].get<typename StatesN::type, typename StatesN::dual_type>()...);
    output.set<T,D>(e);
  });

  newState.set_vjp([vjp](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    vjp(inputs[0].get<typename State0::type, typename State0::dual_type>(),
        inputs[state_indices+1].get<typename StatesN::type, typename StatesN::dual_type>()...,
        output.get<T,D>(),
        inputs[0].get_dual<typename State0::dual_type, typename State0::type>(),
        inputs[state_indices+1].get_dual<typename StatesN::dual_type, typename StatesN::type>()...,
        output.get_dual<D,T>());
  });

  return newState.finalize();
}

template <typename T, typename D, typename ZeroFunc, typename State0, typename... StatesN>
gretl::State<T,D> create_state(const ZeroFunc& zeroFunc,
                               const std::function<T(const typename State0::type&, const typename StatesN::type&...)>& eval,
                               const std::function<void(const typename State0::type&, const typename StatesN::type&..., const T&,
                                                        typename State0::dual_type&, typename StatesN::dual_type&..., const D&)>& vjp,
                               State0 state0, StatesN... statesN)
{
  return create_state_impl<T,D>(zeroFunc, eval, vjp, std::make_integer_sequence<int, sizeof...(StatesN)>(), state0, statesN...);
}


template <typename State0, typename... StatesN, int... state_indices>
gretl::State<typename State0::type,typename State0::dual_type> clone_state_impl(const std::function<typename State0::type(const typename State0::type&, const typename StatesN::type&...)>& eval,
                                                              const std::function<void(const typename State0::type&, const typename StatesN::type&..., const typename State0::type&,
                                                                                       typename State0::dual_type&, typename StatesN::dual_type&..., const typename State0::dual_type&)>& vjp,
                                                              std::integer_sequence<int, state_indices...>,
                                                              State0 state0, StatesN... statesN)
{
  using T = typename State0::type;
  using D = typename State0::dual_type;

  std::vector<gretl::StateBase> state_bases { { state0, statesN... } };

  auto newState = state0.clone(state_bases);

  newState.set_eval([eval](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    const T e = eval(inputs[0].get<typename State0::type, typename State0::dual_type>(),
                     inputs[state_indices+1].get<typename StatesN::type, typename StatesN::dual_type>()...);
    output.set<T,D>(e);
  });

  newState.set_vjp([vjp](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    vjp(inputs[0].get<typename State0::type, typename State0::dual_type>(),
        inputs[state_indices+1].get<typename StatesN::type, typename StatesN::dual_type>()...,
        output.get<T,D>(),
        inputs[0].get_dual<typename State0::dual_type, typename State0::type>(),
        inputs[state_indices+1].get_dual<typename StatesN::dual_type, typename StatesN::type>()...,
        output.get_dual<D,T>());
  });

  return newState.finalize();
}

template <typename State0, typename... StatesN>
gretl::State<typename State0::type, typename State0::dual_type>
clone_state(const std::function<typename State0::type(const typename State0::type&, const typename StatesN::type&...)>& eval,
                                                         const std::function<void(const typename State0::type&, const typename StatesN::type&..., const typename State0::type&,
                                                                                  typename State0::dual_type&, typename StatesN::dual_type&..., const typename State0::dual_type&)>& vjp,
                                                         State0 state0, StatesN... statesN)
{
  return clone_state_impl(eval, vjp, std::make_integer_sequence<int, sizeof...(StatesN)>(), state0, statesN...);
}

}