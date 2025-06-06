// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state.hpp
 */

#pragma once

#include <functional>
#include "state_base.hpp"
#include "upstream_state.hpp"

namespace gretl {

template <typename T, typename D>
struct State : public StateBase {
  using type = T;
  using dual_type = D;

  inline void set(const T& t) { get_data().set_primal(t); }

  inline void set(T&& t) { get_data().set_primal(std::move(t)); }

  inline void set_dual(const D& d) { get_data().set_dual(d); }

  inline void set_dual(D&& d) { get_data().set_dual(std::move(d)); }

  inline const T& get() const { return get_data().get_primal(); }

  inline const D& get_dual() const { return get_data().get_dual(); }

  State<T, D> clone(const std::vector<StateBase>& upstreams) const { return get_data().clone(upstreams); }

  void set_eval(const std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)>& e)
  {
    get_data().eval = e;
  }

  void set_vjp(const std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)>& v)
  {
    get_data().vjp = v;
  }

  State<T, D> finalize()
  {
    this->evaluate_and_remove_disposable_checkpoints();
    return *this;
  }

  friend class DataStore;

 protected:
  template <typename InitDualFromValue>
  State(DataStore& store, const T& t, size_t step, InitDualFromValue initialize_zero_dual,
        const std::vector<StateBase>& ustreams)
  {
    stateData = std::make_shared<StateData<T, D>>(store, t, step, initialize_zero_dual, ustreams);
  }

  template <typename InitDualFromValue>
  State(DataStore& store, size_t step, InitDualFromValue initialize_zero_dual, const std::vector<StateBase>& ustreams)
  {
    stateData = std::make_shared<StateData<T, D>>(store, step, initialize_zero_dual, ustreams);
  }

  inline StateData<T, D>& get_data() const
  {
    auto typedData = std::dynamic_pointer_cast<StateData<T, D>>(stateData);
    gretl_assert(typedData);
    return *typedData;
  }
};

inline State<double> set_as_objective(State<double> o)
{
  o.set_dual(1.0);
  return o;
}

}  // namespace gretl