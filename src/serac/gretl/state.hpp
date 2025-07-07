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

using Int = unsigned int;

template <typename T, typename D>
struct State : public StateBase {
  using type = T;
  using dual_type = D;

  inline void set(const T& t) { dataStore_->set_primal(step_, t); }

  inline const T& get() const { return dataStore_->get_primal<T>(step_); }

  inline const D& get_dual() const { return dataStore_->get_dual<D>(step_); }

  void set_eval(const std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)>& e)
  {
    dataStore_->evals_[step_] = e;
  }

  void set_vjp(const std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)>& v)
  {
    dataStore_->vjps_[step_] = v;
  }

  State<T, D> clone(const std::vector<StateBase>& upstreams) const
  {
    gretl_assert(!upstreams.empty());
    auto newVal = std::make_shared<std::any>(*std::any_cast<T>(primal_.get()));
    State<T, D> state(dataStore_, dataStore_->states_.size(), newVal, initialize_zero_dual_);
    dataStore_->add_state(std::make_unique<State<T, D>>(state), upstreams);
    return state;
  }

  State<T, D> finalize()
  {
    this->evaluate_forward();
    return *this;
  }

  friend class DataStore;

 protected:
  State(DataStore* store, size_t step, std::shared_ptr<std::any> val,
        const InitializeZeroDual<T, D>& initialize_zero_dual)
      : StateBase(store, val), initialize_zero_dual_(initialize_zero_dual)
  {
    step_ = static_cast<Int>(step);
  }

  InitializeZeroDual<T, D> initialize_zero_dual_;
};

inline State<double> set_as_objective(State<double> o)
{
  o.set_dual(1.0);
  return o;
}

}  // namespace gretl