// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state.hpp
 */

#pragma once

#include <functional>
#include "upstream_state.hpp"
#include "state_base.hpp"
#include "upstream_state.hpp"

namespace gretl {

using Int = unsigned int;

/// @brief Templated State
/// @tparam T Primal type
/// @tparam D Dual type
template <typename T, typename D>
struct State : public StateBase {
  using type = T;       ///< type
  using dual_type = D;  ///< dual_type

  /// @brief Set primal value of correct type
  inline void set(const T& t) { data_store().set_primal(step(), t); }

  /// @brief Get primal value of correct type
  inline const T& get() const { return data_store().template get_primal<T>(step()); }

  /// @brief Get dual value of correct type
  inline const D& get_dual() const { return data_store().template get_dual<D, T>(step()); }

  /// @brief Set the std::functions which evaluates downstream primals given upstream primals
  void set_eval(const std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)>& e)
  {
    data_store().evals_[step()] = e;
  }

  /// @brief Set the std::functions which computes the action of the jacobian transpose on the downstream dual, and
  /// plus-equals into the upstream duals.
  void set_vjp(const std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)>& v)
  {
    data_store().vjps_[step()] = v;
  }

  /// @brief Helper function to clone an existing state (keeping its type)
  /// @param upstreams The upstream dependencies for this new state
  State<T, D> clone(const std::vector<StateBase>& upstreams) const
  {
    gretl_assert(!upstreams.empty());
    auto primal_ptr = primal().get();
    gretl_assert(primal_ptr);
    std::shared_ptr<std::any> new_val;
    if (primal_ptr) {
      new_val = std::make_shared<std::any>(*std::any_cast<T>(primal_ptr));
    }
    State<T, D> state(&data_store(), data_store().states_.size(), new_val, initialize_zero_dual_);
    data_store().add_state(std::make_unique<State<T, D>>(state), upstreams);
    return state;
  }

  /// @brief After calling set_eval and set_vjp, this actually computes the set_eval.  Typically this is required when
  /// constructing a new state.
  State<T, D> finalize()
  {
    this->evaluate_forward();
    return *this;
  }

  friend class DataStore;

 protected:
  /// @brief Protected constructor for states.  This is called by the DataStore when registering a new state on the
  /// graph.
  /// @param store datastore
  /// @param step step
  /// @param val type-erased value which is the data for the state
  /// @param initialize_zero_dual std::function which takes a primal value type T, and returns a zeroed out, but memory
  /// allocated dual type D
  State(DataStore* store, size_t step, std::shared_ptr<std::any> val,
        const InitializeZeroDual<T, D>& initialize_zero_dual)
      : StateBase(store, val), initialize_zero_dual_(initialize_zero_dual)
  {
    reset_step(static_cast<Int>(step));
  }

  InitializeZeroDual<T, D> initialize_zero_dual_;  ///< std::function which initializes and zeroes a dual value of type
                                                   ///< D, given a primal value of type T.
};

/// @brief Sets the dual value for a 'double' state to 1.  This is equivalent to saying this state appears additively in
/// the objective or constraint of an optimization problem
/// @param o state to be made into an objective
inline State<double> set_as_objective(State<double> o)
{
  o.set_dual(1.0);
  o.data_store().stillConstructingGraph_ = false;
  o.data_store().currentStep_ = o.data_store().size();
  gretl_assert_msg(o.step() == o.data_store().currentStep_ - 1,
                   "Only the last state on the graph can be set as the objective");
  return o;
}

}  // namespace gretl
