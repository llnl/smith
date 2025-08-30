// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file data_store.hpp
 */

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <functional>
#include <memory>
#include <any>
#include "checkpoint.hpp"
#include "print_utils.hpp"

#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace gretl {

using Int = unsigned int;  ///< gretl Int type

struct StateBase;

template <typename T, typename D = T>
struct State;

struct UpstreamStates;

struct DownstreamState;

/// @brief ZeroDual function type
template <typename T, typename D = T>
using InitializeZeroDual = std::function<D(const T&)>;

/// @brief Default zero initializer,
template <typename T, typename D = T>
struct defaultInitializeZeroDual {
  /// @brief functor operator
  D operator()(const T&) { return D{}; }
};

/// @brief DataStore class hold onto states, duals and additional information to represent a computational graph, its
/// checkpointing state information, and its backpropagated sensitivities
class DataStore {
 public:
  /// @brief Constructor
  /// @param maxStates maximum number of states the users is allowing to be allocated for the dynamic checkpointing.
  /// This does not include persistent states, nor states held in scope by the user.
  DataStore(size_t maxStates);

  /// @brief virtual destructor
  virtual ~DataStore() {}

  /// @brief create a new state in the graph, store it, return it
  template <typename T, typename D = T>
  State<T, D> create_state(
      const T& t, InitializeZeroDual<T, D> initial_zero_dual = [](const T&) { return D{}; })
  {
    State<T, D> state(this, states_.size(), std::make_shared<std::any>(t), initial_zero_dual);
    add_state(std::make_unique<State<T, D>>(state), {});
    return state;
  }

  /// @brief  unwind one step of the graph
  virtual void reverse_state();

  /// @brief unwind the entire graph
  void back_prop();

  /// @brief clear all but persistent state, keeping the graph. Returns the number of persistent states.
  void reset();

  /// @brief reevaluates the final state and refills checkpoints to get ready for another back propagation
  void reset_for_backprop();

  /// @brief clear all but persistent state, remove the graph
  void reset_graph();

  /// @brief resize data structures
  void resize(Int newSize);

  /// @brief get total number of states in the graph
  Int size() { return static_cast<Int>(states_.size()); }

  /// @brief print all checkpoint data in data store
  void print_graph() const;

  /// @brief do internal checks of consistency with respect to checkpoints and
  bool check_validity() const;

  /// @brief create a new state in the graph, store it, return it
  template <typename T, typename D>
  State<T, D> create_empty_state(InitializeZeroDual<T, D> initial_zero_dual, const std::vector<StateBase>& upstreams)
  {
    gretl_assert(!upstreams.empty());
    auto t = std::make_shared<std::any>(T{});
    State<T, D> state(this, states_.size(), t, initial_zero_dual);
    add_state(std::make_unique<State<T, D>>(state), upstreams);
    return state;
  }

  /// @brief vjp
  void vjp(StateBase& state);

  /// @brief function for safely adding new states to graph and checkpoint
  void add_state(std::unique_ptr<StateBase> newState, const std::vector<StateBase>& upstreams);

  /// @brief method for fetching states at a particular step
  void fetch_state_data(Int);

  /// @brief erase the data for a particular step
  void erase_step_state_data(Int);

  /// @brief clear usage at a particular step
  void clear_usage(Int step);

  /// @brief std::function for evaluating downstream from upstreams
  using EvalT = std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)>;

  /// @brief std::function for computing vector-jacobian product from downstream dual to upstream duals
  using VjpT = std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)>;

  /// @brief Get the primal data as a shared_ptr to std::any (type-erased)
  /// @param step
  std::shared_ptr<std::any>& any_primal(Int step);

  /// @brief Get primal value
  /// @param step
  template <typename T>
  const T& get_primal(Int step)
  {
    T* tptr = std::any_cast<T>(any_primal(step).get());
    if (stillConstructingGraph_) {
      if (!tptr) {
        gretl_assert(check_validity());
        print_graph();
        print("on reverse, at ", currentStep_, "getting", step);
      }
      gretl_assert_msg(tptr, "bad step " + std::to_string(step));
    } else {
      if (!tptr) {
        fetch_state_data(step);
        tptr = std::any_cast<T>(any_primal(step).get());
      }
      gretl_assert_msg(tptr, "bad step " + std::to_string(step));
    }
    return *tptr;
  }

  /// @brief Set primal value
  /// @param step step
  /// @param t value of type T to set primal to
  template <typename T>
  void set_primal(Int step, const T& t)
  {
    T* tptr = std::any_cast<T>(any_primal(step).get());
    if (!tptr) {
      gretl_assert(!stillConstructingGraph_);
      // MRT, debug reverse pass here
      // if (usageCount_[step] != 1) {
      //   print("step", step);
      //   print_graph();
      // }
      // gretl_assert(usageCount_[step] == 1);
      any_primal(step) = std::make_shared<std::any>(t);
      return;
    }
    gretl_assert(tptr);
    *tptr = t;
  }

  /// @brief Get dual value
  /// @param step
  template <typename D, typename T>
  D& get_dual(Int step)
  {
    if (!duals_[step]) {
      const T& thisPrimal = get_primal<T>(step);
      auto thisState = dynamic_cast<const State<T, D>*>(states_[step].get());
      gretl_assert_msg(thisState, std::string("failed to get primal to this state, step ") + std::to_string(step));
      duals_[step] = std::make_unique<std::any>(thisState->initialize_zero_dual_(thisPrimal));
    }
    auto dualData = std::any_cast<D>(duals_[step].get());
    gretl_assert(dualData);
    return *dualData;
  }

  /// @brief Set dual value
  /// @param step step
  /// @param d value of type D to set dual to
  template <typename D>
  void set_dual(Int step, const D& d)
  {
    if (!duals_[step]) {
      duals_[step] = std::make_unique<std::any>(d);
    }
    auto dualData = std::any_cast<D>(duals_[step].get());
    gretl_assert(dualData);
    *dualData = d;
  }

  /// @brief Deallocate the dual value
  /// @param step
  void clear_dual(Int step) { duals_[step] = nullptr; }

  /// @brief Check if state in use
  /// @param step step
  /// @return bool
  bool state_in_use(Int step) const;

  /// @brief Check if state is persistent
  /// @param step step
  /// @return bool
  bool is_persistent(Int step) const;

  /// @brief Register the graph as being complete.  This is mostly for internal consistency checks.
  void finalize_graph() { stillConstructingGraph_ = false; }

  /// @brief Attempt to free the primal value for this state.  This will happen so long as: 1.) the checkpointer doesn't
  /// have is as an active state; 2.) no downstream state which is active according to checkpointer depends on it as an
  /// upstream; and 3.) an external copy of this state is not being help for potential future use outside of the graph.
  void try_to_free(Int step);

  std::vector<std::unique_ptr<StateBase>> states_;  ///< states for steps
  std::vector<std::unique_ptr<std::any>> duals_;    ///< duals for steps
  std::vector<UpstreamStates> upstreams_;           ///< upstreams dependencies for steps
  std::vector<EvalT> evals_;                        ///< forward evaluation functions for steps
  std::vector<VjpT> vjps_;                          ///< vector-jacobian product functions for steps
  std::vector<bool> active_;                        ///< active status for steps
  std::vector<Int> usageCount_;  ///< count how many times a step is used in some downstream still is the scope of the
                                 ///< checkpoint algorithm

  std::vector<Int>
      lastStepUsed_;  ///< for a given step, records the last known future-step where its used as an upstream
  std::vector<std::vector<Int>> passthroughs_;  ///< at a given step, the list of all the previous steps which are
                                                ///< eventually used in some future step as an upstream

  /// container which track the states in the graph with allocated data
  CheckpointManager checkpointManager_;

  /// step counter
  Int currentStep_;

  /// @brief specifies if graph is in construction or back-prop mode.  This is used for internal asserts.
  bool stillConstructingGraph_ = true;

  friend struct StateBase;

  template <typename T, typename D>
  friend struct State;

  friend struct UpstreamState;
  friend struct DownstreamState;
};

}  // namespace gretl
