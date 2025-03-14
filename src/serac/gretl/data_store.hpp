// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file data_store.hpp
 */

#pragma once

#include <memory>
#include <vector>
#include <functional>
#include "checkpoint.hpp"

namespace gretl {

static constexpr bool debugPrint = false;

struct StateBase;
template <typename T, typename D = T>
struct State;

struct StateDataBase;

template <typename T, typename D>
using ZeroClone = std::function<D(const T&)>;

struct DataStore {
  DataStore(size_t maxStates);
  virtual ~DataStore() {}

  // create a new state in the graph, store it, return it
  template <typename T, typename D = T>
  State<T, D> create_state(const T& t)
  {
    auto zero_clone = [](const T&) -> D { return D(); };
    return create_state<T, D, decltype(zero_clone)>(t, zero_clone);
  }

  // create a new state in the graph, store it, return it
  template <typename T, typename D, typename ZeroCloneFromT>
  State<T, D> create_state(const T& t, ZeroCloneFromT zero_clone)
  {
    State<T, D> newState(*this, t, states.size(), zero_clone, {});
    add_state(newState);
    return newState;
  }

  // clear all but persistent state, remove graph
  void delete_graph();

  // unwind the entire graph
  void back_prop();

  // clear all but persistent state, keep the graph though
  void reset();

  // reset back to the end of the simulation
  void reset_for_backprop();

  size_t num_active_states() const;
  size_t num_dual_states() const;

  void print_state_info() const;
  bool check_consistency() const;
  bool check_consistency_except_last(size_t) const;

  // get a handle to the state at a particular step
  StateBase get_state() const;

  // reverse back a single state, updating the duals along the way
  StateBase virtual reverse_state();

  friend struct StateBase;

  template <typename T, typename D>
  friend struct State;  // MRT, try to get all dataStore access off State, into StateData

  template <typename T, typename D>
  friend struct StateData;

  using GhostState = std::shared_ptr<StateDataBase>;
  using Measure = std::pair<StateBase, std::vector<GhostState> >;

 protected:
  // create a new state in the graph, store it, return it
  template <typename T, typename D, typename ZeroCloneFromT>
  State<T, D> create_empty_state(ZeroCloneFromT zero_clone, const std::vector<StateBase>& upstreams)
  {
    assert(!upstreams.empty());
    State<T, D> newState(*this, states.size(), zero_clone, upstreams);
    add_state(newState);
    return newState;
  }

  // recompute primary state at the specificied step
  void fetch_state_data(size_t stepIndex);

  // recompute primary state at the specificied step
  void clear_disposable_state(double costFactor);

  // internal function for safely adding new states to graph and checkpoint
  void add_state(StateBase& newState);

  std::shared_ptr<StateDataBase> add_ghost_state(const std::shared_ptr<StateDataBase>& parent,
                                                 const std::shared_ptr<StateDataBase>& last, size_t step_to_add);

  void clear_measure(Measure& measure);
  void clear_measure_dual(Measure& measure);
  void vjp_measure(Measure& measure);
  bool measure_primal_active(const Measure& measure) const;
  bool measure_dual_active(const Measure& measure) const;

  // vector of all states in the graph, their data may or may not be allocated
  std::vector<Measure> states;

  // container which track the states in the graph with allocated data
  CheckpointManager checkpoints;

  // index to track the currently relevant state
  size_t currentStep;

  // save off the next state to clear memory for
  size_t nextStepToClear = 0;
};

}  // namespace gretl