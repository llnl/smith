#pragma once

#include <vector>
#include <functional>
#include "checkpoint.hpp"

namespace gretl {

static constexpr bool debugPrint = false;

struct StateBase;
template <typename T>
struct State;

class StateDataBase;

template <typename T>
using ZeroClone = std::function<T(const T&)>;

struct DataStore {
  DataStore(size_t maxStates);
  virtual ~DataStore() {}

  // create a new state in the graph, store it, return it
  template <typename T>
  State<T> create_state(const T& t)
  {
    auto zero_clone = [](const T&) -> T { return T(); };
    return create_state(t, zero_clone);
  }

  // create a new state in the graph, store it, return it
  template <typename T, typename ZeroCloneFromT>
  State<T> create_state(const T& t, ZeroCloneFromT zero_clone)
  {
    bool persistent = true;
    State<T> newState(*this, t, states.size(), zero_clone, {});
    add_state(newState);
    return newState;
  }

  // unwind the entire graph
  void back_prop();

  // clear all but persistent state, keep the graph though
  void reset();

  size_t num_active_states() const;
  size_t num_dual_states() const;

  void print_state_info() const;
  void check_consistency() const;
  void check_consistency_except_last(size_t) const;

  friend class StateBase;

  template <typename T>
  friend struct State;  // MRT, try to get all dataStore access off State, into StateData

  template <typename T>
  friend class StateData;

  using GhostState = std::shared_ptr<StateDataBase>;
  using Measure = std::pair<StateBase, std::vector<GhostState> >;

 protected:
  // create a new state in the graph, store it, return it
  template <typename T, typename ZeroCloneFromT>
  State<T> create_empty_state(ZeroCloneFromT zero_clone, const std::vector<StateBase>& upstreams)
  {
    assert(!upstreams.empty());
    State<T> newState(*this, states.size(), zero_clone, upstreams);
    add_state(newState);
    return newState;
  }

  // recompute primary state at the specificied step
  void fetch_state_data(size_t stepIndex);

  // recompute primary state at the specificied step
  void clear_disposable_state(double costFactor);

  // reverse back a single state, updating the duals along the way
  StateBase virtual reverse_state(size_t);

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

  // save off the next state to clear memory for
  size_t nextStepToClear = 0;
};

}  // namespace gretl