#pragma once

#include <vector>
#include <functional>
#include <memory>
#include "checkpoint.hpp"

namespace gretl {

static constexpr bool debugPrint = false;

struct StateBase;
template <typename T, typename D = T>
struct State;

struct StateDataBase;

template <typename T, typename D>
using InitializeZeroDual = std::function<D(const T&)>;

template <typename T>
struct defaultInitializeZeroDual {
  T operator()(const T&) { return T(0.0); }
};





struct DataStore {
  DataStore(size_t maxStates);
  virtual ~DataStore() {}

  // create a new state in the graph, store it, return it
  template <typename T, typename D = T, typename InitDualFromValue>
  State<T, D> create_state(const T& t, InitDualFromValue initial_zero_dual)
  {
    State<T, D> newState(*this, t, states.size(), initial_zero_dual, {});
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

  friend struct StateBase;

  template <typename T, typename D>
  friend struct State;  // MRT, try to get all dataStore access off State, into StateData

  template <typename T, typename D>
  friend struct StateData;

  using GhostState = std::shared_ptr<StateDataBase>;
  using Measure = std::pair<StateBase, std::vector<GhostState> >;

 protected:
  // create a new state in the graph, store it, return it
  template <typename T, typename D, typename InitDualFromValue>
  State<T, D> create_empty_state(InitDualFromValue initial_zero_dual, const std::vector<StateBase>& upstreams)
  {
    gretl_assert(!upstreams.empty());
    State<T, D> newState(*this, states.size(), initial_zero_dual, upstreams);
    add_state(newState);
    return newState;
  }

  // recompute primary state at the specificied step
  void fetch_state_data(size_t stepIndex);

  // recompute primary state at the specificied step
  void clear_disposable_state();

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