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

class DataStore {
 public:
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

  /// @brief  unwind one step of the graph
  StateBase virtual reverse_state();

  /// @brief unwind the entire graph
  void back_prop();

  /// @brief clear all but persistent state, keeping the graph
  void reset();

  void fetch_state_data(size_t stepIndex) { (void)stepIndex; }

  size_t num_active_states() const;
  size_t num_dual_states() const;

  friend struct StateBase;

  template <typename T, typename D>
  friend struct State;

  template <typename T, typename D>
  friend struct StateData;

  using Measure = std::vector<std::shared_ptr<StateDataBase>>;

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

  // internal function for safely adding new states to graph and checkpoint
  void add_state(StateBase& newState);

  /// @brief vjp
  void vjp(StateBase& state);

  void clear_disposable_state() {}

  // vector of all states in the graph.  states know how to re-evaluate themselves and how to vjp
  std::vector<StateBase> states;

  //
  // std::vector<Measure> checkpointed_states;

  // container which track the states in the graph with allocated data
  CheckpointManager checkpoints;

  //
  size_t step;
};

}  // namespace gretl