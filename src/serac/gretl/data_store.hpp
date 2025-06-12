
#pragma once

#include <vector>
#include <functional>
#include <memory>
#include "checkpoint.hpp"

namespace gretl {

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
  StateBase reverse_state();

  /// @brief unwind the entire graph
  void back_prop();

  /// @brief clear all but persistent state, keeping the graph
  void reset();

  size_t num_allocated_states() const;
  size_t num_active_states() const;
  size_t num_dual_states() const;

  friend struct StateBase;

  template <typename T, typename D>
  friend struct State;

  template <typename T, typename D>
  friend struct StateData;

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

  /// @brief vjp
  void vjp(StateBase& state);

  /// @brief function for safely adding new states to graph and checkpoint
  virtual void add_state(StateBase& newState);

  /// @brief method for clearing states in the past that are no longer needed
  virtual void clear_disposable_state() {}

  /// @brief method for fetching states at a particular moment in time
  virtual void fetch_state_data(size_t stepIndex);

  /// vector of all states in the graph.  States know how to re-evaluate themselves and how to vjp.
  std::vector<StateBase> states;

  /// step counter
  size_t step;
};

class DynamicDataStore : public DataStore {
 public:
  DynamicDataStore(size_t maxStates);
  virtual ~DynamicDataStore() {}

 protected:

  friend struct StateBase;
  friend struct UpstreamState;

  /// @overload
  virtual void add_state(StateBase& newState) override;

  /// @overload
  void clear_disposable_state() override;

  /// @overload
  void fetch_state_data(size_t stepIndex) override;

  /// container which track the states in the graph with allocated data
  CheckpointManager checkpoints;
};

}  // namespace gretl