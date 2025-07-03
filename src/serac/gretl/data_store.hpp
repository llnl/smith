
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

using Int = unsigned int;

struct StateBase;

template <typename T, typename D = T>
struct State;

struct UpstreamStates;

struct DownstreamState;

template <typename T, typename D>
using InitializeZeroDual = std::function<D(const T&)>;

template <typename T>
struct defaultInitializeZeroDual {
  T operator()(const T&) { return T{}; }
};

class DataStore {
 public:
  DataStore(size_t maxStates);
  virtual ~DataStore() {}

  /// @brief create a new state in the graph, store it, return it
  template <typename T, typename D = T>
  State<T, D> create_state(const T& t, InitializeZeroDual<T, D> initial_zero_dual = defaultInitializeZeroDual<D>())
  {
    State<T, D> state(this, states_.size(), std::make_shared<std::any>(t), initial_zero_dual);
    add_state(std::make_unique<State<T, D>>(state), {});

    return state;
  }

  /// @brief  unwind one step of the graph
  virtual void reverse_state();

  /// @brief unwind the entire graph
  void back_prop();

  /// @brief clear all but persistent state, keeping the graph
  void reset();

  Int num_active_states() const;
  Int num_dual_states() const;

  friend struct StateBase;

  template <typename T, typename D>
  friend struct State;

  friend struct UpstreamState;
  friend struct DownstreamState;

  virtual void print() const {}
  virtual bool check_validity() const { return true; }

  // create a new state in the graph, store it, return it
  template <typename T, typename D, typename InitDualFromValue>
  State<T, D> create_empty_state(InitDualFromValue initial_zero_dual, const std::vector<StateBase>& upstreams)
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
  virtual void add_state(std::unique_ptr<StateBase> newState, const std::vector<StateBase>& upstreams);

  /// @brief method for fetching states at a particular moment in time
  virtual void fetch_state_data(Int stepIndex);

  virtual void remove_things(Int) {}

  using EvalT = std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)>;
  using VjpT = std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)>;

  std::vector<std::unique_ptr<StateBase>> states_;
  std::vector<std::unique_ptr<std::any>> duals_;
  std::vector<UpstreamStates> upstreams_;
  std::vector<EvalT> evals_;
  std::vector<VjpT> vjps_;
  mutable std::vector<bool> active_;
  mutable std::vector<Int> activeCount_;

  std::shared_ptr<std::any>& any_primal(Int step);

  template <typename T>
  const T& get_primal(Int step)
  {
    auto tptr = std::any_cast<T>(any_primal(step).get());
    gretl_assert(tptr);
    return *tptr;
  }

  template <typename T>
  void set_primal(Int step, const T& t)
  {
    auto tptr = std::any_cast<T>(any_primal(step).get());
    if (!tptr) {
      gretl_assert(!isGoingForward);
      gretl_assert(activeCount_[step]==0);
      any_primal(step) = std::make_shared<std::any>(t);
      activeCount_[step] = 1;
      return;
    }
    gretl_assert(tptr);
    *tptr = t;
  }

  template <typename D, typename T = D>
  D& get_dual(Int step)
  {
    if (!duals_[step]) {
      const T& thisPrimal = get_primal<T>(step);
      auto thisState = dynamic_cast<const State<T, D>*>(states_[step].get());
      gretl_assert(thisState);
      duals_[step] = std::make_unique<std::any>(thisState->initialize_zero_dual_(thisPrimal));
    }
    auto dualData = std::any_cast<D>(duals_[step].get());
    gretl_assert(dualData);
    return *dualData;
  }

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

  bool is_persistent(Int step) const;

  /// step counter
  Int current_step_;

  /// is going forward
  bool isGoingForward = true;
};

class DynamicDataStore : public DataStore {
 public:
  DynamicDataStore(size_t maxStates);
  virtual ~DynamicDataStore() {}

  /// @overload
  virtual void print() const override;

  /// @overload 
  virtual bool check_validity() const override;

  virtual void reverse_state() override;

  friend struct StateBase;
  friend struct UpstreamState;

  /// @overload
  virtual void add_state(std::unique_ptr<StateBase> newState, const std::vector<StateBase>& upstreams) override;

  /// @overload
  virtual void fetch_state_data(Int stepIndex) override;

  /// @overload
  virtual void remove_things(Int stepIndex) override;

  std::vector<Int> lastStepUsed_;
  std::vector< std::vector<Int> > passthroughs_;

  /// container which track the states in the graph with allocated data
  CheckpointManager checkpointManager;
};

}  // namespace gretl