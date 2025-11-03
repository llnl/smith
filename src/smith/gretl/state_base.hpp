// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state_base.hpp
 */

#pragma once

#include <vector>
#include <set>
#include "data_store.hpp"

namespace gretl {

/// @brief Constainer for DataStore, primal value and step index for a given state
struct StateData {
  /// @brief constructor
  StateData(DataStore* dataStore, std::shared_ptr<std::any> primal) : dataStore_(dataStore), primal_(primal) {}
  DataStore* dataStore_;                        ///< datastore
  std::shared_ptr<std::any> primal_;            ///< value, stores as shared_ptr to std::any
  Int step_ = std::numeric_limits<Int>::max();  ///< step
};

/// @brief Baseclass for State.  State stores type-erased value and step number in the graph.
struct StateBase {
  /// @brief Construct state base from a date store and a type-erased values
  StateBase(DataStore* store, const std::shared_ptr<std::any>& val) : data_(std::make_shared<StateData>(store, val)) {}

  /// @brief copy operator
  StateBase(const StateBase& oldState) { data_ = oldState.data_; }

  /// @brief assignment operator
  StateBase& operator=(const StateBase& oldState)
  {
    if (!data_) {
      data_ = oldState.data_;
      return *this;
    }
    auto* dataStore = &data_store();
    Int s = step();
    data_ = oldState.data_;
    if (dataStore) {
      dataStore->try_to_free(s);
    }
    return *this;
  }

  /// @brief default virtual destructor
  virtual ~StateBase()
  {
    if (!data_) {
      return;
    }
    auto* dataStore = &data_store();
    Int s = step();
    data_ = nullptr;
    if (dataStore) {
      dataStore->try_to_free(s);
    }
  }

  /// @brief get the underlying value
  template <typename T>
  const T& get() const
  {
    return data_store().get_primal<T>(step());
  }

  /// @brief get the underlying dual value, dual template type comes first
  template <typename D, typename T = D>
  const D& get_dual() const
  {
    return data_store().get_dual<D>(step());
  }

  /// @brief set the underlying dual value, dual template type comes first
  template <typename D, typename T = D>
  void set_dual(const D& d)
  {
    data_store().set_dual<D>(step(), d);
  }

  /// @brief method to clear out the memory usage for state's dual value
  void clear_dual() { data_store().clear_dual(step()); }

  /// @brief create a new state, given the upstream input dependencies and a function specifying how to initial the dual
  /// value to zero.
  template <typename T, typename D = T>
  State<T, D> create_state(const std::vector<StateBase>& upstreams, InitializeZeroDual<T, D> initialize_zero_dual) const
  {
    return data_store().create_empty_state<T, D>(initialize_zero_dual, upstreams);
  }

  /// @brief create a new state, given the upstream input dependencies, uses a default dual initializer
  template <typename T, typename D = T>
  State<T, D> create_state(const std::vector<StateBase>& upstreams) const
  {
    return StateBase::create_state<T, D>(upstreams, defaultInitializeZeroDual<T>());
  }

  friend class DataStore;
  friend class DynamicDataStore;

  /// @brief Evaluate graph one step forward, compute primal value at this new state
  void evaluate_forward();

  /// @brief Evaluate graph one step backward, contribute sensitivity to the upstream duals
  void evaluate_vjp();

  /// @brief Datastore accessor
  DataStore& data_store() const { return *data_->dataStore_; }

  /// @brief Get step
  Int step() const { return data_->step_; }

  /// @brief Reset step.
  void reset_step(Int newStep) { data_->step_ = newStep; }

  /// @brief accessor for geting the value of the state off of the StateData
  std::shared_ptr<std::any>& primal() { return data_->primal_; }

  /// @brief const accessor for geting the value of the state off of the StateData
  const std::shared_ptr<std::any>& primal() const { return data_->primal_; }

  /// @brief returns the counted number of usage of this State's StateData in user code by returning the number of
  /// shared pointer instances minus 1 (as the state manager graph always has a copy)
  size_t wild_count() const { return static_cast<size_t>(data_.use_count()) - 1; }

 protected:
  /// @brief state data which store step, and value information.  The shared_ptr allows tracking of the number of
  /// external usages of this state.
  std::shared_ptr<StateData> data_;
};

}  // namespace gretl
