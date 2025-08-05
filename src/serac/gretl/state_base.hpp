// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
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

/// @brief Baseclass for State.  State stores type-erased value and step number in the graph.
struct StateBase {
  /// @brief Construct state base from a date store and a type-erased values
  StateBase(DataStore* store, const std::shared_ptr<std::any>& val) : dataStore_(store), primal_(val) {}

  /// @brief default virtual destructor
  virtual ~StateBase() = default;

  /// @brief get the underlying value
  template <typename T, typename D = T>
  const T& get() const
  {
    return dataStore_->get_primal<T>(step_);
  }

  /// @brief get the underlying dual value, dual template type comes first
  template <typename D, typename T = D>
  const D& get_dual() const
  {
    return dataStore_->get_dual<D>(step_);
  }

  /// @brief set the underlying dual value, dual template type comes first
  template <typename D, typename T = D>
  void set_dual(const D& d)
  {
    dataStore_->set_dual<D>(step_, d);
  }

  /// @brief create a new state, given the upstream input dependencies and a function specifying how to initial the dual
  /// value to zero.
  template <typename T, typename D = T>
  State<T, D> create_state(const std::vector<StateBase>& upstreams, InitializeZeroDual<T, D> initialize_zero_dual) const
  {
    return dataStore_->create_empty_state<T, D>(initialize_zero_dual, upstreams);
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
  DataStore& data_store() const { return *dataStore_; }

  /// @brief Get step
  Int step() const { return step_; }

 protected:
  DataStore* dataStore_;              ///< datastore
  std::shared_ptr<std::any> primal_;  ///< value, stores as shared_ptr to std::any
  Int step_;                          ///< step
};

}  // namespace gretl