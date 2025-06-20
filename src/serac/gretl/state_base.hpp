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

struct StateBase {
  // StateBase(DataStore& store) : dataStore_(&store) {}
  StateBase(DataStore& store) : dataStore_(&store) {}
  StateBase(const StateBase&) = default;
  StateBase& operator=(const StateBase&) = default;
  virtual ~StateBase() = default;

  template <typename T, typename D = T>
  const T& get() const
  {
    return dataStore_->get_primal<T>(step_);
  }

  template <typename T, typename D = T>
  void set(const T& t)
  {
    dataStore_->set_primal<T>(step_, t);
  }

  template <typename D, typename T = D>
  const D& get_dual() const
  {
    return dataStore_->get_dual<D>(step_);
  }
  
  template <typename D, typename T = D>
  void set_dual(const D& d)
  {
    dataStore_->set_dual<D>(step_, d);
  }

  template <typename T, typename D>
  State<T, D> create_state(const std::vector<StateBase>& upstreams, InitializeZeroDual<T,D> initialize_zero_dual) const
  {
    return dataStore_->create_empty_state<T, D>(initialize_zero_dual, upstreams);
  }

  template <typename T, typename D = T>
  State<T, D> create_state(const std::vector<StateBase>& upstreams) const
  {
    return StateBase::create_state<T, D>(upstreams, [](const T&) -> T { return T(); });
  }

  friend class DataStore;
  friend class DynamicDataStore;
  friend struct StateDataBase;

  void evaluate_and_remove_disposable_checkpoints();
  void evaluate_vjp();

  DataStore& data_store() const { return *dataStore_; }

 protected:
  // const StateDataBase* data() const { return dataStore..get(); }

  Int step_index() const;
  void clear_primal();
  void clear_dual();

  DataStore* dataStore_;
  Int step_;
};

}  // namespace gretl