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
#include "state_data.hpp"

namespace gretl {

struct StateBase {
  StateBase() {}
  StateBase(const StateBase&) = default;
  StateBase& operator=(const StateBase&) = default;
  virtual ~StateBase() = default;

  template <typename T, typename D = T>
  const T& get() const
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T, D>>(stateData);
    gretl_assert(typedStateData);
    return typedStateData->get_primal();
  }

  template <typename D, typename T = D>
  const D& get_dual() const
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T, D>>(stateData);
    gretl_assert(typedStateData);
    return typedStateData->get_dual();
  }

  template <typename T, typename D = T>
  void set(const T& t)
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T, D>>(stateData);
    gretl_assert(typedStateData);
    typedStateData->set(t);
  }

  template <typename T, typename D = T>
  void set(T&& t)
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T, D>>(stateData);
    gretl_assert(typedStateData);
    typedStateData->set(std::move(t));
  }

  template <typename D, typename T = D>
  void set_dual(const D& d)
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T, D>>(stateData);
    gretl_assert(typedStateData);
    typedStateData->set_dual(d);
  }

  template <typename D, typename T = D>
  void set_dual(D&& d)
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T, D>>(stateData);
    gretl_assert(typedStateData);
    typedStateData->set_dual(std::move(d));
  }

  template <typename T, typename D, typename ZeroCopyT>
  State<T, D> create_state(const std::vector<StateBase>& upstreams, const ZeroCopyT initialize_zero_dual) const
  {
    return stateData->dataStore.create_empty_state<T, D>(initialize_zero_dual, upstreams);
  }

  template <typename T, typename D = T>
  State<T, D> create_state(const std::vector<StateBase>& upstreams) const
  {
    return StateBase::create_state<T, D>(upstreams, [](const T&) -> T { return T(); });
  }

  const StateDataBase* data() const { return stateData.get(); }

  void clear_dual();

  friend class DataStore;
  friend struct StateDataBase;

 protected:
  size_t step_index() const;
  void clear();

  void evaluate_and_remove_disposable_checkpoints();
  void evaluate_vjp();

  std::shared_ptr<StateDataBase> stateData;
  std::vector<std::shared_ptr<StateDataBase>> upstreamsForNextStep;
};

}  // namespace gretl