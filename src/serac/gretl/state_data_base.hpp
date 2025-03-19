// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state_data_base.hpp
 */

#pragma once

#include "data_store.hpp"

namespace gretl {

struct UpstreamState;
struct DownstreamState;
using UpstreamStates = std::vector<UpstreamState>;

template <typename T, typename D>
struct StateData;

struct StateDataBase {
  virtual ~StateDataBase() {}

  StateDataBase(const StateDataBase&) = delete;
  StateDataBase& operator=(const StateDataBase&) = delete;

  template <typename T, typename D>
  T& get() const
  {
    auto typedStateData = dynamic_cast<const StateData<T, D>*>(this);
    assert(typedStateData);
    return typedStateData->get_primal();
  }

  template <typename T, typename D>
  D& get_dual() const
  {
    auto typedStateData = dynamic_cast<const StateData<T, D>*>(this);
    assert(typedStateData);
    return typedStateData->get_dual();
  }

  template <typename T, typename D = T>
  void set_primal(const T& t) const
  {
    auto typedStateData = dynamic_cast<const StateData<T, D>*>(this);
    assert(typedStateData);
    typedStateData->set_primal(t);
  }

  template <typename T, typename D = T>
  void move_primal(T&& t) const
  {
    auto typedStateData = dynamic_cast<const StateData<T, D>*>(this);
    assert(typedStateData);
    typedStateData->move_primal(std::move(t));
  }

  template <typename T, typename D>
  void set_dual(const D& d) const
  {
    auto typedStateData = dynamic_cast<const StateData<T, D>*>(this);
    assert(typedStateData);
    typedStateData->set_dual(d);
  }

  template <typename T, typename D>
  void move_dual(D&& d) const
  {
    auto typedStateData = dynamic_cast<const StateData<T, D>*>(this);
    assert(typedStateData);
    typedStateData->move_dual(std::move(d));
  }

  virtual void evaluate() = 0;
  virtual void evaluate_vjp() = 0;

  virtual void clear_primal() = 0;
  virtual void clear_dual() = 0;

  virtual bool primal_active() const = 0;
  virtual bool dual_active() const = 0;

  bool persistent() const;

  size_t step_index() const { return stepIndex; }
  virtual size_t parent_step_index() const { return 0; }

  virtual std::shared_ptr<StateDataBase> create_ghost(const std::shared_ptr<StateDataBase>& parentGhost,
                                                      const std::shared_ptr<StateDataBase>& lastGhost,
                                                      size_t step) const = 0;

  size_t stepIndex;
  DataStore& dataStore;
  UpstreamStates upstreams;

  friend struct DataStore;

 protected:
  StateDataBase(DataStore& cpd, size_t step, const std::vector<StateBase>& ustreams);
};

}  // namespace gretl