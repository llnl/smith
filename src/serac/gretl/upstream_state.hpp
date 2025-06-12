// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file upstream_state.hpp
 */

#pragma once

#include <memory>
#include "data_store.hpp"

namespace gretl {

struct UpstreamState {
  UpstreamState(const std::shared_ptr<StateDataBase>& s) : state(s) {}

  bool valid() const { return state->primal_active(); }

  bool dual_valid() const { return state->dual_active(); }

  template <typename T, typename D = T>
  const T& get() const
  {
    return state->template get<T, D>();
  }

  template <typename D, typename T = D>
  D& get_dual()
  {
    return state->template get_dual<T, D>();
  }

  template <typename D, typename T = D>
  const D& get_dual() const
  {
    return state->template get_dual<T, D>();
  }

  template <typename D, typename T = D>
  void set_dual(const D& d)
  {
    return state->set_dual<T, D>(d);
  }

  friend class DataStore;
  friend class DynamicDataStore;

 protected:
  const std::shared_ptr<StateDataBase>& get_state() { return state; }

 private:
  std::shared_ptr<StateDataBase> state;
};

struct DownstreamState {
  DownstreamState(StateDataBase& s) : stateDataBase(s) {}

  bool dual_valid() const { return stateDataBase.dual_active(); }

  template <typename T, typename D = T>
  void set(const T& t)
  {
    return stateDataBase.set_primal<T, D>(t);
  }

  // this call will give incorrect behavior if called on the forward pass.  Consider adding ConstDownStreamState type to
  // allow this on reverse, but not forward
  template <typename T, typename D = T>
  const T& get() const
  {
    return stateDataBase.template get<T, D>();
  }
  template <typename D, typename T = D>
  const D& get_dual() const
  {
    return stateDataBase.template get_dual<T, D>();
  }

  friend class DataStore;

 private:
  StateDataBase& stateDataBase;
};

}  // namespace gretl