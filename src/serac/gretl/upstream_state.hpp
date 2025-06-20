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
  DataStore* dataStore_;
  Int step_;

  template <typename T>
  const T& get() const
  {
    return dataStore_->get_primal<T>(step_);
  }

  template <typename D>
  D& get_dual() const
  {
    return dataStore_->get_dual<D>(step_);
  }
};

struct UpstreamStates {
  UpstreamStates(DataStore& s, std::vector<Int> steps) : dataStore_(&s), steps_(steps) {}

  UpstreamState operator[](Int index) const { return UpstreamState{.dataStore_ = dataStore_, .step_ = steps_[index]}; }

  Int size() const { return static_cast<Int>(steps_.size()); }

 // private:
  DataStore* dataStore_;
  std::vector<Int> steps_;
};

struct DownstreamState {
  DownstreamState(DataStore* s, Int step) : dataStore_(s), step_(step) {}

  template <typename T, typename D = T>
  void set(const T& t)
  {
    return dataStore_->set_primal<T>(step_, t);
  }

  template <typename T, typename D = T>
  const T& get() const
  {
    return *dataStore_->get_primal<T>(step_);
  }

  template <typename D, typename T = D>
  const D& get_dual() const
  {
    return dataStore_->get_dual<D>(step_);
  }

  friend class DataStore;

 private:
  DataStore* dataStore_;
  Int step_;
};

}  // namespace gretl