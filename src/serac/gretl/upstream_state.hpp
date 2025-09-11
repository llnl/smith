// Copyright (c) Lawrence Livermore National Security, LLC and
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

/// @brief UpstreamState is a wrapper for a states.  Its used in external-facing interfaces to ensure const correctness
/// for users to encourage correct usage.
struct UpstreamState {
  Int step_;              ///< step
  DataStore* dataStore_;  ///< datastore

  /// @brief get underlying value
  template <typename T>
  const T& get() const
  {
    return dataStore_->get_primal<T>(step_);
  }

  /// @brief get underlying dual value
  template <typename D, typename T>
  D& get_dual() const
  {
    return dataStore_->get_dual<D, T>(step_);
  }
};

/// @brief UpstreamStates is a wrapper for a vector of states.  Its used in external-facing interfaces to ensure const
/// correctness for users to encourage correct usage.
struct UpstreamStates {
  /// @brief Default constructor to use in std containers
  UpstreamStates() {}

  /// @brief Constructor for upstream states
  /// @param store datastore
  /// @param steps vector of upstream steps
  UpstreamStates(DataStore& store, std::vector<Int> steps)
  {
    for (Int s : steps) {
      states_.push_back({s, &store});
    }
  }

  /// @brief Accessor for individual upstream states
  /// @param index index
  template <typename IntT>
  const UpstreamState& operator[](IntT index) const
  {
    return states_[static_cast<size_t>(index)];
  }

  /// @brief Accessor for individual upstream states
  /// @param index index
  const UpstreamState& operator[](Int index) const { return states_[index]; }

  /// @brief Number of upstream states
  Int size() const { return static_cast<Int>(states_.size()); }

  /// @brief Vector of upstream step indices
  const std::vector<UpstreamState>& states() const { return states_; }

 private:
  std::vector<UpstreamState> states_;  ///< states
};

/// @brief DownstreamState is a wrapper for a state.  Its used in external-facing interfaces to ensure const correctness
/// for users to encourage correct usage.
struct DownstreamState {
  /// @brief Constructor
  /// @param s datastore
  /// @param step step
  DownstreamState(DataStore* s, Int step) : dataStore_(s), step_(step) {}

  /// @brief set underlying value
  template <typename T, typename D = T>
  void set(const T& t)
  {
    return dataStore_->set_primal<T>(step_, t);
  }

  /// @brief get underlying value
  template <typename T, typename D = T>
  const T& get() const
  {
    return dataStore_->get_primal<T>(step_);
  }

  /// @brief get underlying dual value
  template <typename D, typename T = D>
  const D& get_dual() const
  {
    return dataStore_->get_dual<D, T>(step_);
  }

  friend class DataStore;

 private:
  DataStore* dataStore_;  ///< datastore
  Int step_;              ///< step
};

}  // namespace gretl
