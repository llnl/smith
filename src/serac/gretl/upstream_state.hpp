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
  DataStore* dataStore_;  ///< datastore
  Int step_;              ///< step

  /// @brief get underlying value
  template <typename T>
  const T& get() const
  {
    return dataStore_->get_primal<T>(step_);
  }

  /// @brief get underlying dual value
  template <typename D>
  D& get_dual() const
  {
    return dataStore_->get_dual<D>(step_);
  }
};

/// @brief UpstreamStates is a wrapper for a vector of states.  Its used in external-facing interfaces to ensure const
/// correctness for users to encourage correct usage.
struct UpstreamStates {
  /// @brief Constructor for upstream states
  /// @param s datastore
  /// @param steps vector of upstream steps
  UpstreamStates(DataStore& s, std::vector<Int> steps) : dataStore_(&s), steps_(steps) {}

  /// @brief Accessor for individual upstream states
  /// @param index index
  UpstreamState operator[](Int index) const { return UpstreamState{.dataStore_ = dataStore_, .step_ = steps_[index]}; }

  /// @brief Number of upstream states
  Int size() const { return static_cast<Int>(steps_.size()); }

  /// @brief Vector of upstream step indices
  std::vector<Int> steps() const { return steps_; }

 private:
  DataStore* dataStore_;    ///< datastore
  std::vector<Int> steps_;  ///< steps
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
    return *dataStore_->get_primal<T>(step_);
  }

  /// @brief get underlying dual value
  template <typename D, typename T = D>
  const D& get_dual() const
  {
    return dataStore_->get_dual<D>(step_);
  }

  friend class DataStore;

 private:
  DataStore* dataStore_;  ///< datastore
  Int step_;              ///< step
};

}  // namespace gretl
