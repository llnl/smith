// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file common.hpp
 *
 * @brief A file defining some enums and structs that are used by the different physics modules
 */
#pragma once

#include <utility>

namespace serac {

/// @brief struct storing time and timestep information
struct TimeInfo {
  /// @brief constructor
  TimeInfo(double t, double t_step, size_t c = 0)
      : time_(std::make_pair(t, 0.0)), dt_(std::make_pair(t_step, 0.0)), cycle_(c)
  {
  }

  double time() const { return time_.first; }
  double dt() const { return dt_.first; }
  size_t cycle() const { return cycle_; }

  std::pair<double, double> time_;
  std::pair<double, double> dt_;
  size_t cycle_;
};

/**
 * @brief a struct that is used in the physics modules to clarify which template arguments are
 * user-controlled parameters (e.g. for design optimization)
 */
template <typename... T>
struct Parameters {
  static constexpr int n = sizeof...(T);  ///< how many parameters were specified
};

}  // namespace serac
