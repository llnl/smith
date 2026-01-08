// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
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

namespace smith {

/// @brief struct storing time and timestep information
struct TimeInfo {
  /// @brief constructor
  TimeInfo(double t, double t_step, size_t c = 0)
      : time_(std::make_pair(t, 0.0)), dt_(std::make_pair(t_step, 0.0)), cycle_(c)
  {
  }

  /// @brief accessor for time
  double time() const { return time_.first; }

  /// @brief accessor for dt
  double dt() const { return dt_.first; }

  /// @brief accessor for cycle
  size_t cycle() const { return cycle_; }

  /// @brief return the time info corresponding to the end of this cycle (time + dt, dt, cycle)
  TimeInfo endTimeInfo() const { return TimeInfo(time() + dt(), dt(), cycle()); }

  std::pair<double, double> time_;  ///< time and its dual
  std::pair<double, double> dt_;    ///< timestep and its dual
  size_t cycle_;                    ///< cycle, step, iteration count
};

/**
 * @brief a struct that is used in the physics modules to clarify which template arguments are
 * user-controlled parameters (e.g. for design optimization)
 */
template <typename... T>
struct Parameters {
  static constexpr int n = sizeof...(T);  ///< how many parameters were specified
};

}  // namespace smith
