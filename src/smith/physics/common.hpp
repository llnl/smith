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

#include "smith/numerics/functional/tuple.hpp"

namespace smith {

/// @brief struct storing time and timestep information
struct TimeInfo {
  /// @brief Evaluation mode for the current step
  enum class EvaluationMode
  {
    Regular,   ///< Normal evaluation step
    CycleZero  ///< Initialization or cycle zero step
  };

  /// @brief constructor
  SMITH_HOST_DEVICE TimeInfo(double t, double t_step, size_t c = 0,
                             EvaluationMode mode = EvaluationMode::Regular)
      : time_(tuple<double, double>{t, 0.0}),
        dt_(tuple<double, double>{t_step, 0.0}),
        cycle_(c),
        mode_(mode)
  {
  }

  /// @brief accessor for the current time
  SMITH_HOST_DEVICE double time() const
  {
    return get<0>(time_) + get<0>(dt_);
  }

  /// @brief accessor for dt
  SMITH_HOST_DEVICE double dt() const { return get<0>(dt_); }

  /// @brief accessor for cycle
  SMITH_HOST_DEVICE size_t cycle() const { return cycle_; }

  /// @brief true when evaluating the startup acceleration solve.
  SMITH_HOST_DEVICE bool isCycleZeroEvaluation() const { return mode_ == EvaluationMode::CycleZero; }

  /// @brief accessor for residual evaluation mode.
  SMITH_HOST_DEVICE EvaluationMode mode() const { return mode_; }

 private:
  tuple<double, double> time_;  ///< time and its dual
  tuple<double, double> dt_;    ///< timestep and its dual
  size_t cycle_;                              ///< cycle, step, iteration count
  EvaluationMode mode_;                       ///< residual evaluation mode
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
