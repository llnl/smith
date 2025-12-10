// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file timestep_estimator.hpp
 *
 * @brief Base class and implementations of an interface to estimate stable timesteps
 */

#pragma once

#include "gretl/double_state.hpp"
#include "smith/differentiable_numerics/field_state.hpp"

namespace smith {

/// @brief Base class interface for estimating the stable timestep given the current state and parameters
class TimestepEstimator {
 public:
  /// @brief destructor
  virtual ~TimestepEstimator() {}

  /// @brief Interface method for estimating the stable timestep give the current state and parameters
  virtual DoubleState dt(const FieldState& shape_disp, const std::vector<FieldState>& states,
                         const std::vector<FieldState>& params) const = 0;
};

/// @brief TimeStepEstimator which uses a simple and fixed timestep
class ConstantTimeStepEstimator : public TimestepEstimator {
 public:
  /// @brief Constructor
  /// @param dt fixed timestep to use throughout the simulation
  ConstantTimeStepEstimator(double dt) : dt_(dt) {}

  /// @overload
  DoubleState dt([[maybe_unused]] const FieldState& shape_disp, [[maybe_unused]] const std::vector<FieldState>& states,
                 [[maybe_unused]] const std::vector<FieldState>& params) const override
  {
    double dt = dt_;
    DoubleState DT = gretl::create_state<double, double>(
        gretl::defaultInitializeZeroDual<double, double>(), [dt](FEFieldPtr) { return dt; },
        [](FEFieldPtr, double, FEDualPtr&, double) {}, shape_disp);
    return DT;
  }

  double dt_;  ///< fixed timestep
};

}  // namespace smith
