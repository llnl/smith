// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state_advancer.hpp
 *
 * @brief Interface and implementations for advancing from one step to the next.  Typically these are time integrators.
 */

#pragma once

#include <vector>
#include "gretl/double_state.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/physics/common.hpp"

namespace smith {

using ResultantState = ReactionState;

/// Base state advancer class, allows specification for quasi-static solve strategies, or time integration algorithms
class StateAdvancer {
 public:
  /// @brief destructor
  virtual ~StateAdvancer() {}

  /// @brief interface method to advance the states from a given cycle and time, to the next cycle (cycle+1) and time
  /// (time+dt). shape_disp and params are assumed to be fixed in this advance.  Time and time increment (dt) are
  /// gretl::State in order to record the duals on the reverse pass
  virtual std::vector<FieldState> advanceState(const TimeInfo& time_info, const FieldState& shape_disp,
                                               const std::vector<FieldState>& states,
                                               const std::vector<FieldState>& params) const
  {
    return advanceState(shape_disp, states, params, time_info);
  }

  // Backward-compat signature for older examples/tests.
  virtual std::vector<FieldState> advanceState(const FieldState& shape_disp, const std::vector<FieldState>& states,
                                               const std::vector<FieldState>& params,
                                               const TimeInfo& time_info) const
  {
    return advanceState(time_info, shape_disp, states, params);
  }

  /// @brief interface method to compute reactions given previous, current states and
  /// parameters.
  virtual std::vector<ReactionState> computeReactions(const TimeInfo& time_info, const FieldState& shape_disp,
                                                      const std::vector<FieldState>& states,
                                                      const std::vector<FieldState>& params) const
  {
    return computeResultants(shape_disp, states, states, params, time_info);
  }

  // Backward-compat signature for older examples/tests.
  virtual std::vector<ResultantState> computeResultants(const FieldState& shape_disp,
                                                        const std::vector<FieldState>& states,
                                                        const std::vector<FieldState>& states_old,
                                                        const std::vector<FieldState>& params,
                                                        const TimeInfo& time_info) const
  {
    (void)states_old;
    return computeReactions(time_info, shape_disp, states, params);
  }
};

/// @brief creates a time info struct from gretl::State<double>
/// @param t time
/// @param dt timestep
/// @param cycle iteration
/// @return
inline TimeInfo create_time_info(DoubleState t, DoubleState dt, size_t cycle)
{
  return TimeInfo(t.get(), dt.get(), cycle);
}

}  // namespace smith
