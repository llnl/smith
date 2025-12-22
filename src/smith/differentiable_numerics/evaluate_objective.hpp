// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file evaluate_objective.hpp
 *
 * @brief Methods for evaluating objective functions and tracking these operations on the gretl graph with a custom vjp
 */

#pragma once

#include "gretl/double_state.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/physics/scalar_objective.hpp"

namespace smith {

/// @brief Evaluates a DoubleState using a provided ScalarObjective instance, and the input arguments to that objective.
/// This operation is tracked on the gretl graph.
DoubleState evaluateObjective(std::shared_ptr<ScalarObjective> objective, const TimeInfo& time_info,
                              const FieldState& shape_disp, const std::vector<FieldState>& inputs);

/// @brief operation is tracked on the gretl graph.
DoubleState evaluateObjective(std::shared_ptr<ScalarObjective> objective, const FieldState& shape_disp,
                              const std::vector<FieldState>& inputs);

}  // namespace smith
