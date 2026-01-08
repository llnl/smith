// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file reaction.hpp
 *
 * @brief Reaction class which is a names combination of a weak form and a set of dirichlet constrained nodes.
 */

#pragma once

#include <string>
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"

namespace smith {

/// @brief gretl-function implementation which evaluates the residual force (which is minus the mechanical force)
/// given
/// shape displacement, states and params.  The field_for_residual_space Field is only used to set the appropriate size
/// (mfem::ParFiniteElementSpace) for the residual field so it can be returned as a ReactionState
inline auto evaluateWeakForm(const std::shared_ptr<WeakForm>& weak_form, const TimeInfo& time_info,
                             FieldState shape_disp, const std::vector<FieldState>& field_states,
                             FieldState field_for_residual_space)
{
  std::vector<gretl::StateBase> all_state_bases{shape_disp};
  for (auto& f : field_states) all_state_bases.push_back(f);
  all_state_bases.push_back(field_for_residual_space);

  auto z = shape_disp.create_state<FEDualPtr, FEFieldPtr>(all_state_bases, zero_state_from_dual());

  z.set_eval([=](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    SMITH_MARK_FUNCTION;

    size_t num_fields = inputs.size() - 2;
    ConstFieldPtr shape_disp_ = inputs[0].get<FEFieldPtr>().get();
    std::vector<ConstFieldPtr> fields(num_fields);
    for (size_t field_index = 0; field_index < num_fields; ++field_index) {
      fields[field_index] = inputs[field_index + 1].get<FEFieldPtr>().get();
    }
    ConstFieldPtr field_for_residual_space_ = inputs[num_fields + 1].get<FEFieldPtr>().get();

    FEDualPtr R = std::make_shared<FiniteElementDual>(field_for_residual_space_->space(),
                                                      "residual");  // set up output pointer
    // evaluate the residual with zero acceleration contribution
    *R = weak_form->residual(time_info, shape_disp_, fields);
    output.set<FEDualPtr, FEFieldPtr>(R);
  });

  z.set_vjp([=](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    SMITH_MARK_FUNCTION;

    const FEDualPtr Z = output.get<FEDualPtr, FEFieldPtr>();
    const FEFieldPtr Z_dual = output.get_dual<FEFieldPtr, FEDualPtr>();
    FiniteElementState Z_dual_state(Z_dual->space(), Z_dual->name());
    Z_dual_state = *Z_dual;

    size_t num_fields = inputs.size() - 2;
    std::vector<ConstFieldPtr> fields(num_fields);
    for (size_t field_index = 0; field_index < num_fields; ++field_index) {
      fields[field_index] = inputs[field_index + 1].get<FEFieldPtr>().get();
    }

    std::vector<DualFieldPtr> field_sensitivities(num_fields);
    for (size_t field_index = 0; field_index < num_fields; ++field_index) {
      field_sensitivities[field_index] = inputs[field_index + 1].get_dual<FEDualPtr, FEFieldPtr>().get();
    }

    ConstFieldPtr shape_disp_ = inputs[0].get<FEFieldPtr>().get();
    DualFieldPtr shape_disp_sensitivity = inputs[0].get_dual<FEDualPtr, FEFieldPtr>().get();

    // set the dual fields for each input, using the call to residual that pulls the derivative
    weak_form->vjp(time_info, shape_disp_, fields, {}, &Z_dual_state, shape_disp_sensitivity, field_sensitivities, {});
  });

  return z.finalize();
}

}  // namespace smith
