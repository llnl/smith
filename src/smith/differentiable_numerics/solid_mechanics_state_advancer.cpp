// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/reaction.hpp"

namespace smith {

SolidMechanicsStateAdvancer::SolidMechanicsStateAdvancer(
    std::shared_ptr<smith::DifferentiableSolver> solver, std::shared_ptr<smith::DirichletBoundaryConditions> vector_bcs,
    std::shared_ptr<SecondOrderTimeDiscretizedWeakForms> solid_dynamic_weak_forms,
    smith::SecondOrderTimeIntegrationRule time_rule)
    : solver_(solver),
      vector_bcs_(vector_bcs),
      solid_dynamic_weak_forms_(solid_dynamic_weak_forms),
      time_rule_(time_rule)
{
}

std::vector<FieldState> SolidMechanicsStateAdvancer::advanceState(const TimeInfo& time_info,
                                                                  const FieldState& shape_disp,
                                                                  const std::vector<FieldState>& states_old_,
                                                                  const std::vector<FieldState>& params) const
{
  std::vector<FieldState> states_old = states_old_;
  if (time_info.cycle() == 0) {
    states_old[ACCELERATION] = solve(*solid_dynamic_weak_forms_->quasi_static_weak_form, shape_disp, states_old_,
                                     params, time_info, *solver_, *vector_bcs_, ACCELERATION);
  }

  TimeInfo final_time_info = time_info.endTimeInfo();

  std::vector<FieldState> solid_inputs{states_old[DISPLACEMENT], states_old[DISPLACEMENT], states_old[VELOCITY],
                                       states_old[ACCELERATION]};

  auto displacement = solve(*solid_dynamic_weak_forms_->time_discretized_weak_form, shape_disp, solid_inputs, params,
                            final_time_info, *solver_, *vector_bcs_);

  std::vector<FieldState> states = states_old;

  states[DISPLACEMENT] = displacement;
  states[VELOCITY] = time_rule_.derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                           states_old[VELOCITY], states_old[ACCELERATION]);
  states[ACCELERATION] = time_rule_.second_derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                                      states_old[VELOCITY], states_old[ACCELERATION]);

  return states;
}

std::vector<ReactionState> SolidMechanicsStateAdvancer::computeReactions(const TimeInfo& time_info,
                                                                         const FieldState& shape_disp,
                                                                         const std::vector<FieldState>& states,
                                                                         const std::vector<FieldState>& params) const
{
  std::vector<FieldState> solid_inputs{states[DISPLACEMENT], states[VELOCITY], states[ACCELERATION]};
  solid_inputs.insert(solid_inputs.end(), params.begin(), params.end());
  return {evaluateWeakForm(solid_dynamic_weak_forms_->quasi_static_weak_form, time_info, shape_disp, solid_inputs,
                           states[DISPLACEMENT])};
}

}  // namespace smith
