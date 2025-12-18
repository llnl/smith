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

std::vector<FieldState> SolidMechanicsStateAdvancer::advanceState(const FieldState& shape_disp,
                                                                  const std::vector<FieldState>& states_old_,
                                                                  const std::vector<FieldState>& params,
                                                                  const TimeInfo& time_info) const
{
  std::vector<FieldState> states_old = states_old_;
  if (time_info.cycle() == 0) {
    // input fields for solid_weak_form
    std::vector<FieldState> solid_inputs{states_old[DISPLACEMENT], states_old[VELOCITY]};
    solid_inputs.insert(solid_inputs.end(), params.begin(), params.end());
    FieldState accel_guess = states_old[ACCELERATION];
    size_t accel_index = 2;
    states_old[ACCELERATION] =
        solve(accel_guess, shape_disp, solid_inputs, time_info, *solid_dynamic_weak_forms_->quasi_static_weak_form,
              *solver_, *vector_bcs_, accel_index);
  }

  double dt = time_info.dt();
  size_t cycle = time_info.cycle();
  double final_time = time_info.time() + dt;
  TimeInfo final_time_info(final_time, dt, cycle);

  // evaluate initial guesses
  FieldState displacement_guess =
      states_old[DISPLACEMENT] + dt * states_old[VELOCITY] + (0.5 * dt * dt) * states_old[ACCELERATION];

  // input fields for solid_weak_form
  std::vector<FieldState> solid_inputs{states_old[DISPLACEMENT], states_old[VELOCITY], states_old[ACCELERATION]};
  solid_inputs.insert(solid_inputs.end(), params.begin(), params.end());

  auto displacement = solve(displacement_guess, shape_disp, solid_inputs, final_time_info,
                            *solid_dynamic_weak_forms_->time_discretized_weak_form, *solver_, *vector_bcs_);

  std::vector<FieldState> states = states_old;

  states[DISPLACEMENT] = displacement;
  states[VELOCITY] = time_rule_.derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                           states_old[VELOCITY], states_old[ACCELERATION]);
  states[ACCELERATION] = time_rule_.second_derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                                      states_old[VELOCITY], states_old[ACCELERATION]);

  return states;
}

std::vector<ResultantState> SolidMechanicsStateAdvancer::computeResultants(const FieldState& shape_disp,
                                                                           const std::vector<FieldState>& states,
                                                                           const std::vector<FieldState>& states_old,
                                                                           const std::vector<FieldState>& params,
                                                                           const TimeInfo& time_info) const
{
  std::vector<FieldState> solid_inputs{states[DISPLACEMENT], states_old[DISPLACEMENT], states_old[VELOCITY],
                                       states_old[ACCELERATION]};
  solid_inputs.insert(solid_inputs.end(), params.begin(), params.end());
  return {evaluateWeakForm(solid_dynamic_weak_forms_->time_discretized_weak_form, time_info, shape_disp, solid_inputs,
                           states[DISPLACEMENT])};
}

}  // namespace smith
