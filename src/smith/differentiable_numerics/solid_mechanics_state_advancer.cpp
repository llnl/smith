#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/reaction.hpp"

namespace smith {

SolidMechanicsStateAdvancer::SolidMechanicsStateAdvancer(std::shared_ptr<smith::DifferentiableSolver> solver,
                                                         std::shared_ptr<smith::DirichletBoundaryConditions> vector_bcs,
                                                         std::shared_ptr<smith::WeakForm> weak_form,
                                                         smith::SecondOrderTimeIntegrationRule time_rule)
    : solver_(solver), vector_bcs_(vector_bcs), weak_form_(weak_form), time_rule_(time_rule)
{
}

std::vector<FieldState> SolidMechanicsStateAdvancer::advanceState(const FieldState& shape_disp,
                                                                  const std::vector<FieldState>& states_old,
                                                                  const std::vector<FieldState>& params,
                                                                  const TimeInfo& time_info) const
{
  double dt = time_info.dt();
  size_t cycle = time_info.cycle();
  double final_time = time_info.time() + dt;

  TimeInfo final_time_info(final_time, dt, cycle);

  // evaluate initial guesses
  FieldState displacement_guess = states_old[DISPLACEMENT] + dt * states_old[VELOCITY];

  // input fields for solid_weak_form
  std::vector<FieldState> solid_inputs{states_old[DISPLACEMENT], states_old[VELOCITY], states_old[ACCELERATION]};
  solid_inputs.insert(solid_inputs.end(), params.begin(), params.end());

  auto displacement =
      solve(displacement_guess, shape_disp, solid_inputs, final_time_info, *weak_form_, *solver_, *vector_bcs_);

  std::vector<FieldState> states = states_old;

  states[DISPLACEMENT] = displacement;
  states[VELOCITY] = time_rule_.derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                           states_old[VELOCITY], states_old[ACCELERATION]);
  states[ACCELERATION] = time_rule_.second_derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                                      states_old[VELOCITY], states_old[ACCELERATION]);

  return states;
}

std::vector<ResultantState> SolidMechanicsStateAdvancer::computeResultants(const FieldState& shape_disp, const std::vector<FieldState>& states,
                                                                           const std::vector<FieldState>& states_old,
                                                                           const std::vector<FieldState>& params, const TimeInfo& time_info) const
{
  std::vector<FieldState> solid_inputs{states[DISPLACEMENT], states_old[DISPLACEMENT], states_old[VELOCITY], states_old[ACCELERATION]};
  solid_inputs.insert(solid_inputs.end(), params.begin(), params.end());
  return {evaluateWeakForm(weak_form_, time_info, shape_disp, solid_inputs, states[DISPLACEMENT])};
}

}  // namespace smith