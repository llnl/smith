#pragma once

#include "gretl/data_store.hpp"

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"

using namespace smith;

namespace custom_physics {
class QuasistaticSolidThermoMechanicsStateAdvancer : public StateAdvancer {
 public:
  QuasistaticSolidThermoMechanicsStateAdvancer(std::shared_ptr<DifferentiableSolver> solid_solver,
                                               std::shared_ptr<DifferentiableSolver> thermal_solver,
                                               std::shared_ptr<DirichletBoundaryConditions> vector_bcs,
                                               std::shared_ptr<DirichletBoundaryConditions> scalar_bcs,
                                               std::shared_ptr<SecondOrderTimeDiscretizedWeakForms> solid_weak_form,
                                               std::shared_ptr<SecondOrderTimeDiscretizedWeakForms> thermal_weak_form,
                                               SecondOrderTimeIntegrationRule solid_time_rule,
                                               SecondOrderTimeIntegrationRule thermal_time_rule)
      : solid_solver_(solid_solver),
        thermal_solver_(thermal_solver),
        vector_bcs_(vector_bcs),
        scalar_bcs_(scalar_bcs),
        solid_weak_form_(solid_weak_form),
        thermal_weak_form_(thermal_weak_form),
        solid_time_rule_(solid_time_rule),
        thermal_time_rule_(thermal_time_rule)
  {
  }
  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION,
    TEMPERATURE,
    TEMPERATURE_DOT,
    TEMPERATURE_DDOT
  };

  template <typename FirstParamSpace, typename... ParamSpaces>
  static std::vector<FieldState> createParams(gretl::DataStore& graph, const std::string& name,
                                              const std::vector<std::string>& param_names, const std::string& tag,
                                              size_t index = 0)
  {
    FieldState newParam = create_field_state(graph, FirstParamSpace{}, name + "_" + param_names[index], tag);
    std::vector<FieldState> end_spaces{};
    if constexpr (sizeof...(ParamSpaces) > 0) {
      end_spaces = createParams<ParamSpaces...>(graph, name, param_names, tag, ++index);
    }
    end_spaces.insert(end_spaces.begin(), newParam);
    return end_spaces;
  }

  template <int spatial_dim, typename ShapeDispSpace, typename VectorSpace, typename ScalarSpace,
            typename... ParamSpaces>
  static auto buildWeakFormAndStates(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<gretl::DataStore>& graph,
                                     SecondOrderTimeIntegrationRule solid_time_rule,
                                     SecondOrderTimeIntegrationRule thermal_time_rule, std::string physics_name,
                                     const std::vector<std::string>& param_names, double initial_time = 0.0)
  {
    auto shape_disp = create_field_state(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
    auto disp = create_field_state(*graph, VectorSpace{}, physics_name + "_displacement", mesh->tag());
    auto velo = create_field_state(*graph, VectorSpace{}, physics_name + "_velocity", mesh->tag());
    auto acceleration = create_field_state(*graph, VectorSpace{}, physics_name + "_acceleration", mesh->tag());
    auto temperature = create_field_state(*graph, ScalarSpace{}, physics_name + "_temperature", mesh->tag());
    auto temperature_rate = create_field_state(*graph, ScalarSpace{}, physics_name + "_temperature_rate", mesh->tag());
    auto temperature_rate_rate =
        create_field_state(*graph, ScalarSpace{}, physics_name + "_temperature_rate_rate", mesh->tag());
    auto time = graph->create_state<double, double>(initial_time);
    std::vector<FieldState> params =
        createParams<ParamSpaces...>(*graph, physics_name + "_param", param_names, mesh->tag());
    std::vector<FieldState> states{disp, velo, acceleration, temperature, temperature_rate, temperature_rate_rate};

    using SolidWeakFormT =
        SecondOrderTimeDiscretizedWeakForm<spatial_dim, VectorSpace,
                                           Parameters<VectorSpace, VectorSpace, VectorSpace, VectorSpace, ScalarSpace,
                                                      ScalarSpace, ScalarSpace, ParamSpaces...>>;

    auto solid_input_spaces =
        spaces({states[DISPLACEMENT], states[DISPLACEMENT], states[VELOCITY], states[ACCELERATION], states[TEMPERATURE],
                states[TEMPERATURE_DOT], states[TEMPERATURE_DDOT]});

    auto solid_param_spaces = spaces(params);
    solid_input_spaces.insert(solid_input_spaces.end(), solid_param_spaces.begin(), solid_param_spaces.end());

    auto solid_mechanics_weak_form = std::make_shared<SolidWeakFormT>(physics_name, mesh, solid_time_rule,
                                                                      space(states[DISPLACEMENT]), solid_input_spaces);

    using ThermalWeakFormT =
        SecondOrderTimeDiscretizedWeakForm<spatial_dim, ScalarSpace,
                                           Parameters<ScalarSpace, ScalarSpace, ScalarSpace, ScalarSpace, VectorSpace,
                                                      VectorSpace, VectorSpace, ParamSpaces...>>;

    auto thermal_input_spaces =
        spaces({states[TEMPERATURE], states[TEMPERATURE], states[TEMPERATURE_DOT], states[TEMPERATURE_DDOT],
                states[DISPLACEMENT], states[VELOCITY], states[ACCELERATION]});

    auto thermal_param_spaces = spaces(params);
    thermal_input_spaces.insert(thermal_input_spaces.end(), thermal_param_spaces.begin(), thermal_param_spaces.end());

    auto thermal_mechanics_weak_form = std::make_shared<ThermalWeakFormT>(
        physics_name, mesh, thermal_time_rule, space(states[TEMPERATURE]), thermal_input_spaces);

    return std::make_tuple(shape_disp, states, params, time, solid_mechanics_weak_form, thermal_mechanics_weak_form);
  }

  std::vector<FieldState> advanceState(const FieldState& shape_disp, const std::vector<FieldState>& states_old,
                                       const std::vector<FieldState>& params, const TimeInfo& time_info) const override
  {
    double dt = time_info.dt();
    size_t cycle = time_info.cycle();
    double final_time = time_info.time() + dt;
    TimeInfo final_time_info(final_time, dt, cycle);

    // Evolve Temperature
    // evaluate initial guess
    // SLIC_INFO("Thermal Solve");
    FieldState temperature_guess = states_old[TEMPERATURE] + dt * states_old[TEMPERATURE_DOT];
    std::vector<FieldState> thermal_inputs{states_old[TEMPERATURE],      states_old[TEMPERATURE_DOT],
                                           states_old[TEMPERATURE_DDOT], states_old[DISPLACEMENT],
                                           states_old[VELOCITY],         states_old[ACCELERATION]};
    thermal_inputs.insert(thermal_inputs.end(), params.begin(), params.end());

    auto temperature = solve(temperature_guess, shape_disp, thermal_inputs, final_time_info,
                             *thermal_weak_form_->time_discretized_weak_form, *thermal_solver_, *scalar_bcs_);

    // Evolve Deformation
    // evaluate initial guess
    //
    // SLIC_INFO("Solids Solve");
    FieldState displacement_guess = states_old[DISPLACEMENT] + dt * states_old[VELOCITY];
    std::vector<FieldState> solid_inputs{states_old[DISPLACEMENT],    states_old[VELOCITY],
                                         states_old[ACCELERATION],    temperature,
                                         states_old[TEMPERATURE_DOT], states_old[TEMPERATURE_DDOT]};
    solid_inputs.insert(solid_inputs.end(), params.begin(), params.end());
    auto displacement = solve(displacement_guess, shape_disp, solid_inputs, final_time_info,
                              *solid_weak_form_->time_discretized_weak_form, *solid_solver_, *vector_bcs_);

    std::vector<FieldState> states = states_old;
    states[DISPLACEMENT] = displacement;
    states[VELOCITY] = solid_time_rule_.derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                                   states_old[VELOCITY], states_old[ACCELERATION]);
    states[ACCELERATION] = solid_time_rule_.second_derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                                              states_old[VELOCITY], states_old[ACCELERATION]);
    states[TEMPERATURE] = temperature;
    states[TEMPERATURE_DOT] = thermal_time_rule_.derivative(final_time_info, temperature, states_old[TEMPERATURE],
                                                            states_old[TEMPERATURE_DOT], states_old[TEMPERATURE_DDOT]);

    states[TEMPERATURE_DDOT] =
        thermal_time_rule_.second_derivative(final_time_info, temperature, states_old[TEMPERATURE],
                                             states_old[TEMPERATURE_DOT], states_old[TEMPERATURE_DDOT]);
    return states;
  }

  std::vector<ResultantState> computeResultants(const FieldState& shape_disp, const std::vector<FieldState>& states,
                                                const std::vector<FieldState>& params,
                                                const TimeInfo& time_info) const override
  {
    std::vector<FieldState> inputs{states[DISPLACEMENT], states[VELOCITY],        states[ACCELERATION],
                                   states[TEMPERATURE],  states[TEMPERATURE_DOT], states[TEMPERATURE_DDOT]};
    inputs.insert(inputs.end(), params.begin(), params.end());
    return {evaluateWeakForm(solid_weak_form_->quasi_static_weak_form, time_info, shape_disp, inputs,
                             states[DISPLACEMENT])};
  }

 private:
  std::shared_ptr<DifferentiableSolver> solid_solver_;
  std::shared_ptr<DifferentiableSolver> thermal_solver_;
  std::shared_ptr<DirichletBoundaryConditions> vector_bcs_;
  std::shared_ptr<DirichletBoundaryConditions> scalar_bcs_;
  std::shared_ptr<SecondOrderTimeDiscretizedWeakForms> solid_weak_form_;
  std::shared_ptr<SecondOrderTimeDiscretizedWeakForms> thermal_weak_form_;
  SecondOrderTimeIntegrationRule solid_time_rule_;
  SecondOrderTimeIntegrationRule thermal_time_rule_;
};

template <int dim, typename ShapeDispSpace, typename VectorSpace, typename ScalarSpace, typename... ParamSpaces>
auto buildThermoMechanics(std::shared_ptr<smith::Mesh> mesh,
                          std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver,
                          std::shared_ptr<DifferentiableSolver> d_thermal_solver,
                          smith::SecondOrderTimeIntegrationRule solid_time_rule,
                          smith::SecondOrderTimeIntegrationRule thermal_time_rule, std::string physics_name,
                          const std::vector<std::string>& param_names = {})
{
  auto graph = std::make_shared<gretl::DataStore>(100);
  auto [shape_disp, states, params, time, solid_mechanics_weak_form, thermal_mechanics_weak_form] =
      QuasistaticSolidThermoMechanicsStateAdvancer::buildWeakFormAndStates<dim, ShapeDispSpace, VectorSpace,
                                                                           ScalarSpace, ParamSpaces...>(
          mesh, graph, solid_time_rule, thermal_time_rule, physics_name, param_names);

  auto vector_bcs = std::make_shared<DirichletBoundaryConditions>(
      mesh->mfemParMesh(), space(states[QuasistaticSolidThermoMechanicsStateAdvancer::DISPLACEMENT]));

  auto scalar_bcs = std::make_shared<DirichletBoundaryConditions>(
      mesh->mfemParMesh(), space(states[QuasistaticSolidThermoMechanicsStateAdvancer::TEMPERATURE]));
  auto state_advancer = std::make_shared<QuasistaticSolidThermoMechanicsStateAdvancer>(
      d_solid_nonlinear_solver, d_thermal_solver, vector_bcs, scalar_bcs, solid_mechanics_weak_form,
      thermal_mechanics_weak_form, solid_time_rule, thermal_time_rule);
  auto physics =
      std::make_shared<DifferentiablePhysics>(mesh, graph, shape_disp, states, params, state_advancer, physics_name);

  return std::make_tuple(physics, solid_mechanics_weak_form, thermal_mechanics_weak_form, vector_bcs, scalar_bcs);
}

};  // namespace custom_physics
