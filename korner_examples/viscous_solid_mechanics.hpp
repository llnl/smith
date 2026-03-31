#pragma once

#include "gretl/data_store.hpp"

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"

using namespace smith;

namespace custom_physics {

class SolidMechanicsStateAdvancer : public StateAdvancer {
 public:
  SolidMechanicsStateAdvancer(std::shared_ptr<NonlinearBlockSolverBase> solid_solver,
                              std::shared_ptr<DirichletBoundaryConditions> vector_bcs,
                              std::shared_ptr<SecondOrderTimeDiscretizedWeakForms> weak_form,
                              SecondOrderTimeIntegrationRule time_rule)
      : solver_(solid_solver), vector_bcs_(vector_bcs), weak_form_(weak_form), time_rule_(time_rule)
  {
  }

  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION
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

  template <int spatial_dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
  static auto buildWeakFormAndStates(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<gretl::DataStore>& graph,
                                     SecondOrderTimeIntegrationRule time_rule, std::string physics_name,
                                     const std::vector<std::string>& param_names, double initial_time = 0.0)
  {
    auto shape_disp = create_field_state(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
    auto disp = create_field_state(*graph, VectorSpace{}, physics_name + "_displacement", mesh->tag());
    auto velo = create_field_state(*graph, VectorSpace{}, physics_name + "_velocity", mesh->tag());
    auto acceleration = create_field_state(*graph, VectorSpace{}, physics_name + "_acceleration", mesh->tag());
    auto time = graph->create_state<double, double>(initial_time);
    std::vector<FieldState> params =
        createParams<ParamSpaces...>(*graph, physics_name + "_param", param_names, mesh->tag());
    std::vector<FieldState> states{disp, velo, acceleration};

    // weak form unknowns are disp, disp_old, velo_old, accel_old
    using SolidWeakFormT = SecondOrderTimeDiscretizedWeakForm<
        spatial_dim, VectorSpace, Parameters<VectorSpace, VectorSpace, VectorSpace, VectorSpace, ParamSpaces...>>;
    auto input_spaces = spaces({states[DISPLACEMENT], states[DISPLACEMENT], states[VELOCITY], states[ACCELERATION]});
    auto param_spaces = spaces(params);
    input_spaces.insert(input_spaces.end(), param_spaces.begin(), param_spaces.end());

    auto solid_mechanics_weak_form =
        std::make_shared<SolidWeakFormT>(physics_name, mesh, time_rule, space(states[DISPLACEMENT]), input_spaces);

    return std::make_tuple(shape_disp, states, params, time, solid_mechanics_weak_form);
  }

  std::pair<std::vector<FieldState>, std::vector<ReactionState>> advanceState(
      const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states_old,
      const std::vector<FieldState>& params) const override
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
        solve(displacement_guess, shape_disp, solid_inputs, final_time_info, *weak_form_->time_discretized_weak_form,
              *solver_, *vector_bcs_);

    std::vector<FieldState> states = states_old;

    states[DISPLACEMENT] = displacement;
    states[VELOCITY] = time_rule_.derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                             states_old[VELOCITY], states_old[ACCELERATION]);
    // states[VELOCITY] = (1.0 / final_time_info.dt()) * (displacement - states_old[DISPLACEMENT]);
    states[ACCELERATION] = time_rule_.second_derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                                        states_old[VELOCITY], states_old[ACCELERATION]);
    std::vector<FieldState> reaction_inputs{states[DISPLACEMENT], states_old[DISPLACEMENT], states_old[VELOCITY],
                                            states_old[ACCELERATION]};
    reaction_inputs.insert(reaction_inputs.end(), params.begin(), params.end());
    auto reaction =
        evaluateWeakForm(weak_form_->final_reaction_weak_form, time_info, shape_disp, reaction_inputs,
                         states[DISPLACEMENT]);

    return {states, {reaction}};
  }

 private:
  std::shared_ptr<NonlinearBlockSolverBase> solver_;
  std::shared_ptr<DirichletBoundaryConditions> vector_bcs_;
  std::shared_ptr<SecondOrderTimeDiscretizedWeakForms> weak_form_;
  SecondOrderTimeIntegrationRule time_rule_;
};

template <int dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
auto buildSolidMechanics(std::shared_ptr<smith::Mesh> mesh,
                         std::shared_ptr<NonlinearBlockSolverBase> d_solid_nonlinear_solver,
                         smith::SecondOrderTimeIntegrationRule time_rule, std::string physics_name,
                         const std::vector<std::string>& param_names = {})
{
  auto graph = std::make_shared<gretl::DataStore>(100);
  auto [shape_disp, states, params, time, solid_mechanics_weak_form] =
      SolidMechanicsStateAdvancer::buildWeakFormAndStates<dim, ShapeDispSpace, VectorSpace, ParamSpaces...>(
          mesh, graph, time_rule, physics_name, param_names);

  auto vector_bcs = std::make_shared<DirichletBoundaryConditions>(
      mesh->mfemParMesh(), space(states[SolidMechanicsStateAdvancer::DISPLACEMENT]));

  auto state_advancer = std::make_shared<SolidMechanicsStateAdvancer>(d_solid_nonlinear_solver, vector_bcs,
                                                                      solid_mechanics_weak_form, time_rule);

  auto physics =
      std::make_shared<DifferentiablePhysics>(mesh, graph, shape_disp, states, params, state_advancer, physics_name);

  return std::make_tuple(physics, solid_mechanics_weak_form, vector_bcs);
}
};  // namespace custom_physics
