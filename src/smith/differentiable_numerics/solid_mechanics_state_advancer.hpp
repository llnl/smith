// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermomechanical_state_advancer.hpp
 * .hpp
 *
 * @brief Specifies parametrized residuals and various linearized evaluations for arbitrary nonlinear systems of
 * equations
 */

#pragma once

#include "gretl/src/data_store.hpp"
#include "smith/smith_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"

namespace smith {

class DifferentiableSolver;
class DirichletBoundaryConditions;
class WeakForm;

class SolidMechanicsStateAdvancer : public StateAdvancer {
 public:
  SolidMechanicsStateAdvancer(std::shared_ptr<DifferentiableSolver> solid_solver,
                              std::shared_ptr<DirichletBoundaryConditions> vector_bcs,
                              std::shared_ptr<WeakForm> solid_weak_form, SecondOrderTimeIntegrationRule time_rule);

  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION
  };

  template <typename FirstParamSpace, typename... ParamSpaces>
  static std::vector<FieldState> createParams(gretl::DataStore& graph, std::string name, std::string tag, int index = 0)
  {
    FieldState newParam = create_field_state(graph, FirstParamSpace{}, name + "_" + std::to_string(index), tag);
    std::vector<FieldState> end_spaces{};
    if constexpr (sizeof...(ParamSpaces) > 0) {
      end_spaces = createParams<ParamSpaces...>(graph, name, tag, ++index);
    }
    end_spaces.insert(end_spaces.begin(), newParam);
    return end_spaces;
  }

  template <int spatial_dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
  static auto buildWeakFormAndStates(std::string physics_name, const std::shared_ptr<Mesh>& mesh,
                                     const std::shared_ptr<gretl::DataStore>& graph,
                                     SecondOrderTimeIntegrationRule time_rule, double initial_time = 0.0)
  {
    auto shape_disp = create_field_state(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
    auto disp = create_field_state(*graph, VectorSpace{}, physics_name + "_displacement", mesh->tag());
    auto velo = create_field_state(*graph, VectorSpace{}, physics_name + "_velocity", mesh->tag());
    auto acceleration = create_field_state(*graph, VectorSpace{}, physics_name + "_acceleration", mesh->tag());
    auto time = graph->create_state<double, double>(initial_time);
    std::vector<FieldState> params = createParams<ParamSpaces...>(*graph, physics_name + "_param", mesh->tag());
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

  std::vector<FieldState> advanceState(const FieldState& shape_disp, const std::vector<FieldState>& states_old,
                                       const std::vector<FieldState>& params, const TimeInfo& time_info) const override;

 private:
  std::shared_ptr<DifferentiableSolver> solver_;
  std::shared_ptr<DirichletBoundaryConditions> vector_bcs_;
  std::shared_ptr<WeakForm> weak_form_;
  SecondOrderTimeIntegrationRule time_rule_;
};

}  // namespace smith