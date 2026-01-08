// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_state_advancer.hpp
 * .hpp
 *
 * @brief Specifies parameterized residuals and various linearized evaluations for arbitrary nonlinear systems of
 * equations
 */

#pragma once

#include "gretl/data_store.hpp"
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
class SecondOrderTimeDiscretizedWeakForms;

/// @brief Implementation of the StateAdvancer interface for advancing the solution of solid mechanics problems
class SolidMechanicsStateAdvancer : public StateAdvancer {
 public:
  /// @brief Constructor
  /// @param solid_solver differentiable solve
  /// @param vector_bcs Dirichlet boundary conditions that can be applies to vector unknowns
  /// @param solid_dynamic_weak_forms The weak-forms for time discretized solid mechanics equations
  /// @param time_rule The specific time-integration rule, typically Implicit Newmark or Quasi-static
  SolidMechanicsStateAdvancer(std::shared_ptr<DifferentiableSolver> solid_solver,
                              std::shared_ptr<DirichletBoundaryConditions> vector_bcs,
                              std::shared_ptr<SecondOrderTimeDiscretizedWeakForms> solid_dynamic_weak_forms,
                              SecondOrderTimeIntegrationRule time_rule);

  /// State enum for indexing convenience
  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION
  };

  /// @brief Recursive function for constructing parameter FieldStates of the appropriate space and name, register it on
  /// the gretl graph.
  template <typename FirstParamSpace, typename... ParamSpaces>
  static std::vector<FieldState> createParams(gretl::DataStore& graph, const std::string& name,
                                              const std::vector<std::string>& param_names, const std::string& tag,
                                              size_t index = 0)
  {
    FieldState newParam = createFieldState(graph, FirstParamSpace{}, name + "_" + param_names[index], tag);
    std::vector<FieldState> end_spaces{};
    if constexpr (sizeof...(ParamSpaces) > 0) {
      end_spaces = createParams<ParamSpaces...>(graph, name, param_names, tag, ++index);
    }
    end_spaces.insert(end_spaces.begin(), newParam);
    return end_spaces;
  }

  /// @brief Utility function to consistently construct all the weak forms and FieldStates for a solid mechanics
  /// application you will get back: shape_disp, states, params, time, and solid_mechanics_weak_form
  template <int spatial_dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
  static auto buildWeakFormAndStates(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<gretl::DataStore>& graph,
                                     SecondOrderTimeIntegrationRule time_rule, std::string physics_name,
                                     const std::vector<std::string>& param_names, double initial_time = 0.0)
  {
    auto shape_disp = createFieldState(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
    auto disp = createFieldState(*graph, VectorSpace{}, physics_name + "_displacement", mesh->tag());
    auto velo = createFieldState(*graph, VectorSpace{}, physics_name + "_velocity", mesh->tag());
    auto acceleration = createFieldState(*graph, VectorSpace{}, physics_name + "_acceleration", mesh->tag());
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

  /// @overload
  std::vector<FieldState> advanceState(const TimeInfo& time_info, const FieldState& shape_disp,
                                       const std::vector<FieldState>& states_old,
                                       const std::vector<FieldState>& params) const override;

  /// @overload
  std::vector<ReactionState> computeReactions(const TimeInfo& time_info, const FieldState& shape_disp,
                                              const std::vector<FieldState>& states,
                                              const std::vector<FieldState>& params) const override;

 private:
  std::shared_ptr<DifferentiableSolver> solver_;             ///< Differentiable solver
  std::shared_ptr<DirichletBoundaryConditions> vector_bcs_;  ///< Dirichlet boundary conditions on a vector-field
  std::shared_ptr<SecondOrderTimeDiscretizedWeakForms>
      solid_dynamic_weak_forms_;  ///< Solid mechanics time discretized weak forms, user must setup the appropriate
                                  ///< integrals.  Has both the time discretized and the undiscretized weak forms.
  SecondOrderTimeIntegrationRule time_rule_;  ///< second order time integration rule.  Can compute u, u_dot, u_dot_dot,
                                              ///< given the current predicted u and the previous u, u_dot, u_dot_dot
};

}  // namespace smith
