// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file time_discretized_weak_form.hpp
 *
 * @brief Specifies parametrized residuals and various linearized evaluations for arbitrary nonlinear systems of
 * equations
 */

#pragma once

#include "smith/physics/functional_weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"

namespace smith {

template <int spatial_dim, typename OutputSpace, typename inputs = Parameters<>>
class TimeDiscretizedWeakForm;

/// @brief A time discretized weakform gets a TimeInfo object passed as arguments to q-function (lambdas which are
/// integrated over quadrature points) so users can have access to time increments, and timestep cycle.  These
/// quantities are often valuable for time integrated PDEs.
/// @tparam OutputSpace The output residual for the weak form (test-space)
/// @tparam ...InputSpaces All the input FiniteElementState fields (trial-spaces)
/// @tparam spatial_dim The spatial dimension for the problem
template <int spatial_dim, typename OutputSpace, typename... InputSpaces>
class TimeDiscretizedWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>>
    : public FunctionalWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>> {
 public:
  using WeakFormT = FunctionalWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>>;  ///< using

  /// Constructor
  TimeDiscretizedWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh,
                          const mfem::ParFiniteElementSpace& output_mfem_space,
                          const typename WeakFormT::SpacesT& input_mfem_spaces)
      : WeakFormT(physics_name, mesh, output_mfem_space, input_mfem_spaces)
  {
  }

  /// @overload
  template <int... active_parameters, typename BodyIntegralType>
  void addBodyIntegral(DependsOn<active_parameters...> depends_on, std::string body_name, BodyIntegralType integrand)
  {
    const double* dt = &this->dt_;
    const size_t* cycle = &this->cycle_;
    WeakFormT::addBodyIntegral(depends_on, body_name, [dt, cycle, integrand](double t, auto X, auto... inputs) {
      TimeInfo time_info(t, *dt, *cycle);
      return integrand(time_info, X, inputs...);
    });
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyIntegral(std::string body_name, BodyForceType body_integral)
  {
    addBodyIntegral(DependsOn<>{}, body_name, body_integral);
  }
};

/// @brief A container holding the two types of weak forms useful for solving time discretized second order (in time)
/// systems of equations
class SecondOrderTimeDiscretizedWeakForms {
 public:
  std::shared_ptr<WeakForm> time_discretized_weak_form;  ///< this publically available abstract weak form is a
                                                         ///< functions of the current u, u_old, v_old, and a_old,
  std::shared_ptr<WeakForm> quasi_static_weak_form;      ///< this publically available abstract weak form is structly a
                                                     ///< function of the current u, v, and a (no time discretization)
};

template <int spatial_dim, typename OutputSpace, typename inputs = Parameters<>>
class SecondOrderTimeDiscretizedWeakForm;

/// @brief Useful for time-discretized PDEs of second order (involves for first and second derivatives of time).  Users
/// write q-functions in terns of u, u_dot, u_dot_dot, and the weak form is transformed by the
/// SecondOrderTimeIntegrationRule so that is it globally a function of u, u_old, u_dot_old, u_dot_dot_old, with u as
/// the distinct unknown for the time discretized system.
/// @tparam spatial_dim Spatial dimension, 2 or 3.
/// @tparam OutputSpace The space corresponding to the output residual for the weak form (test-space).
/// @tparam TrialInputSpace The space corresponding to the predicted solution u, i.e., the trial solution, the unique
/// unknown of the time discretized equation.
/// @tparam ...InputSpaces Spaces for all the remaining input FiniteElementState fields.
/// @tparam spatial_dim The spatial dimension for the problem.
template <int spatial_dim, typename OutputSpace, typename TrialInputSpace, typename... InputSpaces>
class SecondOrderTimeDiscretizedWeakForm<spatial_dim, OutputSpace, Parameters<TrialInputSpace, InputSpaces...>>
    : public SecondOrderTimeDiscretizedWeakForms {
 public:
  static constexpr int NUM_STATE_VARS = 4;  ///< u, u_old, v_old, a_old

  using TimeDiscretizedWeakFormT =
      TimeDiscretizedWeakForm<spatial_dim, OutputSpace, Parameters<TrialInputSpace, InputSpaces...>>;  ///< using
  using QuasiStaticWeakFormT =
      TimeDiscretizedWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>>;  ///< using

  /// @brief Constructor
  SecondOrderTimeDiscretizedWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh,
                                     SecondOrderTimeIntegrationRule time_rule,
                                     const mfem::ParFiniteElementSpace& output_mfem_space,
                                     const typename TimeDiscretizedWeakFormT::SpacesT& input_mfem_spaces)
      : time_rule_(time_rule)
  {
    time_discretized_weak_form_ =
        std::make_shared<TimeDiscretizedWeakFormT>(physics_name, mesh, output_mfem_space, input_mfem_spaces);
    time_discretized_weak_form = time_discretized_weak_form_;

    typename TimeDiscretizedWeakFormT::SpacesT input_mfem_spaces_trial_removed(std::next(input_mfem_spaces.begin()),
                                                                               input_mfem_spaces.end());
    quasi_static_weak_form_ =
        std::make_shared<QuasiStaticWeakFormT>(physics_name, mesh, output_mfem_space, input_mfem_spaces_trial_removed);
    quasi_static_weak_form = quasi_static_weak_form_;
  }

  /// @overload
  template <int... active_parameters, typename BodyIntegralType>
  void addBodyIntegral(DependsOn<active_parameters...> /*depends_on*/, std::string body_name,
                       BodyIntegralType integrand)
  {
    auto time_rule = time_rule_;
    time_discretized_weak_form_->addBodyIntegral(
        DependsOn<0, 1, 2, 3, NUM_STATE_VARS + active_parameters...>{}, body_name,
        [integrand, time_rule](const TimeInfo& t, auto X, auto U, auto U_old, auto U_dot_old, auto U_dot_dot_old,
                               auto... inputs) {
          return integrand(t, X, time_rule.value(t, U, U_old, U_dot_old, U_dot_dot_old),
                           time_rule.derivative(t, U, U_old, U_dot_old, U_dot_dot_old),
                           time_rule.second_derivative(t, U, U_old, U_dot_old, U_dot_dot_old), inputs...);
        });
    quasi_static_weak_form_->addBodyIntegral(DependsOn<0, 1, 2, NUM_STATE_VARS - 1 + active_parameters...>{}, body_name,
                                             integrand);
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyIntegral(std::string body_name, BodyForceType body_integral)
  {
    addBodyIntegral(DependsOn<>{}, body_name, body_integral);
  }

 private:
  std::shared_ptr<TimeDiscretizedWeakFormT>
      time_discretized_weak_form_;  ///< fully templated time discretized weak form (with time integration rule injected
                                    ///< into the q-function)
  std::shared_ptr<QuasiStaticWeakFormT>
      quasi_static_weak_form_;  ///< fully template underlying weak form (no time integration included, a function of
                                ///< current u, v, and a)

  SecondOrderTimeIntegrationRule time_rule_;  ///< encodes the time integration rule
};

}  // namespace smith
