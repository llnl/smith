// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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

/// @brief A time discretized weakform get a TimeInfo object passed as arguments to q-function (lambdas which are
/// integrated over quadrature points) so users can have access to time increments, and timestep cycle.  These
/// quantities are often valuable for time integrated PDEs.
/// @tparam OutputSpace The output residual for the weak form (test-space)
/// @tparam ...InputSpaces All the input FiniteElementState fields (test-spaces)
/// @tparam spatial_dim The spatial dimension for the problem
template <int spatial_dim, typename OutputSpace, typename... InputSpaces>
class TimeDiscretizedWeakForm : public FunctionalWeakForm<spatial_dim, OutputSpace, InputSpaces...> {
 public:
  using WeakFormT = FunctionalWeakForm<spatial_dim, OutputSpace, InputSpaces...>;

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

/// @brief Useful for time-discretized PDE of second order (involves for first and second derivatives of time).  Users
/// write q-functions in terns of u, u_dot, u_dot_dot, and the weak form is transformed by the
/// SecondOrderTimeIntegrationRule so that is it globally a function of u, u_old, u_dot_old, u_dot_dot_old, with u as
/// the distinct unknown for the time discretized system.
/// @tparam OutputSpace The output residual for the weak form (test-space)
/// @tparam ...InputSpaces All the input FiniteElementState fields (test-spaces)
/// @tparam spatial_dim The spatial dimension for the problem
template <int spatial_dim, typename OutputSpace, typename... InputSpaces>
class SecondOrderTimeDiscretizedWeakForm : public TimeDiscretizedWeakForm<spatial_dim, OutputSpace, InputSpaces...> {
 public:
  static constexpr int NUM_STATE_VARS = 4;  //< u, u_old, v_old, a_old

  using WeakFormT = TimeDiscretizedWeakForm<spatial_dim, OutputSpace, InputSpaces...>;

  /// @brief Constructor
  SecondOrderTimeDiscretizedWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh,
                                     SecondOrderTimeIntegrationRule time_rule,
                                     const mfem::ParFiniteElementSpace& output_mfem_space,
                                     const typename WeakFormT::SpacesT& input_mfem_spaces)
      : WeakFormT(physics_name, mesh, output_mfem_space, input_mfem_spaces), time_rule_(time_rule)
  {
  }

  /// @overload
  template <int... active_parameters, typename BodyIntegralType>
  void addBodyIntegral(DependsOn<active_parameters...> /*depends_on*/, std::string body_name,
                       BodyIntegralType integrand)
  {
    auto time_rule = time_rule_;
    WeakFormT::addBodyIntegral(DependsOn<0, 1, 2, 3, active_parameters + NUM_STATE_VARS...>{}, body_name,
                               [integrand, time_rule](const TimeInfo& t, auto X, auto F, auto F_old, auto F_dot_old,
                                                      auto F_dot_dot_old, auto... inputs) {
                                 return integrand(t, X, time_rule.value(t, F, F_old, F_dot_old, F_dot_dot_old),
                                                  time_rule.derivative(t, F, F_old, F_dot_old, F_dot_dot_old),
                                                  time_rule.second_derivative(t, F, F_old, F_dot_old, F_dot_dot_old),
                                                  inputs...);
                               });
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyIntegral(std::string body_name, BodyForceType body_integral)
  {
    addBodyIntegral(DependsOn<>{}, body_name, body_integral);
  }

  SecondOrderTimeIntegrationRule time_rule_;  /// encodes the time integration rule
};

}  // namespace smith