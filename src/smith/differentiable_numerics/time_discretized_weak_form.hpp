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
  template <typename BodyForceType, int... all_active_parameters>
  void addBodyIntegralImpl(std::string body_name, BodyForceType body_integral,
                           std::integer_sequence<int, all_active_parameters...>)
  {
    addBodyIntegral(DependsOn<all_active_parameters...>{}, body_name, body_integral);
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyIntegral(std::string body_name, BodyForceType body_integral)
  {
    addBodyIntegralImpl(body_name, body_integral, std::make_integer_sequence<int, sizeof...(InputSpaces)>{});
  }
};

}  // namespace smith
