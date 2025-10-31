// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file material_verification_tools.hpp
 *
 * @brief Utility for testing material model output
 */

#pragma once

#include <functional>

#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"

namespace smith {

/**
 * @brief Drive the material model thorugh a uniaxial tension experiment
 *
 * Drives material model through specified axial displacement gradient history.
 * The time elaspses from 0 up to t_max.
 * Currently only implemented for isotropic materials (or orthotropic materials with the
 * principal axes aligned with the coordinate directions).
 *
 * @param t_max upper limit of the time interval.
 * @param num_steps The number of discrete time points at which the response is sampled (uniformly spaced).
 *        This is inclusive of the point at time zero.
 * @param material The material model to use
 * @param initial_state The state variable collection for this material, set to the desired initial
 *        condition.
 * @param epsilon_xx A function describing the desired axial displacement gradient as a function of time.
 *        (NB axial displacement gradient is equivalent to engineering strain).
 * @param parameter_functions Pack of functions that return each parameter as a function of time. Leave
 *        empty if the material has no parameters.
 */
template <typename MaterialType, typename StateType, typename... parameter_types>
auto uniaxial_stress_test(double t_max, size_t num_steps, const MaterialType material, const StateType initial_state,
                          std::function<double(double)> epsilon_xx, const parameter_types... parameter_functions)
{
  double t = 0;
  const double dt = t_max / double(num_steps - 1);

  auto state = initial_state;

  auto sigma_yy_and_zz = [&](auto x) {
    auto epsilon_yy = x[0];
    auto epsilon_zz = x[1];
    using T = decltype(epsilon_yy);
    tensor<T, 3, 3> du_dx{};
    du_dx[0][0] = epsilon_xx(t);
    du_dx[1][1] = epsilon_yy;
    du_dx[2][2] = epsilon_zz;
    auto state_copy = state;
    auto stress = material(state_copy, du_dx, parameter_functions(t)...);
    return tensor{{stress[1][1], stress[2][2]}};
  };

  std::vector<tuple<double, tensor<double, 3, 3>, tensor<double, 3, 3>, StateType> > output_history;
  output_history.reserve(num_steps);

  tensor<double, 3, 3> dudx{};
  for (size_t i = 0; i < num_steps; i++) {
    auto initial_guess = tensor<double, 2>{dudx[1][1], dudx[2][2]};
    auto epsilon_yy_and_zz = find_root(sigma_yy_and_zz, initial_guess);
    dudx[0][0] = epsilon_xx(t);
    dudx[1][1] = epsilon_yy_and_zz[0];
    dudx[2][2] = epsilon_yy_and_zz[1];

    auto stress = material(state, dudx, parameter_functions(t)...);
    output_history.push_back(tuple{t, dudx, stress, state});

    t += dt;
  }

  return output_history;
}

/**
 * @brief Drive a rate-dependent material model thorugh a uniaxial tension experiment
 *
 * Drives material model through specified axial displacement gradient history.
 * The time elapses from 0 up to t_max.
 * Currently only implemented for isotropic materials (or orthotropic materials with the
 * principal axes aligned with the coordinate directions).
 */
template <typename MaterialType, typename StateType, typename... parameter_types>
auto uniaxial_stress_test_rate_dependent(double t_max, size_t num_steps, const MaterialType material,
                                         const StateType initial_state, std::function<double(double)> epsilon_xx,
                                         const parameter_types... parameter_functions)
{
  const double dt = t_max / double(num_steps - 1);
  auto mat_with_dt = [&material, dt](auto& state, auto du_dx, parameter_types... parameters) {
    return material(state, dt, du_dx, parameters...);
  };
  return uniaxial_stress_test(t_max, num_steps, mat_with_dt, initial_state, epsilon_xx, parameter_functions...);
}

/**
 * @brief This function takes a material model (and associate state variables),
 *        subjects it to a time history of stimuli, described by `functions ... f`,
 *        and returns the outputs at each step. This is intended to be used for testing
 *        materials, to ensure their response is in agreement with known data (analytic or
 *        experimental).
 *
 * @tparam MaterialType the type of the material model under test
 * @tparam StateType the associated state variables to be provided to the material
 * @tparam functions a variadic list of callables
 * @param t_max the final time value for
 * @param num_steps the number of timesteps to be
 * @param material an instance of a material model under test
 * @param initial_state the initial conditions for materials that exhibit hysteresis
 * @param f the functions (e.g. std::function, lambdas, etc) that are used to
 *          generate the inputs to the material model at each timestep
 * @return a std::vector of tuples of the form: ( time, state(t), f(t) ... , output(t) )
 *         evaluated at each step
 */
template <typename MaterialType, typename StateType, typename... functions>
auto single_quadrature_point_test(double t_max, size_t num_steps, const MaterialType material,
                                  const StateType initial_state, const functions... f)
{
  double t = 0;
  const double dt = t_max / double(num_steps - 1);
  auto state = initial_state;

  using output_type = decltype(std::tuple{t, state, f(0.0)..., decltype(material(state, dt, f(0.0)...)){}});
  std::vector<output_type> history;
  history.reserve(num_steps);

  for (size_t i = 0; i < num_steps; i++) {
    auto material_output = material(state, dt, f(t)...);
    history.push_back(std::tuple{t, state, f(t)..., material_output});
    t += dt;
  }

  return history;
}

}  // namespace smith
