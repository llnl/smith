// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"

namespace smith::thermomechanics {

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T, int dim>
auto greenStrain(const tensor<T, dim, dim>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

/// @brief Green-Saint Venant isotropic thermoelastic model
struct GreenSaintVenantThermoelasticMaterial {
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity

  /// internal variables for the material model
  struct State {
    double strain_trace;  ///< trace of Green-Saint Venant strain tensor
  };

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename T1, typename T2, typename T3, int dim>
  auto operator()(State& state, const tensor<T1, dim, dim>& grad_u, T2 theta, const tensor<T3, dim>& grad_theta) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    static constexpr auto I = Identity<dim>();
    auto F = grad_u + I;
    const auto Eg = greenStrain(grad_u);
    const auto trEg = tr(Eg);

    // stress
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    const auto Piola = dot(F, S);

    // internal heat source
    const auto s0 = -3.0 * K * alpha * theta * (trEg - state.strain_trace);

    // heat flux
    const auto q0 = -kappa * grad_theta;

    state.strain_trace = get_value(trEg);

    return smith::tuple{Piola, C_v, s0, q0};
  }

  /**
   * @brief evaluate free energy density
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   */
  template <typename T1, typename T2, int dim>
  auto calculateFreeEnergy(const tensor<T1, dim, dim>& grad_u, T2 theta) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    auto strain = greenStrain(grad_u);
    auto trE = tr(strain);
    auto psi_1 = G * squared_norm(dev(strain)) + 0.5 * K * trE * trE;
    using std::log;
    auto logT = log(theta / theta_ref);
    auto psi_2 = C_v * (theta - theta_ref - theta * logT);
    auto psi_3 = -3.0 * K * alpha * (theta - theta_ref) * trE;
    return psi_1 + psi_2 + psi_3;
  }
};

/// @brief Green-Saint Venant isotropic thermoelastic model
struct ParameterizedGreenSaintVenantThermoelasticMaterial {
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha0;     ///< reference value of thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity

  /// internal variables for the material model
  struct State {
    double strain_trace;  ///< trace of Green-Saint Venant strain tensor
  };

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   * @tparam T4 Type of the coefficient of thermal expansion scale factor
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in] thermal_expansion_scaling Parameterized scale factor on the coefficient of thermal expansion
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(State& state, const tensor<T1, dim, dim>& grad_u, T2 theta, const tensor<T3, dim>& grad_theta,
                  T4 thermal_expansion_scaling) const
  {
    auto [scale, unused] = thermal_expansion_scaling;
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    static constexpr auto I = Identity<dim>();
    auto F = grad_u + I;
    const auto Eg = greenStrain(grad_u);
    const auto trEg = tr(Eg);
    auto alpha = alpha0 * scale;

    // stress
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    const auto Piola = dot(F, S);

    // internal heat source
    const auto s0 = -3.0 * K * alpha * theta * (trEg - state.strain_trace);

    // heat flux
    const auto q0 = -kappa * grad_theta;

    state.strain_trace = get_value(trEg);

    return smith::tuple{Piola, C_v, s0, q0};
  }

  /**
   * @brief evaluate free energy density
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] thermal_expansion_scaling a scaling factor to be applied to alpha0
   */
  template <typename T1, typename T2, typename T3, int dim>
  auto calculateFreeEnergy(const tensor<T1, dim, dim>& grad_u, T2 theta, T3 thermal_expansion_scaling) const
  {
    auto [scale, unused] = thermal_expansion_scaling;
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    auto strain = greenStrain(grad_u);
    auto trE = tr(strain);
    const double alpha = alpha0 * scale;
    auto psi_1 = G * squared_norm(dev(strain)) + 0.5 * K * trE * trE;
    using std::log;
    auto logT = log(theta / theta_ref);
    auto psi_2 = C_v * (theta - theta_ref - theta * logT);
    auto psi_3 = -3.0 * K * alpha * (theta - theta_ref) * trE;
    return psi_1 + psi_2 + psi_3;
  }
};

}  // namespace smith::thermomechanics
