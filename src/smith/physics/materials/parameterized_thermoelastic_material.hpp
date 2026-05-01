// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/physics/materials/green_saint_venant_thermoelastic.hpp"

/// Thermomechanics helper data types
namespace smith::thermomechanics {

/**
 * @brief Green-Saint Venant isotropic thermoelastic material model
 *
 */
struct ParameterizedThermoelasticMaterial {
  double density;    ///< density
  double E0;         ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha0;     ///< reference value of thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa0;     ///< thermal conductivity

  /// internal variables for the material model
  struct State {
    double strain_trace;  ///< trace of Green-Saint Venant strain tensor
  };

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 3; }

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
   * @param[in] DeltaE Parameterized Young's modulus offset
   * @param[in] DeltaKappa Parameterized thermal conductivity offset
   * @param[in] ScaleAlpha Parameterized thermal conductivity offset
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename DispGradType, typename TempType, typename TempGradType, typename YoungsType, typename ConductType,
            typename CoupleType, int dim>
  auto operator()(State& state, const tensor<DispGradType, dim, dim>& grad_u, TempType theta,
                  const tensor<TempGradType, dim>& grad_theta, YoungsType DeltaE, ConductType DeltaKappa,
                  CoupleType ScaleAlpha) const
  {
    auto E = E0 * get<0>(DeltaE);
    auto kappa = kappa0 + get<0>(DeltaKappa);
    auto alpha = alpha0 * get<0>(ScaleAlpha);

    auto K = E / (3.0 * (1.0 - 2.0 * nu));
    auto G = 0.5 * E / (1.0 + nu);
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
};
}  // namespace smith::thermomechanics
