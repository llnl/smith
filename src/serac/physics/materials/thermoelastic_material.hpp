// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/physics/materials/green_saint_venant_thermoelastic.hpp"
#include "serac/physics/materials/solid_material.hpp"

/// Thermomechanics helper data types
namespace serac::thermomechanics {
/// @brief Alternative Green-Saint Venant isotropic thermoelastic model
struct AlternativeGreenSaintVenantThermoelasticMaterial {
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity

  using State = Empty;

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the velocity gradient components (number-like)
   * @tparam T3 Type of the temperature (number-like)
   * @tparam T4 Type of the temperature gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] grad_v Velocity gradient
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
  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain(grad_u);
    const auto trEg = tr(Eg);

    // stress
    static constexpr auto I = Identity<dim>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - dim * alpha * (theta - theta_ref)) * I;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);

    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);
    // const auto s0 = -dim * K * alpha* 1000.;

    // heat flux
    const auto q0 = -kappa * grad_theta;

    return serac::tuple{Piola, C_v, s0, q0};
  }
};

///////////////////////////////////////////////////////////////////////////////

struct NeoHookeanThermoelasticMaterial {
  static constexpr int dim = 3;
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity
  double mu;         ///< viscous parameter

  using State = Empty;
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
  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    using std::log1p;
    // constexpr double eps = 1.0e-12;
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    constexpr auto I = serac::DenseIdentity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(grad_u, transpose(grad_u)) + transpose(grad_u) + grad_u;

    auto logJ = log1p(detApIm1(grad_u));
    // Kirchoff stress, in form that avoids cancellation error when F is near I

    // Pull back to Piola
    auto F = grad_u + I;

    auto L = dot(grad_v, inv(F));
    auto D = sym(L);

    auto TK = lambda * logJ * I + G * B_minus_I + 0.5 * det(F) * mu * D;  // dot(L, inv(transpose(F)));

    // state.F_old = get_value(F);
    const auto S = -1.0 * K * (dim * alpha * (theta - theta_ref)) * I;
    auto Piola = dot(TK, inv(transpose(F))) + dot(F, S);
    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);
    // const auto s0 = -dim * K * alpha * (theta + 273.1);

    // heat flux
    const auto q0 = -kappa * grad_theta;
    return serac::tuple{Piola, C_v, s0, q0};
  }
};

}  // namespace serac::thermomechanics