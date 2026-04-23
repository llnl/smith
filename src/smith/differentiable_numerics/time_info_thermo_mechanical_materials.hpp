// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/physics/common.hpp"
#include "smith/physics/materials/green_saint_venant_thermoelastic.hpp"

namespace smith::thermomechanics {

/// @brief TimeInfo-aware wrapper for `GreenSaintVenantThermoelasticMaterial`.
struct TimeInfoGreenSaintVenantThermoelasticMaterial {
  /// State type reused from wrapped material.
  using State = GreenSaintVenantThermoelasticMaterial::State;

  double density;    ///< Mass density.
  double E;          ///< Young's modulus.
  double nu;         ///< Poisson ratio.
  double C_v;        ///< Heat capacity.
  double alpha;      ///< Thermal expansion coefficient.
  double theta_ref;  ///< Reference temperature.
  double kappa;      ///< Thermal conductivity.

  template <typename T1, typename GradVType, typename T2, typename T3, int dim>
  /// @brief Evaluate wrapped material, ignoring velocity gradient.
  auto operator()(const TimeInfo& /*t_info*/, State& state, const tensor<T1, dim, dim>& grad_u,
                  const GradVType& /*grad_v*/, T2 theta, const tensor<T3, dim>& grad_theta) const
  {
    return GreenSaintVenantThermoelasticMaterial{density, E, nu, C_v, alpha, theta_ref, kappa}(state, grad_u, theta,
                                                                                               grad_theta);
  }
};

/// @brief TimeInfo-aware wrapper for `ParameterizedGreenSaintVenantThermoelasticMaterial`.
struct TimeInfoParameterizedGreenSaintVenantThermoelasticMaterial {
  /// State type reused from wrapped material.
  using State = ParameterizedGreenSaintVenantThermoelasticMaterial::State;

  double density;    ///< Mass density.
  double E;          ///< Young's modulus.
  double nu;         ///< Poisson ratio.
  double C_v;        ///< Heat capacity.
  double alpha0;     ///< Reference thermal expansion coefficient.
  double theta_ref;  ///< Reference temperature.
  double kappa;      ///< Thermal conductivity.

  template <typename T1, typename GradVType, typename T2, typename T3, typename T4, int dim>
  /// @brief Evaluate wrapped material, ignoring velocity gradient.
  auto operator()(const TimeInfo& /*t_info*/, State& state, const tensor<T1, dim, dim>& grad_u,
                  const GradVType& /*grad_v*/, T2 theta, const tensor<T3, dim>& grad_theta,
                  T4 thermal_expansion_scaling) const
  {
    return ParameterizedGreenSaintVenantThermoelasticMaterial{density, E, nu, C_v, alpha0, theta_ref, kappa}(
        state, grad_u, theta, grad_theta, thermal_expansion_scaling);
  }
};

}  // namespace smith::thermomechanics
