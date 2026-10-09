// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermoelastic_material.hpp
 * @brief Thermoelastic material shared by the prescribed- and dynamic-temperature examples.
 */

#pragma once

#include "smith/smith_config.hpp"

#include "smith/numerics/functional/quadrature_data.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/physics/common.hpp"

namespace smith::examples::kevin {

/**
 * @brief Green-Saint Venant thermoelastic material with Fourier heat conduction.
 *
 * The first call operator makes temperature an externally managed mechanics parameter. The second provides the
 * constitutive tuple required by a coupled thermal-mechanics system. Both add a fixed initial-displacement gradient
 * to the incremental displacement gradient before evaluating stress.
 */

template <int dim = 2>
struct ThermomechanicalLCE {
  using State = Empty;  ///< This material has no history variables.

  /**
   * @brief Evaluate stress when temperature is supplied as a finite-element parameter field.
   */
  template <typename GradUType, typename GradVType, typename TemperatureFieldType,
            typename InitialDisplacementFieldType>
  auto operator()(const TimeInfo&, State&, const GradUType& grad_u, const GradVType&,
                  const TemperatureFieldType& temperature_field,
                  const InitialDisplacementFieldType& initial_displacement_field) const
  {
    const auto total_grad_u = grad_u + get<DERIVATIVE>(initial_displacement_field);
    return firstPiolaStress(total_grad_u, get<VALUE>(temperature_field));
  }

  /**
   * @brief Evaluate the stress, heat capacity, heat source, and heat flux for coupled thermo-mechanics.
   */
  template <typename GradUType, typename GradVType, typename TemperatureType, typename TemperatureGradientType,
            typename InitialDisplacementFieldType>
  auto operator()(const TimeInfo&, State&, const GradUType& grad_u, const GradVType& grad_v,
                  const TemperatureType& temperature, const TemperatureGradientType& grad_temperature,
                  const InitialDisplacementFieldType& initial_displacement_field) const
  {
    const auto total_grad_u = grad_u + get<DERIVATIVE>(initial_displacement_field);
    auto stress = firstPiolaStress(total_grad_u, temperature);
    auto heat_source = 0.0 * tr(grad_v);
    auto heat_flux = -thermal_conductivity * grad_temperature;
    return tuple{stress, heat_capacity, heat_source, heat_flux};
  }

  double density = 1.0;                ///< Reference mass density.
  double youngs_modulus = 100.0;       ///< Young's modulus.
  double poissons_ratio = 0.25;        ///< Poisson's ratio.
  double heat_capacity = 10.0;         ///< Volumetric heat capacity.
  double thermal_expansion = 0.0e-3;   ///< Coefficient of thermal expansion.
  double reference_temperature = 0.0;  ///< Stress-free temperature.
  double thermal_conductivity = 0.25;  ///< Isotropic thermal conductivity.

  double Smax = 0.6;
  double Tni = 350.0;
  double DT = 10.0;
  double alpha = 0.4;
  double T0 = 500.0;  ///< High-temperature reference state.
  tensor<double, dim> normal = make_tensor<dim>([](int i) { return i == 0 ? 1.0 : 0.0; });  ///< LCE director.

 private:
  template <typename T>
  auto orderParameter(const T temperature) const
  {
    using std::exp;
    return Smax / (1.0 + exp(2.0 * (temperature - Tni) / DT));
  }

  template <typename T>
  auto anisotropyRatio(const T temperature) const
  {
    const auto op = orderParameter(temperature);
    return (1.0 + 2.0 * alpha * op) / (1.0 - alpha * op);
  }

  template <typename T>
  auto spontaneousStretch(const T temperature) const
  {
    using std::pow;
    return pow(anisotropyRatio(temperature) / anisotropyRatio(T0), 1.0 / 3.0);
  }

  template <typename T>
  auto spontaneousStretchTensor(const T temperature, const tensor<double, dim>& n0) const
  {
    using std::pow;
    const auto N = outer(n0, n0);
    const auto eye_minus_N = Identity<dim>() - N;
    const auto spontaneous_stretch = spontaneousStretch(temperature);
    return spontaneous_stretch * N + pow(spontaneous_stretch, -0.5) * eye_minus_N;
  }

  template <typename T, typename TemperatureType>
  auto firstPiolaStress(const tensor<T, dim, dim>& grad_u, const TemperatureType& temperature) const
  {
    const double bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poissons_ratio));
    const double shear_modulus = youngs_modulus / (2.0 * (1.0 + poissons_ratio));
    static constexpr auto I = Identity<dim>();

    const auto F = grad_u + I;
    const auto Fs = spontaneousStretchTensor(temperature, normal);
    const auto Fsinv = inv(Fs);
    const auto Fe = dot(F, Fsinv);
    const auto strain = 0.5 * (dot(transpose(Fe), Fe) - I);
    const auto second_piola =
        2.0 * shear_modulus * dev(strain) +
        bulk_modulus *
            (tr(strain) - static_cast<double>(dim) * thermal_expansion * (temperature - reference_temperature)) * I;
    return dot(dot(Fe, second_piola), transpose(Fsinv));
  }
};

struct ThermoelasticMaterial {
  using State = Empty;  ///< This material has no history variables.

  /**
   * @brief Evaluate stress when temperature is supplied as a finite-element parameter field.
   */
  template <typename GradUType, typename GradVType, typename TemperatureFieldType,
            typename InitialDisplacementFieldType>
  auto operator()(const TimeInfo&, State&, const GradUType& grad_u, const GradVType&,
                  const TemperatureFieldType& temperature_field,
                  const InitialDisplacementFieldType& initial_displacement_field) const
  {
    const auto total_grad_u = grad_u + get<DERIVATIVE>(initial_displacement_field);
    return firstPiolaStress(total_grad_u, get<VALUE>(temperature_field));
  }

  /**
   * @brief Evaluate the stress, heat capacity, heat source, and heat flux for coupled thermo-mechanics.
   */
  template <typename GradUType, typename GradVType, typename TemperatureType, typename TemperatureGradientType,
            typename InitialDisplacementFieldType>
  auto operator()(const TimeInfo&, State&, const GradUType& grad_u, const GradVType& grad_v,
                  const TemperatureType& temperature, const TemperatureGradientType& grad_temperature,
                  const InitialDisplacementFieldType& initial_displacement_field) const
  {
    const auto total_grad_u = grad_u + get<DERIVATIVE>(initial_displacement_field);
    auto stress = firstPiolaStress(total_grad_u, temperature);
    auto heat_source = 0.0 * tr(grad_v);
    auto heat_flux = -thermal_conductivity * grad_temperature;
    return tuple{stress, heat_capacity, heat_source, heat_flux};
  }

  double density = 1.0;                ///< Reference mass density.
  double youngs_modulus = 100.0;       ///< Young's modulus.
  double poissons_ratio = 0.25;        ///< Poisson's ratio.
  double heat_capacity = 10.0;         ///< Volumetric heat capacity.
  double thermal_expansion = 1.0e-3;   ///< Coefficient of thermal expansion.
  double reference_temperature = 0.0;  ///< Stress-free temperature.
  double thermal_conductivity = 0.25;  ///< Isotropic thermal conductivity.

 private:
  template <typename T, typename TemperatureType, int dim>
  auto firstPiolaStress(const tensor<T, dim, dim>& grad_u, const TemperatureType& temperature) const
  {
    const double bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poissons_ratio));
    const double shear_modulus = youngs_modulus / (2.0 * (1.0 + poissons_ratio));
    static constexpr auto I = Identity<dim>();

    const auto F = grad_u + I;
    const auto strain = 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
    const auto second_piola =
        2.0 * shear_modulus * dev(strain) +
        bulk_modulus *
            (tr(strain) - static_cast<double>(dim) * thermal_expansion * (temperature - reference_temperature)) * I;
    return dot(F, second_piola);
  }
};

}  // namespace smith::examples::kevin
