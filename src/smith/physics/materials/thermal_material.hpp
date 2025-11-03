// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_material.hpp
 *
 * @brief The material and load types for the thermal functional physics module
 */

#pragma once

#include "smith/numerics/functional/functional.hpp"

namespace smith::heat_transfer {

/// Linear isotropic heat transfer material model
struct LinearIsotropicConductor {
  /**
   * @brief Construct a new Linear Isotropic Conductor object
   *
   * @param density Density of the material (mass/volume)
   * @param specific_heat_capacity Specific heat capacity of the material (energy / (mass * temp))
   * @param conductivity Thermal conductivity of the material (power / (length * temp))
   */
  LinearIsotropicConductor(double density = 1.0, double specific_heat_capacity = 1.0, double conductivity = 1.0)
      : density_(density), specific_heat_capacity_(specific_heat_capacity), conductivity_(conductivity)
  {
    SLIC_ERROR_ROOT_IF(conductivity_ < 0.0,
                       "Conductivity must be positive in the linear isotropic conductor material model.");

    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear isotropic conductor material model.");

    SLIC_ERROR_ROOT_IF(specific_heat_capacity_ < 0.0,
                       "Specific heat capacity must be positive in the linear isotropic conductor material model.");
  }

  /**
   * @brief Material response call for a linear isotropic material
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Temperature type
   * @tparam T3 Temperature gradient type
   * @param[in] temperature_gradient Temperature gradient
   * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a linear
   * isotropic material
   */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* temperature */,
                                    const T3& temperature_gradient) const
  {
    return smith::tuple{density_ * specific_heat_capacity_, -1.0 * conductivity_ * temperature_gradient};
  }

 private:
  /// Density
  double density_;

  /// Specific heat capacity
  double specific_heat_capacity_;

  /// Constant isotropic thermal conductivity
  double conductivity_;
};

/// Nonlinear isotropic heat transfer material model
struct IsotropicConductorWithLinearConductivityVsTemperature {
  /**
   * @brief Construct a Isotropic Conductor with Conductivity linear with Temparture object
   *
   * @param density Density of the material (mass/volume)
   * @param specific_heat_capacity Specific heat capacity of the material (energy / (mass * temp))
   * @param reference_conductivity Reference thermal conductivity of the material at temp = 0 (power / (length * temp))
   * @param d_conductivity_d_temperature Slope for the thermal conductivity as a function of temperature
   */
  IsotropicConductorWithLinearConductivityVsTemperature(double density = 1.0, double specific_heat_capacity = 1.0,
                                                        double reference_conductivity = 1.0,
                                                        double d_conductivity_d_temperature = 0.0)
      : density_(density),
        specific_heat_capacity_(specific_heat_capacity),
        reference_conductivity_(reference_conductivity),
        d_conductivity_d_temperature_(d_conductivity_d_temperature)
  {
    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear isotropic conductor material model.");
    SLIC_ERROR_ROOT_IF(specific_heat_capacity_ < 0.0,
                       "Specific heat capacity must be positive in the linear isotropic conductor material model.");
  }

  /**
   * @brief Material response call for a linear isotropic material with linear conductivity vs temperature
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Temperature type
   * @tparam T3 Temperature gradient type
   * @param[in] temperature
   * @param[in] temperature_gradient
   * @return The calculated material response (tuple of volumetric heat capacity and thermal flux)
   */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const T2& temperature, const T3& temperature_gradient) const
  {
    const auto currentConductivity = reference_conductivity_ + d_conductivity_d_temperature_ * temperature;
    SLIC_ERROR_ROOT_IF(
        smith::get_value(currentConductivity) < 0.0,
        "Conductivity in the IsotropicConductorWithLinearConductivityVsTemperature model has gone negative.");
    return smith::tuple{density_ * specific_heat_capacity_, -1.0 * currentConductivity * temperature_gradient};
  }

 private:
  /// Density
  double density_;

  /// Specific heat capacity
  double specific_heat_capacity_;

  /// Reference isotropic thermal conductivity
  double reference_conductivity_;

  /// Slope of nonlinear thermal conductivity dependence on temperature
  double d_conductivity_d_temperature_;
};

/**
 * @brief Linear anisotropic thermal material model
 *
 * @tparam dim Spatial dimension
 */
template <int dim>
struct LinearConductor {
  /**
   * @brief Construct a new Linear Isotropic Conductor object
   *
   * @param density Density of the material (mass/volume)
   * @param specific_heat_capacity Specific heat capacity of the material (energy / (mass * temp))
   * @param conductivity Thermal conductivity of the material (power / (length * temp))
   */
  LinearConductor(double density = 1.0, double specific_heat_capacity = 1.0,
                  tensor<double, dim, dim> conductivity = Identity<dim>())
      : density_(density), specific_heat_capacity_(specific_heat_capacity), conductivity_(conductivity)
  {
    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear conductor material model.");

    SLIC_ERROR_ROOT_IF(specific_heat_capacity_ < 0.0,
                       "Specific heat capacity must be positive in the linear conductor material model.");

    SLIC_ERROR_ROOT_IF(!is_symmetric_and_positive_definite(conductivity_),
                       "Conductivity tensor must be symmetric and positive definite.");
  }

  /**
   * @brief Material response call for a linear anisotropic material
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Temperature type
   * @tparam T3 Temperature gradient type
   * @param[in] temperature_gradient Temperature gradient
   * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a linear
   * anisotropic material
   */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* temperature */,
                                    const T3& temperature_gradient) const
  {
    return smith::tuple{density_ * specific_heat_capacity_, -1.0 * conductivity_ * temperature_gradient};
  }

 private:
  /// Density
  double density_;

  /// Specific heat capacity
  double specific_heat_capacity_;

  /// Constant thermal conductivity
  tensor<double, dim, dim> conductivity_;
};

/// Constant thermal source model
struct ConstantSource {
  /// The constant source
  double source_ = 0.0;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the position vector
   * @tparam T2 type of the temperature
   * @tparam T3 type of the temperature gradient
   * @tparam dim The dimension of the problem
   * @return The thermal source value
   */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const double /* time */, const T2& /* temperature */,
                                    const T3& /* temperature_gradient */) const
  {
    return source_;
  }
};

/// Constant thermal flux boundary model
struct ConstantFlux {
  /// The constant flux applied to the boundary
  double flux_ = 0.0;

  /**
   * @brief Evaluation function for the thermal flux on a boundary
   *
   * @tparam T1 Type of the position vector
   * @tparam T2 Type of the normal vector
   * @tparam T3 Type of the temperature on the boundary
   * @tparam dim The dimension of the problem
   * @return The flux applied to the boundary
   */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* normal */, const double /* time */,
                                    const T3& /* temperature */) const
  {
    return flux_;
  }
};

}  // namespace smith::heat_transfer
