// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file parameterized_thermal_material.hpp
 *
 * @brief The material and load types for the thermal functional physics module
 */

#pragma once

#include "smith/physics/materials/thermal_material.hpp"

/// HeatTransfer helper structs
namespace smith::heat_transfer {

/// Linear isotropic conductor with a parameterized conductivity
class ParameterizedLinearIsotropicConductor {
 public:
  /**
   * @brief Construct a new Parameterized Linear Isotropic Conductor object
   *
   * @param density Density of the material (mass/volume)
   * @param specific_heat_capacity Specific heat capacity of the material (energy / (mass * temp))
   * @param conductivity_offset Thermal conductivity offset of the material (power / (length * temp)). This is
   * added to the parameter value to get the total conductivity.
   */
  ParameterizedLinearIsotropicConductor(double density = 1.0, double specific_heat_capacity = 1.0,
                                        double conductivity_offset = 0.0)
      : density_(density), specific_heat_capacity_(specific_heat_capacity), conductivity_offset_(conductivity_offset)
  {
    SLIC_ERROR_ROOT_IF(conductivity_offset_ < 0.0,
                       "Conductivity must be positive in the linear isotropic conductor material model.");

    SLIC_ERROR_ROOT_IF(density_ < 0.0, "Density must be positive in the linear isotropic conductor material model.");

    SLIC_ERROR_ROOT_IF(specific_heat_capacity_ < 0.0,
                       "Specific heat capacity must be positive in the linear isotropic conductor material model.");
  }

  /**
   * @brief Thermal material response operator
   *
   * @tparam T1 Spatial position type
   * @tparam T2 Temperature type
   * @tparam T3 Temperature gradient type
   * @tparam T4 Parameter type
   * @param temperature_gradient The spatial gradient of the temperature (d temperature dx)
   * @param parameter The user-defined parameter used to compute the conductivity (total conductivity = conductivity
   * offset + parameter)
   * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a parameterized
   * linear isotropic material
   */
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* temperature */, const T3& temperature_gradient,
                                    const T4& parameter) const
  {
    return smith::tuple{density_ * specific_heat_capacity_,
                        -1.0 * (conductivity_offset_ + get<0>(parameter)) * temperature_gradient};
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 1; }

 private:
  /// Density
  double density_;

  /// Specific heat capacity
  double specific_heat_capacity_;

  /// Conductivity offset
  double conductivity_offset_;
};

/// Nonlinear isotropic heat transfer material model
struct ParameterizedIsotropicConductorWithLinearConductivityVsTemperature {
  /**
   * @brief Construct a Parameterized Isotropic Conductor with Conductivity linear with Temparture object
   *
   * @param density Density of the material (mass/volume)
   * @param specific_heat_capacity Specific heat capacity of the material (energy / (mass * temp))
   * @param conductivity_offset Thermal conductivity offset of the material (power / (length * temp)). This is
   * added to the parameter value to get the total conductivity.
   * @param d_conductivity_d_temperature Slope for the thermal conductivity as a function of temperature
   */
  ParameterizedIsotropicConductorWithLinearConductivityVsTemperature(double density = 1.0,
                                                                     double specific_heat_capacity = 1.0,
                                                                     double conductivity_offset = 1.0,
                                                                     double d_conductivity_d_temperature = 0.0)
      : density_(density),
        specific_heat_capacity_(specific_heat_capacity),
        conductivity_offset_(conductivity_offset),
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
   * @tparam T4 Parameter type
   * @param[in] temperature
   * @param[in] temperature_gradient
   * @param parameter The user-defined parameter used to compute the conductivity (total conductivity = conductivity
   * offset + parameter)
   * @return The calculated material response (tuple of volumetric heat capacity and thermal flux)
   */
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const T2& temperature, const T3& temperature_gradient,
                                    const T4& parameter) const
  {
    const auto currentConductivity =
        conductivity_offset_ + get<0>(parameter) + d_conductivity_d_temperature_ * temperature;
    SLIC_ERROR_ROOT_IF(
        smith::get_value(currentConductivity) < 0.0,
        "Conductivity in the IsotropicConductorWithLinearConductivityVsTemperature model has gone negative.");
    return smith::tuple{density_ * specific_heat_capacity_, -1.0 * currentConductivity * temperature_gradient};
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 1; }

 private:
  /// Density
  double density_;

  /// Specific heat capacity
  double specific_heat_capacity_;

  /// Reference isotropic thermal conductivity
  double conductivity_offset_;

  /// Slope of nonlinear thermal conductivity dependence on temperature
  double d_conductivity_d_temperature_;
};

/// Parameterized thermal source model
struct ParameterizedSource {
  /// The constant source offset
  double source_offset_ = 0.0;

  /**
   * @brief Evaluation function for the constant thermal source model
   *
   * @tparam T1 type of the position vector
   * @tparam T2 type of the temperature
   * @tparam T3 type of the temperature gradient
   * @tparam T4 type of the parameter
   * @param parameter user-defined parameter
   * @return The thermal source value
   */
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const double /* time */, const T2& /* temperature */,
                                    const T3& /* temperature_gradient */, const T4& parameter) const
  {
    return source_offset_ + parameter;
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 1; }
};

/// Constant thermal flux boundary model
struct ParameterizedFlux {
  /// The constant flux applied to the boundary
  double flux_offset_ = 0.0;

  /**
   * @brief Evaluation function for the thermal flux on a boundary
   *
   * @tparam T1 Type of the position vector
   * @tparam T2 Type of the normal vector
   * @tparam T3 Type of the temperature on the boundary
   * @tparam T4 Type of the parameter
   * @param parameter The user-defined parameter value
   * @return The flux applied to the boundary
   */
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const T2& /* normal */, const double /* time */,
                                    const T3& /* temperature */, const T4& parameter) const
  {
    return flux_offset_ + parameter;
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 1; }
};

}  // namespace smith::heat_transfer
