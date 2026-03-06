// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermomechanics_input.hpp
 *
 * @brief An object containing all input file options for the solver for
 * thermal structural solver
 */

#pragma once

#include <optional>

#include "smith/infrastructure/input.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/solid_mechanics_input.hpp"
#include "smith/physics/heat_transfer_input.hpp"

namespace smith {

/**
 * @brief Stores all information held in the input file that
 * is used to configure the thermal structural solver
 */
struct ThermomechanicsInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet's Container that input files will be added to
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief Solid mechanics input options
   */
  SolidMechanicsInputOptions solid_options;

  /**
   * @brief Heat transfer input options
   *
   */
  HeatTransferInputOptions thermal_options;

  /**
   * @brief The isotropic coefficient of thermal expansion
   */
  std::optional<input::CoefficientInputOptions> coef_thermal_expansion;

  /**
   * @brief The reference temperature for thermal expansion
   */
  std::optional<input::CoefficientInputOptions> reference_temperature;
};

}  // namespace smith

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<smith::ThermomechanicsInputOptions> {
  /// @brief Returns created object from Inlet container
  smith::ThermomechanicsInputOptions operator()(const axom::inlet::Container& base);
};
