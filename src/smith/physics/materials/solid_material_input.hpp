// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_material_input.hpp
 *
 * @brief This file contains functions for reading a material from input files
 */

#pragma once

#include <string>
#include <variant>

#include "smith/infrastructure/input.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/materials/hardening_input.hpp"

namespace smith {

/// @brief All possible solid mechanics materials that can be utilized in our input file
using var_solid_material_t = std::variant<solid_mechanics::NeoHookean, solid_mechanics::LinearIsotropic,
                                          solid_mechanics::J2SmallStrain<solid_mechanics::LinearHardening>,
                                          solid_mechanics::J2SmallStrain<solid_mechanics::PowerLawHardening>,
                                          solid_mechanics::J2SmallStrain<solid_mechanics::VoceHardening>>;

/// @brief Contains function that defines the schema for solid mechanics materials
struct SolidMaterialInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet's Container to which fields should be added
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);
};

}  // namespace smith

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<smith::var_solid_material_t> {
  /// @brief Returns created object from Inlet container
  smith::var_solid_material_t operator()(const axom::inlet::Container& base);
};
