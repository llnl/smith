// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file heat_transfer_input.hpp
 *
 * @brief An object containing all input file options for the solver for
 *  a heat transfer PDE
 */

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "smith/physics/materials/thermal_material_input.hpp"
#include "smith/numerics/odes.hpp"
#include "smith/infrastructure/input.hpp"
#include "smith/numerics/solver_config.hpp"

namespace smith {

/**
 * @brief Stores all information held in the input file that
 * is used to configure the solver
 */
struct HeatTransferInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet's Container that input files will be added to
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief The order of the discretized field
   *
   */
  int order;

  /**
   * @brief The linear solver options
   *
   */
  LinearSolverOptions lin_solver_options;

  /**
   * @brief The linear solver options
   *
   */
  NonlinearSolverOptions nonlin_solver_options;

  /**
   * @brief The timestepping options
   *
   */
  TimesteppingOptions timestepping_options;

  /**
   * @brief The material options
   *
   */
  std::vector<var_thermal_material_t> materials;

  /**
   * @brief Source function coefficient
   *
   */
  std::optional<input::CoefficientInputOptions> source_coef;

  /**
   * @brief The boundary condition information
   */
  std::unordered_map<std::string, input::BoundaryConditionInputOptions> boundary_conditions;

  /**
   * @brief The initial temperature field
   * @note This can be used as either an intialization for dynamic simulations or an
   *       initial guess for quasi-static ones
   *
   */
  std::optional<input::CoefficientInputOptions> initial_temperature;
};

}  // namespace smith

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<smith::HeatTransferInputOptions> {
  /// @brief Returns created object from Inlet container
  smith::HeatTransferInputOptions operator()(const axom::inlet::Container& base);
};
