// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_input.hpp
 *
 * @brief An object containing all input file options for the solver for
 * total Lagrangian finite deformation solid mechanics
 */

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "smith/physics/materials/solid_material_input.hpp"
#include "smith/numerics/odes.hpp"
#include "smith/infrastructure/input.hpp"
#include "smith/numerics/solver_config.hpp"

namespace smith {

/**
 * @brief Stores all information held in the input file that
 * is used to configure the solver
 */
struct SolidMechanicsInputOptions {
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet container on which the input schema will be defined
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);

  /**
   * @brief The order of the discretization
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
  std::vector<var_solid_material_t> materials;

  /**
   * @brief Boundary condition information
   *
   */
  std::unordered_map<std::string, input::BoundaryConditionInputOptions> boundary_conditions;

  /**
   * @brief The initial displacement
   * @note This can be used as an initialization field for dynamic problems or an initial guess
   *       for quasi-static solves
   *
   */
  std::optional<input::CoefficientInputOptions> initial_displacement;

  /**
   * @brief The initial velocity
   *
   */
  std::optional<input::CoefficientInputOptions> initial_velocity;
};

}  // namespace smith

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<smith::SolidMechanicsInputOptions> {
  /// @brief Returns created object from Inlet container
  smith::SolidMechanicsInputOptions operator()(const axom::inlet::Container& base);
};
