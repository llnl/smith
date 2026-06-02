// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file contact_config.hpp
 *
 * @brief This file contains enumerations and record types for contact configuration
 */

#pragma once

namespace smith {

/**
 * @brief Methodology for enforcing contact constraints (i.e. how you form the constraint equations)
 */
enum class ContactMethod
{
  SingleMortar,     /**< Puso and Laursen 2004 */
  EnergyMortar,     /**< Energy-based mortar (g_tilde/A formulation) */
  EnergyAreaPenalty /**< Energy-based area-integrated penalty (k/2 * integral H(gn)^2 dA) */
};

/**
 * @brief Describes how to enforce the contact constraint equations
 */
enum class ContactEnforcement
{
  Penalty,            /**< Equal penalty applied to all constrained dofs */
  LagrangeMultiplier, /**< Solve for exact pressures to satisfy constraints */
  NotRequired
};

/**
 * @brief Mechanical constraint type on contact surfaces
 */
enum class ContactType
{
  TiedNormal,  /**< Tied contact in the normal direction, no friction */
  Frictionless /**< Enforce gap >= 0, pressure <= 0, gap * pressure = 0 in the normal direction */
};

/**
 * @brief Method for computing Jacobian of contact terms
 */
enum class ContactJacobian
{
  Approximate, /**< Ignore higher order contributions to the Jacobian */
  Exact        /**< Compute exact Jacobian (needed for quadratic convergence with Newton) */
};

/**
 * @brief Smoothing type for isoparametric integration bounds (EnergyMortar only)
 */
enum class BoundsSmoothing
{
  Hermite,  /**< C1 cubic Hermite ramp (identity in middle section) */
  Quadratic /**< C1 quadratic ramp (slope 1/(1-del) in middle) */
};

/**
 * @brief Penalty smoothing at the contact/separation transition (EnergyMortar only)
 */
enum class PenaltySmoothing
{
  Hard,  /**< Hard clamp: p = k * max(-g, 0). C0 kink in Jacobian. */
  Smooth /**< C1 quadratic ramp over [-del/2, +del/2]: penalty decays smoothly to zero. */
};

/**
 * @brief Stores the options for a contact pair
 */
struct ContactOptions {
  /// The contact methodology to be applied
  ContactMethod method = ContactMethod::SingleMortar;

  /// The contact enforcement strategy to use
  ContactEnforcement enforcement = ContactEnforcement::Penalty;

  /// The type of contact constraint
  ContactType type = ContactType::Frictionless;

  /// Penalty parameter (only used when enforcement == ContactEnforcement::Penalty)
  double penalty = 1.0e3;
  /// Secondary penalty parameter (reserved for future use).
  double penalty2 = 0.0;

  /// The method to use for Jacobian calculations
  ContactJacobian jacobian = ContactJacobian::Approximate;

  /// Integration bounds smoothing type (EnergyMortar only)
  BoundsSmoothing bounds_smoothing = BoundsSmoothing::Quadratic;

  /// Penalty smoothing at the contact/separation boundary (EnergyMortar only)
  PenaltySmoothing penalty_smoothing = PenaltySmoothing::Smooth;

  /// Width of the smooth penalty transition band (EnergyMortar only, used when penalty_smoothing == Smooth)
  double penalty_smoothing_del = 1.0e-5;
};

}  // namespace smith
