// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file scalar_objective.hpp
 *
 * @brief Specifies interface for evaluating scalar objective from fields and their field gradients
 */

#pragma once

#include <vector>
#include "serac/physics/common.hpp"

namespace mfem {
class Vector;
}  // namespace mfem

namespace serac {

class FiniteElementState;

/// @brief Abstract residual class
class ScalarObjective {
 public:
  /// @brief base constructor takes the name of the physics
  ScalarObjective(const std::string& name) : name_(name) {}

  /// @brief destructor
  virtual ~ScalarObjective() {}

  /// @brief using
  using FieldPtr = FiniteElementState*;

  /** @brief Virtual interface for computing objective gradient from a vector of serac::FiniteElementState*
   *
   * @param time time
   * @param dt  time step
   * @param fields vector of serac::FiniteElementState* as arguments to the residual
   * @return double
   */
  virtual double evaluate(double time, double dt, const std::vector<FieldPtr>& fields) const = 0;

  /** @brief Virtual interface for computing objective gradient from a vector of serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState* as arguments to the residual
   * @param direction index for which field to take the gradient with respect to
   * @return mfem::Vector
   */
  virtual mfem::Vector gradient(double time, double dt, const std::vector<FieldPtr>& fields, int direction) const = 0;

  /// @brief name
  std::string name() const { return name_; }

 private:
  /// @brief name provided to objective
  std::string name_;
};

}  // namespace serac