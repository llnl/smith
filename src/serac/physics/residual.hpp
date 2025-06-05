// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file residual.hpp
 *
 * @brief Specifies interface for evaluating residuals and their gradients
 */

#pragma once

// reverse residual sign
// enums on residuals for state?
// shared_ptr for field inputs
// proper const correctness
// shape_disp on mesh

// action of the mass...
// force, residual, inertial terms
// lumped-mass residual, acceleration needs to be 0

// C(u) u_dot + K(u) u = 0

#include <vector>
#include <string>
#include "serac/physics/common.hpp"
#include "serac/physics/field_types.hpp"

namespace mfem {
class Vector;
class HypreParMatrix;
}  // namespace mfem

namespace serac {

class FiniteElementState;
class FiniteElementDual;

/// @brief Abstract residual class
class Residual {
 public:
  /** @brief base constructor takes the name of the physics
   * @param name provide a name corresponding to the physics
   */
  Residual(std::string name) : name_(name) {}

  /// @brief destructor
  virtual ~Residual() {}

  /** @brief Virtual interface for computing residual from a vector of serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState* as arguments to the residual
   * @param block_row integer which specifies which row of a block system to get the residual for, defaults to 0
   * @return mfem::Vector
   */
  virtual mfem::Vector residual(double time, double dt, const std::vector<ConstFieldPtr>& fields,
                                int block_row = 0) const = 0;

  /** @brief Derivative of the residual with respect to specified field arguments: sum_j d{r}_i/d{fields}_j *
   * argument_tangents[j], i is row, j are input fields (columns)
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState* as arguments to the residual
   * @param argument_tangents specifies the weighting of the residual derivative with respect to each field
   * @param block_row specifies which block row of the residual to compute the jacobian for
   * the call will error if a non-zero argument_tangent weight is provided for two input fields with different sizes
   * @return std::unique_ptr<mfem::HypreParMatrix> returns sum_j d{r}_i/d{fields}_j * argument_tangents[j], where
   * {fields}_j is the jth field, {r}_i is the ith residual block row
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> jacobian(double time, double dt,
                                                         const std::vector<ConstFieldPtr>& fields,
                                                         const std::vector<double>& argument_tangents,
                                                         int block_row = 0) const = 0;

  /**
   * @brief Jacobian-vector product, will overwrite any existing values in jvpReactions
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState* as arguments to the residual
   * @param v_fields right hand side 'v' fields
   * @param jvp_reactions output vjps, 1 per row of a block system: d{r}_i / d{fields}_j * fieldsV[j]
   * nullptr fieldsV are assumed to be all zero to avoid extra calculations
   */
  virtual void jvp(double time, double dt, const std::vector<ConstFieldPtr>& fields,
                   const std::vector<ConstFieldPtr>& v_fields,
                   const std::vector<DualFieldPtr>& jvp_reactions) const = 0;

  /**
   * @brief Vector-Jacobian product, will += into existing values in vjpFields
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState* as arguments to the residual
   * @param v_fields left hand side 'v' fields
   * @param vjp_sensitivities output jvps, 1 per input field: v_fields[i] * d{r}_i / d{fields}_j
   */
  virtual void vjp(double time, double dt, const std::vector<ConstFieldPtr>& fields,
                   const std::vector<ConstFieldPtr>& v_fields,
                   const std::vector<DualFieldPtr>& vjp_sensitivities) const = 0;

  /// @brief name
  std::string name() const { return name_; }

 private:
  /// name
  std::string name_;
};

}  // namespace serac
