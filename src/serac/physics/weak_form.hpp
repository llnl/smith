// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file weak_form.hpp
 *
 * @brief Specifies interface for evaluating residuals and their gradients for weak forms
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include "serac/physics/common.hpp"
#include "serac/physics/field_types.hpp"

namespace mfem {
class Vector;
class HypreParMatrix;
}  // namespace mfem

namespace serac {

class FiniteElementState;
class FiniteElementDual;

using QuadratureField = double;                 ///< This is a placeholder for quadrature fields
using QuadratureFieldPtr = double*;             ///< This is a placeholder for quadrature field pointers
using ConstQuadratureFieldPtr = const double*;  ///< This is a placeholder for quadrature field pointers

/// @brief Abstract WeakForm class
class WeakForm {
 public:
  /** @brief base constructor takes the name of the physics
   * @param name provide a name corresponding to the physics
   */
  WeakForm(std::string name) : name_(name) {}

  /// @brief destructor
  virtual ~WeakForm() {}

  /** @brief Virtual interface for computing the residual vector residual of a weak form
   *
   * @param time time
   * @param dt time step
   * @param shape_disp serac::FiniteElementState*, change in model coordinates relative to the initially read in mesh
   * @param fields vector of serac::FiniteElementState*
   * @param quad_fields vector of ConstQuadratureFieldPtr
   * @param block_row integer which specifies which row of a block system to get the residual for, defaults to 0
   * @return mfem::Vector
   */
  virtual mfem::Vector residual(double time, double dt, ConstFieldPtr shape_disp,
                                const std::vector<ConstFieldPtr>& fields,
                                const std::vector<ConstQuadratureFieldPtr>& quad_fields = {},
                                int block_row = 0) const = 0;

  /** @brief Derivative of the residual with respect to specified field arguments: sum_j d{r}_i/d{fields}_j *
   * argument_tangents[j], i is row, j are input fields (columns)
   * @param time time
   * @param dt time step
   * @param shape_disp serac::FiniteElementState*, change in model coordinates relative to the initially read in mesh
   * @param fields vector of serac::FiniteElementState*
   * @param field_argument_tangents specifies the weighting of the residual derivative with respect to each field
   * @param quad_fields vector of ConstQuadratureFieldPtr
   * @param block_row specifies which block row of the residual to compute the jacobian for
   * the call will error if a non-zero argument_tangent weight is provided for two input fields with different sizes
   * @return std::unique_ptr<mfem::HypreParMatrix> returns sum_j d{r}_i/d{fields}_j * argument_tangents[j], where
   * {fields}_j is the jth field, {r}_i is the ith residual block row
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> jacobian(double time, double dt, ConstFieldPtr shape_disp,
                                                         const std::vector<ConstFieldPtr>& fields,
                                                         const std::vector<double>& field_argument_tangents,
                                                         const std::vector<ConstQuadratureFieldPtr>& quad_fields = {},
                                                         int block_row = 0) const = 0;

  /**
   * @brief Jacobian-vector product, will overwrite any existing values in jvp_reactions
   * @param time time
   * @param dt time step
   * @param shape_disp serac::FiniteElementState*, change in model coordinates relative to the initially read in mesh
   * @param fields vector of serac::FiniteElementState*
   * @param quad_fields vector of ConstQuadratureFieldPtr
   * @param v_shape_disp shape_displacement tangent
   * @param v_fields field tangents, right hand side 'v' fields
   * @param v_quad_fields quadrature_field_tangents
   * @param jvp_reactions output jvps, 1 per row of a block system: d{r}_i / d{fields}_j * fieldsV[j]
   * nullptr fieldsV are assumed to be all zero to avoid extra calculations
   */
  virtual void jvp(double time, double dt, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                   const std::vector<ConstQuadratureFieldPtr>& quad_fields, ConstFieldPtr v_shape_disp,
                   const std::vector<ConstFieldPtr>& v_fields,
                   const std::vector<ConstQuadratureFieldPtr>& v_quad_fields,
                   const std::vector<DualFieldPtr>& jvp_reactions) const = 0;

  /**
   * @brief Vector-Jacobian product, will += into existing values in vjpFields
   * @param time time
   * @param dt time step
   * @param shape_disp serac::FiniteElementState*, change in model coordinates relative to the initially read in mesh
   * @param fields vector of serac::FiniteElementState*
   * @param quad_fields vector of ConstQuadratureFieldPtr
   * @param v_fields left hand side 'v' fields
   * @param vjp_shape_disp_sensitivity vjp for shape_displacement: v_fields[i] * d{r}_i / d{shape_disp}
   * @param vjp_sensitivities output vjps, 1 per input field: v_fields[i] * d{r}_i / d{fields}_j
   * @param vjp_quadrature_sensivities output vjps, 1 per input quadrature field: v_fields[i] * d{r}_i /
   * d{quadrature_field}_j
   */
  virtual void vjp(double time, double dt, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                   const std::vector<ConstQuadratureFieldPtr>& quad_fields, const std::vector<ConstFieldPtr>& v_fields,
                   DualFieldPtr vjp_shape_disp_sensitivity, const std::vector<DualFieldPtr>& vjp_sensitivities,
                   const std::vector<QuadratureFieldPtr>& vjp_quadrature_sensivities) const = 0;

  /// @brief name
  std::string name() const { return name_; }

 private:
  /// name
  std::string name_;
};

}  // namespace serac
