// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file weak_form.hpp
 *
 * @brief Specifies interface for evaluating weak form residuals and their gradients
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include "smith/physics/common.hpp"
#include "smith/physics/field_types.hpp"

namespace mfem {
class Vector;
class HypreParMatrix;
}  // namespace mfem

namespace smith {

class FiniteElementState;
class FiniteElementDual;

using QuadratureField = double;                 ///< This is a placeholder for quadrature fields
using QuadratureFieldPtr = double*;             ///< This is a placeholder for quadrature field pointers
using ConstQuadratureFieldPtr = const double*;  ///< This is a placeholder for const quadrature field pointers

/// @brief Abstract WeakForm class
class WeakForm {
 public:
  /** @brief base constructor takes the name of the physics
   * @param name provide a name corresponding to the physics
   */
  WeakForm(std::string name) : name_(name) {}

  /// @brief destructor
  virtual ~WeakForm() {}

  /** @brief Virtual interface for computing the residual vector of a weak form
   *
   * @param time_info time and timestep information
   * @param shape_disp smith::FiniteElementState*, change in model coordinates relative to the initially read in mesh
   * @param fields vector of smith::FiniteElementState*
   * @param quad_fields vector of ConstQuadratureFieldPtr
   * @return mfem::Vector
   */
  virtual mfem::Vector residual(TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                                const std::vector<ConstQuadratureFieldPtr>& quad_fields = {}) const = 0;

  /** @brief Derivative of the residual with respect to specified field arguments: sum_j d{r}/d{fields}_j *
   * argument_tangents[j], j are input fields (columns)
   * @param time_info time and timestep information
   * @param shape_disp smith::FiniteElementState*, change in model coordinates relative to the initially read in mesh
   * @param fields vector of smith::FiniteElementState*
   * @param field_argument_tangents specifies the weighting of the residual derivative with respect to each field
   * @param quad_fields vector of ConstQuadratureFieldPtr
   * @return std::unique_ptr<mfem::HypreParMatrix> returns sum_j d{r}/d{fields}_j * argument_tangents[j], where
   * {fields}_j is the jth field, {r} is the residual
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> jacobian(
      TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
      const std::vector<double>& field_argument_tangents,
      const std::vector<ConstQuadratureFieldPtr>& quad_fields = {}) const = 0;

  /**
   * @brief Jacobian-vector product, will overwrite any existing values in jvp_reactions
   * @param time_info time and timestep information
   * @param shape_disp smith::FiniteElementState*, change in model coordinates relative to the initially read in mesh
   * @param fields vector of smith::FiniteElementState*
   * @param quad_fields vector of ConstQuadratureFieldPtr
   * @param v_shape_disp shape_displacement tangent
   * @param v_fields field tangents, right hand side 'v' fields
   * @param v_quad_fields quadrature_field_tangents
   * @param jvp_reaction output jvps: d{r} / d{fields}_j * fieldsV[j]
   * nullptr fieldsV are assumed to be all zero to avoid extra calculations
   */
  virtual void jvp(TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                   const std::vector<ConstQuadratureFieldPtr>& quad_fields, ConstFieldPtr v_shape_disp,
                   const std::vector<ConstFieldPtr>& v_fields,
                   const std::vector<ConstQuadratureFieldPtr>& v_quad_fields, DualFieldPtr jvp_reaction) const = 0;

  /**
   * @brief Vector-Jacobian product, will += into existing values in vjpFields
   * @param time_info time and timestep information
   * @param shape_disp smith::FiniteElementState*, change in model coordinates relative to the initially read in mesh
   * @param fields vector of smith::FiniteElementState*
   * @param quad_fields vector of ConstQuadratureFieldPtr
   * @param v_field left hand side 'v' field
   * @param vjp_shape_disp_sensitivity vjp for shape_displacement: v_fields * d{r} / d{shape_disp}
   * @param vjp_sensitivities output vjps, 1 per input field: v_fields * d{r} / d{fields}_j
   * @param vjp_quadrature_sensivities output vjps, 1 per input quadrature field: v_field * d{r} /
   * d{quadrature_field}_j
   */
  virtual void vjp(TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                   const std::vector<ConstQuadratureFieldPtr>& quad_fields, ConstFieldPtr v_field,
                   DualFieldPtr vjp_shape_disp_sensitivity, const std::vector<DualFieldPtr>& vjp_sensitivities,
                   const std::vector<QuadratureFieldPtr>& vjp_quadrature_sensivities) const = 0;

  /// @brief name
  std::string name() const { return name_; }

 private:
  /// name
  std::string name_;
};

}  // namespace smith
