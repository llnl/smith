// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
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
#include "smith/physics/common.hpp"
#include "smith/physics/field_types.hpp"

namespace mfem {
class Vector;
}  // namespace mfem

namespace smith {

class FiniteElementState;

/// @brief Abstract residual class
class ScalarObjective {
 public:
  /// @brief base constructor takes the name of the physics
  ScalarObjective(const std::string& name) : name_(name) {}

  /// @brief destructor
  virtual ~ScalarObjective() {}

  /** @brief Virtual interface for computing the scale value for the objective/constrant, given a vector of
   * smith::FiniteElementState*
   *
   * @param time_info time and timestep information
   * @param shape_disp shape displacement
   * @param fields inputs to residual operator
   * @return double which is the scalar objective value
   */
  virtual double evaluate(TimeInfo time_info, ConstFieldPtr shape_disp,
                          const std::vector<ConstFieldPtr>& fields) const = 0;

  /** @brief Virtual interface for computing objective gradient from a vector of smith::FiniteElementState*
   *
   * @param time_info time and timestep information
   * @param shape_disp shape displacement
   * @param fields inputs to residual operator
   * @param field_ordinal index for which field to take the gradient with respect to
   * @return mfem::Vector
   */
  virtual mfem::Vector gradient(TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                                size_t field_ordinal) const = 0;

  /** @brief Compute objective gradient from a vector of FiniteElementState*, using int for index
   *
   * @param time_info time and timestep information
   * @param shape_disp shape displacement
   * @param fields inputs to residual operator
   * @param field_ordinal index for which field to take the gradient with respect to
   * @return mfem::Vector
   */
  virtual mfem::Vector gradient(TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                                int field_ordinal) const
  {
    return gradient(time_info, shape_disp, fields, static_cast<size_t>(field_ordinal));
  }

  /** @brief Virtual interface for computing objective gradient with respect to the mesh coordinates
   *
   * @param time_info time and timestep information
   * @param shape_disp shape displacement
   * @param fields inputs to residual operator
   * @return mfem::Vector
   */
  virtual mfem::Vector mesh_coordinate_gradient(TimeInfo time_info, ConstFieldPtr shape_disp,
                                                const std::vector<ConstFieldPtr>& fields) const = 0;

  /// @brief name
  std::string name() const { return name_; }

 private:
  /// @brief name provided to objective
  std::string name_;
};

}  // namespace smith
