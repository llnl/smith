// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file constraint.hpp
 *
 * @brief Specifies interface for evaluating distributed constriants from fields as well as
 * their Jacobians and Hessian-vector products
 */

#pragma once

#include <vector>
#include "smith/physics/common.hpp"
#include "smith/physics/field_types.hpp"

namespace mfem {
class Vector;
class HypreParMatrix;
}  // namespace mfem

namespace smith {

class FiniteElementState;

/// @brief Abstract constraint class
class Constraint {
 public:
  /// @brief base constructor takes the name of the physics
  Constraint(const std::string& name = "constraint") : name_(name) {}

  /// @brief destructor
  virtual ~Constraint() {}

  /** @brief Virtual interface for computing the constraint, given a vector of
   * smith::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of smith::FiniteElementState*
   * @param fresh_evaluation boolean indicating if we re-evaluate or use previously cached evaluation
   * @return mfem::Vector which is the constraint evaluation
   */
  virtual mfem::Vector evaluate(double time, double dt, const std::vector<ConstFieldPtr>& fields,
                                bool fresh_evaluation = true) const = 0;

  /** @brief Virtual interface for computing constraint Jacobian from a vector of smith::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of smith::FiniteElementState*
   * @param direction index for which field to take the gradient with respect to
   * @param fresh_evaluation boolean indicating if we re-evaluate or use previously cached evaluation
   * @return std::unique_ptr<mfem::HypreParMatrix>
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> jacobian(double time, double dt,
                                                         const std::vector<ConstFieldPtr>& fields, int direction,
                                                         bool fresh_evaluation = true) const = 0;

  /** @brief Virtual interface for computing constraint Jacobian_tilde from a vector of smith::FiniteElementState*
   *         Jacobian_tilde is an optional approximation of the true Jacobian
   *
   * @param time time
   * @param dt time step
   * @param fields vector of smith::FiniteElementState*
   * @param direction index for which field to take the gradient with respect to
   * @param fresh_evaluation boolean indicating if we re-evaluate or use previously cached evaluation
   * @return std::unique_ptr<mfem::HypreParMatrix>
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> jacobian_tilde(double time, double dt,
                                                               const std::vector<ConstFieldPtr>& fields, int direction,
                                                               bool fresh_evaluation = true) const
  {
    return jacobian(time, dt, fields, direction, fresh_evaluation);
  };

  /** @brief Virtual interface for computing residual contribution Jacobian_tilde^(Transpose) * (Lagrange multiplier)
   * smith::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of smith::FiniteElementState*
   * @param multipliers mfem::Vector of Lagrange multipliers
   * @param direction index for which field to take the gradient with respect to
   * @param fresh_evaluation boolean indicating if we re-evaluate or use previously cached evaluation
   * @return std::Vector
   */
  virtual mfem::Vector residual_contribution(double time, double dt, const std::vector<ConstFieldPtr>& fields,
                                             const mfem::Vector& multipliers, int direction,
                                             bool fresh_evaluation = true) const
  {
    std::unique_ptr<mfem::HypreParMatrix> jac = jacobian_tilde(time, dt, fields, direction, fresh_evaluation);
    mfem::Vector y(jac->Width());
    y = 0.0;
    SLIC_ERROR_ROOT_IF(jac->Height() != multipliers.Size(), "Incompatible matrix and vector sizes.");
    jac->MultTranspose(multipliers, y);
    return y;
  };

  /** @brief Virtual interface for computing residual contribution Jacobian_tilde^(Transpose) * (Lagrange multiplier)
   * from a vector of smith::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of smith::FiniteElementState*
   * @param multipliers mfem::Vector of Lagrange multipliers
   * @param direction index for which field to take the gradient with respect to
   * @param fresh_evaluation boolean indicating if we re-evaluate or use previously cached evaluation
   * @return std::unique_ptr<mfem::HypreParMatrix>
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> residual_contribution_jacobian(
      [[maybe_unused]] double time, [[maybe_unused]] double dt,
      [[maybe_unused]] const std::vector<ConstFieldPtr>& fields, [[maybe_unused]] const mfem::Vector& multipliers,
      [[maybe_unused]] int direction, [[maybe_unused]] bool fresh_evaluation = true) const
  {
    SLIC_ERROR_ROOT(axom::fmt::format("Base class must override residual_contribution_jacobian before usage"));
    std::unique_ptr<mfem::HypreParMatrix> res_contr_jacobian = nullptr;
    return res_contr_jacobian;
  };

  /// @brief name
  std::string name() const { return name_; }

 private:
  /// @brief name provided to constraint
  std::string name_;
};

}  // namespace smith
