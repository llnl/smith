// Copyright Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file constraints.hpp
 *
 * @brief Specifies interface for evaluating vector of constriants from fields and their field gradients
 */

#pragma once

#include <vector>
#include "serac/physics/common.hpp"
#include "serac/physics/field_types.hpp"

namespace mfem {
class Vector;
class HypreParMatrix;
}  // namespace mfem

namespace serac {

class FiniteElementState;
class FiniteElementDual;

/// @brief Abstract constraint class
class Constraint {
 public:
  /// @brief base constructor takes the name of the physics
  Constraint(const std::string& name) : name_(name) {}

  /// @brief destructor
  virtual ~Constraint() {}

  /** @brief Virtual interface for computing the scale value for the objective/constrant, given a vector of
   * serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState* as arguments to the residual
   * @return mfem::Vector which is the constraint evaluation
   */
  virtual mfem::Vector evaluate(double time, double dt, const std::vector<ConstFieldPtr>& fields) const = 0;

  /** @brief Virtual interface for computing objective gradient from a vector of serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState* as arguments to the residual
   * @param direction index for which field to take the gradient with respect to
   * @return std::unique_ptr<mfem::HypreParMatrix>
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> jacobian(double time, double dt,
                                                         const std::vector<ConstFieldPtr>& fields,
                                                         int direction) const = 0;

  /** @brief constraint Hessian-vector productVirtual interface for computing objective gradient from a vector of
   * serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState* as arguments to the residual
   * @return std::unique_ptr<mfem::HypreParMatrix>
   */
  virtual std::unique_ptr<mfem::HypreParMatrix> hvp(double time, double dt, const std::vector<ConstFieldPtr>& fields);

  /// @brief name
  std::string name() const { return name_; }

 private:
  /// @brief name provided to objective
  std::string name_;
};

}  // namespace serac
