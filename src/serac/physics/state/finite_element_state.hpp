// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element_state.hpp
 *
 * @brief This file contains the declaration of structure that manages the MFEM objects
 * that make up the state for a given field
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "mfem.hpp"

#include "serac/infrastructure/variant.hpp"
#include "serac/physics/state/finite_element_vector.hpp"
#include "serac/numerics/functional/domain.hpp"

namespace serac {

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_scalar_valued(const GeneralCoefficient& coef)
{
  return holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef);
}

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_vector_valued(const GeneralCoefficient& coef)
{
  return holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef);
}

/**
 * @brief Class for encapsulating the critical MFEM components of a primal finite element field
 *
 * Namely: Mesh, FiniteElementCollection, FiniteElementState, and the true vector of the solution
 */
class FiniteElementState : public FiniteElementVector {
 public:
  using FiniteElementVector::FiniteElementVector;
  using mfem::Vector::Print;

  /**
   * @brief Copy constructor
   *
   * @param[in] rhs The input state used for construction
   */
  FiniteElementState(const FiniteElementState& rhs) : FiniteElementVector(rhs) {}

  /**
   * @brief Move construct a new Finite Element State object
   *
   * @param[in] rhs The input vector used for construction
   */
  FiniteElementState(FiniteElementState&& rhs) : FiniteElementVector(std::move(rhs)) {}

  /**
   * @brief Copy assignment
   *
   * @param rhs The right hand side input state
   * @return The assigned FiniteElementState
   */
  FiniteElementState& operator=(const FiniteElementState& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Move assignment
   *
   * @param rhs The right hand side input State
   * @return The assigned FiniteElementState
   */
  FiniteElementState& operator=(FiniteElementState&& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Copy assignment with HypreParVector
   *
   * @param rhs The right hand side input HypreParVector
   * @return The assigned FiniteElementState
   */
  FiniteElementState& operator=(const mfem::HypreParVector& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Copy assignment with mfem::Vector
   *
   * @param rhs The right hand side input State
   * @return The assigned FiniteElementState
   */
  FiniteElementState& operator=(const mfem::Vector& rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Copy assignment with double
   *
   * @param rhs The right hand side input double
   * @return The assigned FiniteElementState
   */
  FiniteElementState& operator=(double rhs)
  {
    FiniteElementVector::operator=(rhs);
    return *this;
  }

  /**
   * @brief Fill a user-provided grid function based on the underlying true vector
   *
   * This distributes true vector dofs to the finite element (local) dofs  by multiplying the true dofs
   * by the prolongation operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   */
  void fillGridFunction(mfem::ParGridFunction& grid_function) const { grid_function.SetFromTrueDofs(*this); }

  /**
   * @brief Initialize the true vector in the FiniteElementState based on an input grid function
   *
   * This distributes the grid function dofs to the true vector dofs by multiplying by the
   * restriction operator.
   *
   * @see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for details
   *
   * @param grid_function The grid function used to initialize the underlying true vector.
   */
  void setFromGridFunction(const mfem::ParGridFunction& grid_function) { grid_function.GetTrueDofs(*this); }

  /**
   * @brief Project a vector coefficient onto a set of dofs
   *
   * @param coef The vector coefficient to project
   * @param dof_list A list of true degrees of freedom to set. Note this is the scalar dof (not vdof) numbering.
   *
   * @note This only sets nodal values based on the coefficient at that point. It does not perform
   * a full least squares projection.
   */
  void project(mfem::VectorCoefficient& coef, mfem::Array<int>& dof_list);

  /**
   * @brief Project a scalar coefficient onto a set of dofs
   *
   * @param coef The vector coefficient to project
   * @param dof_list A list of true degrees of freedom to set. Note this is the scalar dof (not vdof) numbering.
   * @param component The component to set
   *
   * @note This only sets nodal values based on the coefficient at that point. It does not perform
   * a full least squares projection.
   */
  void project(mfem::Coefficient& coef, mfem::Array<int>& dof_list, std::optional<int> component = {});

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   *
   * @note This only sets nodal values based on the coefficient at that point. It does not perform
   * a full least squares projection.
   */
  void project(const GeneralCoefficient& coef);

  /// \overload
  void project(mfem::Coefficient& coef);

  /// \overload
  void project(mfem::VectorCoefficient& coef);

  /**
   * @brief Project a coefficient on a specific set of marked boundaries
   *
   * @param coef The coefficient to project
   * @param markers A marker array of the boundaries to set
   *
   * @note This only sets nodal values based on the coefficient at that point. It does not perform
   * a full least squares projection.
   */
  void projectOnBoundary(mfem::Coefficient& coef, const mfem::Array<int>& markers);

  /// \overload
  void projectOnBoundary(mfem::VectorCoefficient& coef, const mfem::Array<int>& markers);

  /// \overload
  void project(mfem::Coefficient& coef, const Domain& d);

  /// \overload
  void project(mfem::VectorCoefficient& coef, const Domain& d);

  /**
   * @brief Construct a grid function from the finite element state true vector
   *
   * @return The constructed grid function
   */
  mfem::ParGridFunction& gridFunction() const;

 protected:
  /**
   * @brief An optional container for a grid function (L-vector) view of the finite element state.
   *
   * If a user requests it, it is constructed and potentially reused during subsequent calls. It is
   * not updated unless specifically requested via the @a gridFunction method.
   */
  mutable std::unique_ptr<mfem::ParGridFunction> grid_func_;
};

/**
 * @brief Find the Lp norm of a finite element state across all dofs
 *
 * @param state The state variable to compute a norm of
 * @param p The order norm to compute
 * @return The Lp norm of the finite element state
 */
double norm(const FiniteElementState& state, const double p = 2);

/**
 * @brief Find the L2 norm of the error of a vector-valued finite element state with respect to an exact solution
 *
 * @param state The numerical solution
 * @param exact_solution The exact solution to measure error against
 * @return The L2 norm of the difference between \p state and \p exact_solution
 */
double computeL2Error(const FiniteElementState& state, mfem::VectorCoefficient& exact_solution);

/**
 * @brief Find the L2 norm of the error of a scalar-valued finite element state with respect to an exact solution
 *
 * @param state The numerical solution
 * @param exact_solution The exact solution to measure error against
 * @return The L2 norm of the difference between \p state and \p exact_solution
 */
double computeL2Error(const FiniteElementState& state, mfem::Coefficient& exact_solution);

}  // namespace serac
