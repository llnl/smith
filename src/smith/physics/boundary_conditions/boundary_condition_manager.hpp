// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file boundary_condition_manager.hpp
 *
 * @brief This file contains the declaration of the boundary condition manager class
 */

#pragma once

#include <memory>
#include <set>
#include <optional>
#include <vector>

#include "smith/physics/boundary_conditions/boundary_condition.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

/**
 * @brief A container for the boundary condition information relating to a specific physics module
 */
class BoundaryConditionManager {
 public:
  /**
   * @brief Construct a new Boundary Condition Manager object
   *
   * @param mesh The mesh for the underlying physics module
   */
  explicit BoundaryConditionManager(const mfem::ParMesh& mesh) : num_attrs_(mesh.bdr_attributes.Max()) {}

  /**
   * @brief Set the essential boundary conditions from a list of boundary markers and a coefficient
   *
   * @param[in] ess_bdr The set of essential BC attributes
   * @param[in] ess_bdr_coef The essential BC value coefficient
   * @param[in] space The finite element space to which the BC should be applied
   * @param[in] component The component to set (null implies all components are set)
   */
  void addEssential(const std::set<int>& ess_bdr, smith::GeneralCoefficient ess_bdr_coef,
                    mfem::ParFiniteElementSpace& space, const std::optional<int> component = {});

  /**
   * @brief Set a list of local degrees of freedom from a scalar coefficient
   *
   * @param[in] local_dofs Indices of ldofs to set with a Dirichlet condition
   * @param[in] ess_bdr_coef The scalar coefficient that evaluates to the Dirichlet condition
   * @param[in] space The finite element space to which the BC should be applied
   * @param[in] component The component to set (null implies all components are set)
   */
  void addEssential(const mfem::Array<int>& local_dofs, std::shared_ptr<mfem::Coefficient> ess_bdr_coef,
                    mfem::ParFiniteElementSpace& space, std::optional<int> component = {});

  /**
   * @ brief Set a list of true degrees of feedom from a vector coefficient
   *
   * @param[in] true_dofs Indices of tdofs to set with a Dirichlet condition
   * @param[in] ess_bdr_coef The coefficient that evaluates to the Dirichlet condition
   * @param[in] space The finite element space to which the BC should be applied
   */
  void addEssentialByTrueDofs(const mfem::Array<int>& true_dofs, std::shared_ptr<mfem::VectorCoefficient> ess_bdr_coef,
                              mfem::ParFiniteElementSpace& space);

  /**
   * @brief Returns all the true degrees of freedom associated with all the essential BCs
   * @return A const reference to the list of true DOF indices, without duplicates and sorted
   */
  const mfem::Array<int>& allEssentialTrueDofs() const
  {
    if (!all_dofs_valid_) {
      updateAllDofs();
    }
    return all_true_dofs_;
  }

  /**
   * @brief Returns all the local degrees of freedom associated with all the essential BCs
   * @return A const reference to the list of local DOF indices, without duplicates and sorted
   */
  const mfem::Array<int>& allEssentialLocalDofs() const
  {
    if (!all_dofs_valid_) {
      updateAllDofs();
    }
    return all_local_dofs_;
  }

  /**
   * @brief Eliminates all essential BCs from a matrix
   * @param[inout] matrix The matrix to eliminate from, will be modified
   * @return The eliminated matrix entries
   * @note The sum of the eliminated matrix and the modified parameter is
   * equal to the initial state of the parameter
   */
  std::unique_ptr<mfem::HypreParMatrix> eliminateAllEssentialDofsFromMatrix(mfem::HypreParMatrix& matrix) const
  {
    return std::unique_ptr<mfem::HypreParMatrix>(matrix.EliminateRowsCols(allEssentialTrueDofs()));
  }

  /**
   * @brief Accessor for the essential BC objects
   */
  std::vector<BoundaryCondition>& essentials() { return ess_bdr_; }

  /**
   * @brief Accessor for the essential BC objects
   */
  const std::vector<BoundaryCondition>& essentials() const { return ess_bdr_; }

 private:
  /**
   * @brief Updates the "cached" list of all DOF indices
   */
  void updateAllDofs() const;

  /**
   * @brief The total number of boundary attributes for a mesh
   */
  const int num_attrs_;

  /**
   * @brief The vector of essential boundary conditions
   */
  std::vector<BoundaryCondition> ess_bdr_;

  /**
   * @brief The set of boundary attributes associated with
   * already-registered BCs
   * @see https://mfem.org/mesh-formats/
   */
  std::set<int> attrs_in_use_;

  /**
   * @brief The set of true DOF indices corresponding
   * to all registered essential BCs
   */
  mutable mfem::Array<int> all_true_dofs_;

  /**
   * @brief The set of local DOF indices corresponding
   * to all registered essential BCs
   */
  mutable mfem::Array<int> all_local_dofs_;

  /**
   * @brief Whether the set of stored total DOFs is valid
   */
  mutable bool all_dofs_valid_ = false;
};

}  // namespace smith
