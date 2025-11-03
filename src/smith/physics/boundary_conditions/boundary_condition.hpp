// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file boundary_condition.hpp
 *
 * @brief This file contains the declaration of the boundary condition class
 */

#pragma once

#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>
#include <cstddef>

#include "smith/infrastructure/logger.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_vector.hpp"

namespace smith {

/**
 * @brief Boundary condition information bundle
 */
class BoundaryCondition {
 public:
  /**
   * @brief Constructor for setting up a boundary condition using a set of attributes
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be null for all components
   * @param[in] attrs The set of boundary condition attributes in the mesh that the BC applies to
   * @param[in] space The finite element space on which this BC is applied. This is used to calculate the DOFs from the
   * attribute list.
   */
  BoundaryCondition(GeneralCoefficient coef, const std::optional<int> component,
                    const mfem::ParFiniteElementSpace& space, const std::set<int>& attrs);

  /**
   * @brief Minimal constructor for setting the true DOFs directly
   * @param[in] coef Either a mfem::Coefficient or mfem::VectorCoefficient representing the BC
   * @param[in] component The zero-indexed vector component if the BC applies to just one component,
   * should be null for all components
   * @param[in] true_dofs The vector indices of the relevant DOFs
   * @param[in] space The finite element space on which this BC is applied.
   */
  BoundaryCondition(GeneralCoefficient coef, const std::optional<int> component,
                    const mfem::ParFiniteElementSpace& space, const mfem::Array<int>& true_dofs);

  /**
   * @brief Determines whether a boundary condition is associated with a tag
   * @tparam Tag The type of the tag to compare against
   * @param[in] tag The tag to compare against
   * @pre Template type "Tag" must be an enumeration
   */
  template <typename Tag>
  bool tagEquals(const Tag tag) const
  {
    static_assert(std::is_enum_v<Tag>, "Only enumerations can be used to tag a boundary condition.");
    SLIC_ERROR_ROOT_IF(!tag_, "No tag has been configured for this boundary condition");
    bool tags_same_type = typeid(tag).hash_code() == tag_->second;
    SLIC_WARNING_ROOT_IF(!tags_same_type, "Attempting to compare tags of two different enum types (always false)");
    return (static_cast<int>(tag) == tag_->first) && tags_same_type;
  }

  /**
   * @brief Sets the tag for the BC
   * @tparam Tag The template type for the tag (label)
   * @param[in] tag The new tag
   * @pre Template type "Tag" must be an enumeration
   */
  template <typename Tag>
  void setTag(const Tag tag)
  {
    static_assert(std::is_enum_v<Tag>, "Only enumerations can be used to tag a boundary condition.");
    tag_ = {static_cast<int>(tag), typeid(tag).hash_code()};
  }

  /**
   * @brief Returns a non-owning reference to the array of boundary
   * attribute markers
   */
  const mfem::Array<int>& markers() const { return attr_markers_; }

  /**
   * @brief Returns a non-owning reference to the array of boundary
   * attribute markers
   */
  mfem::Array<int>& markers() { return attr_markers_; }

  /**
   * @brief Returns the DOF indices for an essential boundary condition
   * @return A non-owning reference to the array of indices
   *
   * @note True and local dofs are described in the <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a>
   */
  const mfem::Array<int>& getTrueDofList() const { return true_dofs_; }

  /**
   * @brief Returns the DOF indices for an essential boundary condition
   * @return A non-owning reference to the array of indices
   *
   * @note True and local dofs are described in the <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a>
   */
  const mfem::Array<int>& getLocalDofList() const { return local_dofs_; }

  /**
   * @brief Projects the associated coefficient over a solution vector on the DOFs constrained by the boundary condition
   * @param[in] time The time at which to project the boundary condition
   * @param[inout] state The field to project over
   */
  void setDofs(mfem::Vector& state, const double time = 0.0) const;

  /**
   * @brief Modify the system of equations \f$Ax=b\f$ by replacing equations that correspond to
   * essential boundary conditions with ones that prescribe the desired values. The rows of the matrix containing
   * essential dofs are set to zero with a one on the diagonal. To preserve symmetry,
   * the off-diagonal entries of associated columns of A are also zeroed out, and b is modified accordingly.
   * This function is equivalent to:
   *
   * \f[
   * A = \tilde{A} + A_e
   * (\tilde{A} + A_e) x = b
   * \tilde{A} x = b - A_e x
   * \f]
   *
   * where \f$ A_e \f$ contains the eliminated columns of \f$ A \f$ for the essential degrees of freedom. Note that
   * these equations only apply to the non-essential dofs. For the essential rows, \f$ A \f$ is modified to contain one
   * on the diagonal and the right hand side is modified to contain the essential value originally contained in \f$ x
   * \f$. If \f$ A \f$ is given as the input @a k_mat , \f$ A_e \f$ is returned in @a k_mat .
   * @param[inout] k_mat A stiffness (system) matrix. The rows and cols of the essential dofs will be set to zero with a
   * one on the diagonal after the return of this method.
   * @param[inout] rhs The RHS vector for the system. At return, this vector contains \f$ b - A_e x \f$.
   * @param[in] state The state from which the solution DOF values are extracted and used to modify @a k_mat
   * @pre The input state solution must contain the correct essential DOFs. This can be done by calling the @a
   * BoundaryCondition::setDofs method.
   */
  void apply(mfem::HypreParMatrix& k_mat, mfem::Vector& rhs, mfem::Vector& state) const;

 private:
  /**
   * @brief Uses mfem::ParFiniteElementSpace::GetEssentialTrueDofs to
   * determine the DOFs for the boundary condition from the stored attribute marker list
   *
   * @note This will set both the true and local dof values.
   */
  void setDofListsFromAttributeMarkers();

  /**
   * @brief "Manually" set the DOF indices without specifying the field to which they apply
   * @param[in] true_dofs The true vector indices of the DOFs constrained by the boundary condition
   *
   * @note This will set both the true and local internal dof index arrays.
   * @note True and local dofs are described in the <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a>
   */
  void setTrueDofList(const mfem::Array<int>& true_dofs);

  /**
   * @brief A coefficient containing either a mfem::Coefficient or an mfem::VectorCoefficient
   */
  GeneralCoefficient coef_;
  /**
   * @brief The vector component affected by this BC (empty implies all components)
   */
  std::optional<int> component_;
  /**
   * @brief The attribute marker array where this BC is active
   */
  mfem::Array<int> attr_markers_;
  /**
   * @brief The true DOFs affected by this BC
   * @note Only used for essential (Dirichlet) BCs
   */
  mfem::Array<int> true_dofs_;
  /**
   * @brief The local (finite element) DOFs affected by this BC
   * @note Only used for essential (Dirichlet) BCs
   */
  mfem::Array<int> local_dofs_;
  /**
   * @brief The state (field) affected by this BC
   * @note Only used for essential (Dirichlet) BCs
   */
  const mfem::ParFiniteElementSpace& space_;

  /**
   * @brief A label for the BC, for filtering purposes, in addition to its type hash
   * @note This should always correspond to an enum
   * The first element is the enum val, the second is the hash of the corresponding enum type
   */
  std::optional<std::pair<int, std::size_t>> tag_;
};

}  // namespace smith
