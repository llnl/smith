// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file contact_interaction.hpp
 *
 * @brief Class for storing a contact interaction and interfacing with Tribol
 */

#pragma once

#include "smith/smith_config.hpp"

#ifdef SMITH_USE_TRIBOL

#include <memory>
#include <set>

#include "mfem.hpp"
#include "tribol/common/Parameters.hpp"

#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

/**
 * @brief A ContactInteraction defines a single contact interaction between two surfaces
 *
 * This class stores the details of a contact interaction between two surfaces.  It also interfaces with the Tribol
 * interface physics library, defining the Tribol coupling scheme for the interaction.  A problem can have multiple
 * ContactInteractions defined on it with different contact surfaces and enforcement schemes.  See the ContactData class
 * for the container holding all contact interactions and for Tribol API calls acting on all contact interactions.
 **/
class ContactInteraction {
 public:
  /**
   * @brief The constructor
   *
   * @param interaction_id Unique identifier for the ContactInteraction (used in Tribol)
   * @param mesh Mesh of the entire domain
   * @param bdry_attr_surf1 MFEM boundary attributes for the first (mortar) surface
   * @param bdry_attr_surf2 MFEM boundary attributes for the second (nonmortar) surface
   * @param current_coords Reference to the grid function holding current mesh
   * @param contact_opts Defines contact method, enforcement, type, and penalty
   * coordinates
   */
  ContactInteraction(int interaction_id, const mfem::ParMesh& mesh, const std::set<int>& bdry_attr_surf1,
                     const std::set<int>& bdry_attr_surf2, const mfem::ParGridFunction& current_coords,
                     ContactOptions contact_opts);

  /**
   * @brief Get the integer identifier of the contact interaction
   *
   * @return Contact interaction ID
   */
  int getInteractionId() const { return interaction_id_; }

  /**
   * @brief Get the contact options for the contact interaction
   *
   * @return Struct of contact options
   */
  const ContactOptions& getContactOptions() const { return contact_opts_; }

  /**
   * @brief Get the contact constraint residual (i.e. nodal forces) from this contact interaction
   *
   * @return Nodal contact forces on the true DOFs
   */
  FiniteElementDual forces() const;

  /**
   * @brief Get the pressure true degrees of freedom on the contact surface for the contact interaction
   *
   * The type of pressure (normal or vector-valued) is set by the ContactType in the ContactOptions struct for the
   * contact interaction. TiedNormal and Frictionless (the two type supported in Tribol) correspond to scalar normal
   * pressure. Only linear (order = 1) pressure fields are supported.
   *
   * @return Pressure true degrees of freedom as a FiniteElementState
   */
  FiniteElementState pressure() const;

  /**
   * @brief Get the nodal gaps on the true degrees of freedom of the contact surface for the contact interaction
   *
   * The type of gap (normal or vector-valued) is set by the ContactType in the ContactOptions struct for the contact
   * interaction. TiedNormal and Frictionless (the two type supported in Tribol) correspond to scalar gap normal.  Only
   * linear (order = 1) gap fields are supported.
   *
   * @return Nodal gaps on the true DOFs on the contact surface as a FiniteElementDual
   */
  FiniteElementDual gaps() const;

  /**
   * @brief Get the (2x2) block Jacobian for the contact interaction
   *
   * Block row/col 0: displacement space
   * Block row/col 1: pressure space
   *
   * The element Jacobian contributions are computed upon calling ContactData::update(). This method does MPI
   * communication to move Jacobian contributions to the correct rank, then assembles the contributions.
   *
   * @return Contact Jacobian as a BlockOperator
   */
  std::unique_ptr<mfem::BlockOperator> jacobian() const;

  /**
   * @brief Get the finite element space of the pressure DOFs
   *
   * @return mfem::ParFiniteElementSpace of the pressure DOFs
   */
  mfem::ParFiniteElementSpace& pressureSpace() const;

  /**
   * @brief Updates the pressure DOFs stored in Tribol
   *
   * @param pressure FiniteElementState holding pressure DOF values
   */
  void setPressure(const FiniteElementState& pressure) const;

  /**
   * @brief Returns the number of pressure DOFs on this rank
   *
   * @return Number of pressure DOFs as an integer
   */
  int numPressureDofs() const;

  /**
   * @brief List of pressure/gap DOFs that are not active
   *
   * @return Array of inactive DOFs
   */
  const mfem::Array<int>& inactiveDofs() const;

  /**
   * @brief List of pressure/gap DOFs that are not active
   *
   * @param pressure A custom pressure field
   * @return Array of inactive DOFs
   */
  const mfem::Array<int>& inactiveDofs(const FiniteElementState& pressure) const;

  /**
   * @brief Turn on/off Jacobian evaluation
   *
   * @param eval True if Jacobian should be evaluated, false otherwise
   */
  void evalJacobian(bool eval) const;

 private:
  /**
   * @brief Get the Tribol enforcement method given a Smith enforcement method
   *
   * @return Tribol enforcement method
   */
  tribol::ContactMethod getMethod() const;

  /**
   * @brief Unique identifier for the contact interaction
   */
  int interaction_id_;

  /**
   * @brief Defines contact method, enforcement, type, and penalty
   */
  ContactOptions contact_opts_;

  /**
   * @brief Reference to the current coords GridFunction
   */
  const mfem::ParGridFunction& current_coords_;

  /**
   * @brief List of true DOFs currently not in the active set
   */
  mutable mfem::Array<int> inactive_tdofs_;
};

}  // namespace smith

#endif
