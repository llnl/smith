// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file contact_data.hpp
 *
 * @brief Class for storing contact data
 */

#pragma once

#include <memory>
#include <set>
#include <vector>

#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#ifdef SMITH_USE_TRIBOL
#include "smith/physics/contact/contact_interaction.hpp"
#endif

namespace smith {

namespace contact {

/**
 * @brief Default contact options: frictionless mortar with penalty = 1000
 * enforcement
 */
const ContactOptions default_contact_options = {.method = ContactMethod::SingleMortar,
                                                .enforcement = ContactEnforcement::Penalty,
                                                .type = ContactType::Frictionless,
                                                .penalty = 1.0e3};

}  // namespace contact

/**
 * @brief This class stores all ContactInteractions for a problem, calls Tribol functions that act on all contact
 * interactions, and agglomerates fields that exist over different ContactInteractions.
 */
class ContactData {
 public:
  /**
   * @brief The constructor
   *
   * @param mesh The volume mesh for the problem
   */
  ContactData(const mfem::ParMesh& mesh);

  /**
   * @brief Destructor to finalize Tribol
   */
  ~ContactData();

  /**
   * @brief Add another contact interaction
   *
   * @param interaction_id Unique identifier for the ContactInteraction (used in Tribol)
   * @param bdry_attr_surf1 MFEM boundary attributes for the first (mortar) surface
   * @param bdry_attr_surf2 MFEM boundary attributes for the second (nonmortar) surface
   * @param contact_opts Defines contact method, enforcement, type, and penalty
   */
  void addContactInteraction(int interaction_id, const std::set<int>& bdry_attr_surf1,
                             const std::set<int>& bdry_attr_surf2, ContactOptions contact_opts);

  /**
   * @brief Updates the positions, forces, and Jacobian contributions associated with contact
   *
   * @param cycle The current simulation cycle
   * @param time The current time
   * @param dt The timestep size to attempt
   */
  void update(int cycle, double time, double& dt);

  /**
   * @brief Resets the contact pressures to zero
   *
   */
  void reset();

  /**
   * @brief Get the contact constraint residual (i.e. nodal forces) from all contact interactions
   *
   * @return Nodal contact forces on the true DOFs
   *
   * @pre update() must be called with the current configuration so the force contributions are up-to-date
   */
  FiniteElementDual forces() const;

  /**
   * @brief Returns pressures from all contact interactions on the contact surface true degrees of freedom
   *
   * The type of pressure (normal or vector-valued) is set by the ContactType in the ContactOptions struct for the
   * contact interaction. TiedNormal and Frictionless (the two type supported in Tribol) correspond to scalar normal
   * pressure. Only linear (order = 1) pressure fields are supported.
   *
   * @return Pressure true degrees of freedom on each contact interaction (merged into one mfem::HypreParVector)
   */
  mfem::HypreParVector mergedPressures() const;

  /**
   * @brief Returns nodal gaps from all contact interactions on the contact surface true degrees of freedom
   *
   * The type of gap (normal or vector-valued) is set by the ContactType in the ContactOptions struct for the contact
   * interaction. TiedNormal and Frictionless (the two type supported in Tribol) correspond to scalar gap normal.  Only
   * linear (order = 1) gap fields are supported.
   *
   * @param [in] zero_inactive Sets inactive t-dofs to zero gap
   * @return Nodal gap true degrees of freedom on each contact interaction (merged into one mfem::HypreParVector)
   *
   * @pre update() must be called with the current configuration so the gap values are up-to-date
   */
  mfem::HypreParVector mergedGaps(bool zero_inactive = false) const;

  /**
   * @brief Returns a 2x2 block Jacobian on displacement/pressure true degrees of
   * freedom from contact constraints
   *
   * The element Jacobian contributions are computed upon calling update(). This method does MPI communication to move
   * Jacobian contributions to the correct rank, then assembles the contributions.  The pressure degrees of freedom for
   * all contact interactions are merged into a single block.
   *
   * @note The blocks are owned by the BlockOperator
   *
   * @return Pointer to block Jacobian (2x2 BlockOperator of HypreParMatrix)
   */
  std::unique_ptr<mfem::BlockOperator> mergedJacobian() const;

  /**
   * @brief Computes the residual including contact terms
   *
   * @param [in] u_shape Shape displacement vector (size of [displacement] block)
   * @param [in] u Solution vector ([displacement; pressure] block vector)
   * @param [in,out] r Residual vector ([force; gap] block vector); takes in initialized residual force vector and adds
   * contact contributions
   *
   * @pre The current coordinates must be up-to-date
   *
   * @note This method calls update() to compute residual and Jacobian contributions based on the current configuration
   */
  void residualFunction(const mfem::Vector& u_shape, const mfem::Vector& u, mfem::Vector& r);

  /**
   * @brief Computes the Jacobian including contact terms, given the non-contact Jacobian terms
   *
   * @param orig_J The non-contact terms of the Jacobian, not including essential boundary conditions
   * @return Jacobian with contact terms, not including essential boundary conditions
   *
   * @pre update() must be called with the current configuration so the Jacobian contributions are up-to-date
   */
  std::unique_ptr<mfem::BlockOperator> jacobianFunction(mfem::HypreParMatrix* orig_J) const;

  /**
   * @brief Set the pressure field
   *
   * This sets Tribol's pressure degrees of freedom based on
   *  1) the values in merged_pressure for Lagrange multiplier enforcement
   *  2) the nodal gaps and penalty for penalty enforcement
   *
   * @note The nodal gaps must be up-to-date for penalty enforcement
   *
   * @param merged_pressures Current pressure true dof values in a merged mfem::Vector
   */
  void setPressures(const mfem::Vector& merged_pressures) const;

  /**
   * @brief Update the current coordinates based on the new displacement field
   *
   * @param u_shape Shape displacement vector
   * @param u Current displacement dof values
   */
  void setDisplacements(const mfem::Vector& u_shape, const mfem::Vector& u);

  /**
   * @brief Have there been contact interactions added?
   *
   * @return true if contact interactions have been added
   * @return false if there are no contact interactions
   */
  bool haveContactInteractions() const
  {
#ifdef SMITH_USE_TRIBOL
    return !interactions_.empty();
#else
    return false;
#endif
  }

#ifdef SMITH_USE_TRIBOL
  /**
   * @brief Get the contact interactions
   *
   * @return Vector of contact interactions
   */
  const std::vector<ContactInteraction>& getContactInteractions() const { return interactions_; }
#endif

  /**
   * @brief Are any contact interactions enforced using Lagrange multipliers?
   *
   * @return true: at least one contact interaction is using Lagrange multiplier enforcement
   * @return false: no contact interactions are using Lagrange multipliers
   */
  bool haveLagrangeMultipliers() const { return have_lagrange_multipliers_; }

  /**
   * @brief Get the number of Lagrange multiplier true degrees of freedom
   *
   * @return Number of Lagrange multiplier true degrees of freedom
   */
  int numPressureDofs() const { return num_pressure_dofs_; };

 private:
#ifdef SMITH_USE_TRIBOL
  /**
   * @brief Computes interaction pressure T-dof offsets and global pressure T-dof offsets
   *
   * @note Only non-zero on constraints with Lagrange multiplier enforcement
   */
  void updateDofOffsets() const;

  /**
   * @brief The volume mesh for the problem
   */
  const mfem::ParMesh& mesh_;
#endif

  /**
   * @brief Reference coordinates of the mesh
   */
  const mfem::ParGridFunction* reference_nodes_;

#ifdef SMITH_USE_TRIBOL
  /**
   * @brief Current coordinates of the mesh
   */
  mfem::ParGridFunction current_coords_;

  /**
   * @brief The contact boundary condition information
   */
  std::vector<ContactInteraction> interactions_;
#endif

  /**
   * @brief True if any of the contact interactions are enforced using Lagrange
   * multipliers
   */
  bool have_lagrange_multipliers_;

  /**
   * @brief Pressure T-dof count (only including pressure fields with Lagrange multiplier enforcement)
   */
  int num_pressure_dofs_;

  /**
   * @brief Tracks whether the Jacobian and pressure offsets need to be updated
   *
   * The offsets need to be updated when a new contact interaction is added.
   *
   */
  mutable bool offsets_up_to_date_;

  /**
   * @brief Offsets giving size of each block Jacobian contribution
   *
   * Size = 3, first block: displacement, second block: pressure Lagrange multipliers
   *
   * @note This is mutable so it can be updated when pressures/gaps/Jacobians are retrieved.
   */
  mutable mfem::Array<int> jacobian_offsets_;

  /**
   * @brief Get the interaction offset vector tracking number of on-rank pressure T-dofs per interaction
   *
   * Sized to number of contact interactions + 1. Only non-zero on contact interactions with Lagrange multiplier
   * enforcement.
   *
   * @note This is mutable so it can be updated when pressures/gaps/Jacobians are retrieved.
   */
  mutable mfem::Array<int> pressure_dof_offsets_;

  /**
   * @brief Get global offset vector for pressure T-dofs
   *
   * Sized to number of ranks if HYPRE_AssumedPartitionCheck() is false; sized to 3 if true. Tracks global offsets of
   * total Lagrange multiplier-enforced pressure dofs.
   *
   * @note This is mutable so it can be updated when pressures/gaps/Jacobians are retrieved.
   */
  mutable mfem::Array<HYPRE_BigInt> global_pressure_dof_offsets_;

  int cycle_{0};
  double time_{0.0};
  double dt_{1.0};
};

}  // namespace smith
