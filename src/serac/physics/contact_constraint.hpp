// Copyright Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file contact_constraint.hpp
 *
 * @brief Specifies interface for evaluating contact constriants from fields as well as
 * their Jacobians
 */

#pragma once

#include <vector>

#include "serac/serac_config.hpp"

#ifdef SERAC_USE_TRIBOL

#include "tribol/interface/tribol.hpp"

#include "serac/physics/constraint.hpp"
#include "serac/physics/field_types.hpp"
#include "serac/physics/contact/contact_config.hpp"
#include "serac/physics/contact/contact_data.hpp"

namespace mfem {
class Vector;
class HypreParMatrix;
}  // namespace mfem

namespace serac {

/**
 * @brief Enumerates ContactFields for ContactConstraint class
 */
enum ContactFields
{
  SHAPE,
  DISP,
};

/** @brief Interface for extracting the iblock, jblock block from a std::unique_ptr<mfem::BlockOperator>
 *         said block is returned as a std::unique_ptr<mfem::HypreParMatrix> if possible
 *         All other blocks are deleted
 *
 * @param block_operator the block operator
 * @param iblock row block index
 * @param jblock column block index
 * @return std::unique_ptr<mfem::HypreParMatrix> The requested block of block_operator
 */
static std::unique_ptr<mfem::HypreParMatrix> safelyObtainBlock(mfem::BlockOperator* block_operator, int iblock,
                                                               int jblock)
{
  SLIC_ERROR_IF(iblock < 0 || jblock < 0, "block indicies must be non-negative");
  SLIC_ERROR_IF(iblock > block_operator->NumRowBlocks() || jblock > block_operator->NumColBlocks(),
                "one or more block indicies are too large and the requested block does not exist");
  SLIC_ERROR_IF(block_operator->IsZeroBlock(iblock, jblock), "attempting to extract a null block");
  block_operator->owns_blocks = false;
  for (int i = 0; i < block_operator->NumRowBlocks(); i++) {
    for (int j = 0; j < block_operator->NumColBlocks(); j++) {
      if (i == iblock && j == jblock) {
        continue;
      }
      if (!block_operator->IsZeroBlock(i, j)) {
        delete &block_operator->GetBlock(i, j);
      }
    }
  }
  auto Ablk = dynamic_cast<mfem::HypreParMatrix*>(&block_operator->GetBlock(iblock, jblock));
  SLIC_ERROR_IF(!Ablk, "failed cast to mfem::HypreParMatrix");
  std::unique_ptr<mfem::HypreParMatrix> Ablk_unique(Ablk);
  return Ablk_unique;
};

class FiniteElementState;

/**
 * @brief A ContactConstraint defines a gap constraint associated to contact problem
 *
 * This class stores the details of a single contact interaction between two surfaces. It also interfaces provides a
 * description of a contact constraint given by a single contact interaction. A ContactConstraint can have a single
 * ContactInteraction and will default to LagrangeMultiplier as it will be up to the solver that takes this
 * ContactConstraint to determine how it will enforce the constraint.
 */
class ContactConstraint : public Constraint {
 public:
  /**
   * @brief The constructor
   *
   * @param interaction_id Unique identifier for the ContactInteraction (used in Tribol)
   * @param mesh Mesh of the entire domain
   * @param bdry_attr_surf1 MFEM boundary attributes for the first (mortar) surface
   * @param bdry_attr_surf2 MFEM boundary attributes for the second (nonmortar) surface
   * @param contact_opts Defines contact method
   * @param name provides a name to associate to the contact constraint
   */
  ContactConstraint(int interaction_id, const mfem::ParMesh& mesh, const std::set<int>& bdry_attr_surf1,
                    const std::set<int>& bdry_attr_surf2, ContactOptions contact_opts,
                    const std::string& name = "contact_constraint")
      : Constraint(name), contact_(mesh), contact_opts_{contact_opts}
  {
    contact_opts_.enforcement = ContactEnforcement::LagrangeMultiplier;
    contact_.addContactInteraction(interaction_id, bdry_attr_surf1, bdry_attr_surf2, contact_opts_);
    interaction_id_ = interaction_id;
  }

  /// @brief destructor
  virtual ~ContactConstraint() {}

  /** @brief Interface for computing the gap contact constraint, given a vector of
   * serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState*
   * @return mfem::Vector which is the constraint evaluation
   */
  mfem::Vector evaluate(double time, double dt, const std::vector<ConstFieldPtr>& fields) const
  {
    contact_.setDisplacements(*fields[ContactFields::SHAPE], *fields[ContactFields::DISP]);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_GAP);

    // note: Tribol does not use cycle.
    int cycle = 0;
    contact_.update(cycle, time, dt);
    auto merged_gaps = contact_.mergedGaps(false);
    merged_gaps.SetOwnership(false);
    mfem::Vector merged_gaps_vector = std::move(merged_gaps);
    merged_gaps_vector.MakeDataOwner();
    return merged_gaps_vector;
  };

  /** @brief Interface for computing contact gap constraint Jacobian from a vector of serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState*
   * @param direction index for which field to take the gradient with respect to
   * @return std::unique_ptr<mfem::HypreParMatrix> The true Jacobian
   */
  std::unique_ptr<mfem::HypreParMatrix> jacobian(double time, double dt, const std::vector<ConstFieldPtr>& fields,
                                                 int direction) const
  {
    SLIC_ERROR_IF(direction != ContactFields::DISP, "requesting a non displacement-field derivative");
    contact_.setDisplacements(*fields[ContactFields::SHAPE], *fields[ContactFields::DISP]);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_JACOBIAN);

    int cycle = 0;
    contact_.update(cycle, time, dt);
    auto J_contact = contact_.mergedJacobian();

    int iblock = 1;
    int jblock = 0;
    auto dgdu = safelyObtainBlock(J_contact.get(), iblock, jblock);
    return dgdu;
  };

  /** @brief Interface for computing residual contribution Jacobian_tilde^T multiplier from a vector of
   * serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState*
   * @param multipliers mfem::Vector of Lagrange multipliers
   * @param direction index for which field to take the gradient with respect to
   * @return std::Vector
   */
  mfem::Vector residual_contribution(double time, double dt, const std::vector<ConstFieldPtr>& fields,
                                     const mfem::Vector& multipliers, int direction) const
  {
    SLIC_ERROR_IF(direction != ContactFields::DISP, "requesting a non displacement-field derivative");
    contact_.setDisplacements(*fields[ContactFields::SHAPE], *fields[ContactFields::DISP]);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_GAP);

    int cycle = 0;
    // we need to call update first to update gaps
    for (auto& interaction : contact_.getContactInteractions()) {
      interaction.evalJacobian(false);
    }
    contact_.update(cycle, time, dt);
    // with updated gaps, we can update pressure for contact interactions with penalty enforcement
    contact_.setPressures(multipliers);
    // call update again with the right pressures
    for (auto& interaction : contact_.getContactInteractions()) {
      interaction.evalJacobian(true);
    }
    contact_.update(cycle, time, dt);

    return contact_.forces();
  };

  /** @brief Interface for computing Jacobians of the residual contribution from a vector of
   * serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState*
   * @param multipliers mfem::Vector of Lagrange multipliers
   * @param direction index for which field to take the gradient with respect to
   * @return std::unique_ptr<mfem::HypreParMatrix>
   */
  std::unique_ptr<mfem::HypreParMatrix> residual_contribution_jacobian(double time, double dt,
                                                                       const std::vector<ConstFieldPtr>& fields,
                                                                       const mfem::Vector& multipliers,
                                                                       int direction) const
  {
    SLIC_ERROR_IF(direction != ContactFields::DISP, "requesting a non displacement-field derivative");
    contact_.setDisplacements(*fields[ContactFields::SHAPE], *fields[ContactFields::DISP]);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_JACOBIAN);

    int cycle = 0;
    // we need to call update first to update gaps
    for (auto& interaction : contact_.getContactInteractions()) {
      interaction.evalJacobian(false);
    }
    contact_.update(cycle, time, dt);
    // with updated gaps, we can update pressure for contact interactions with penalty enforcement
    contact_.setPressures(multipliers);
    // call update again with the right pressures
    for (auto& interaction : contact_.getContactInteractions()) {
      interaction.evalJacobian(true);
    }
    contact_.update(cycle, time, dt);

    auto J_contact = contact_.mergedJacobian();
    int iblock = 0;
    int jblock = 0;
    auto Hessian = safelyObtainBlock(J_contact.get(), iblock, jblock);
    return Hessian;
  };

  /** @brief Interface for computing contact constraint Jacobian_tilde from a vector of serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState*
   * @param direction index for which field to take the gradient with respect to
   * @return std::unique_ptr<mfem::HypreParMatrix>
   */
  std::unique_ptr<mfem::HypreParMatrix> jacobian_tilde(double time, double dt, const std::vector<ConstFieldPtr>& fields,
                                                       int direction) const
  {
    SLIC_ERROR_IF(direction != ContactFields::DISP, "requesting a non displacement-field derivative");
    contact_.setDisplacements(*fields[ContactFields::SHAPE], *fields[ContactFields::DISP]);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_JACOBIAN);

    int cycle = 0;
    contact_.update(cycle, time, dt);
    auto J_contact = contact_.mergedJacobian();
    int iblock = 0;
    int jblock = 1;
    auto dgduT = safelyObtainBlock(J_contact.get(), iblock, jblock);
    std::unique_ptr<mfem::HypreParMatrix> dgdu(dgduT->Transpose());
    return dgdu;
  };

  int numPressureDofs() const { return contact_.numPressureDofs(); }

 protected:
  /**
   * @brief ContactData which has various contact calls
   */
  mutable ContactData contact_;

  /**
   * @brief contact_opts Defines contact method, enforcement, type, and penalty
   */
  ContactOptions contact_opts_;

  /**
   * @brief interaction_id Unique identifier for the ContactInteraction (used in Tribol)
   */
  int interaction_id_;
};

}  // namespace serac

#endif
