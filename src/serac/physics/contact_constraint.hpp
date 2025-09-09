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
    return contact_.mergedGaps(false);
  };

  /** @brief Interface for computing contact gap constraint Jacobian from a vector of serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState*
   * @param direction index for which field to take the gradient with respect to
   * @return std::unique_ptr<mfem::HypreParMatrix>
   */
  std::unique_ptr<mfem::HypreParMatrix> jacobian(double time, double dt, const std::vector<ConstFieldPtr>& fields,
                                                 [[maybe_unused]] int direction) const
  {
    // TODO: should direction be optional? should we check that the user has been an acceptable value?
    contact_.setDisplacements(*fields[ContactFields::SHAPE], *fields[ContactFields::DISP]);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_JACOBIAN);

    int cycle = 0;
    contact_.update(cycle, time, dt);
    auto J_contact = contact_.mergedJacobian();
    J_contact->owns_blocks = false;
    delete &J_contact->GetBlock(0, 0);
    delete &J_contact->GetBlock(0, 1);
    delete &J_contact->GetBlock(1, 1);

    auto dgdu = dynamic_cast<mfem::HypreParMatrix*>(&J_contact->GetBlock(1, 0));
    std::unique_ptr<mfem::HypreParMatrix> dgdu_unique(dgdu);
    return dgdu_unique;
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
                                     const mfem::Vector& multipliers, [[maybe_unused]] int direction) const
  {
    contact_.setDisplacements(*fields[ContactFields::SHAPE], *fields[ContactFields::DISP]);
    contact_.setPressures(multipliers);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_GAP);

    int cycle = 0;
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
                                                                       [[maybe_unused]] int direction) const
  {
    contact_.setDisplacements(*fields[ContactFields::SHAPE], *fields[ContactFields::DISP]);
    contact_.setPressures(multipliers);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_JACOBIAN);

    int cycle = 0;
    contact_.update(cycle, time, dt);
    auto J_contact = contact_.mergedJacobian();
    J_contact->owns_blocks = false;
    delete &J_contact->GetBlock(0, 1);
    delete &J_contact->GetBlock(1, 0);
    delete &J_contact->GetBlock(1, 1);

    auto dgdu = dynamic_cast<mfem::HypreParMatrix*>(&J_contact->GetBlock(0, 0));
    std::unique_ptr<mfem::HypreParMatrix> dgdu_unique(dgdu);
    return dgdu_unique;
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
                                                       [[maybe_unused]] int direction) const
  {
    contact_.setDisplacements(*fields[ContactFields::SHAPE], *fields[ContactFields::DISP]);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_JACOBIAN);

    int cycle = 0;
    contact_.update(cycle, time, dt);
    auto J_contact = contact_.mergedJacobian();
    J_contact->owns_blocks = false;
    delete &J_contact->GetBlock(0, 0);
    delete &J_contact->GetBlock(1, 0);
    delete &J_contact->GetBlock(1, 1);

    auto dgdu = dynamic_cast<mfem::HypreParMatrix*>(&J_contact->GetBlock(0, 1));
    std::unique_ptr<mfem::HypreParMatrix> dgdu_unique(dgdu);
    return dgdu_unique;
  };

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
