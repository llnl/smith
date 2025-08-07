// Copyright Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file constraints.hpp
 *
 * @brief Specifies interface for evaluating distributed constriants from fields as well as
 * their Jacobians and Hessian-vector products
 */

#pragma once

#include <vector>
#include "serac/physics/common.hpp"
#include "serac/physics/field_types.hpp"
#include "tribol/interface/tribol.hpp"

namespace mfem {
class Vector;
class HypreParMatrix;
}  // namespace mfem

namespace serac {

class FiniteElementState;

/// @brief Abstract constraint class
class ContactConstraint : public Constraint {
 public:
  /// @brief base constructor takes the name of the physics
  ContactConstraint(const mfem::ParMesh& mesh, const std::string& name, int interaction_id, 
		  const std::set<int>& bdry_attr_surf1, const std::set<int>& bdry_attr_surf2, ContactOptions contact_opts) : Constraint(name), contact_(mesh), mesh_{mesh}
  {
     contact_opts.enforcement = ContactEnforcement::LagrangeMultiplier;
     contact_.addContactInteraction(interaction_id, bdry_attr_surf1, bdry_attr_surf2, contact_opts);
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
  mfem::Vector evaluate([[maybe_unused]] double time, [[maybe_unused]] double dt,
                        [[maybe_unused]] const std::vector<ConstFieldPtr>& fields)
  {
    // todo: use enum?
    //       will all contact simulations involve the same fields?
    int SHAPE = 0;
    int DISP = 1;
    contact_.setDisplacements(*fields[SHAPE], *fields[DISP]);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_GAP);
    
    // TODO: how to specify the right cycle?
    int cycle = 0;
    contact_.update(cycle, time, dt);
    return contact_.mergedGaps(true);
  };

  /** @brief Interface for computing contact gap constraint Jacobian from a vector of serac::FiniteElementState*
   *
   * @param time time
   * @param dt time step
   * @param fields vector of serac::FiniteElementState*
   * @param direction index for which field to take the gradient with respect to
   * @return std::unique_ptr<mfem::HypreParMatrix>
   */
  std::unique_ptr<mfem::HypreParMatrix> jacobian([[maybe_unused]] double time, [[maybe_unused]] double dt,
                                                 [[maybe_unused]] const std::vector<ConstFieldPtr>& fields,
                                                 [[maybe_unused]] int direction) const
  {
    // todo: use enum?
    //       will all contact simulations involve the same fields?
    int SHAPE = 0;
    int DISP = 1;
    contact_.setDisplacements(*fields[SHAPE], *fields[DISP]);
    tribol::setLagrangeMultiplierOptions(interaction_id_, tribol::ImplicitEvalMode::MORTAR_JACOBIAN);
    
    // TODO: how to specify the right cycle?
    int cycle = 0;
    contact_.update(cycle, time, dt);
    auto J_contact = contact_.mergedJacobian();
    J_contact->owns_blocks = false;
    delete &J_contact->GetBlock(0, 0);
    delete &J_contact->GetBlock(1, 0);
    delete &J_contact->GetBlock(1, 1);
    
    auto dgdu = dynamic_cast<mfem::HypreParMatrix *>(&J_contact->GetBlock(0, 1));
    std::unique_ptr<mfem::HypreParMatrix> dgdu_unique(dgdu);
    return dgdu_unique;
  };

 protected:
  /// @brief Class holding contact constraint data
  mutable ContactData contact_;
  int interaction_id_;

 private:
  const mfem::ParMesh& mesh_;
};

}  // namespace serac
