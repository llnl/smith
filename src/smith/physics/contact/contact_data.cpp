// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/contact/contact_data.hpp"

#include <cstddef>

#include "axom/slic.hpp"
#include "mpi.h"

#include "smith/physics/state/finite_element_state.hpp"

#ifdef SMITH_USE_TRIBOL
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#endif

namespace smith {

#ifdef SMITH_USE_TRIBOL

ContactData::ContactData(const mfem::ParMesh& mesh)
    : mesh_{mesh},
      reference_nodes_{static_cast<const mfem::ParGridFunction*>(mesh.GetNodes())},
      current_coords_{*reference_nodes_},
      have_lagrange_multipliers_{false},
      num_pressure_dofs_{0},
      offsets_up_to_date_{false}
{
}

ContactData::~ContactData() { tribol::finalize(); }

void ContactData::addContactInteraction(int interaction_id, const std::set<int>& bdry_attr_surf1,
                                        const std::set<int>& bdry_attr_surf2, ContactOptions contact_opts)
{
  interactions_.emplace_back(interaction_id, mesh_, bdry_attr_surf1, bdry_attr_surf2, current_coords_, contact_opts);
  if (contact_opts.enforcement == ContactEnforcement::LagrangeMultiplier) {
    have_lagrange_multipliers_ = true;
    num_pressure_dofs_ += interactions_.back().numPressureDofs();
    offsets_up_to_date_ = false;
  }
}

void ContactData::reset()
{
  for (auto& interaction : interactions_) {
    FiniteElementState zero = interaction.pressure();
    zero = 0.0;
    interaction.setPressure(zero);
  }
}

void ContactData::update(int cycle, double time, double& dt)
{
  cycle_ = cycle;
  time_ = time;
  dt_ = dt;
  // This updates the redecomposed surface mesh based on the current displacement, then transfers field quantities to
  // the updated mesh.
  tribol::updateMfemParallelDecomposition();
  // This function computes forces, gaps, and Jacobian contributions based on the current field quantities. Note the
  // fields (with the exception of pressure) are stored on the redecomposed surface mesh until transferred by calling
  // forces(), mergedGaps(), etc.
  tribol::update(cycle, time, dt);
}

FiniteElementDual ContactData::forces() const
{
  FiniteElementDual f(*reference_nodes_->ParFESpace(), "contact force");
  for (const auto& interaction : interactions_) {
    f += interaction.forces();
  }
  return f;
}

mfem::HypreParVector ContactData::mergedPressures() const
{
  updateDofOffsets();
  mfem::HypreParVector merged_p(mesh_.GetComm(), global_pressure_dof_offsets_[global_pressure_dof_offsets_.Size() - 1],
                                global_pressure_dof_offsets_.GetData());
  for (size_t i{0}; i < interactions_.size(); ++i) {
    if (interactions_[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier) {
      mfem::Vector p_interaction;
      p_interaction.MakeRef(
          merged_p, pressure_dof_offsets_[static_cast<int>(i)],
          pressure_dof_offsets_[static_cast<int>(i) + 1] - pressure_dof_offsets_[static_cast<int>(i)]);
      p_interaction.Set(1.0, interactions_[i].pressure());
    }
  }
  return merged_p;
}

mfem::HypreParVector ContactData::mergedGaps(bool zero_inactive) const
{
  updateDofOffsets();
  mfem::HypreParVector merged_g(mesh_.GetComm(), global_pressure_dof_offsets_[global_pressure_dof_offsets_.Size() - 1],
                                global_pressure_dof_offsets_.GetData());
  for (size_t i{0}; i < interactions_.size(); ++i) {
    if (interactions_[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier) {
      auto g = interactions_[i].gaps();
      if (zero_inactive) {
        for (auto dof : interactions_[i].inactiveDofs()) {
          g[dof] = 0.0;
        }
      }
      mfem::Vector g_interaction(
          merged_g, pressure_dof_offsets_[static_cast<int>(i)],
          pressure_dof_offsets_[static_cast<int>(i) + 1] - pressure_dof_offsets_[static_cast<int>(i)]);
      g_interaction.Set(1.0, g);
    }
  }
  return merged_g;
}

std::unique_ptr<mfem::BlockOperator> ContactData::mergedJacobian() const
{
  updateDofOffsets();
  // this is the BlockOperator we are returning with the following blocks:
  //  | df_(contact)/dx  df_(contact)/dp |
  //  | dg/dx            I_(inactive)    |
  // where I_(inactive) is a matrix with ones on the diagonal of inactive pressure true degrees of freedom
  auto block_J = std::make_unique<mfem::BlockOperator>(jacobian_offsets_);
  block_J->owns_blocks = true;
  // rather than returning different blocks for each contact interaction with Lagrange multipliers, merge them all into
  // a single block
  mfem::Array2D<const mfem::HypreParMatrix*> dgdu_blocks(static_cast<int>(interactions_.size()), 1);
  mfem::Array2D<const mfem::HypreParMatrix*> dfdp_blocks(1, static_cast<int>(interactions_.size()));
  for (size_t i{0}; i < interactions_.size(); ++i) {
    dgdu_blocks(static_cast<int>(i), 0) = nullptr;
    dfdp_blocks(0, static_cast<int>(i)) = nullptr;
  }

  for (size_t i{0}; i < interactions_.size(); ++i) {
    // this is the BlockOperator for one of the contact interactions
    auto interaction_J = interactions_[i].jacobian();
    interaction_J->owns_blocks = false;  // we'll manage the ownership of the blocks on our own...
    // add the contact interaction's contribution to df_(contact)/dx (the 0, 0 block)
    if (!interaction_J->IsZeroBlock(0, 0)) {
      SLIC_ERROR_ROOT_IF(!dynamic_cast<mfem::HypreParMatrix*>(&interaction_J->GetBlock(0, 0)),
                         "Only HypreParMatrix constraint matrix blocks are currently supported.");
      if (block_J->IsZeroBlock(0, 0)) {
        block_J->SetBlock(0, 0, &interaction_J->GetBlock(0, 0));
      } else {
        block_J->SetBlock(0, 0,
                          mfem::Add(1.0, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(0, 0)), 1.0,
                                    static_cast<mfem::HypreParMatrix&>(interaction_J->GetBlock(0, 0))));
        delete &interaction_J->GetBlock(0, 0);
      }
    }
    // add the contact interaction's (other) contribution to df_(contact)/dx (for penalty) or to df_(contact)/dp and
    // dg/dx (for Lagrange multipliers)
    if (!interaction_J->IsZeroBlock(1, 0) && !interaction_J->IsZeroBlock(0, 1)) {
      auto dgdu = dynamic_cast<mfem::HypreParMatrix*>(&interaction_J->GetBlock(1, 0));
      auto dfdp = dynamic_cast<mfem::HypreParMatrix*>(&interaction_J->GetBlock(0, 1));
      SLIC_ERROR_ROOT_IF(!dgdu, "Only HypreParMatrix constraint matrix blocks are currently supported.");
      SLIC_ERROR_ROOT_IF(!dfdp, "Only HypreParMatrix constraint matrix blocks are currently supported.");
      // zero out rows and cols not in the active set
      auto inactive_dofs = interactions_[i].inactiveDofs();
      dgdu->EliminateRows(inactive_dofs);
      dfdp->EliminateCols(inactive_dofs);
      if (interactions_[i].getContactOptions().enforcement == ContactEnforcement::Penalty) {
        // compute contribution to df_(contact)/dx (the 0, 0 block) for penalty
        std::unique_ptr<mfem::HypreParMatrix> BTB(mfem::ParMult(dfdp, dgdu, true));
        delete &interaction_J->GetBlock(1, 0);
        delete &interaction_J->GetBlock(0, 1);
        if (block_J->IsZeroBlock(0, 0)) {
          mfem::Vector penalty(reference_nodes_->ParFESpace()->GetTrueVSize());
          penalty = interactions_[i].getContactOptions().penalty;
          BTB->ScaleRows(penalty);
          block_J->SetBlock(0, 0, BTB.release());
        } else {
          block_J->SetBlock(0, 0,
                            mfem::Add(1.0, static_cast<mfem::HypreParMatrix&>(block_J->GetBlock(0, 0)),
                                      interactions_[i].getContactOptions().penalty, *BTB));
        }
      } else  // enforcement == ContactEnforcement::LagrangeMultiplier
      {
        // compute contribution to off-diagonal blocks for Lagrange multiplier
        dgdu_blocks(static_cast<int>(i), 0) = dgdu;
        dfdp_blocks(0, static_cast<int>(i)) = dfdp;
      }
      if (!interaction_J->IsZeroBlock(1, 1)) {
        // we track our own active set, so get rid of the tribol inactive dof block
        delete &interaction_J->GetBlock(1, 1);
      }
    }
  }
  if (haveLagrangeMultipliers()) {
    // merge all of the contributions from all of the contact interactions
    block_J->SetBlock(1, 0, mfem::HypreParMatrixFromBlocks(dgdu_blocks));
    // store the transpose explicitly (rather than as a TransposeOperator) for solvers that need HypreParMatrixs
    block_J->SetBlock(0, 1, mfem::HypreParMatrixFromBlocks(dfdp_blocks));
    // explicitly delete the blocks
    for (size_t i{0}; i < interactions_.size(); ++i) {
      delete dgdu_blocks(static_cast<int>(i), 0);
      delete dfdp_blocks(0, static_cast<int>(i));
    }
    // build I_(inactive): a diagonal matrix with ones on inactive dofs and zeros elsewhere
    mfem::Array<const mfem::Array<int>*> inactive_tdofs_vector(static_cast<int>(interactions_.size()));
    int inactive_tdofs_ct = 0;
    for (int i{0}; i < inactive_tdofs_vector.Size(); ++i) {
      inactive_tdofs_vector[i] = &interactions_[static_cast<size_t>(i)].inactiveDofs();
      inactive_tdofs_ct += inactive_tdofs_vector[i]->Size();
    }
    mfem::Array<int> inactive_tdofs(inactive_tdofs_ct);
    inactive_tdofs_ct = 0;
    for (int i{0}; i < inactive_tdofs_vector.Size(); ++i) {
      if (inactive_tdofs_vector[i]) {
        for (int d{0}; d < inactive_tdofs_vector[i]->Size(); ++d) {
          inactive_tdofs[d + inactive_tdofs_ct] = (*inactive_tdofs_vector[i])[d] + pressure_dof_offsets_[i];
        }
        inactive_tdofs_ct += inactive_tdofs_vector[i]->Size();
      }
    }
    inactive_tdofs.GetMemory().SetHostPtrOwner(false);
    mfem::Array<int> rows(numPressureDofs() + 1);
    rows = 0;
    inactive_tdofs_ct = 0;
    for (int i{0}; i < numPressureDofs(); ++i) {
      if (inactive_tdofs_ct < inactive_tdofs.Size() && inactive_tdofs[inactive_tdofs_ct] == i) {
        ++inactive_tdofs_ct;
      }
      rows[i + 1] = inactive_tdofs_ct;
    }
    rows.GetMemory().SetHostPtrOwner(false);
    mfem::Vector ones(inactive_tdofs_ct);
    ones = 1.0;
    ones.GetMemory().SetHostPtrOwner(false);
    mfem::SparseMatrix inactive_diag(rows.GetData(), inactive_tdofs.GetData(), ones.GetData(), numPressureDofs(),
                                     numPressureDofs(), false, false, true);
    // if the size of ones is zero, SparseMatrix creates its own memory which it
    // owns.  explicitly prevent this...
    inactive_diag.SetDataOwner(false);
    auto block_1_1 =
        new mfem::HypreParMatrix(mesh_.GetComm(), global_pressure_dof_offsets_[global_pressure_dof_offsets_.Size() - 1],
                                 global_pressure_dof_offsets_, &inactive_diag);
    block_1_1->SetOwnerFlags(3, 3, 1);
    block_J->SetBlock(1, 1, block_1_1);
    // end building I_(inactive)
  }
  return block_J;
}

void ContactData::residualFunction(const mfem::Vector& u_shape, const mfem::Vector& u, mfem::Vector& r)
{
  const int disp_size = reference_nodes_->ParFESpace()->GetTrueVSize();

  // u_const should not change in this method; const cast is to create vector views which are copied to Tribol
  // displacements and pressures and used to compute the (non-contact) residual
  auto& u_const = const_cast<mfem::Vector&>(u);
  const mfem::Vector u_blk(u_const, 0, disp_size);
  const mfem::Vector p_blk(u_const, disp_size, numPressureDofs());

  mfem::Vector r_blk(r, 0, disp_size);
  mfem::Vector g_blk(r, disp_size, numPressureDofs());

  setDisplacements(u_shape, u_blk);

  // we need to call update first to update gaps
  for (auto& interaction : interactions_) {
    interaction.evalJacobian(false);
  }
  update(cycle_, time_, dt_);
  // with updated gaps, we can update pressure for contact interactions with penalty enforcement
  setPressures(p_blk);
  // call update again with the right pressures
  for (auto& interaction : interactions_) {
    interaction.evalJacobian(true);
  }
  update(cycle_, time_, dt_);

  r_blk += forces();
  // calling mergedGaps() with true will zero out gap on inactive dofs (so the residual converges and the linearized
  // system makes sense)
  g_blk.Set(1.0, mergedGaps(true));
}

std::unique_ptr<mfem::BlockOperator> ContactData::jacobianFunction(mfem::HypreParMatrix* orig_J) const
{
  auto J_contact = mergedJacobian();
  if (J_contact->IsZeroBlock(0, 0)) {
    J_contact->SetBlock(0, 0, orig_J);
  } else {
    J_contact->SetBlock(0, 0,
                        mfem::Add(1.0, *orig_J, 1.0, static_cast<mfem::HypreParMatrix&>(J_contact->GetBlock(0, 0))));
  }

  return J_contact;
}

void ContactData::setPressures(const mfem::Vector& merged_pressures) const
{
  updateDofOffsets();
  for (size_t i{0}; i < interactions_.size(); ++i) {
    FiniteElementState p_interaction(interactions_[i].pressureSpace());
    if (interactions_[i].getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier) {
      // merged_pressures_const should not change; const cast is to create a vector view for copying to tribol pressures
      auto& merged_pressures_const = const_cast<mfem::Vector&>(merged_pressures);
      const mfem::Vector p_interaction_ref(
          merged_pressures_const, pressure_dof_offsets_[static_cast<int>(i)],
          pressure_dof_offsets_[static_cast<int>(i) + 1] - pressure_dof_offsets_[static_cast<int>(i)]);
      p_interaction.Set(1.0, p_interaction_ref);
    } else  // enforcement == ContactEnforcement::Penalty
    {
      p_interaction.Set(interactions_[i].getContactOptions().penalty, interactions_[i].gaps());
    }
    for (auto dof : interactions_[i].inactiveDofs()) {
      p_interaction[dof] = 0.0;
    }
    interactions_[i].setPressure(p_interaction);
  }
}

void ContactData::setDisplacements(const mfem::Vector& shape_u, const mfem::Vector& u)
{
  mfem::ParGridFunction prolonged_shape_disp{current_coords_};
  reference_nodes_->ParFESpace()->GetProlongationMatrix()->Mult(u, current_coords_);
  reference_nodes_->ParFESpace()->GetProlongationMatrix()->Mult(shape_u, prolonged_shape_disp);

  current_coords_ += *reference_nodes_;
  current_coords_ += prolonged_shape_disp;
}

void ContactData::updateDofOffsets() const
{
  if (offsets_up_to_date_) {
    return;
  }
  jacobian_offsets_ = mfem::Array<int>({0, reference_nodes_->ParFESpace()->GetTrueVSize(),
                                        numPressureDofs() + reference_nodes_->ParFESpace()->GetTrueVSize()});
  pressure_dof_offsets_.SetSize(static_cast<int>(interactions_.size()) + 1);
  pressure_dof_offsets_ = 0;
  for (size_t i{0}; i < interactions_.size(); ++i) {
    pressure_dof_offsets_[static_cast<int>(i + 1)] =
        pressure_dof_offsets_[static_cast<int>(i)] + interactions_[i].numPressureDofs();
  }
  global_pressure_dof_offsets_.SetSize(mesh_.GetNRanks() + 1);
  global_pressure_dof_offsets_ = 0;
  global_pressure_dof_offsets_[mesh_.GetMyRank() + 1] = numPressureDofs();
  MPI_Allreduce(MPI_IN_PLACE, global_pressure_dof_offsets_.GetData(), global_pressure_dof_offsets_.Size(), MPI_INT,
                MPI_SUM, mesh_.GetComm());
  for (int i{1}; i < mesh_.GetNRanks(); ++i) {
    global_pressure_dof_offsets_[i + 1] += global_pressure_dof_offsets_[i];
  }
  if (HYPRE_AssumedPartitionCheck()) {
    auto total_dofs = global_pressure_dof_offsets_[global_pressure_dof_offsets_.Size() - 1];
    // If the number of ranks is less than 2, ensure the size of global_pressure_dof_offsets_ is large enough
    if (mesh_.GetNRanks() < 2) {
      global_pressure_dof_offsets_.SetSize(3);
    }
    global_pressure_dof_offsets_[0] = global_pressure_dof_offsets_[mesh_.GetMyRank()];
    global_pressure_dof_offsets_[1] = global_pressure_dof_offsets_[mesh_.GetMyRank() + 1];
    global_pressure_dof_offsets_[2] = total_dofs;
    // If the number of ranks is greater than 2, shrink the size of global_pressure_dof_offsets_
    if (mesh_.GetNRanks() > 2) {
      global_pressure_dof_offsets_.SetSize(3);
    }
  }
  offsets_up_to_date_ = true;
}

#else

ContactData::ContactData([[maybe_unused]] const mfem::ParMesh& mesh)
    : have_lagrange_multipliers_{false}, num_pressure_dofs_{0}
{
}

ContactData::~ContactData() {}

void ContactData::addContactInteraction([[maybe_unused]] int interaction_id,
                                        [[maybe_unused]] const std::set<int>& bdry_attr_surf1,
                                        [[maybe_unused]] const std::set<int>& bdry_attr_surf2,
                                        [[maybe_unused]] ContactOptions contact_opts)
{
  SLIC_WARNING_ROOT("Smith built without Tribol support. No contact interaction will be added.");
}

void ContactData::update([[maybe_unused]] int cycle, [[maybe_unused]] double time, [[maybe_unused]] double& dt) {}

FiniteElementDual ContactData::forces() const
{
  FiniteElementDual f(*reference_nodes_->ParFESpace(), "contact force");
  f = 0.0;
  return f;
}

mfem::HypreParVector ContactData::mergedPressures() const { return mfem::HypreParVector(); }

mfem::HypreParVector ContactData::mergedGaps([[maybe_unused]] bool zero_inactive) const
{
  return mfem::HypreParVector();
}

std::unique_ptr<mfem::BlockOperator> ContactData::mergedJacobian() const
{
  jacobian_offsets_ = mfem::Array<int>(
      {0, reference_nodes_->ParFESpace()->GetTrueVSize(), reference_nodes_->ParFESpace()->GetTrueVSize()});
  return std::make_unique<mfem::BlockOperator>(jacobian_offsets_);
}

void ContactData::residualFunction([[maybe_unused]] const mfem::Vector& u_shape, [[maybe_unused]] const mfem::Vector& u,
                                   [[maybe_unused]] mfem::Vector& r)
{
}

std::unique_ptr<mfem::BlockOperator> ContactData::jacobianFunction(mfem::HypreParMatrix* orig_J) const
{
  auto J_contact = mergedJacobian();
  if (J_contact->IsZeroBlock(0, 0)) {
    J_contact->SetBlock(0, 0, orig_J);
  } else {
    J_contact->SetBlock(0, 0,
                        mfem::Add(1.0, *orig_J, 1.0, static_cast<mfem::HypreParMatrix&>(J_contact->GetBlock(0, 0))));
  }

  return J_contact;
}

void ContactData::setPressures([[maybe_unused]] const mfem::Vector& true_pressures) const {}

void ContactData::setDisplacements([[maybe_unused]] const mfem::Vector& u_shape,
                                   [[maybe_unused]] const mfem::Vector& true_displacement)
{
}

#endif

}  // namespace smith
