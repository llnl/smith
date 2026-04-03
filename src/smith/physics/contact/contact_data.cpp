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
  // specify all contact boundaries
  mfem::Array<int> contact_bdry_attribs;
  contact_bdry_attribs.SetSize(mesh_.bdr_attributes.Max());
  contact_bdry_attribs = 0;
  // attributes start at 1,
  // shift by -1 to account for zero-based array indexing
  for (const auto& bdry_attr : bdry_attr_surf1) {
    contact_bdry_attribs[bdry_attr - 1] = 1;
  }
  for (const auto& bdry_attr : bdry_attr_surf2) {
    contact_bdry_attribs[bdry_attr - 1] = 1;
  }
  // dofs for the current contact interaction
  mfem::Array<int> contact_interaction_dofs_;
  reference_nodes_->ParFESpace()->GetEssentialTrueDofs(contact_bdry_attribs, contact_interaction_dofs_);
  // add dofs for current contact interaction call to all contact_dofs_
  contact_dofs_.Append(contact_interaction_dofs_.GetData(), contact_interaction_dofs_.Size());
  // sort and delete duplicates
  contact_dofs_.Sort();
  contact_dofs_.Unique();
}

void ContactData::reset()
{
  for (auto& interaction : interactions_) {
    FiniteElementState zero = interaction.pressure();
    zero = 0.0;
    interaction.setPressure(zero);
  }
}

void ContactData::updateGaps(int cycle, double time, double& dt,
                             std::optional<std::reference_wrapper<const mfem::Vector>> u_shape,
                             std::optional<std::reference_wrapper<const mfem::Vector>> u, bool eval_jacobian)
{
  cycle_ = cycle;
  time_ = time;
  dt_ = dt;

  if (u_shape && u) {
    setDisplacements(u_shape->get(), u->get());
  }

  for (auto& interaction : interactions_) {
    interaction.evalJacobian(eval_jacobian);
  }
  // This updates the redecomposed surface mesh based on the current displacement, then transfers field quantities to
  // the updated mesh.
  if (u_shape && u) {
    tribol::updateMfemParallelDecomposition();
  }
  // This function computes gaps (and optionally geometric Jacobian blocks) based on the current mesh.
  tribol::update(cycle, time, dt);
}

void ContactData::update(int cycle, double time, double& dt,
                         std::optional<std::reference_wrapper<const mfem::Vector>> u_shape,
                         std::optional<std::reference_wrapper<const mfem::Vector>> u,
                         std::optional<std::reference_wrapper<const mfem::Vector>> p)
{
  // First pass: update gaps if coordinates are provided
  if (u_shape && u) {
    updateGaps(cycle, time, dt, u_shape, u, false);
  } else {
    // Ensure internal timing is updated even if coordinates are not
    cycle_ = cycle;
    time_ = time;
    dt_ = dt;
  }

  // second pass: update pressures and compute forces/Jacobians if p is provided
  if (p) {
    // with updated gaps, we can update pressure for contact interactions (active set detection and penalty)
    setPressures(p->get());

    for (auto& interaction : interactions_) {
      interaction.evalJacobian(true);
    }
    // This second call is required to synchronize the updated pressures to Tribol's internal redecomposed surface mesh
    // and to ensure Tribol's internal state is correctly reset for the second pass.
    tribol::updateMfemParallelDecomposition();
    tribol::update(cycle, time, dt);
  }
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
    // this is the BlockOperator for one of the contact interactions, post-processed for Smith's assembly conventions
    auto interaction_J = interactions_[i].jacobianContribution();
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

    // add the contact interaction's contribution to df_(contact)/dp and dg/dx (for Lagrange multipliers)
    if (!interaction_J->IsZeroBlock(1, 0) && !interaction_J->IsZeroBlock(0, 1)) {
      auto dgdu = dynamic_cast<mfem::HypreParMatrix*>(&interaction_J->GetBlock(1, 0));
      auto dfdp = dynamic_cast<mfem::HypreParMatrix*>(&interaction_J->GetBlock(0, 1));
      SLIC_ERROR_ROOT_IF(!dgdu, "Only HypreParMatrix constraint matrix blocks are currently supported.");
      SLIC_ERROR_ROOT_IF(!dfdp, "Only HypreParMatrix constraint matrix blocks are currently supported.");

      dgdu_blocks(static_cast<int>(i), 0) = dgdu;
      dfdp_blocks(0, static_cast<int>(i)) = dfdp;
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
    mfem::Array<int> rows(numPressureDofs() + 1);
    rows = 0;
    inactive_tdofs_ct = 0;
    for (int i{0}; i < numPressureDofs(); ++i) {
      if (inactive_tdofs_ct < inactive_tdofs.Size() && inactive_tdofs[inactive_tdofs_ct] == i) {
        ++inactive_tdofs_ct;
      }
      rows[i + 1] = inactive_tdofs_ct;
    }
    mfem::Vector ones(inactive_tdofs_ct);
    ones = 1.0;
    mfem::SparseMatrix inactive_diag(rows.GetData(), inactive_tdofs.GetData(), ones.GetData(), numPressureDofs(),
                                     numPressureDofs(), false, false, true);
    rows.GetMemory().ClearOwnerFlags();
    inactive_tdofs.GetMemory().ClearOwnerFlags();
    ones.GetMemory().ClearOwnerFlags();
    inactive_diag.GetMemoryI().ClearOwnerFlags();
    inactive_diag.GetMemoryJ().ClearOwnerFlags();
    inactive_diag.GetMemoryData().ClearOwnerFlags();
    auto block_1_1 =
        new mfem::HypreParMatrix(mesh_.GetComm(), global_pressure_dof_offsets_[global_pressure_dof_offsets_.Size() - 1],
                                 global_pressure_dof_offsets_, &inactive_diag);
    constexpr int mfem_owned_host_flag = 3;
    block_1_1->SetOwnerFlags(mfem_owned_host_flag, block_1_1->OwnsOffd(), block_1_1->OwnsColMap());
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

  update(cycle_, time_, dt_, u_shape, u_blk, p_blk);

  r_blk += forces();
  // calling mergedGaps() with true will zero out gap on inactive dofs (so the residual converges and the linearized
  // system makes sense)
  g_blk.Set(1.0, mergedGaps(true));
}

std::unique_ptr<mfem::BlockOperator> ContactData::jacobianFunction(std::unique_ptr<mfem::HypreParMatrix> orig_J) const
{
  auto J_contact = mergedJacobian();
  if (J_contact->IsZeroBlock(0, 0)) {
    J_contact->SetBlock(0, 0, orig_J.release());
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

std::unique_ptr<mfem::HypreParMatrix> ContactData::contactSubspaceTransferOperator()
{
  const MPI_Comm comm = reference_nodes_->ParFESpace()->GetComm();
  HYPRE_BigInt* col_offsets = reference_nodes_->ParFESpace()->GetTrueDofOffsets();
  HYPRE_BigInt ncols_glb = reference_nodes_->ParFESpace()->GlobalTrueVSize();

  // number of rows of the restriction
  // operator owned by the local MPI process
  int nrows_loc = contact_dofs_.Size();
  // should nrows_glb be of type HYPRE_BigInt?
  // global number of rows of restriction
  // operator
  int nrows_glb = 0;
  MPI_Allreduce(&nrows_loc, &nrows_glb, 1, MPI_INT, MPI_SUM, comm);
  // determine rows offsets of the restriction operator
  int row_offset = 0;
  MPI_Scan(&nrows_loc, &row_offset, 1, MPI_INT, MPI_SUM, comm);
  row_offset -= nrows_loc;
  HYPRE_BigInt row_offsets[2];
  row_offsets[0] = row_offset;
  row_offsets[1] = row_offset + nrows_loc;

  // create mfem::SparseMatrix restriction matrix
  // restriction from displacement dofs to
  // contact dofs
  // one nonzero (unit) entry per row
  mfem::SparseMatrix Rsparse(nrows_loc, ncols_glb);
  mfem::Array<int> col(1);
  col = 0;
  mfem::Vector entry(1);
  entry = 1.0;
  HYPRE_BigInt col_offset = col_offsets[0];
  for (int k = 0; k < nrows_loc; k++) {
    // local per process contact dof
    // to global column number
    col[0] = col_offset + contact_dofs_[k];
    Rsparse.SetRow(k, col, entry);
  }
  Rsparse.Finalize();

  // convert local sparse restriction matrix
  // to distributed mfem::HypreParMatrix
  int* I = Rsparse.GetI();
  HYPRE_BigInt* J = Rsparse.GetJ();
  double* data = Rsparse.GetData();

  std::unique_ptr<mfem::HypreParMatrix> restriction_operator = std::make_unique<mfem::HypreParMatrix>(
      comm, nrows_loc, nrows_glb, ncols_glb, I, J, data, row_offsets, col_offsets);
  // convert restriction operator
  // to a contact dof to displacement
  // dof transfer operator
  std::unique_ptr<mfem::HypreParMatrix> transfer_operator;
  transfer_operator.reset(restriction_operator->Transpose());
  return transfer_operator;
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

void ContactData::updateGaps([[maybe_unused]] int cycle, [[maybe_unused]] double time, [[maybe_unused]] double& dt,
                             [[maybe_unused]] std::optional<std::reference_wrapper<const mfem::Vector>> u_shape,
                             [[maybe_unused]] std::optional<std::reference_wrapper<const mfem::Vector>> u,
                             [[maybe_unused]] bool eval_jacobian)
{
}

void ContactData::update([[maybe_unused]] int cycle, [[maybe_unused]] double time, [[maybe_unused]] double& dt,
                         [[maybe_unused]] std::optional<std::reference_wrapper<const mfem::Vector>> u_shape,
                         [[maybe_unused]] std::optional<std::reference_wrapper<const mfem::Vector>> u,
                         [[maybe_unused]] std::optional<std::reference_wrapper<const mfem::Vector>> p)
{
}

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

std::unique_ptr<mfem::BlockOperator> ContactData::jacobianFunction(std::unique_ptr<mfem::HypreParMatrix> orig_J) const
{
  auto J_contact = mergedJacobian();
  if (J_contact->IsZeroBlock(0, 0)) {
    J_contact->SetBlock(0, 0, orig_J.release());
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

std::unique_ptr<mfem::HypreParMatrix> ContactData::contactSubspaceTransferOperator()
{
  std::unique_ptr<mfem::HypreParMatrix> transfer_operator = nullptr;
  return transfer_operator;
}

#endif

}  // namespace smith
