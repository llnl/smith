// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/contact/contact_interaction.hpp"

#ifdef SMITH_USE_TRIBOL

#include <cstddef>
#include <algorithm>
#include <vector>

#include "axom/slic.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

#include "smith/physics/contact/contact_config.hpp"

namespace smith {

static void mark_dofs(const mfem::Array<int>& dofs, mfem::Array<int>& mark_array)
{
  for (int i = 0; i < dofs.Size(); i++) {
    int k = dofs[i];
    if (k < 0) {
      k = -1 - k;
    }
    mark_array[k] = 0;
  }
}

ContactInteraction::ContactInteraction(int interaction_id, const mfem::ParMesh& mesh,
                                       const std::set<int>& bdry_attr_surf1, const std::set<int>& bdry_attr_surf2,
                                       const mfem::ParGridFunction& current_coords, ContactOptions contact_opts)
    : interaction_id_{interaction_id}, contact_opts_{contact_opts}, current_coords_{current_coords}
{
  int mesh1_id = 2 * interaction_id;      // unique id for the first Tribol mesh
  int mesh2_id = 2 * interaction_id + 1;  // unique id for the second Tribol mesh
  tribol::registerMfemCouplingScheme(interaction_id, mesh1_id, mesh2_id, mesh, current_coords, bdry_attr_surf1,
                                     bdry_attr_surf2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, getMethod(),
                                     tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER);
  tribol::setLagrangeMultiplierOptions(interaction_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN);

  // get true DOFs only associated with surface 1 (i.e. surface 1 \ surface 2)
  if (getContactOptions().type == ContactType::TiedNormal) {
    // this block essentially returns the complement of GetEssentialTrueDofsFromElementAttribute(surface 2) (def'd in
    // boundary_condition_helper)
    auto& pressure_space = *tribol::getMfemPressure(interaction_id).ParFESpace();
    mfem::Array<int> dof_markers(pressure_space.GetVSize());
    dof_markers = -1;
    mfem::Array<int> surf2_markers(pressure_space.GetMesh()->attributes.Max());
    for (auto attr : bdry_attr_surf2) {
      surf2_markers[attr - 1] = 1;
    }
    for (int e{0}; e < pressure_space.GetNE(); ++e) {
      if (surf2_markers[pressure_space.GetAttribute(e) - 1]) {
        mfem::Array<int> vdofs;
        pressure_space.GetElementVDofs(e, vdofs);
        mark_dofs(vdofs, dof_markers);
      }
    }
    mfem::Array<int> tdof_markers(pressure_space.GetTrueVSize());
    pressure_space.GetRestrictionMatrix()->BooleanMult(dof_markers, tdof_markers);
    mfem::FiniteElementSpace::MarkerToList(tdof_markers, inactive_tdofs_);
  }

  // set up Tribol to compute exact Jacobian if requested
  if (getContactOptions().jacobian == ContactJacobian::Exact) {
#ifdef SMITH_USE_ENZYME
    tribol::enableEnzyme(interaction_id, true);
#endif
    tribol::registerMfemReferenceCoords(interaction_id, static_cast<const mfem::ParGridFunction&>(*mesh.GetNodes()));
  }
}

FiniteElementDual ContactInteraction::forces() const
{
  FiniteElementDual f(*current_coords_.ParFESpace());
  auto& f_loc = f.linearForm();
  tribol::getMfemResponse(getInteractionId(), f_loc);
  f.setFromLinearForm(f_loc);
  return f;
}

FiniteElementState ContactInteraction::pressure() const
{
  auto& p_tribol = tribol::getMfemPressure(getInteractionId());
  FiniteElementState p(*p_tribol.ParFESpace());
  p.setFromGridFunction(p_tribol);
  return p;
}

FiniteElementDual ContactInteraction::gaps() const
{
  FiniteElementDual g(pressureSpace());
  auto& g_loc = g.linearForm();
  tribol::getMfemGap(getInteractionId(), g_loc);
  g.setFromLinearForm(g_loc);
  return g;
}

std::unique_ptr<mfem::BlockOperator> ContactInteraction::jacobian() const
{
  return tribol::getMfemBlockJacobian(getInteractionId());
}

int ContactInteraction::numPressureDofs() const
{
  return getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier
             ? tribol::getMfemPressure(getInteractionId()).ParFESpace()->GetTrueVSize()
             : 0;
}

mfem::ParFiniteElementSpace& ContactInteraction::pressureSpace() const
{
  return *tribol::getMfemPressure(getInteractionId()).ParFESpace();
}

void ContactInteraction::setPressure(const FiniteElementState& pressure) const
{
  tribol::getMfemPressure(getInteractionId()) = pressure.gridFunction();
}

const mfem::Array<int>& ContactInteraction::inactiveDofs() const { return inactiveDofs(pressure()); }

const mfem::Array<int>& ContactInteraction::inactiveDofs(const FiniteElementState& pressure) const
{
  if (getContactOptions().type == ContactType::Frictionless) {
    auto g = gaps();
    std::vector<int> inactive_tdofs_vector;
    inactive_tdofs_vector.reserve(static_cast<size_t>(pressure.Size()));
    for (int d{0}; d < pressure.Size(); ++d) {
      // don't check pressure for penalty; it's just the gap * a constant
      if (getContactOptions().enforcement == ContactEnforcement::Penalty && g[d] >= -1.0e-14) {
        inactive_tdofs_vector.push_back(d);
      } else if (getContactOptions().enforcement == ContactEnforcement::LagrangeMultiplier && pressure[d] >= 0.0 &&
                 g[d] >= -1.0e-14) {
        inactive_tdofs_vector.push_back(d);
      }
    }
    inactive_tdofs_ = mfem::Array<int>(static_cast<int>(inactive_tdofs_vector.size()));
    std::copy(inactive_tdofs_vector.begin(), inactive_tdofs_vector.end(), inactive_tdofs_.begin());
  }
  return inactive_tdofs_;
}

void ContactInteraction::evalJacobian(bool eval) const
{
  if (eval) {
    tribol::setLagrangeMultiplierOptions(getInteractionId(), tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN);
  } else {
    tribol::setLagrangeMultiplierOptions(getInteractionId(), tribol::ImplicitEvalMode::MORTAR_GAP);
  }
}

tribol::ContactMethod ContactInteraction::getMethod() const
{
  switch (contact_opts_.method) {
    case ContactMethod::SingleMortar:
      return tribol::SINGLE_MORTAR;
      break;
    default:
      SLIC_ERROR_ROOT("Unsupported contact method.");
      // return something so we don't get an error
      return tribol::SINGLE_MORTAR;
  }
}

}  // namespace smith

#endif
