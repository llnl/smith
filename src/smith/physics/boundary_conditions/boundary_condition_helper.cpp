// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/boundary_conditions/boundary_condition_helper.hpp"

namespace smith::mfem_ext {

void GetEssentialTrueDofsFromElementAttribute(const mfem::ParFiniteElementSpace& fespace,
                                              const mfem::Array<int>& elem_attr_is_ess, mfem::Array<int>& ess_tdof_list,
                                              int component)
{
  mfem::Array<int> ess_dofs, true_ess_dofs;

  GetEssentialVDofsFromElementAttribute(fespace, elem_attr_is_ess, ess_dofs, component);
  fespace.GetRestrictionMatrix()->BooleanMult(ess_dofs, true_ess_dofs);
  fespace.MarkerToList(true_ess_dofs, ess_tdof_list);
}

static void mark_dofs(const mfem::Array<int>& dofs, mfem::Array<int>& mark_array)
{
  for (int i = 0; i < dofs.Size(); i++) {
    int k = dofs[i];
    if (k < 0) {
      k = -1 - k;
    }
    mark_array[k] = -1;
  }
}

void GetEssentialVDofsFromElementAttribute(const mfem::ParFiniteElementSpace& fespace,
                                           const mfem::Array<int>& elem_attr_is_ess, mfem::Array<int>& ess_vdofs,
                                           int component)
{
  MFEM_ASSERT(
      fespace.GetParMesh()->attributes.Max() == elem_attr_is_ess.Size(),
      "Length of elem_attr_is_ess must match the number of element attributes on the mesh associated with fespace");

  mfem::Array<int> vdofs, dofs;
  ess_vdofs.SetSize(fespace.GetVSize());
  ess_vdofs = 0;
  for (int elem = 0; elem < fespace.GetNE(); elem++) {
    if (elem_attr_is_ess[fespace.GetAttribute(elem) - 1]) {
      if (component < 0)  // mark all components
      {
        fespace.GetElementVDofs(elem, vdofs);
      } else  // mark only desired component
      {
        fespace.GetElementDofs(elem, dofs);
        vdofs.SetSize(dofs.Size());
        for (int d = 0; d < dofs.Size(); d++) {
          vdofs[d] = fespace.DofToVDof(dofs[d], component);
        }
      }
      mark_dofs(vdofs, ess_vdofs);
    }
  }
  fespace.Synchronize(ess_vdofs);
}

}  // namespace smith::mfem_ext
