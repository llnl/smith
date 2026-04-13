// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"

namespace smith {

DirichletBoundaryConditions::DirichletBoundaryConditions(const mfem::ParMesh& mfem_mesh,
                                                         mfem::ParFiniteElementSpace& space)
    : bcs_(mfem_mesh), space_(space)
{
}

DirichletBoundaryConditions::DirichletBoundaryConditions(const Mesh& mesh, mfem::ParFiniteElementSpace& space)
    : DirichletBoundaryConditions(mesh.mfemParMesh(), space)
{
}

void DirichletBoundaryConditions::setZeroBCsMatchingDofs(const BoundaryConditionManager& source)
{
  const auto& true_dofs = source.allEssentialTrueDofs();
  if (true_dofs.Size() == 0) {
    return;
  }
  int vdim = space_.GetVDim();
  mfem::Vector zero_vec(vdim);
  zero_vec = 0.0;
  auto zero_coef = std::make_shared<mfem::VectorConstantCoefficient>(zero_vec);
  bcs_.addEssentialByTrueDofs(true_dofs, zero_coef, space_);
}

}  // namespace smith
