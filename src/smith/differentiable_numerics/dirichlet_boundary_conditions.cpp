// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"

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

}  // namespace smith
