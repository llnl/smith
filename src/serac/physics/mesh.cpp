// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/mesh.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/numerics/functional/domain.hpp"

namespace serac {

Mesh::Mesh(mfem::Mesh&& mesh, const std::string& meshtag, int refine_serial, int refine_parallel) : mesh_tag_(meshtag)
{
  auto meshtmp = serac::mesh::refineAndDistribute(std::move(mesh), refine_serial, refine_parallel);
  mfem_mesh_ = &serac::StateManager::setMesh(std::move(meshtmp), mesh_tag_);
  createDomains();
}

Mesh::Mesh(mfem::ParMesh& mesh, const std::string& meshtag) : mesh_tag_(meshtag)
{
  mfem_mesh_ = &mesh;
  createDomains();
}

Mesh::Mesh(const std::string& meshfile, const std::string& meshtag, int refine_serial, int refine_parallel)
    : mesh_tag_(meshtag)
{
  auto meshtmp = mesh::refineAndDistribute(buildMeshFromFile(meshfile), refine_serial, refine_parallel);
  mfem_mesh_ = &serac::StateManager::setMesh(std::move(meshtmp), mesh_tag_);
  createDomains();
}

MPI_Comm Mesh::getComm() const { return mfem_mesh_->GetComm(); }

void Mesh::createDomains() { domains_.insert({entireDomainName(), serac::EntireDomain(*mfem_mesh_)}); }

serac::Domain& Mesh::entireDomain() const { return domain(entireDomainName()); }

serac::Domain& Mesh::domain(const std::string& domain_name) const
{
  SLIC_ERROR_IF(domains_.find(domain_name) == domains_.end(),
                axom::fmt::format("Could not find domain named {0} in mesh with tag {1}", domain_name, mesh_tag_));
  return domains_.at(domain_name);
}

serac::Domain& Mesh::addDomainOfBoundaryElements(const std::string& domain_name,
                                                 std::function<bool(std::vector<vec2>, int)> func)
{
  SLIC_ERROR_IF(domains_.find(domain_name) != domains_.end(),
                axom::fmt::format("A domain named {0} already exists in mesh with tag {1}", domain_name, mesh_tag_));
  domains_.emplace(domain_name, Domain::ofBoundaryElements(*mfem_mesh_, func));
  return domain(domain_name);
}

serac::Domain& Mesh::addDomainOfBoundaryElements(const std::string& domain_name,
                                                 std::function<bool(std::vector<vec3>, int)> func)
{
  SLIC_ERROR_IF(domains_.find(domain_name) != domains_.end(),
                axom::fmt::format("A domain named {0} already exists in mesh with tag {1}", domain_name, mesh_tag_));
  domains_.emplace(domain_name, Domain::ofBoundaryElements(*mfem_mesh_, func));
  return domain(domain_name);
}

}  // namespace serac
