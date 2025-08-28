// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/mesh.hpp"

#include <utility>

#include <axom/fmt.hpp>
#include <axom/slic.hpp>

#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/state/finite_element_dual.hpp"
#include "serac/mesh_utils/mesh_utils.hpp"
#include "serac/numerics/functional/finite_element.hpp"

namespace serac {

Mesh::Mesh(const std::string& meshfile, const std::string& meshtag, int refine_serial, int refine_parallel,
           MPI_Comm comm)
    : mesh_tag_(meshtag)
{
  auto meshtmp = mesh::refineAndDistribute(buildMeshFromFile(meshfile), refine_serial, refine_parallel, comm);
  mfem_mesh_ = &serac::StateManager::setMesh(std::move(meshtmp), mesh_tag_);
  createDomains();
}

Mesh::Mesh(mfem::Mesh&& mesh, const std::string& meshtag, int refine_serial, int refine_parallel, MPI_Comm comm)
    : mesh_tag_(meshtag)
{
  auto meshtmp = serac::mesh::refineAndDistribute(std::move(mesh), refine_serial, refine_parallel, comm);
  mfem_mesh_ = &serac::StateManager::setMesh(std::move(meshtmp), mesh_tag_);
  createDomains();
}

Mesh::Mesh(mfem::ParMesh&& mesh, const std::string& meshtag) : mesh_tag_(meshtag)
{
  auto meshtmp = std::make_unique<mfem::ParMesh>(std::move(mesh));
  meshtmp->EnsureNodes();
  meshtmp->ExchangeFaceNbrData();
  mfem_mesh_ = &serac::StateManager::setMesh(std::move(meshtmp), mesh_tag_);
  createDomains();
}

MPI_Comm Mesh::getComm() const { return mfem_mesh_->GetComm(); }

void Mesh::createDomains()
{
  domains_.insert({entireBodyName(), serac::EntireDomain(*mfem_mesh_)});
  domains_.insert({entireBoundaryName(), serac::EntireBoundary(*mfem_mesh_)});
  domains_.insert({internalBoundaryName(), serac::InteriorFaces(*mfem_mesh_)});
}

serac::Domain& Mesh::entireBody() const { return domain(entireBodyName()); }

serac::Domain& Mesh::entireBoundary() const { return domain(entireBoundaryName()); }

serac::Domain& Mesh::internalBoundary() const { return domain(internalBoundaryName()); }

void Mesh::errorIfDomainExists(const std::string &domain_name) const
{
  SLIC_ERROR_IF(domains_.find(domain_name) != domains_.end(),
                axom::fmt::format("A domain named {0} already exists in mesh with tag {1}", domain_name, mesh_tag_));
}

void Mesh::insertDomain(const std::string &domain_name, const Domain &domain)
{
  SLIC_ERROR_IF(&this->mfemParMesh() != &domain.mesh_, "A domain inserted onto a mesh must be defined on that mesh");
  errorIfDomainExists(domain_name);
  domains_.insert({domain_name, domain});
}

serac::Domain& Mesh::domain(const std::string& domain_name) const
{
  SLIC_ERROR_IF(domains_.find(domain_name) == domains_.end(),
                axom::fmt::format("Could not find domain named {0} in mesh with tag {1}", domain_name, mesh_tag_));
  return domains_.at(domain_name);
}

serac::Domain& Mesh::addDomainOfBoundaryElements(const std::string& domain_name,
                                                 std::function<bool(std::vector<vec2>, int)> func)
{
  errorIfDomainExists(domain_name);
  domains_.emplace(domain_name, Domain::ofBoundaryElements(*mfem_mesh_, func));
  return domain(domain_name);
}

serac::Domain& Mesh::addDomainOfBoundaryElements(const std::string& domain_name,
                                                 std::function<bool(std::vector<vec3>, int)> func)
{
  errorIfDomainExists(domain_name);
  domains_.emplace(domain_name, Domain::ofBoundaryElements(*mfem_mesh_, func));
  return domain(domain_name);
}

serac::Domain& Mesh::addDomainOfBodyElements(const std::string& domain_name,
                                             std::function<bool(std::vector<vec2>, int)> func)
{
  errorIfDomainExists(domain_name);
  domains_.emplace(domain_name, Domain::ofElements(*mfem_mesh_, func));
  return domain(domain_name);
}

serac::Domain& Mesh::addDomainOfBodyElements(const std::string& domain_name,
                                             std::function<bool(std::vector<vec3>, int)> func)
{
  errorIfDomainExists(domain_name);
  domains_.emplace(domain_name, Domain::ofElements(*mfem_mesh_, func));
  return domain(domain_name);
}

const mfem::ParFiniteElementSpace& Mesh::shapeDisplacementSpace()
{
  return serac::StateManager::shapeDisplacement(tag()).space();
}

serac::FiniteElementState Mesh::newShapeDisplacement() { return StateManager::shapeDisplacement(tag()); }

serac::FiniteElementDual Mesh::newShapeDisplacementDual() { return StateManager::shapeDisplacementDual(tag()); }

}  // namespace serac
