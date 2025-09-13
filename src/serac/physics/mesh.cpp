// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/mesh.hpp"
#include "serac/mesh_utils/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

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
  int rank = 0;
MPI_Comm_rank(comm, &rank); // Get the rank of the current process

if (rank == 0) { std::cout << ".... Before starting refineAndDistribute!" << std::endl;}
  auto meshtmp = serac::mesh::refineAndDistribute(std::move(mesh), refine_serial, refine_parallel, comm);
if (rank == 0) { std::cout << ".... Before starting setMesh!" << std::endl;}
  mfem_mesh_ = &serac::StateManager::setMesh(std::move(meshtmp), mesh_tag_);
if (rank == 0) { std::cout << ".... Before starting createDomains!" << std::endl;}
  createDomains();
}

Mesh::Mesh(mfem::ParMesh&& mesh, const std::string& meshtag) : mesh_tag_(meshtag)
{
  auto meshtmp = std::make_unique<mfem::ParMesh>(std::move(mesh));
  int rank = 0;
MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process

if (rank == 0) { std::cout << ".... Before starting EnsureNodes!" << std::endl;}
  meshtmp->EnsureNodes();
if (rank == 0) { std::cout << ".... Before starting ExchangeFaceNbrData!" << std::endl;}
  meshtmp->ExchangeFaceNbrData();
if (rank == 0) { std::cout << ".... Before starting setMesh!" << std::endl;}
  mfem_mesh_ = &serac::StateManager::setMesh(std::move(meshtmp), mesh_tag_);
if (rank == 0) { std::cout << ".... Before starting createDomains!" << std::endl;}
  createDomains();
}

MPI_Comm Mesh::getComm() const { return mfem_mesh_->GetComm(); }

void Mesh::createDomains()
{
  domains_.insert({entireBodyName(), serac::EntireDomain(*mfem_mesh_)});
  domains_.insert({entireBoundaryName(), serac::EntireBoundary(*mfem_mesh_)});
  domains_.insert({internalBoundaryName(), serac::InteriorFaces(*mfem_mesh_)});

  int dim = mfem_mesh_->Dimension();

  auto addPrefix = [](const std::string& prefix, const std::string& target) {
    if (prefix.empty()) {
      return target;
    }
    return prefix + "_" + target;
  };

  auto shape_disp_name = addPrefix(mesh_tag_, "shape_displacement");
  if (dim == 2) {
    shape_displacement_ =
        std::shared_ptr<FiniteElementState>(&StateManager::shapeDisplacement(mesh_tag_), [](FiniteElementState*) {});
  } else if (dim == 3) {
    shape_displacement_ =
        std::shared_ptr<FiniteElementState>(&StateManager::shapeDisplacement(mesh_tag_), [](FiniteElementState*) {});
  }

  auto shape_disp_dual_name = addPrefix(mesh_tag_, "shape_displacement_dual");
  if (dim == 2) {
    shape_displacement_dual_ =
        std::make_shared<FiniteElementDual>(StateManager::newDual(H1<1, 2>{}, shape_disp_dual_name, mesh_tag_));
  } else if (dim == 3) {
    shape_displacement_dual_ =
        std::make_shared<FiniteElementDual>(StateManager::newDual(H1<1, 3>{}, shape_disp_dual_name, mesh_tag_));
  }
}

serac::Domain& Mesh::entireBody() const { return domain(entireBodyName()); }

serac::Domain& Mesh::entireBoundary() const { return domain(entireBoundaryName()); }

serac::Domain& Mesh::internalBoundary() const { return domain(internalBoundaryName()); }

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

serac::Domain& Mesh::addDomainOfBodyElements(const std::string& domain_name,
                                             std::function<bool(std::vector<vec2>, int)> func)
{
  SLIC_ERROR_IF(domains_.find(domain_name) != domains_.end(),
                axom::fmt::format("A domain named {0} already exists in mesh with tag {1}", domain_name, mesh_tag_));
  domains_.emplace(domain_name, Domain::ofElements(*mfem_mesh_, func));
  return domain(domain_name);
}

serac::Domain& Mesh::addDomainOfBodyElements(const std::string& domain_name,
                                             std::function<bool(std::vector<vec3>, int)> func)
{
  SLIC_ERROR_IF(domains_.find(domain_name) != domains_.end(),
                axom::fmt::format("A domain named {0} already exists in mesh with tag {1}", domain_name, mesh_tag_));
  domains_.emplace(domain_name, Domain::ofElements(*mfem_mesh_, func));
  return domain(domain_name);
}

serac::FiniteElementState& Mesh::shapeDisplacement() { return *shape_displacement_; }

const serac::FiniteElementState& Mesh::shapeDisplacement() const { return *shape_displacement_; }

serac::FiniteElementDual& Mesh::shapeDisplacementDual() { return *shape_displacement_dual_; }

const serac::FiniteElementDual& Mesh::shapeDisplacementDual() const { return *shape_displacement_dual_; }

}  // namespace serac
