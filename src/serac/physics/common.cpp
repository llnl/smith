#include "serac/physics/common.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "mfem.hpp"

namespace serac {

Mesh::Mesh(mfem::Mesh&& mesh, const std::string& meshtag, int refine_serial, int refine_parallel) : mesh_tag(meshtag)
{
  auto meshtmp = serac::mesh::refineAndDistribute(std::move(mesh), refine_serial, refine_parallel);
  mfem_mesh = &serac::StateManager::setMesh(std::move(meshtmp), mesh_tag);
  createDomains();
}

Mesh::Mesh(mfem::ParMesh& mesh, const std::string& meshtag) : mesh_tag(meshtag)
{
  mfem_mesh = &mesh;
  createDomains();
}

Mesh::Mesh(const std::string& meshfile, const std::string& meshtag, int refine_serial, int refine_parallel)
    : mesh_tag(meshtag)
{
  auto meshtmp = mesh::refineAndDistribute(buildMeshFromFile(meshfile), refine_serial, refine_parallel);
  mfem_mesh = &serac::StateManager::setMesh(std::move(meshtmp), mesh_tag);
  createDomains();
}

MPI_Comm Mesh::getComm() const { return mfem_mesh->GetComm(); }

void Mesh::createDomains() { domains_.insert({"whole_mesh", serac::EntireDomain(*mfem_mesh)}); }

serac::Domain& Mesh::entireDomain() const { return domain("whole_mesh"); }

serac::Domain& Mesh::domain(const std::string& domain_name) const
{
  SLIC_ERROR_IF(domains_.find(domain_name) == domains_.end(),
                axom::fmt::format("Could not find domain named {0} in mesh with tag {1}", domain_name, mesh_tag));
  return domains_.at(domain_name);
}

serac::Domain& Mesh::addDomainOfBoundaryElements(const std::string& domain_name,
                                                 std::function<bool(std::vector<vec3>, int)> func)
{
  domains_.insert({domain_name, Domain::ofBoundaryElements(*mfem_mesh, func)});
  return domain(domain_name);
}

}  // namespace serac