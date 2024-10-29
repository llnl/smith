// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

#include "axom/core.hpp"

namespace serac {

// Initialize StateManager's static members - these will be fully initialized in StateManager::initialize
std::unordered_map<std::string, axom::sidre::MFEMSidreDataCollection> StateManager::datacolls_;
std::unordered_map<std::string, std::unique_ptr<FiniteElementState>>  StateManager::shape_displacements_;
bool                                                                  StateManager::is_restart_ = false;
axom::sidre::DataStore*                                               StateManager::ds_         = nullptr;
std::string                                                           StateManager::output_dir_ = "";
std::unordered_map<std::string, mfem::ParGridFunction*>               StateManager::named_states_;
std::unordered_map<std::string, mfem::ParGridFunction*>               StateManager::named_duals_;

double StateManager::newDataCollection(const std::string& name, const std::optional<int> cycle_to_load)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Cannot construct a DataCollection without a DataStore");
  std::string coll_name = name + "_datacoll";

  auto global_grp   = ds_->getRoot()->createGroup(coll_name + "_global");
  auto bp_index_grp = global_grp->createGroup("blueprint_index/" + coll_name);
  auto domain_grp   = ds_->getRoot()->createGroup(coll_name);

  // Needs to be configured to own the mesh data so all mesh data is saved to datastore/output file
  constexpr bool owns_mesh_data = true;
  auto [iter, _]                = datacolls_.emplace(std::piecewise_construct, std::forward_as_tuple(name),
                                                     std::forward_as_tuple(coll_name, bp_index_grp, domain_grp, owns_mesh_data));
  auto& datacoll                = iter->second;
  datacoll.SetComm(MPI_COMM_WORLD);

  datacoll.SetPrefixPath(output_dir_);

  if (cycle_to_load) {
    // NOTE: Load invalidates previous Sidre pointers
    datacoll.Load(*cycle_to_load);
    datacoll.SetGroupPointers(ds_->getRoot()->getGroup(coll_name + "_global/blueprint_index/" + coll_name),
                              ds_->getRoot()->getGroup(coll_name));
    SLIC_ERROR_ROOT_IF(datacoll.GetBPGroup()->getNumGroups() == 0,
                       "Loaded datastore is empty, was the datastore created on a "
                       "different number of nodes?");

    datacoll.UpdateStateFromDS();
    datacoll.UpdateMeshAndFieldsFromDS();

    // Functional needs the nodal grid function and neighbor data in the mesh

    // Determine if the existing nodal grid function is discontinuous. This
    // indicates that the mesh is periodic and the new nodal grid function must also
    // be discontinuous.
    bool is_discontinuous = false;
    auto nodes            = mesh(name).GetNodes();
    if (nodes) {
      is_discontinuous = nodes->FESpace()->FEColl()->GetContType() == mfem::FiniteElementCollection::DISCONTINUOUS;
      SLIC_WARNING_ROOT_IF(
          is_discontinuous,
          "Periodic mesh detected! This will only work on translational periodic surfaces for vector H1 fields and "
          "has not been thoroughly tested. Proceed at your own risk.");
    }

    // This mfem call ensures the mesh contains an H1 grid function describing nodal
    // cordinates. The parameters do the following:
    // 1. Sets the order of the mesh to  p = 1
    // 2. Uses the existing continuity of the mesh finite element space (periodic meshes are discontinuous)
    // 3. Uses the spatial dimension as the mesh dimension (i.e. it is not a lower dimension manifold)
    // 4. Uses the ordering set by serac::ordering
    mesh(name).SetCurvature(1, is_discontinuous, -1, serac::ordering);

    // Sidre will destruct the nodal grid function instead of the mesh
    mesh(name).SetNodesOwner(false);

    // Generate the face neighbor information in the mesh. This is needed by the face restriction
    // operators used by Functional
    mesh(name).ExchangeFaceNbrData();

    // Construct and store the shape displacement fields and sensitivities associated with this mesh
    constructShapeFields(name);

  } else {
    datacoll.SetCycle(0);   // Iteration counter
    datacoll.SetTime(0.0);  // Simulation time
  }

  return datacoll.GetTime();
}

void StateManager::loadCheckpointedStates(int cycle_to_load, std::vector<FiniteElementState*> states_to_load)
{
  SERAC_MARK_FUNCTION;
  mfem::ParMesh* meshPtr   = &(*states_to_load.begin())->mesh();
  std::string    mesh_name = collectionID(meshPtr);

  std::string coll_name = mesh_name + "_datacoll";

  axom::sidre::MFEMSidreDataCollection previous_datacoll(coll_name);

  previous_datacoll.SetComm(meshPtr->GetComm());
  previous_datacoll.SetPrefixPath(output_dir_);
  previous_datacoll.Load(cycle_to_load);

  for (auto state : states_to_load) {
    meshPtr = &state->mesh();
    SLIC_ERROR_ROOT_IF(collectionID(meshPtr) != mesh_name,
                       "Loading FiniteElementStates from two different meshes at one time is not allowed.");
    mfem::ParGridFunction* datacoll_owned_grid_function = previous_datacoll.GetParField(state->name());

    state->setFromGridFunction(*datacoll_owned_grid_function);
  }
}

void StateManager::initialize(axom::sidre::DataStore& ds, const std::string& output_directory)
{
  // If the global object has already been initialized, clear it out
  if (ds_) {
    reset();
  }
  ds_         = &ds;
  output_dir_ = output_directory;
  if (output_directory.empty()) {
    SLIC_ERROR_ROOT(
        "DataCollection output directory cannot be empty - this will result in problems if executables are run in "
        "parallel");
  }
}

FiniteElementState& StateManager::shapeDisplacement(const std::string& mesh_tag)
{
  return *shape_displacements_[mesh_tag];
}

void StateManager::storeState(FiniteElementState& state)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  auto mesh_tag = collectionID(&state.mesh());
  SLIC_ERROR_ROOT_IF(hasState(state.name()),
                     axom::fmt::format("StateManager already contains a state named '{}'", state.name()));
  auto&                  datacoll = datacolls_.at(mesh_tag);
  const std::string      name     = state.name();
  mfem::ParGridFunction* grid_function;
  if (is_restart_) {
    grid_function = datacoll.GetParField(name);
    state.setFromGridFunction(*grid_function);
  } else {
    SLIC_ERROR_ROOT_IF(datacoll.HasField(name),
                       axom::fmt::format("StateManager already given a field named '{0}'", name));

    // Create a new grid function with unallocated data. This will be managed by sidre.
    grid_function = new mfem::ParGridFunction(&state.space(), static_cast<double*>(nullptr));
    datacoll.RegisterField(name, grid_function);
    state.setFromGridFunction(*grid_function);
  }
  named_states_[name] = grid_function;
}

FiniteElementState StateManager::newState(const mfem::ParFiniteElementSpace& space, const std::string& state_name)
{
  std::string mesh_tag = collectionID(space.GetParMesh());

  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(!hasMesh(mesh_tag), axom::fmt::format("Mesh tag '{}' not found in the data store", mesh_tag));
  SLIC_ERROR_ROOT_IF(hasState(state_name),
                     axom::fmt::format("StateManager already contains a state named '{}'", state_name));
  auto state = FiniteElementState(space, state_name);
  storeState(state);
  return state;
}

void StateManager::storeDual(FiniteElementDual& dual)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  auto mesh_tag = collectionID(&dual.mesh());
  SLIC_ERROR_ROOT_IF(hasDual(dual.name()),
                     axom::fmt::format("StateManager already contains a state named '{}'", dual.name()));
  auto&                  datacoll = datacolls_.at(mesh_tag);
  const std::string      name     = dual.name();
  mfem::ParGridFunction* grid_function;
  if (is_restart_) {
    grid_function = datacoll.GetParField(name);
    std::unique_ptr<mfem::HypreParVector> true_dofs(grid_function->GetTrueDofs());
    dual = *true_dofs;
  } else {
    SLIC_ERROR_ROOT_IF(datacoll.HasField(name),
                       axom::fmt::format("StateManager already given a field named '{0}'", name));

    // Create a new grid function with unallocated data. This will be managed by sidre.
    grid_function = new mfem::ParGridFunction(&dual.space(), static_cast<double*>(nullptr));
    datacoll.RegisterField(name, grid_function);
    std::unique_ptr<mfem::HypreParVector> true_dofs(grid_function->GetTrueDofs());
    dual = *true_dofs;
  }
  named_duals_[name] = grid_function;
}

FiniteElementDual StateManager::newDual(const mfem::ParFiniteElementSpace& space, const std::string& dual_name)
{
  std::string mesh_tag = collectionID(space.GetParMesh());

  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(!hasMesh(mesh_tag), axom::fmt::format("Mesh tag '{}' not found in the data store", mesh_tag));
  SLIC_ERROR_ROOT_IF(hasDual(dual_name),
                     axom::fmt::format("StateManager already contains a dual named '{}'", dual_name));
  auto dual = FiniteElementDual(space, dual_name);
  storeDual(dual);
  return dual;
}

void StateManager::save(const double t, const int cycle, const std::string& mesh_tag)
{
  SERAC_MARK_FUNCTION;
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(!hasMesh(mesh_tag), axom::fmt::format("Mesh tag '{}' not found in the data store", mesh_tag));
  auto&       datacoll  = datacolls_.at(mesh_tag);
  std::string file_path = axom::utilities::filesystem::joinPath(datacoll.GetPrefixPath(), datacoll.GetCollectionName());
  SLIC_INFO_ROOT(
      axom::fmt::format("Saving data collection at time: '{}' and cycle: '{}' to path: '{}'", t, cycle, file_path));

  datacoll.SetTime(t);
  datacoll.SetCycle(cycle);
  datacoll.Save();
}

mfem::ParMesh& StateManager::setMesh(std::unique_ptr<mfem::ParMesh> pmesh, const std::string& mesh_tag)
{
  // Determine if the existing nodal grid function is discontinuous. This
  // indicates that the mesh is periodic and the new nodal grid function must also
  // be discontinuous.
  bool is_discontinuous = false;
  auto nodes            = pmesh->GetNodes();
  if (nodes) {
    is_discontinuous = nodes->FESpace()->FEColl()->GetContType() == mfem::FiniteElementCollection::DISCONTINUOUS;
    SLIC_WARNING_ROOT_IF(
        is_discontinuous,
        "Periodic mesh detected! This will only work on translational periodic surfaces for vector H1 fields and "
        "has not been thoroughly tested. Proceed at your own risk.");
  }

  // This mfem call ensures the mesh contains an H1 grid function describing nodal
  // cordinates. The parameters do the following:
  // 1. Sets the order of the mesh to  p = 1
  // 2. Uses the existing continuity of the mesh finite element space (periodic meshes are discontinuous)
  // 3. Uses the spatial dimension as the mesh dimension (i.e. it is not a lower dimension manifold)
  // 4. Uses the ordering set by serac::ordering
  pmesh->SetCurvature(1, is_discontinuous, -1, serac::ordering);

  // Sidre will destruct the nodal grid function instead of the mesh
  pmesh->SetNodesOwner(false);

  newDataCollection(mesh_tag);
  auto& datacoll = datacolls_.at(mesh_tag);
  datacoll.SetMesh(pmesh.release());
  datacoll.SetOwnData(true);

  // Functional needs the nodal grid function and neighbor data in the mesh
  auto& new_pmesh = mesh(mesh_tag);

  // Generate the face neighbor information in the mesh. This is needed by the face restriction
  // operators used by Functional
  new_pmesh.ExchangeFaceNbrData();

  // We must construct the shape fields here as the mesh did not exist during the newDataCollection call
  // for the non-restart case
  constructShapeFields(mesh_tag);

  return new_pmesh;
}

void StateManager::constructShapeFields(const std::string& mesh_tag)
{
  // Construct the shape displacement field associated with this mesh
  auto& new_mesh = mesh(mesh_tag);

  if (new_mesh.Dimension() == 2) {
    shape_displacements_[mesh_tag] =
        std::make_unique<FiniteElementState>(new_mesh, SHAPE_DIM_2, mesh_tag + "_shape_displacement");
  } else if (new_mesh.Dimension() == 3) {
    shape_displacements_[mesh_tag] =
        std::make_unique<FiniteElementState>(new_mesh, SHAPE_DIM_3, mesh_tag + "_shape_displacement");
  } else {
    SLIC_ERROR_ROOT(axom::fmt::format("Mesh of dimension {} given, only dimensions 2 or 3 are available in Serac.",
                                      new_mesh.Dimension()));
  }

  storeState(*shape_displacements_[mesh_tag]);

  *shape_displacements_[mesh_tag] = 0.0;
}

mfem::ParMesh& StateManager::mesh(const std::string& mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!hasMesh(mesh_tag), axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto mesh = datacolls_.at(mesh_tag).GetMesh();
  SLIC_ERROR_ROOT_IF(!mesh, "The datacollection does not contain a mesh object");
  return static_cast<mfem::ParMesh&>(*mesh);
}

std::string StateManager::collectionID(const mfem::ParMesh* pmesh)
{
  for (auto& [name, datacoll] : datacolls_) {
    if (datacoll.GetMesh() == pmesh) {
      return name;
    }
  }
  SLIC_ERROR_ROOT("The mesh has not been registered with StateManager");
  return {};
}

int StateManager::cycle(std::string mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!hasMesh(mesh_tag), axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  return datacolls_.at(mesh_tag).GetCycle();
}

double StateManager::time(std::string mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!hasMesh(mesh_tag), axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  return datacolls_.at(mesh_tag).GetTime();
}

}  // namespace serac
