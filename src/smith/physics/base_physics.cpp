// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/base_physics.hpp"

#include <cmath>
#include <algorithm>
#include <tuple>

#include "smith/infrastructure/about.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_vector.hpp"
#include "smith/infrastructure/logger.hpp"

namespace smith {

BasePhysics::BasePhysics(std::string physics_name, std::shared_ptr<smith::Mesh> mesh, int cycle, double time,
                         bool checkpoint_to_disk)
    : name_(physics_name),
      mesh_(mesh),
      comm_(mesh_->getComm()),
      shape_displacement_(mesh_->newShapeDisplacement()),
      shape_displacement_dual_(mesh_->newShapeDisplacementDual()),
      bcs_(mesh_->mfemParMesh()),
      checkpoint_to_disk_(checkpoint_to_disk)
{
  std::tie(mpi_size_, mpi_rank_) = getMPIInfo(comm_);

  initializeBasePhysicsStates(cycle, time);
}

double BasePhysics::time() const { return time_; }

int BasePhysics::cycle() const { return cycle_; }

double BasePhysics::maxTime() const { return max_time_; }

int BasePhysics::maxCycle() const { return max_cycle_; }

double BasePhysics::minTime() const { return min_time_; }

int BasePhysics::minCycle() const { return min_cycle_; }

const std::vector<double>& BasePhysics::timesteps() const { return timesteps_; }

const smith::Mesh& BasePhysics::mesh() const { return *mesh_; }

const mfem::ParMesh& BasePhysics::mfemParMesh() const { return mesh_->mfemParMesh(); }

mfem::ParMesh& BasePhysics::mfemParMesh() { return mesh_->mfemParMesh(); }

void BasePhysics::initializeBasePhysicsStates(int cycle, double time)
{
  timesteps_.clear();

  time_ = time;
  dt_ = 0.0;
  max_time_ = time;
  min_time_ = time;
  cycle_ = cycle;
  max_cycle_ = cycle;
  min_cycle_ = cycle;
  ode_time_point_ = time;

  shape_displacement_dual_ = 0.0;
  for (auto& p : parameters_) {
    *p.sensitivity = 0.0;
  }
}

void BasePhysics::setParameter(const size_t parameter_index, const FiniteElementState& parameter_state)
{
  SLIC_ERROR_ROOT_IF(
      parameter_index >= parameters_.size(),
      axom::fmt::format("Parameter '{}' requested when only '{}' parameters exist in physics module '{}'",
                        parameter_index, parameters_.size(), name_));

  SLIC_ERROR_ROOT_IF(&parameter_state.mesh() != &mfemParMesh(),
                     axom::fmt::format("Mesh of parameter '{}' is not the same as the physics mesh", parameter_index));

  SLIC_ERROR_ROOT_IF(
      parameter_state.space().GetTrueVSize() != parameters_[parameter_index].state->space().GetTrueVSize(),
      axom::fmt::format(
          "Physics module parameter '{}' has size '{}' while given state has size '{}'. The finite element "
          "spaces are inconsistent.",
          parameter_index, parameters_[parameter_index].state->space().GetTrueVSize(),
          parameter_state.space().GetTrueVSize()));
  *parameters_[parameter_index].state = parameter_state;
}

const FiniteElementState& BasePhysics::shapeDisplacement() const { return shape_displacement_; }

void BasePhysics::setShapeDisplacement(const FiniteElementState& shape_displacement)
{
  shape_displacement_ = shape_displacement;
}

const FiniteElementDual& BasePhysics::shapeDisplacementSensitivity() const { return shape_displacement_dual_; }

FiniteElementDual BasePhysics::computeTimestepSensitivity(size_t parameter_index)
{
  SLIC_ERROR_ROOT(axom::fmt::format("Parameter sensitivities not enabled in physics module {}", name_));
  return *parameters_[parameter_index].sensitivity;
}

const FiniteElementDual& BasePhysics::computeTimestepShapeSensitivity()
{
  SLIC_ERROR_ROOT(axom::fmt::format("Shape sensitivities not enabled in physics module {}", name_));
  return shapeDisplacementSensitivity();
}

void BasePhysics::CreateParaviewDataCollection() const
{
  std::string output_name = name_.empty() ? "default" : name_;

  paraview_dc_ =
      std::make_unique<mfem::ParaViewDataCollection>(output_name, const_cast<mfem::ParMesh*>(&mfemParMesh()));

  // Register finite element fields

  paraview_dc_->RegisterField(shapeDisplacement().name(), &shapeDisplacement().gridFunction());

  for (const FiniteElementState* state : states_) {
    paraview_dc_->RegisterField(state->name(), &state->gridFunction());
  }

  for (auto& parameter : parameters_) {
    paraview_dc_->RegisterField(parameter.state->name(), &parameter.state->gridFunction());
  }

  // Register dual fields. These don't have gridfunction views already, so create them

  const mfem::ParFiniteElementSpace& shape_sensitivity_space = shapeDisplacementSensitivity().space();
  shape_sensitivity_grid_function_ =
      std::make_unique<mfem::ParGridFunction>(&const_cast<mfem::ParFiniteElementSpace&>(shape_sensitivity_space));
  paraview_dc_->RegisterField(shapeDisplacementSensitivity().name(), shape_sensitivity_grid_function_.get());

  for (const FiniteElementDual* dual : duals_) {
    paraview_dual_grid_functions_[dual->name()] =
        std::make_unique<mfem::ParGridFunction>(const_cast<mfem::ParFiniteElementSpace*>(&dual->space()));
    paraview_dc_->RegisterField(dual->name(), paraview_dual_grid_functions_[dual->name()].get());
  }

  // Identify maximum polynomial order in output fields in order to set detail level

  int max_order_in_fields = mfemParMesh().GetNodalFESpace()->GetMaxElementOrder();

  for (const auto& [_, field] : paraview_dc_->GetFieldMap()) {
    max_order_in_fields = std::max(field->FESpace()->GetMaxElementOrder(), max_order_in_fields);
  }

  // Set the options for the paraview output files
  paraview_dc_->SetLevelsOfDetail(max_order_in_fields);
  paraview_dc_->SetHighOrderOutput(true);
  paraview_dc_->SetDataFormat(mfem::VTKFormat::BINARY);
  paraview_dc_->SetCompression(true);
}

void BasePhysics::UpdateParaviewDataCollection(const std::string& paraview_output_dir) const
{
  for (const FiniteElementState* state : states_) {
    state->gridFunction();  // update grid function values
  }
  for (const FiniteElementDual* dual : duals_) {
    smith::FiniteElementDual* non_const_dual = const_cast<smith::FiniteElementDual*>(dual);
    non_const_dual->linearForm().ParallelAssemble(paraview_dual_grid_functions_[dual->name()]->GetTrueVector());
    paraview_dual_grid_functions_[dual->name()]->SetFromTrueVector();
  }
  for (auto& parameter : parameters_) {
    parameter.state->gridFunction();
  }
  shapeDisplacement().gridFunction();
  shapeDisplacementSensitivity().linearForm().ParallelAssemble(shape_sensitivity_grid_function_->GetTrueVector());
  shape_sensitivity_grid_function_->SetFromTrueVector();

  // Set the current time, cycle, and requested paraview directory
  paraview_dc_->SetCycle(cycle_);
  paraview_dc_->SetTime(time_);
  paraview_dc_->SetPrefixPath(paraview_output_dir);
}

void BasePhysics::outputStateToDisk(std::optional<std::string> paraview_output_dir) const
{
  // Update the states and duals in the state manager
  for (auto& state : states_) {
    StateManager::updateState(*state);
  }

  for (auto& dual : duals_) {
    StateManager::updateDual(*dual);
  }

  for (auto& parameter : parameters_) {
    StateManager::updateState(*parameter.state);
    StateManager::updateDual(*parameter.sensitivity);
  }

  StateManager::updateState(shapeDisplacement());
  StateManager::updateDual(shapeDisplacementSensitivity());

  // Save the restart/Sidre file
  StateManager::save(time_, cycle_, mesh_->tag());

  // Optionally output a paraview datacollection for visualization
  if (paraview_output_dir) {
    // Check to see if the paraview data collection exists. If not, create it.
    if (!paraview_dc_) {
      CreateParaviewDataCollection();
    }

    UpdateParaviewDataCollection(*paraview_output_dir);

    // Write the paraview file
    paraview_dc_->Save();
  }
}

void BasePhysics::initializeSummary(axom::sidre::DataStore& datastore, double t_final, double dt) const
{
  // Summary Sidre Structure
  // Sidre root
  // └── smith_summary
  //     ├── mpi_rank_count : int
  //     └── curves
  //         ├── t : Sidre::Array<axom::IndexType>
  //         ├── <FiniteElementState name>
  //         │    ├── l1norm : Sidre::Array<double>
  //         │    └── l2norm : Sidre::Array<double>
  //         ...
  //         └── <FiniteElementState name>
  //              ├── l1norm : Sidre::Array<double>
  //              └── l2norm : Sidre::Array<double>

  auto [count, rank] = getMPIInfo();
  if (rank != 0) {
    // Don't initialize except on root node
    return;
  }
  const std::string summary_group_name = "smith_summary";
  axom::sidre::Group* sidre_root = datastore.getRoot();
  SLIC_ERROR_ROOT_IF(
      sidre_root->hasGroup(summary_group_name),
      axom::fmt::format("Sidre Group '{0}' cannot exist when initializeSummary is called", summary_group_name));
  axom::sidre::Group* summary_group = sidre_root->createGroup(summary_group_name);

  // Write run info
  summary_group->createViewScalar("mpi_rank_count", count);

  // Write curves info
  axom::sidre::Group* curves_group = summary_group->createGroup("curves");

  // Calculate how many time steps which is the array size
  axom::IndexType array_size = static_cast<axom::IndexType>(ceil(t_final / dt));

  // t: array of each time step value
  axom::sidre::View* t_array_view = curves_group->createView("t");
  axom::sidre::Array<double> ts(t_array_view, 0, array_size);

  for (const FiniteElementState* state : states_) {
    // Group for this Finite Element State (Field)
    axom::sidre::Group* state_group = curves_group->createGroup(state->name());

    // Create an array for each stat type to hold a value at each time step
    for (std::string stat_name : {"l1norms", "l2norms", "linfnorms", "avgs", "mins", "maxs"}) {
      axom::sidre::View* curr_array_view = state_group->createView(stat_name);
      axom::sidre::Array<double> array(curr_array_view, 0, array_size);
    }
  }
}

void BasePhysics::saveSummary(axom::sidre::DataStore& datastore, const double t) const
{
  auto [_, rank] = getMPIInfo();

  // Find curves sidre group
  axom::sidre::Group* curves_group = nullptr;
  // Only save on root node
  if (rank == 0) {
    axom::sidre::Group* sidre_root = datastore.getRoot();
    const std::string curves_group_name = "smith_summary/curves";
    SLIC_ERROR_IF(!sidre_root->hasGroup(curves_group_name),
                  axom::fmt::format("Sidre Group '{0}' did not exist when saveCurves was called", curves_group_name));
    curves_group = sidre_root->getGroup(curves_group_name);

    // Save time step
    axom::sidre::Array<double> ts(curves_group->getView("t"));
    ts.push_back(t);
  }

  // For each Finite Element State (Field)
  double l1norm_value, l2norm_value, linfnorm_value, avg_value, max_value, min_value;
  for (const FiniteElementState* state : states_) {
    // Calculate current stat value
    // Note: These are collective operations.
    l1norm_value = norm(*state, 1.0);
    l2norm_value = norm(*state, 2.0);
    linfnorm_value = norm(*state, mfem::infinity());
    avg_value = avg(*state);
    max_value = max(*state);
    min_value = min(*state);

    // Only save on root node
    if (rank == 0) {
      // Group for this Finite Element State (Field)
      axom::sidre::Group* state_group = curves_group->getGroup(state->name());

      // Save all current stat values in their respective sidre arrays
      axom::sidre::View* l1norms_view = state_group->getView("l1norms");
      axom::sidre::Array<double> l1norms(l1norms_view);
      l1norms.push_back(l1norm_value);

      axom::sidre::View* l2norms_view = state_group->getView("l2norms");
      axom::sidre::Array<double> l2norms(l2norms_view);
      l2norms.push_back(l2norm_value);

      axom::sidre::View* linfnorms_view = state_group->getView("linfnorms");
      axom::sidre::Array<double> linfnorms(linfnorms_view);
      linfnorms.push_back(linfnorm_value);

      axom::sidre::View* avgs_view = state_group->getView("avgs");
      axom::sidre::Array<double> avgs(avgs_view);
      avgs.push_back(avg_value);

      axom::sidre::View* maxs_view = state_group->getView("maxs");
      axom::sidre::Array<double> maxs(maxs_view);
      maxs.push_back(max_value);

      axom::sidre::View* mins_view = state_group->getView("mins");
      axom::sidre::Array<double> mins(mins_view);
      mins.push_back(min_value);
    }
  }
}

FiniteElementState BasePhysics::loadCheckpointedState(const std::string& state_name, int cycle)
{
  if (checkpoint_to_disk_) {
    // See if the requested cycle has been checkpointed previously
    if (!cached_checkpoint_cycle_ || *cached_checkpoint_cycle_ != cycle) {
      // If not, get the checkpoint from disk
      cached_checkpoint_states_ = getCheckpointedStates(cycle);
      cached_checkpoint_cycle_ = cycle;
    }

    // Ensure that the state name exists in this physics module
    SLIC_ERROR_ROOT_IF(
        cached_checkpoint_states_.find(state_name) == cached_checkpoint_states_.end(),
        axom::fmt::format("Requested state name {} does not exist in physics module {}.", state_name, name_));
    return cached_checkpoint_states_.at(state_name);
  }

  // Ensure that the state name exists in this physics module
  SLIC_ERROR_ROOT_IF(
      checkpoint_states_.find(state_name) == checkpoint_states_.end(),
      axom::fmt::format("Requested state name {} does not exist in physics module {}.", state_name, name_));

  return checkpoint_states_.at(state_name)[static_cast<size_t>(cycle)];
}

void BasePhysics::loadCheckpointedStatesFromDisk(int cycle_to_load)
{
  std::vector<FiniteElementState*> previous_states_ptrs;
  for (const auto& state_name : stateNames()) {
    previous_states_ptrs.emplace_back(const_cast<FiniteElementState*>(&state(state_name)));
  }
  StateManager::loadCheckpointedStates(cycle_to_load, previous_states_ptrs);
}

std::unordered_map<std::string, FiniteElementState> BasePhysics::getCheckpointedStates(int cycle_to_load)
{
  std::unordered_map<std::string, FiniteElementState> previous_states_map;

  if (checkpoint_to_disk_) {
    loadCheckpointedStatesFromDisk(cycle_to_load);
    for (const auto& state_name : stateNames()) {
      previous_states_map.emplace(state_name, state(state_name));
    }
    return previous_states_map;
  } else {
    for (const auto& state_name : stateNames()) {
      previous_states_map.emplace(state_name, checkpoint_states_.at(state_name)[static_cast<size_t>(cycle_to_load)]);
    }
  }

  return previous_states_map;
}

double BasePhysics::getCheckpointedTimestep(int cycle) const
{
  SLIC_ERROR_ROOT_IF(cycle < 0, axom::fmt::format("Negative cycle number requested for physics module {}.", name_));
  SLIC_ERROR_ROOT_IF(cycle > static_cast<int>(timesteps_.size()),
                     axom::fmt::format("Timestep for cycle {} requested, but physics module has only reached cycle {}.",
                                       cycle, timesteps_.size()));
  return cycle < static_cast<int>(timesteps_.size()) ? timesteps_[static_cast<size_t>(cycle)] : 0.0;
}

namespace detail {
std::string addPrefix(const std::string& prefix, const std::string& target)
{
  if (prefix.empty()) {
    return target;
  }
  return prefix + "_" + target;
}

std::string removePrefix(const std::string& prefix, const std::string& target)
{
  std::string modified_target{target};
  // Ensure the prefix isn't an empty string
  if (!prefix.empty()) {
    // Ensure the prefix is at the beginning of the string
    auto index = modified_target.find(prefix + "_");
    if (index == 0) {
      // Remove the prefix
      modified_target.erase(0, prefix.size() + 1);
    }
  }
  return modified_target;
}

}  // namespace detail

}  // namespace smith
