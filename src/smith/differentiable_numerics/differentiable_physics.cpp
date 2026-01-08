// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gretl/data_store.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "gretl/data_store.hpp"

namespace smith {

/// @brief gretl-function to create a dummy-state which records all states and params of interest to the mechanics. This
/// is used to inject additional adjoint loads and evaluate individual timestep sensitivities for the BasePhysics
/// interface.
gretl::State<int> make_milestone(const std::vector<FieldState>& states)
{
  std::vector<gretl::StateBase> base_states;
  for (const auto& s : states) {
    base_states.push_back(s);
  }

  auto milestone = states[0].create_state<int, int>(base_states);

  milestone.set_eval(
      []([[maybe_unused]] const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) { output.set<int>(0); });
  milestone.set_vjp(
      []([[maybe_unused]] gretl::UpstreamStates& inputs, [[maybe_unused]] const gretl::DownstreamState& output) {});

  return milestone.finalize();
}

// mesh, equation, fields, parameters, state advancer, solver
DifferentiablePhysics::DifferentiablePhysics(std::shared_ptr<Mesh> mesh, std::shared_ptr<gretl::DataStore> graph,
                                             const FieldState& shape_disp, const std::vector<FieldState>& states,
                                             const std::vector<FieldState>& params,
                                             std::shared_ptr<StateAdvancer> advancer, std::string mech_name,
                                             const std::vector<std::string>& reaction_names)
    : BasePhysics(mech_name, mesh, 0, 0.0, false),  // the false is checkpoint_to_disk
      checkpointer_(graph),
      advancer_(advancer),
      reaction_names_(reaction_names)
{
  SLIC_ERROR_IF(states.size() == 0, "Must have a least 1 state for a mechanics.");
  field_shape_displacement_ = std::make_unique<FieldState>(shape_disp);
  for (size_t i = 0; i < states.size(); ++i) {
    const auto& s = states[i];
    field_states_.push_back(s);
    initial_field_states_.push_back(s);
    state_name_to_field_index_[s.get()->name()] = i;
    state_names_.push_back(s.get()->name());
  }

  for (size_t i = 0; i < params.size(); ++i) {
    const auto& p = params[i];
    field_params_.push_back(p);
    param_name_to_field_index_[p.get()->name()] = i;
    param_names_.push_back(p.get()->name());
  }

  for (size_t i = 0; i < reaction_names_.size(); ++i) {
    reaction_name_to_reaction_index_[reaction_names_[i]] = i;
  }

  completeSetup();
}

void DifferentiablePhysics::completeSetup()
{
  SLIC_ERROR_IF(field_states_.empty(), "Empty field state during completeSetup()");
}

void DifferentiablePhysics::resetStates(int cycle, double time)
{
  for (size_t i = 0; i < initial_field_states_.size(); ++i) {
    field_states_[i] = initial_field_states_[i];
  }
  milestones_.clear();
  checkpointer_->reset_graph();
  time_ = time;
  cycle_ = cycle;
}

void DifferentiablePhysics::resetAdjointStates()
{
  checkpointer_->finalize_graph();
  checkpointer_->reset_for_backprop();
  gretl_assert(checkpointer_->check_validity());
}

std::vector<std::string> DifferentiablePhysics::stateNames() const { return state_names_; }

std::vector<std::string> DifferentiablePhysics::parameterNames() const { return param_names_; }

std::vector<std::string> DifferentiablePhysics::dualNames() const { return reaction_names_; }

const FiniteElementState& DifferentiablePhysics::state([[maybe_unused]] const std::string& field_name) const
{
  SLIC_ERROR_IF(
      state_name_to_field_index_.find(field_name) == state_name_to_field_index_.end(),
      axom::fmt::format("Could not find field named {0} in mesh with tag \"{1}\" to get", field_name, mesh_->tag()));
  size_t state_index = state_name_to_field_index_.at(field_name);
  return *field_states_[state_index].get();
}

const FiniteElementDual& DifferentiablePhysics::dual(const std::string& dual_name) const
{
  SLIC_ERROR_IF(
      reaction_name_to_reaction_index_.find(dual_name) == reaction_name_to_reaction_index_.end(),
      axom::fmt::format("Could not find dual named {0} in mesh with tag \"{1}\" to get", dual_name, mesh_->tag()));
  size_t reaction_index = reaction_name_to_reaction_index_.at(dual_name);
  SLIC_ERROR_IF(
      reaction_index >= reaction_names_.size(),
      "Dual reactions not correctly allocated yet, cannot get dual until after initializationStep is called.");

  TimeInfo time_info(time_prev_, dt_prev_, static_cast<size_t>(cycle_prev_));
  reaction_states_ = advancer_->computeReactions(time_info, *field_shape_displacement_, field_states_, field_params_);
  return *reaction_states_[reaction_index].get();
}

FiniteElementState DifferentiablePhysics::loadCheckpointedState(const std::string& state_name, int cycle)
{
  SLIC_ERROR_IF(cycle != cycle_,
                axom::fmt::format("Due to checkpointing restrictions in smith::Mechanics, cannot ask for an arbitrary "
                                  "checkpointed cycle, asking for cycle {}, but physics is at cycle {}",
                                  cycle, cycle_));
  return state(state_name);
}

const FiniteElementState& DifferentiablePhysics::shapeDisplacement() const { return *field_shape_displacement_->get(); }

const FiniteElementState& DifferentiablePhysics::parameter(std::size_t parameter_index) const
{
  SLIC_ERROR_IF(parameter_index >= field_params_.size(),
                axom::fmt::format("Parameter index {} requested, but only {} parameters exist in physics module {}.",
                                  parameter_index, field_params_.size(), name_));
  return *field_params_[parameter_index].get();
}

const FiniteElementState& DifferentiablePhysics::parameter(const std::string& parameter_name) const
{
  SLIC_ERROR_IF(param_name_to_field_index_.find(parameter_name) == param_name_to_field_index_.end(),
                axom::fmt::format("Could not find parameter named {0} in mesh with tag \"{1}\" to get", parameter_name,
                                  mesh_->tag()));
  size_t param_index = param_name_to_field_index_.at(parameter_name);
  return parameter(param_index);
}

void DifferentiablePhysics::setParameter(const size_t parameter_index, const FiniteElementState& parameter_state)
{
  SLIC_ERROR_IF(parameter_index >= field_params_.size(),
                axom::fmt::format("Parameter '{}' requested when only '{}' parameters exist in physics module '{}'",
                                  parameter_index, field_params_.size(), name_));
  *field_params_[parameter_index].get() = parameter_state;
}

void DifferentiablePhysics::setShapeDisplacement(const FiniteElementState& s) { *field_shape_displacement_->get() = s; }

void DifferentiablePhysics::setState([[maybe_unused]] const std::string& field_name,
                                     [[maybe_unused]] const FiniteElementState& s)
{
  SLIC_ERROR_IF(
      state_name_to_field_index_.find(field_name) == state_name_to_field_index_.end(),
      axom::fmt::format("Could not find field named {0} in mesh with tag {1} to set", field_name, mesh_->tag()));
  size_t state_index = state_name_to_field_index_.at(field_name);
  *field_states_[state_index].get() = s;
  *initial_field_states_[state_index].get() = s;
}

void DifferentiablePhysics::setAdjointLoad(
    std::unordered_map<std::string, const smith::FiniteElementDual&> string_to_dual)
{
  for (auto string_dual_pair : string_to_dual) {
    std::string field_name = string_dual_pair.first;
    const smith::FiniteElementDual& dual = string_dual_pair.second;
    SLIC_ERROR_IF(state_name_to_field_index_.find(field_name) == state_name_to_field_index_.end(),
                  axom::fmt::format("Could not find dual named {0} in mesh with tag {1}", field_name, mesh_->tag()));
    size_t state_index = state_name_to_field_index_.at(field_name);
    *field_states_[state_index].get_dual() += dual;
  }
}

void DifferentiablePhysics::setDualAdjointBcs(
    std::unordered_map<std::string, const smith::FiniteElementState&> string_to_bc)
{
  for (auto string_bc_pair : string_to_bc) {
    std::string reaction_name = string_bc_pair.first;
    const smith::FiniteElementState& reaction_dual = string_bc_pair.second;
    SLIC_ERROR_IF(
        reaction_name_to_reaction_index_.find(reaction_name) == reaction_name_to_reaction_index_.end(),
        axom::fmt::format("When calling setDualAdjointBcs, could not find reaction named {0} in mesh with tag {1}",
                          reaction_name, mesh_->tag()));
    size_t reaction_index = reaction_name_to_reaction_index_.at(reaction_name);
    *reaction_states_[reaction_index].get_dual() += reaction_dual;
  }
}

const FiniteElementState& DifferentiablePhysics::adjoint([[maybe_unused]] const std::string& adjoint_name) const
{
  // MRT, not implemented
  SLIC_ERROR("What is the use case for asking for the adjoint solution field directly?");
  return *adjoints_[0];
}

void DifferentiablePhysics::advanceTimestep(double dt)
{
  if (cycle_ == 0) {
    field_states_ = initial_field_states_;
    milestones_.push_back(make_milestone(field_states_).step());
  }

  cycle_prev_ = cycle_;
  time_prev_ = time_;
  dt_prev_ = dt;

  TimeInfo time_info(time_, dt, static_cast<size_t>(cycle_));
  field_states_ = advancer_->advanceState(time_info, *field_shape_displacement_, field_states_, field_params_);

  cycle_++;
  time_ += dt;
  milestones_.push_back(make_milestone(field_states_).step());
}

void DifferentiablePhysics::reverseAdjointTimestep()
{
  --cycle_;
  const gretl::Int milestone = milestones_[static_cast<size_t>(cycle_)];

  field_shape_displacement_->clear_dual();
  for (auto& p : field_params_) {
    p.clear_dual();
  }

  gretl::Int current_step = checkpointer_->currentStep_;
  while (milestone != current_step) {
    checkpointer_->reverse_state();
    current_step = checkpointer_->currentStep_;
  }

  auto& upstreams = checkpointer_->upstreams_[milestone];

  SLIC_ERROR_IF(field_states_.size() != upstreams.size(), "field states and upstream sizes do not match.");
  // recreate the upstream field states with upstream step, field, and dual values.
  for (size_t s = 0; s < upstreams.size(); ++s) {
    field_states_[s].reset_step(upstreams[s].step_);
    field_states_[s].set(upstreams[s].get<FEFieldPtr>());
    field_states_[s].set_dual(upstreams[s].get_dual<FEDualPtr, FEFieldPtr>());
  }
}

FiniteElementDual DifferentiablePhysics::computeTimestepSensitivity(size_t parameter_index)
{
  return *field_params_[parameter_index].get_dual();
}

const FiniteElementDual& DifferentiablePhysics::computeTimestepShapeSensitivity()
{
  return *field_shape_displacement_->get_dual();
}

const std::unordered_map<std::string, const smith::FiniteElementDual&>
DifferentiablePhysics::computeInitialConditionSensitivity() const
{
  std::unordered_map<std::string, const smith::FiniteElementDual&> map;
  for (auto& name : stateNames()) {
    auto state_index = state_name_to_field_index_.at(name);
    map.insert({name, *initial_field_states_[state_index].get_dual()});
  }
  return map;
}

std::vector<FieldState> DifferentiablePhysics::getFieldStatesAndParamStates() const
{
  std::vector<FieldState> fields;
  fields.insert(fields.end(), field_states_.begin(), field_states_.end());
  fields.insert(fields.end(), field_params_.begin(), field_params_.end());
  return fields;
}

FieldState DifferentiablePhysics::getShapeDispFieldState() const { return *field_shape_displacement_; }

}  // namespace smith
