// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_physics.hpp
 *
 * @brief Defines a `BasePhysics` implementation backed by `FieldState` objects and a gretl computational graph.
 */

#pragma once

#include "gretl/data_store.hpp"
#include "smith/physics/base_physics.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include <vector>
#include <map>

namespace smith {

class Mesh;
class WeakForm;
class StateAdvancer;
class TimestepEstimator;
class Reaction;

/// @brief `BasePhysics` implementation that stores differentiable states in gretl for checkpointed reverse solves.
class DifferentiablePhysics : public BasePhysics {
 public:
  /**
   * @brief Construct a differentiable physics wrapper around a state advancer and its tracked fields.
   *
   * @param mesh The mesh shared by all tracked fields.
   * @param graph The gretl data store used to record forward operations and replay them in reverse.
   * @param shape_disp The shape displacement field held fixed during a forward solve.
   * @param states The primal state fields advanced by `advancer`.
   * @param params The parameter fields treated as differentiable inputs.
   * @param advancer The algorithm that advances the tracked state fields by one timestep.
   * @param physics_name The `BasePhysics` name for this module.
   * @param reaction_names Names of any differentiable reaction outputs produced during `advanceTimestep`.
   */
  DifferentiablePhysics(std::shared_ptr<Mesh> mesh, std::shared_ptr<gretl::DataStore> graph,
                        const FieldState& shape_disp, const std::vector<FieldState>& states,
                        const std::vector<FieldState>& params, std::shared_ptr<StateAdvancer> advancer,
                        std::string physics_name, const std::vector<std::string>& reaction_names = {});
  /// @brief Destructor.
  ~DifferentiablePhysics() {}

  /// @copydoc smith::BasePhysics::resetStates(int, double)
  void resetStates(int cycle = 0, double time = 0.0) override;

  /// @copydoc smith::BasePhysics::resetAdjointStates()
  virtual void resetAdjointStates() override;

  /// @copydoc smith::BasePhysics::completeSetup()
  void completeSetup() override;

  /// @copydoc smith::BasePhysics::stateNames() const
  std::vector<std::string> stateNames() const override;

  /// @copydoc smith::BasePhysics::parameterNames() const
  std::vector<std::string> parameterNames() const override;

  /// @copydoc smith::BasePhysics::dualNames() const
  std::vector<std::string> dualNames() const override;

  /// @copydoc smith::BasePhysics::state(const std::string&) const
  const FiniteElementState& state(const std::string& state_name) const override;

  /// @copydoc smith::BasePhysics::dual(const std::string&) const
  const FiniteElementDual& dual(const std::string& dual_name) const override;

  /// @copydoc smith::BasePhysics::shapeDisplacement() const
  const FiniteElementState& shapeDisplacement() const override;

  /// @copydoc smith::BasePhysics::parameter(std::size_t) const
  const FiniteElementState& parameter(std::size_t parameter_index) const override;

  /// @copydoc smith::BasePhysics::parameter(const std::string&) const
  const FiniteElementState& parameter(const std::string& parameter_name) const override;

  /**
   * @brief Return a state for a stored checkpoint cycle.
   *
   * @param state_name Name of the state to retrieve.
   * @param cycle Cycle index to load.
   * @return A copy of the requested state at the requested cycle.
   *
   * @note This implementation only supports the current cycle stored in the gretl checkpoint state.
   */
  FiniteElementState loadCheckpointedState(const std::string& state_name, int cycle) override;

  /// @copydoc smith::BasePhysics::setState(const std::string&, const FiniteElementState&)
  void setState(const std::string& state_name, const FiniteElementState& s) override;

  /// @copydoc smith::BasePhysics::setShapeDisplacement(const FiniteElementState&)
  void setShapeDisplacement(const FiniteElementState& shape_displacement) override;

  /// @copydoc smith::BasePhysics::setParameter(const size_t, const FiniteElementState&)
  void setParameter(const size_t parameter_index, const FiniteElementState& parameter_state) override;

  /// @copydoc smith::BasePhysics::setAdjointLoad(std::unordered_map<std::string, const smith::FiniteElementDual&>)
  void setAdjointLoad(std::unordered_map<std::string, const smith::FiniteElementDual&> string_to_dual) override;

  /// @copydoc smith::BasePhysics::setDualAdjointBcs(std::unordered_map<std::string, const smith::FiniteElementState&>)
  void setDualAdjointBcs(std::unordered_map<std::string, const smith::FiniteElementState&> string_to_bc) override;

  /// @copydoc smith::BasePhysics::adjoint(const std::string&) const
  const FiniteElementState& adjoint(const std::string& adjoint_name) const override;

  /// @copydoc smith::BasePhysics::advanceTimestep(double)
  virtual void advanceTimestep(double dt) override;

  /**
   * @brief Reverse one recorded timestep through the gretl graph.
   *
   * Restores the primal state values and adjoint values at the start of the just-reversed step.
   */
  void reverseAdjointTimestep() override;

  /// @copydoc smith::BasePhysics::computeTimestepSensitivity(size_t)
  FiniteElementDual computeTimestepSensitivity(size_t parameter_index) override;

  /// @copydoc smith::BasePhysics::computeTimestepShapeSensitivity()
  const FiniteElementDual& computeTimestepShapeSensitivity() override;

  /// @copydoc smith::BasePhysics::computeInitialConditionSensitivity() const
  const std::unordered_map<std::string, const smith::FiniteElementDual&> computeInitialConditionSensitivity()
      const override;

  /// @brief Get the initial state fields captured before any timesteps were advanced.
  /// @return Copies of the tracked initial state fields.
  std::vector<FieldState> getInitialFieldStates() const { return initial_field_states_; }

  /// @brief Get the current primal state fields.
  /// @return Copies of the tracked state fields at the current cycle.
  std::vector<FieldState> getFieldStates() const { return field_states_; }

  /// @brief Get all the parameter FieldStates
  std::vector<FieldState> getFieldParams() const { return field_params_; }

  /// @brief Get the tracked state fields followed by the tracked parameter fields.
  /// @return A concatenated vector of state fields then parameter fields.
  std::vector<FieldState> getFieldStatesAndParamStates() const;

  /// @brief Get the tracked shape displacement field.
  /// @return A copy of the shape displacement `FieldState`.
  FieldState getShapeDispFieldState() const;

  /// @brief Get the current differentiable reaction outputs.
  /// @return Copies of the current reaction states produced by the last forward advance.
  std::vector<ReactionState> getReactionStates() const { return reaction_states_; }

  /// @brief Get the state advancer used for forward solves.
  /// @return The shared state advancer.
  std::shared_ptr<StateAdvancer> getStateAdvancer() const { return advancer_; }

 private:
  std::shared_ptr<gretl::DataStore> checkpointer_;  ///< gretl data store manages dynamic checkpointing logic
  std::shared_ptr<StateAdvancer> advancer_;  ///< abstract interface for advancing state from one cycle to the next

  std::vector<FieldState> initial_field_states_;  ///< hold a copy of the initial states, mostly to have a record of
                                                  ///< initial condition sensitivities
  std::vector<FieldState> field_states_;          ///< all the states that may be changed by the StateAdvancer
  std::vector<FieldState> field_params_;  ///< all the parameters which should not be changed by the StateAdvancer
  std::unique_ptr<FieldState>
      field_shape_displacement_;  ///< shape displacement which is also fixed for a given simulation

  std::map<std::string, size_t> state_name_to_field_index_;  ///< map from state names to field index
  std::map<std::string, size_t> param_name_to_field_index_;  ///< map from param names to param index
  std::vector<std::string> state_names_;                     ///< names of all the states in order
  std::vector<std::string> param_names_;                     ///< names of all the states in order

  mutable std::vector<ReactionState> reaction_states_;             ///< all the reactions registered for the physics
  std::map<std::string, size_t> reaction_name_to_reaction_index_;  ///< map from reaction names to reaction index
  std::vector<std::string> reaction_names_;                        ///< names for all the relevant reactions/reactions

  std::vector<gretl::Int> milestones_;  ///< a record of the steps in the graph that represent the end conditions of
                                        ///< advanceTimestep(dt). this information is used to halt the gretl graph when
                                        ///< back-propagating to allow users of reverseAdjointTimestep to specify
                                        ///< adjoint loads and to retrieve timestep sensitivity information.

  double time_prev_ =
      0.0;  ///< previous time, saved to reconstruct the start of step time used in computing reaction forces
  double dt_prev_ =
      0.0;  ///< previous time increment, saved to reconstruct the start of step time used in computing reaction forces
  int cycle_prev_ =
      0;  ///< previous cycle, saved to reconstruct the start of step time used in computing reaction forces
};

}  // namespace smith
