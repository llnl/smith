// Copyright (c), Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_physics.hpp
 *
 * @brief Implementation of BasePhysics which uses FieldStates and gretl to track the computational graph, dynamically
 * checkpoint, and backpropagate sensitivities.
 */

#pragma once

#include "smith/physics/base_physics.hpp"
#include "smith/gretl/data_store.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include <vector>
#include <map>

namespace smith {

class Mesh;
class WeakForm;
class DifferentiableSolver;
class StateAdvancer;
class TimestepEstimator;

/// @brief Implementation of BasePhysics which uses FieldStates and gretl to track the computational graph, dynamically
/// checkpoint, and back-propagate sensitivities.
class DifferentiablePhysics : public BasePhysics {
 public:
  /// @brief constructor
  DifferentiablePhysics(std::shared_ptr<Mesh> mesh, std::shared_ptr<gretl::DataStore> graph,
                        const FieldState& shape_disp, const std::vector<FieldState>& states,
                        const std::vector<FieldState>& params, std::shared_ptr<StateAdvancer> advancer,
                        std::shared_ptr<TimestepEstimator> dt_estimate, std::string mech_name);
  /// @brief destructor
  ~DifferentiablePhysics() {}

  /// overload
  void resetStates(int cycle = 0, double time = 0.0) override;

  /// overload
  virtual void resetAdjointStates() override;

  /// @overload
  void completeSetup() override;

  /// @overload
  std::vector<std::string> stateNames() const override;

  /// @overload
  std::vector<std::string> parameterNames() const override;

  /// @overload
  const FiniteElementState& state(const std::string& state_name) const override;

  /// @overload
  FiniteElementState loadCheckpointedState(const std::string& state_name, int cycle) override;

  /// @overload
  const FiniteElementState& shapeDisplacement() const override;

  /// @overload
  const FiniteElementState& parameter(std::size_t parameter_index) const override;

  /// @overload
  const FiniteElementState& parameter(const std::string& parameter_name) const override;

  /// @overload
  void setState(const std::string& state_name, const FiniteElementState& s) override;

  /// @overload
  void setShapeDisplacement(const FiniteElementState& s) override;

  /// @overload
  void setParameter(const size_t parameter_index, const FiniteElementState& parameter_state) override;

  /// @overload
  void setAdjointLoad(std::unordered_map<std::string, const smith::FiniteElementDual&> string_to_dual) override;

  /// @overload
  const FiniteElementState& adjoint(const std::string& adjoint_name) const override;

  /// @overload
  virtual void advanceTimestep(double dt) override;

  /// @overload
  void reverseAdjointTimestep() override;

  /// @overload
  FiniteElementDual computeTimestepSensitivity(size_t parameter_index) override;

  /// @overload
  const FiniteElementDual& computeTimestepShapeSensitivity() override;

  /// @overload
  const std::unordered_map<std::string, const smith::FiniteElementDual&> computeInitialConditionSensitivity()
      const override;

  /// @brief Get all the FieldStates... states first, parameters next
  std::vector<FieldState> getAllFieldStates() const;

  /// @brief Get the shape displacement FieldState
  FieldState getShapeDispFieldState() const;

  std::shared_ptr<gretl::DataStore> checkpointer_;  ///< gretl data store manages dynamic checkpointing logic
  std::shared_ptr<StateAdvancer> advancer_;  ///< abstract interface for advancing state from one cycle to the next
  std::shared_ptr<TimestepEstimator>
      dt_estimator_;  ///< abstract interface for estimating stable timestep from the current states

  std::vector<FieldState> initial_field_states_;  ///< hold a copy of the initial states, mostly to have a record of
                                                  ///< initial condition sensitivities
  std::vector<FieldState> field_states_;          ///< all the states that may be changed by the StateAdvancer
  std::vector<FieldState> field_params_;  ///< all the parameters which should not be changed by the StateAdvancer
  std::unique_ptr<FieldState>
      field_shape_displacement_;  ///< shape displacement which is also fixed for a given simulation

  std::map<std::string, size_t> state_name_to_field_index_;  ///< map from state names to field index
  std::map<std::string, size_t> param_name_to_field_index_;  ///< map from param names to param index
  std::vector<std::string> state_names;                      ///< names of all the states in order
  std::vector<std::string> param_names;                      ///< names of all the states in order

  std::vector<gretl::Int> milestones_;  ///< a record of the steps in the graph that represent the end conditions of
                                        ///< advanceTimestep(dt). this information is used to halt the gretl graph when
                                        ///< back-propagating to allow users of reverseAdjointTimestep to specify
                                        ///< adjoint loads and to retrieve timestep sensitivity information.

  size_t sub_cycle_;  ///< counts all calls to state advancer's advance step (on the forward pass)
};

}  // namespace smith
