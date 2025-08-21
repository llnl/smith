// Copyright (c), Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file serac_mechanics.hpp
 *
 * @brief Implementation of BasePhysics which uses FieldStates and gretl to track the computational graph, dynamically
 * checkpoint, and backpropagate sensitivities.
 */

#pragma once

#include "serac/physics/base_physics.hpp"
#include "serac/gretl/data_store.hpp"
#include "serac/differentiable_numerics/field_state.hpp"
#include "serac/differentiable_numerics/timestep_estimator.hpp"
#include <vector>
#include <map>

namespace serac {

class Mesh;
class WeakForm;
class DifferentiableSolver;
class StateAdvancer;

class Mechanics : public BasePhysics {
 public:
  Mechanics(std::shared_ptr<Mesh> mesh, std::shared_ptr<gretl::DataStore> graph, std::shared_ptr<WeakForm> res,
            const FieldState& shape_disp, const std::vector<FieldState>& states, const std::vector<FieldState>& params,
            std::shared_ptr<StateAdvancer> advancer, std::shared_ptr<TimestepEstimator> dt_estimate);
  ~Mechanics() {}

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
  void setAdjointLoad(std::unordered_map<std::string, const serac::FiniteElementDual&> string_to_dual) override;

  /// @overload
  const FiniteElementState& adjoint(const std::string& adjoint_name) const override;

  /// @brief Initialize time integrator
  virtual void initializationStep() override;

  /// @brief Compute adjoint sensitivity corresponding to time integrator initialization
  virtual void reverseAdjointInitializationStep() override;

  /// @overload
  virtual void advanceTimestep(double dt) override;

  /// @overload
  void reverseAdjointTimestep() override;

  /// @overload
  FiniteElementDual computeTimestepSensitivity(size_t parameter_index) override;

  /// @overload
  const FiniteElementDual& computeTimestepShapeSensitivity() override;

  /// @overload
  const std::unordered_map<std::string, const serac::FiniteElementDual&> computeInitialConditionSensitivity()
      const override;

  /// @brief Get all the fields, states first, parameters next
  /// @return vector of all states
  std::vector<FieldState> getAllFieldStates() const;
  FieldState getShapeDispFieldState() const;

  std::shared_ptr<gretl::DataStore> checkpointer_;
  std::shared_ptr<WeakForm> residual_;
  std::shared_ptr<StateAdvancer> advancer_;
  std::shared_ptr<TimestepEstimator> dt_estimator_;

  std::vector<FieldState> initial_field_states_;
  std::vector<FieldState> field_states_;
  std::vector<FieldState> field_params_;
  std::unique_ptr<FieldState> field_shape_displacement_;

  std::map<std::string, size_t> state_name_to_field_index_;
  std::map<std::string, size_t> param_name_to_field_index_;

  std::vector<gretl::Int> milestones_;
};

}  // namespace serac
