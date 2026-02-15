// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_time_integrator.hpp
 *
 * @brief Specifies parameterized residuals and various linearized evaluations for arbitrary nonlinear systems of
 * equations
 */

#pragma once

#include "smith/smith_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"

namespace smith {

class DifferentiableBlockSolver;
class DirichletBoundaryConditions;

/// @brief Implementation of the StateAdvancer interface for advancing the solution of solid mechanics problems
class SolidMechanicsTimeIntegrator : public StateAdvancer {
 public:
  /**
   * @brief Construct a new SolidMechanicsTimeIntegrator object.
   * @param field_store Field store containing the fields.
   * @param solid_weak_form Primary solid mechanics weak form.
   * @param cycle_zero_weak_form Weak form for initial acceleration solve at cycle=0.
   * @param solver The block solver to use.
   */
  SolidMechanicsTimeIntegrator(std::shared_ptr<FieldStore> field_store, std::shared_ptr<WeakForm> solid_weak_form,
                              std::shared_ptr<WeakForm> cycle_zero_weak_form,
                              std::shared_ptr<smith::DifferentiableBlockSolver> solver);

  /// State enum for indexing convenience (deprecated, use FieldType instead)
  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION
  };

  /**
   * @brief Utility function to consistently construct all the weak forms and FieldStates for a solid mechanics
   * application.
   */
  template <int spatial_dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
  static auto buildWeakFormAndStates([[maybe_unused]] std::shared_ptr<Mesh> mesh, std::shared_ptr<FieldStore> field_store,
                                     ImplicitNewmarkSecondOrderTimeIntegrationRule time_rule, std::string physics_name,
                                     FieldType<ParamSpaces>... parameter_types)
  {
    // Add shape displacement
    FieldType<ShapeDispSpace> shape_disp_type(physics_name + "_shape_displacement");
    field_store->addShapeDisp(shape_disp_type);

    // Add displacement as independent (unknown) with time integration rule
    auto time_rule_ptr = std::make_shared<ImplicitNewmarkSecondOrderTimeIntegrationRule>(time_rule);
    FieldType<VectorSpace> disp_type(physics_name + "_displacement");
    field_store->addIndependent(disp_type, time_rule_ptr);

    // Add dependent fields for time integration history
    auto disp_old_type =
        field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE, physics_name + "_displacement_old");
    auto velo_old_type =
        field_store->addDependent(disp_type, FieldStore::TimeDerivative::DOT, physics_name + "_velocity_old");
    auto accel_old_type =
        field_store->addDependent(disp_type, FieldStore::TimeDerivative::DDOT, physics_name + "_acceleration_old");

    // Add parameters
    (field_store->addParameter(parameter_types), ...);

    // Create solid mechanics weak form (u, u_old, v_old, a_old)
    field_store->addWeakFormTestField(physics_name, disp_type.name);
    const mfem::ParFiniteElementSpace& test_space = field_store->getField(disp_type.name).get()->space();
    std::vector<const mfem::ParFiniteElementSpace*> input_spaces;
    createSpaces(physics_name, *field_store, input_spaces, 0, disp_type, disp_old_type, velo_old_type, accel_old_type,
                 parameter_types...);

    using SolidWeakFormType = TimeDiscretizedWeakForm<
        spatial_dim, VectorSpace,
        Parameters<VectorSpace, VectorSpace, VectorSpace, VectorSpace, ParamSpaces...>>;

    auto solid_weak_form = std::make_shared<SolidWeakFormType>(
        physics_name, field_store->getMesh(), test_space, input_spaces);

    // Create cycle-zero weak form (u, v, a) for initial acceleration solve at cycle=0
    auto cycle_zero_weak_form = createWeakForm<spatial_dim>(physics_name + "_reaction", accel_old_type, *field_store,
                                                            disp_type, velo_old_type, accel_old_type, parameter_types...);

    return std::make_tuple(solid_weak_form, cycle_zero_weak_form);
  }

  /// @overload
  std::pair<std::vector<FieldState>, std::vector<ReactionState>> advanceState(
      const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
      const std::vector<FieldState>& params) const override;

 private:
  std::shared_ptr<FieldStore> field_store_;
  std::shared_ptr<WeakForm> cycle_zero_weak_form_;
  std::shared_ptr<DifferentiableBlockSolver> solver_;
  std::shared_ptr<MultiphysicsTimeIntegrator> integrator_;
};

}  // namespace smith
