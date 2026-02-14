// Copyright (c) Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_solid_mechanics.hpp
 *
 */

#pragma once

#include <memory>
#include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"

namespace smith {

/// @brief Helper to expand parameter spaces and build weak form
template <int dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces, size_t... Is>
auto buildSolidMechanicsHelper(std::shared_ptr<smith::Mesh> mesh, std::shared_ptr<FieldStore> field_store,
                               std::shared_ptr<DifferentiableBlockSolver> d_solid_nonlinear_solver,
                               smith::ImplicitNewmarkSecondOrderTimeIntegrationRule time_rule, std::string physics_name,
                               const std::vector<std::string>& param_names, std::index_sequence<Is...>)
{
  auto [solid_weak_form, cycle_zero_weak_form] =
      SolidMechanicsStateAdvancer::buildWeakFormAndStates<dim, ShapeDispSpace, VectorSpace, ParamSpaces...>(
          mesh, field_store, time_rule, physics_name,
          FieldType<ParamSpaces>(physics_name + "_param_" + param_names[Is])...);

  auto state_advancer = std::make_shared<SolidMechanicsStateAdvancer>(field_store, solid_weak_form, cycle_zero_weak_form,
                                                                      d_solid_nonlinear_solver);

  auto vector_bcs = field_store->getBoundaryConditions(0);  // Displacement was the first independent field added

  // For DifferentiablePhysics, we need the vectors of fields
  std::vector<FieldState> states = {field_store->getField(physics_name + "_displacement"),
                                    field_store->getField(physics_name + "_displacement_old"),
                                    field_store->getField(physics_name + "_velocity_old"),
                                    field_store->getField(physics_name + "_acceleration_old")};

  std::vector<FieldState> params;
  for (size_t i = 0; i < sizeof...(ParamSpaces); ++i) {
    params.push_back(field_store->getField(physics_name + "_param_" + param_names[i]));
  }

  auto physics = std::make_shared<DifferentiablePhysics>(mesh, field_store->graph(), field_store->getShapeDisp(), states,
                                                         params, state_advancer, physics_name,
                                                         std::vector<std::string>{"reactions"});

  return std::make_tuple(physics, solid_weak_form, cycle_zero_weak_form, vector_bcs);
}

/// @brief Helper function to generate the base-physics for solid mechanics
/// @tparam ShapeDispSpace Space for shape displacement, must be H1<1, dim> in most cases
/// @tparam VectorSpace Space for displacement, velocity, acceleration field, typically H1<order, dim>
/// @tparam ...ParamSpaces Additional parameter spaces, either H1<param_order, param_dim> or L1<param_order, param_dim>
/// @tparam dim Spatial dimension
/// @param mesh smith::Mesh
/// @param d_solid_nonlinear_solver Abstract differentiable block solver
/// @param time_rule Time integration rule for second order systems.  Likely either quasi-static or implicit Newmark
/// @param num_checkpoints Number of checkpointed states for gretl to store for reverse mode derivatives
/// @param physics_name Name of the physics/WeakForm
/// @param param_names Names for the parameter fields with a one-to-one correspondence with the templated ParamSpaces
/// @return tuple of shared pointers to the: BasePhysics, WeakForm, and DirichetBoundaryConditions
template <int dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
auto buildSolidMechanics(std::shared_ptr<smith::Mesh> mesh,
                         std::shared_ptr<DifferentiableBlockSolver> d_solid_nonlinear_solver,
                         smith::ImplicitNewmarkSecondOrderTimeIntegrationRule time_rule, size_t num_checkpoints,
                         std::string physics_name, const std::vector<std::string>& param_names = {})
{
  auto field_store = std::make_shared<FieldStore>(mesh, num_checkpoints);

  return buildSolidMechanicsHelper<dim, ShapeDispSpace, VectorSpace, ParamSpaces...>(
      mesh, field_store, d_solid_nonlinear_solver, time_rule, physics_name, param_names,
      std::make_index_sequence<sizeof...(ParamSpaces)>{});
}

}  // namespace smith
