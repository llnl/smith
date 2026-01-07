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

namespace smith {

/// @brief Helper function to generate the base-physics for solid mechanics
/// @tparam ShapeDispSpace Space for shape displacement, must be H1<1, dim> in most cases
/// @tparam VectorSpace Space for displacement, velocity, acceleration field, typically H1<order, dim>
/// @tparam ...ParamSpaces Additional parameter spaces, either H1<param_order, param_dim> or L1<param_order, param_dim>
/// @tparam dim Spatial dimension
/// @param mesh smith::Mesh
/// @param d_solid_nonlinear_solver Abstract differentiable solver
/// @param time_rule Time integration rule for second order systems.  Likely either quasi-static or implicit Newmark
/// @param num_checkpoints Number of checkpointed states for gretl to store for reverse mode derivatives
/// @param physics_name Name of the physics/WeakForm
/// @param param_names Names for the parameter fields with a one-to-one correspondence with the templated ParamSpaces
/// @return tuple of shared pointers to the: BasePhysics, WeakForm, and DirichetBoundaryConditions
/// @note Only the BasePhysics needs to stay in scope.  The others are returned to the user so they can define the
/// WeakForm integrals, and to specify space and time varying boundary conditions
template <int dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
auto buildSolidMechanics(std::shared_ptr<smith::Mesh> mesh,
                         std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver,
                         smith::SecondOrderTimeIntegrationRule time_rule, size_t num_checkpoints,
                         std::string physics_name, const std::vector<std::string>& param_names = {})
{
  auto graph = std::make_shared<gretl::DataStore>(num_checkpoints);
  auto [shape_disp, states, params, time, solid_mechanics_weak_form] =
      SolidMechanicsStateAdvancer::buildWeakFormAndStates<dim, ShapeDispSpace, VectorSpace, ParamSpaces...>(
          mesh, graph, time_rule, physics_name, param_names);

  auto vector_bcs = std::make_shared<DirichletBoundaryConditions>(
      mesh->mfemParMesh(), space(states[SolidMechanicsStateAdvancer::DISPLACEMENT]));

  auto state_advancer = std::make_shared<SolidMechanicsStateAdvancer>(d_solid_nonlinear_solver, vector_bcs,
                                                                      solid_mechanics_weak_form, time_rule);

  auto physics = std::make_shared<DifferentiablePhysics>(mesh, graph, shape_disp, states, params, state_advancer,
                                                         physics_name, std::vector<std::string>{"reactions"});

  return std::make_tuple(physics, solid_mechanics_weak_form, vector_bcs);
}

}  // namespace smith
