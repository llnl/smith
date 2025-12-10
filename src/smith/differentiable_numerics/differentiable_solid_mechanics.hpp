// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_solid_mechanics.hpp
 * .hpp
 *
 */

#pragma once

#include <memory>
#include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"

namespace smith {

template <int dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
auto buildSolidMechanics(std::shared_ptr<smith::Mesh> mesh,
                         std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver,
                         smith::SecondOrderTimeIntegrationRule time_rule, std::string physics_name,
                         const std::vector<std::string>& param_names = {})
{
  auto graph = std::make_shared<gretl::DataStore>(100);
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