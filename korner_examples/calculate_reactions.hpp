#pragma once
#include <iostream>

#include "gretl/data_store.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/functional_objective.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"

using namespace smith;

double CalculateReaction(FiniteElementDual& reactions, std::shared_ptr<smith::Mesh> mesh, std::string domain_name,
                         const int direction)
{
  FiniteElementState reactionDirections(reactions.space(), "reaction_directions");
  const int dim = mesh->mfemParMesh().Dimension();

  reactionDirections = 0.0;
  mfem::VectorFunctionCoefficient func(dim, [direction](const mfem::Vector& /*x*/, mfem::Vector& u) {
    u = 0.0;
    u[direction] = 1.0;
  });

  reactionDirections.project(func, mesh->domain(domain_name));

  return innerProduct(reactions, reactionDirections);
}
