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
#include "smith/differentiable_numerics/differentiable_solver.hpp"

using namespace smith;

double CalculateReaction(FiniteElementDual& reactions, const int face, const int direction)
{
  auto fespace = reactions.space();

  mfem::ParGridFunction mask_gf(&fespace);
  mask_gf = 0.0;
  auto fhan = [direction](const mfem::Vector&, mfem::Vector& y) {
    y = 0.0;
    y[direction] = 1.0;
  };

  mfem::VectorFunctionCoefficient boundary_coeff(3, fhan);
  mfem::Array<int> ess_bdr(fespace.GetParMesh()->bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[face] = 1;
  mask_gf.ProjectBdrCoefficient(boundary_coeff, ess_bdr);

  mfem::Vector projections = reactions;
  mfem::Vector mask_t(fespace.GetTrueVSize());
  mask_t = 0.0;
  fespace.GetRestrictionMatrix()->Mult(mask_gf, mask_t);
  projections *= mask_t;

  double local_sum = projections.Sum();
  MPI_Comm mycomm = fespace.GetParMesh()->GetComm();
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mycomm);

  return global_sum;
}
