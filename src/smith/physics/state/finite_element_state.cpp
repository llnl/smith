// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/state/finite_element_state.hpp"

#include "smith/numerics/functional/domain.hpp"
#include "smith/infrastructure/logger.hpp"

namespace smith {

void FiniteElementState::project(mfem::VectorCoefficient& coef, mfem::Array<int>& dof_list)
{
  mfem::ParGridFunction& grid_function = gridFunction();
  grid_function.ProjectCoefficient(coef, dof_list);
  setFromGridFunction(grid_function);
}

void FiniteElementState::project(mfem::Coefficient& coef, mfem::Array<int>& dof_list, std::optional<int> component)
{
  mfem::ParGridFunction& grid_function = gridFunction();

  if (component) {
    grid_function.ProjectCoefficient(coef, dof_list, *component);
  } else {
    grid_function.ProjectCoefficient(coef, dof_list);
  }

  setFromGridFunction(grid_function);
}

void FiniteElementState::project(const GeneralCoefficient& coef)
{
  mfem::ParGridFunction& grid_function = gridFunction();

  // The generic lambda parameter, auto&&, allows the component type (mfem::Coef or mfem::VecCoef)
  // to be deduced, and the appropriate version of ProjectCoefficient is dispatched.
  visit(
      [this, &grid_function](auto&& concrete_coef) {
        grid_function.ProjectCoefficient(*concrete_coef);
        setFromGridFunction(grid_function);
      },
      coef);
}

void FiniteElementState::project(mfem::Coefficient& coef)
{
  mfem::ParGridFunction& grid_function = gridFunction();
  grid_function.ProjectCoefficient(coef);
  setFromGridFunction(grid_function);
}

void FiniteElementState::project(mfem::VectorCoefficient& coef)
{
  mfem::ParGridFunction& grid_function = gridFunction();
  grid_function.ProjectCoefficient(coef);
  setFromGridFunction(grid_function);
}

void FiniteElementState::projectOnBoundary(mfem::Coefficient& coef, const mfem::Array<int>& markers)
{
  mfem::ParGridFunction& grid_function = gridFunction();
  // markers should be const param in mfem, but it's not
  grid_function.ProjectBdrCoefficient(coef, const_cast<mfem::Array<int>&>(markers));
  setFromGridFunction(grid_function);
}

void FiniteElementState::projectOnBoundary(mfem::VectorCoefficient& coef, const mfem::Array<int>& markers)
{
  mfem::ParGridFunction& grid_function = gridFunction();
  // markers should be const param in mfem, but it's not
  grid_function.ProjectBdrCoefficient(coef, const_cast<mfem::Array<int>&>(markers));
  setFromGridFunction(grid_function);
}

void FiniteElementState::project(mfem::Coefficient& coef, const Domain& domain)
{
  mfem::Array<int> uniq_dof_ids = domain.dof_list(gridFunction().ParFESpace());
  mfem::ParGridFunction& grid_function = gridFunction();
  grid_function.ProjectCoefficient(coef, uniq_dof_ids);
  setFromGridFunction(grid_function);
}

void FiniteElementState::project(mfem::VectorCoefficient& coef, const Domain& domain)
{
  mfem::Array<int> uniq_dof_ids = domain.dof_list(gridFunction().ParFESpace());
  mfem::ParGridFunction& grid_function = gridFunction();
  grid_function.ProjectCoefficient(coef, uniq_dof_ids);
  setFromGridFunction(grid_function);
}

mfem::ParGridFunction& FiniteElementState::gridFunction() const
{
  if (!grid_func_) {
    grid_func_ = std::make_unique<mfem::ParGridFunction>(space_.get());
  }

  fillGridFunction(*grid_func_);
  return *grid_func_;
}

double norm(const FiniteElementState& state, const double p)
{
  if (state.space().GetVDim() == 1) {
    mfem::ConstantCoefficient zero(0.0);
    return state.gridFunction().ComputeLpError(p, zero);
  } else {
    mfem::Vector zero(state.space().GetVDim());
    zero = 0.0;
    mfem::VectorConstantCoefficient zerovec(zero);
    return state.gridFunction().ComputeLpError(p, zerovec);
  }
}

double computeL2Error(const FiniteElementState& state, mfem::VectorCoefficient& exact_solution)
{
  return state.gridFunction().ComputeL2Error(exact_solution);
}

double computeL2Error(const FiniteElementState& state, mfem::Coefficient& exact_solution)
{
  return state.gridFunction().ComputeL2Error(exact_solution);
}

}  // namespace smith
