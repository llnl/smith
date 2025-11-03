// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/solid_mechanics_input.hpp"

#include <map>
#include <utility>

#include "smith/numerics/equation_solver.hpp"

namespace smith {

void SolidMechanicsInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // interpolation order - currently up to 3rd order is allowed
  container.addInt("order", "polynomial order of the basis functions.").defaultValue(1).range(1, 3);

  auto& material_container = container.addStructArray("materials", "Container for array of materials");
  SolidMaterialInputOptions::defineInputFileSchema(material_container);

  auto& equation_solver_container =
      container.addStruct("equation_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  EquationSolver::defineInputFileSchema(equation_solver_container);

  auto& dynamics_container = container.addStruct("dynamics", "Parameters for mass matrix inversion");
  dynamics_container.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_container.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_container = container.addStructDictionary("boundary_conds", "Container of boundary conditions");
  input::BoundaryConditionInputOptions::defineInputFileSchema(bc_container);

  auto& init_displ = container.addStruct("initial_displacement", "Coefficient for initial condition");
  input::CoefficientInputOptions::defineInputFileSchema(init_displ);
  auto& init_velo = container.addStruct("initial_velocity", "Coefficient for initial condition");
  input::CoefficientInputOptions::defineInputFileSchema(init_velo);
}

}  // namespace smith

smith::SolidMechanicsInputOptions FromInlet<smith::SolidMechanicsInputOptions>::operator()(
    const axom::inlet::Container& base)
{
  smith::SolidMechanicsInputOptions result;

  result.order = base["order"];

  // Solver parameters
  auto equation_solver = base["equation_solver"];
  result.lin_solver_options = equation_solver["linear"].get<smith::LinearSolverOptions>();
  result.nonlin_solver_options = equation_solver["nonlinear"].get<smith::NonlinearSolverOptions>();

  if (base.contains("dynamics")) {
    smith::TimesteppingOptions timestepping_options;
    auto dynamics = base["dynamics"];

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, smith::TimestepMethod> timestep_methods = {
        {"AverageAcceleration", smith::TimestepMethod::AverageAcceleration},
        {"NewmarkBeta", smith::TimestepMethod::Newmark},
        {"BackwardEuler", smith::TimestepMethod::BackwardEuler}};
    std::string timestep_method = dynamics["timestepper"];
    SLIC_ERROR_ROOT_IF(timestep_methods.count(timestep_method) == 0,
                       "Unrecognized timestep method: " << timestep_method);
    timestepping_options.timestepper = timestep_methods.at(timestep_method);

    // FIXME: Implement all supported methods as part of an ODE schema
    const static std::map<std::string, smith::DirichletEnforcementMethod> enforcement_methods = {
        {"RateControl", smith::DirichletEnforcementMethod::RateControl}};
    std::string enforcement_method = dynamics["enforcement_method"];
    SLIC_ERROR_ROOT_IF(enforcement_methods.count(enforcement_method) == 0,
                       "Unrecognized enforcement method: " << enforcement_method);
    timestepping_options.enforcement_method = enforcement_methods.at(enforcement_method);

    result.timestepping_options = std::move(timestepping_options);
  }

  result.materials = base["materials"].get<std::vector<smith::var_solid_material_t>>();

  if (base.contains("boundary_conds")) {
    result.boundary_conditions =
        base["boundary_conds"].get<std::unordered_map<std::string, smith::input::BoundaryConditionInputOptions>>();
  }

  if (base.contains("initial_displacement")) {
    result.initial_displacement = base["initial_displacement"].get<smith::input::CoefficientInputOptions>();
  }
  if (base.contains("initial_velocity")) {
    result.initial_velocity = base["initial_velocity"].get<smith::input::CoefficientInputOptions>();
  }
  return result;
}
