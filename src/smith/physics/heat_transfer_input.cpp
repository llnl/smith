// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/heat_transfer_input.hpp"

#include <map>

#include "smith/numerics/equation_solver.hpp"

namespace smith {

void HeatTransferInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Polynomial interpolation order - currently up to 8th order is allowed
  container.addInt("order", "Order degree of the finite elements.").defaultValue(1).range(1, 8);

  auto& material_container = container.addStructArray("materials", "Container for array of materials");
  ThermalMaterialInputOptions::defineInputFileSchema(material_container);

  auto& source = container.addStruct("source", "Scalar source term (RHS of the heat transfer PDE)");
  input::CoefficientInputOptions::defineInputFileSchema(source);

  auto& equation_solver_container =
      container.addStruct("equation_solver", "Linear and Nonlinear stiffness Solver Parameters.");
  EquationSolver::defineInputFileSchema(equation_solver_container);

  auto& dynamics_container = container.addStruct("dynamics", "Parameters for mass matrix inversion");
  dynamics_container.addString("timestepper", "Timestepper (ODE) method to use");
  dynamics_container.addString("enforcement_method", "Time-varying constraint enforcement method to use");

  auto& bc_container = container.addStructDictionary("boundary_conds", "Container of boundary conditions");
  input::BoundaryConditionInputOptions::defineInputFileSchema(bc_container);

  auto& init_temp = container.addStruct("initial_temperature", "Coefficient for initial condition");
  input::CoefficientInputOptions::defineInputFileSchema(init_temp);
}

}  // namespace smith

smith::HeatTransferInputOptions FromInlet<smith::HeatTransferInputOptions>::operator()(
    const axom::inlet::Container& base)
{
  smith::HeatTransferInputOptions result;

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
        {"BackwardEuler", smith::TimestepMethod::BackwardEuler},
        {"ForwardEuler", smith::TimestepMethod::ForwardEuler}};
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

    result.timestepping_options = timestepping_options;
  }

  if (base.contains("source")) {
    result.source_coef = base["source"].get<smith::input::CoefficientInputOptions>();
  }

  result.materials = base["materials"].get<std::vector<smith::var_thermal_material_t>>();

  result.boundary_conditions =
      base["boundary_conds"].get<std::unordered_map<std::string, smith::input::BoundaryConditionInputOptions>>();

  if (base.contains("initial_temperature")) {
    result.initial_temperature = base["initial_temperature"].get<smith::input::CoefficientInputOptions>();
  }
  return result;
}
