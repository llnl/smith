import os
import glob
import re

tests = glob.glob('src/smith/differentiable_numerics/tests/test_*.cpp')
for t in tests:
    with open(t, 'r') as file:
        content = file.read()
    
    # Needs to include SystemSolver
    if '#include "smith/differentiable_numerics/system_solver.hpp"' not in content:
        content = content.replace('#include "smith/differentiable_numerics/differentiable_solver.hpp"', 
                                  '#include "smith/differentiable_numerics/differentiable_solver.hpp"\n#include "smith/differentiable_numerics/system_solver.hpp"')
    
    # We will search for all 'buildXSystem' and try to replace the solver argument.
    # It's better to just do it via regex
    # Usually it's `buildXYZ(..., solver, ...)`
    
    # In test_solid_dynamics.cpp
    content = content.replace(
        'auto system = buildSolidDynamicsSystem<dim, order>(\n      mesh, d_solid_nonlinear_solver,',
        'auto system_solver = std::make_shared<SystemSolver>(1e-8, 1);\n  system_solver->addStage({0}, d_solid_nonlinear_solver);\n  auto system = buildSolidDynamicsSystem<dim, order>(\n      mesh, system_solver,'
    )
    content = content.replace(
        'auto system = buildSolidDynamicsSystem<dim, order>(mesh, d_solid_nonlinear_solver,',
        'auto system_solver = std::make_shared<SystemSolver>(1e-8, 1);\n  system_solver->addStage({0}, d_solid_nonlinear_solver);\n  auto system = buildSolidDynamicsSystem<dim, order>(mesh, system_solver,'
    )

    # In test_solid_static_with_internal_vars.cpp
    content = content.replace(
        'buildSolidStaticsWithL2StateSystem<dim, disp_order, StateSpace>(mesh, solver,',
        'auto sys_solver = std::make_shared<SystemSolver>(1e-8, 1);\n      sys_solver->addStage({0, 1}, solver);\n      auto sys = buildSolidStaticsWithL2StateSystem<dim, disp_order, StateSpace>(mesh, sys_solver,'
    )
    # the original code was: auto sys = buildSolidStaticsWithL2StateSystem...
    content = content.replace('auto sys = auto sys_solver', 'auto sys_solver')

    # test_thermal_static.cpp
    content = content.replace(
        'auto thermal_system = buildThermalSystem<2, temp_order>(mesh, block_solver);',
        'auto sys_solver = std::make_shared<SystemSolver>(1e-8, 1);\n    sys_solver->addStage({0}, block_solver);\n    auto thermal_system = buildThermalSystem<2, temp_order>(mesh, sys_solver);'
    )

    # test_thermo_mechanics.cpp
    content = content.replace(
        'auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, solver, youngs_modulus);',
        'auto sys_solver = std::make_shared<SystemSolver>(1e-8, 1);\n  sys_solver->addStage({0, 1}, solver);\n  auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, sys_solver, youngs_modulus);'
    )

    with open(t, 'w') as file:
        file.write(content)
