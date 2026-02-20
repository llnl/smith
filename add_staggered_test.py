import os

with open('src/smith/differentiable_numerics/tests/test_thermo_mechanics.cpp', 'r') as f:
    content = f.read()

# find RunThermoMechanicalCoupled test and duplicate it
start = content.find('TEST_F(ThermoMechanicsMeshFixture, RunThermoMechanicalCoupled)')
end = content.find('TEST_F(ThermoMechanicsMeshFixture, TransientHeatEquationAnalytic)')

coupled_test = content[start:end]
staggered_test = coupled_test.replace('RunThermoMechanicalCoupled', 'RunThermoMechanicalStaggered')

# change the solver setup
old_solver = '''  auto solver = buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);

  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  auto sys_solver = std::make_shared<SystemSolver>(1e-8, 1);
  sys_solver->addStage({0, 1}, solver);'''

new_solver = '''  auto solver_mech = buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);
  auto solver_therm = buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);

  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  auto sys_solver = std::make_shared<SystemSolver>(1e-8, 3); // 3 staggered iterations max
  sys_solver->addStage({0}, solver_mech); // mechanics
  sys_solver->addStage({1}, solver_therm); // thermal'''

staggered_test = staggered_test.replace(old_solver, new_solver)

new_content = content[:end] + staggered_test + content[end:]

with open('src/smith/differentiable_numerics/tests/test_thermo_mechanics.cpp', 'w') as f:
    f.write(new_content)
