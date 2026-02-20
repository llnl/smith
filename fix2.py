import os

with open('src/smith/differentiable_numerics/tests/test_solid_static_with_internal_vars.cpp', 'r') as f:
    content = f.read()

old = '''  auto system =
      auto sys_solver = std::make_shared<SystemSolver>(1e-8, 1);
      sys_solver->addStage({0, 1}, solver);
      auto sys = buildSolidStaticsWithL2StateSystem<dim, disp_order, StateSpace>(mesh, sys_solver, "solid_static_with_internal_vars");'''

new = '''  auto sys_solver = std::make_shared<SystemSolver>(1e-8, 1);
  sys_solver->addStage({0, 1}, solver);
  auto system = buildSolidStaticsWithL2StateSystem<dim, disp_order, StateSpace>(mesh, sys_solver, "solid_static_with_internal_vars");'''

with open('src/smith/differentiable_numerics/tests/test_solid_static_with_internal_vars.cpp', 'w') as f:
    f.write(content.replace(old, new))
