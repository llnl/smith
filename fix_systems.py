import os
import glob

files = glob.glob('src/smith/differentiable_numerics/*system.hpp')
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    content = content.replace('std::shared_ptr<DifferentiableBlockSolver> solver', 'std::shared_ptr<SystemSolver> solver')
    with open(f, 'w') as file:
        file.write(content)

tests = glob.glob('src/smith/differentiable_numerics/tests/test_*.cpp')
for t in tests:
    with open(t, 'r') as file:
        content = file.read()
    # Replace usages of build...BlockSolver
    # e.g.: auto solver = buildDifferentiableNonlinearBlockSolver(...)
    # with: auto block_solver = build...; auto solver = std::make_shared<SystemSolver>(1e-8, 1); solver->addStage({0}, block_solver);
    # Actually, the number of stages depends on the system.
    pass

