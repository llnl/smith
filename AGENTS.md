# Repository Guidelines

## Project Structure & Module Organization
- `src/smith/`: core C++ libraries (physics, numerics, infrastructure).
- `src/drivers/`: driver entrypoints (currently mostly disabled in CMake).
- `examples/`: runnable example executables (e.g., `examples/conduction/`).
- `src/tests/` and `src/smith/**/tests/`: GoogleTest-based unit/smoke tests.
- `tests/`: external test-data submodule (not compiled; used by CI/integration).
- `cmake/` + `host-configs/`: BLT-based CMake build system and machine configs.
