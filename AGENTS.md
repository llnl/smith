# Repository Guidelines

## Project Structure & Module Organization
- `src/smith/`: core C++ libraries (physics, numerics, infrastructure).
- `src/drivers/`: driver entrypoints (currently mostly disabled in CMake).
- `examples/`: runnable example executables (e.g., `examples/conduction/`).
- `src/tests/` and `src/smith/**/tests/`: GoogleTest-based unit/smoke tests.
- `tests/`: external test-data submodule (not compiled; used by CI/integration).
- `cmake/` + `host-configs/`: BLT-based CMake build system and machine configs.

## Coding Style & Naming Conventions
- C++ formatting is enforced via `.clang-format`; prefer running `cmake --build build --target style` before pushing.
- Use existing naming patterns: `PascalCase` types, `camelCase` methods, `snake_case_` members, `SCREAMING_SNAKE_CASE` constants.
- Keep changes minimal and localized; avoid drive-by refactors.

## Testing Guidelines
- Tests use GoogleTest (`TEST(...)`) and are registered via CMake/BLT.
- Add or update tests near the code they cover (e.g., `src/smith/physics/state/tests/`).
- If a test requires MPI/GPU/TPLs, gate it behind the existing CMake options and keep a CPU-only unit test when possible.

## Commit & Pull Request Guidelines
- Commit messages in this repo are typically short and imperative (e.g., “fix …”, “update …”, “add …”).
- PRs should include: a clear description, how to reproduce/verify, and the relevant `ctest` command(s) you ran.
- Update documentation in `src/docs/` when changing user-facing behavior or APIs.
