# Smith FEM Modernization Plan

**Mission**: Migrate from legacy monolithic interfaces to a composable, differentiable architecture for advanced multiphysics simulations.

**Current branch**: `tupek/solver_and_subsolvers`

---

## Current Focus: Sub-Solver Structure

### What's Done

- `SystemSolver` with stages and staggered loop (`9ecdabdb`)
- Index-based state propagation replacing fragile name-matching (`31bc99c7`)
- Global L2 norm convergence check (`f85e8b33`)
- Replaced global norm with per-stage `checkConvergence` in `DifferentiableBlockSolver`
- `exact_staggered_steps` mode (skip convergence check, run fixed iterations)
- 0.6x inner-tolerance factor in `buildDifferentiableNonlinearBlockSolver`
- All existing test call-sites updated to new `SystemSolver` constructor

### Immediate TODOs

#### A. Nits — DONE
- [x] Add `&& iter < max_staggered_iterations_ - 1` guard in `system_solver.cpp:98`
- [x] Add `SLIC_INFO_ROOT` logging of per-stage residual norms and iteration count
- [x] Delete `system_solver.cpp.orig`
- [x] Parallel norms: already uses `mfem::ParNormlp` correctly throughout

#### B. Bug Fix — DONE
`system_solver.cpp:129-133` — convergence-check residual eval omits `params` from `input_ptrs`. The initial-norm-capture loop (lines 61-63) correctly includes params; the convergence-check is a drifted copy that doesn't. Any parameterized system with `max_staggered_iterations > 1` will SLIC_ERROR. Undetected because all tests use `SystemSolver(comm, 1)`.

**Fix**: added `for (const auto& param_state : params[global_row]) input_ptrs.push_back(param_state.get().get());` after the states loop at line 133.

#### C. Staggered Convergence Tests — DONE
All existing tests use `SystemSolver(1)` — zero coverage of the new staggered convergence logic.

1. **Thermo-Mechanics Staggered vs Monolithic** (most important)
   - File: `test_thermo_mechanics.cpp`
   - Staggered: `SystemSolver(10)`, two stages (thermal + mech separately)
   - Monolithic reference: `SystemSolver(1)`, single stage with both blocks
   - Check: staggered converges, final fields match monolithic within O(tol)

2. **Exact Staggered Steps Test**
   - File: `test_thermo_mechanics.cpp`
   - Same setup but `SystemSolver(3, /*exact_staggered_steps=*/true)`
   - Check: completes without error, solution is reasonable

#### D. MacOS Apple Clang Lapack Linking Issue — DONE
Tests on macOS with Apple Clang (specifically `test_solid_dynamics` and `test_thermo_mechanics`) are crashing immediately with:
`dyld[...]: Library not loaded: @rpath/liblapack.3.dylib`
even though `liblapack.3.dylib` exists in the Spack install tree (e.g. `smith-tpls/apple-clang-17.0.0/netlib-lapack-3.12.1-.../lib/liblapack.3.dylib`).

**Workaround**: Setting `DYLD_LIBRARY_PATH` or `DYLD_FALLBACK_LIBRARY_PATH` to the Lapack lib dir allows tests to pass.
**Fix**: Modified `cmake/thirdparty/FindMFEM.cmake` to specifically scrape `MFEM_LIBRARIES` for `-L` paths containing `lapack` and injected them firmly into `CMAKE_BUILD_RPATH` and `CMAKE_INSTALL_RPATH` if on Apple frameworks `(APPLE)`.

### Backlog (lower priority)
- Unit Test for `checkConvergence` (Synthetic vectors with known norms; test abs/rel tolerance logic, `resetConvergenceState` in `test_differentiable_block_solver.cpp`)
- Extract helper lambda for "build input_ptrs → call residual() → zero BC dofs" in `system_solver.cpp` — duplication is what caused Bug B
- `SLIC_ERROR_IF(max_staggered_iterations <= 0, ...)` in `SystemSolver` constructor
- `1e12` trick in `checkConvergence(1e12, ...)` call: consider named constant or dedicated `recordInitialResidual()` method
- Error messages in `nonlinear_solve.cpp` — add actual size values via `axom::fmt::format`
- Per-block tolerance vector (only needed for mixed-physics stages with staggered iterations)
- Adjoint correctness through staggered iterations (VJP for outer loop)
- Pressure BC total net force investigation
- L2 stress visualization projection

---

## Stress Output Notes

See [stress_output_plan.md](stress_output_plan.md) for full details.

**Status**: Largely complete.
- `SolidStaticsWithL2StateSystem` has `stress_predicted` field and `stress_weak_form` for L2 projection
- `addStressProjection` method added; verified in `test_solid_static_with_state.cpp`
- Differentiable via `DifferentiableBlockSolver` / gretl graph

**Future enhancements** (when needed):
- Cauchy stress output for finite-strain problems: `σ = (1/J) P F^T`
- Other derived quantities: equivalent plastic strain, damage energy release rate

---

## Broader Roadmap

### COMPLETED
1. ✅ **Validation** — Runtime field/space checking (`FunctionalWeakForm`, `FieldStore`, `block_solve`)
2. ✅ **Convert Tests** — New interface replaces legacy `SecondOrderTimeDiscretizedWeakForm`
3. ✅ **Verify Derivatives** — `DifferentiablePhysics` wrapper matches pure gretl
7. ✅ **Thermal System** — Static `ThermalSystem` with manufactured solution test
12. ✅ **L2 State Variables** — History-dependent materials via additional weak forms
13. ✅ **SystemSolver** — Initial implementation with stages and staggered dispatch (in progress on current branch)

### IN PROGRESS
6. ⚙️ **Clean Interface** — Simplify user-facing `addBodyForce`/`addTraction` with automatic time integration handling
9. ⚙️ **Pressure BCs** — Follower forces exist; investigating non-zero net force on closed body
11. ⚙️ **Stress Output** — L2 projection of quadrature fields to L2 nodal fields (see above)

### NOT STARTED — Next Up
5. **BDF Integrators** — BDF1/BDF2 for stiff thermal problems
8. **Integral Support** — Expose `addInteriorFaceIntegral` for DG interior penalty
10. **Angled Dirichlet BCs** — Roller BCs: constrain normal, free tangent
14. **Constraints** — Penalty contact, homotopy continuation, augmented Lagrangian
15. **Jacobian Caching** — Cache matrices/factorizations for linear problems
27. **Warm-Start Solver** — Safeguarded BC updates, BC continuation line search

### LATER PHASES
- **Phase 6**: Command-line driver (YAML/JSON), expanded example library
- **Phase 7**: Phase field fracture, 3D mortar contact, enhanced DG
- **Phase 8**: GPU backends, in-situ visualization, in-app meshing
- **Phase 9**: Delete legacy `HeatTransferWeakForm`, migrate/delete `SolidMechanics.hpp` (1690 lines)
- **Phase 10**: Enhanced logging, Smith AI agent

---

## Key Files

| Area | File |
|------|------|
| Sub-solver dispatch | `src/smith/differentiable_numerics/system_solver.hpp/.cpp` |
| Block solver interface | `src/smith/differentiable_numerics/differentiable_solver.hpp/.cpp` |
| Thermo-mech system | `src/smith/differentiable_numerics/thermo_mechanics_system.hpp` |
| Solid dynamics system | `src/smith/differentiable_numerics/solid_dynamics_system.hpp` |
| Solid statics + L2 state | `src/smith/differentiable_numerics/solid_statics_with_L2_state_system.hpp` |
| Thermal system | `src/smith/differentiable_numerics/thermal_system.hpp` |
| Weak form core | `src/smith/physics/functional_weak_form.hpp` |
| Nonlinear solve | `src/smith/differentiable_numerics/nonlinear_solve.cpp` |

## Build Notes (macOS)

See [build_smith_summary.md](build_smith_summary.md) for full details on building with `apple-clang` + OpenMP on macOS Apple Silicon (Spack, RPATH, `-Xpreprocessor` flags).
