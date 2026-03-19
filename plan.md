LLM implementation plan: nonlinear convergence behavior

What was already done

Per-block nonlinear convergence tolerances were added to `NonlinearBlockSolver`. `CoupledSystemSolver` also gained stage-local per-block convergence overrides, plus validation for tolerance-vector sizes and the rule that stage-local tolerances cannot be tighter than the solver's own tolerances. Focused regression tests were added for these behaviors.

Update after implementation

The main OR-semantics refactor is now landed:

- Added shared Smith convergence types/helpers in `src/smith/numerics/nonlinear_convergence.{hpp,cpp}`
- Inner custom Newton now stops on:
  - global scalar convergence OR
  - all-block per-block convergence
- Inner trust-region now uses the same OR rule
- PETSc SNES now uses a Smith-owned convergence callback when block tolerances are active
- `CoupledSystemSolver` outer staggered early exit now uses the OR rule and owns explicit per-stage convergence contexts
- Removed the old fake "huge multiplier" pattern for seeding initial norms
- Scaled solver-local inner per-block tolerances by the existing 0.6 inner-tolerance factor
- Fixed the manual/prebuilt `EquationSolver` path so injected nonlinear solvers can still use Smith-managed block convergence state

Regression coverage now includes:

- helper-level convergence semantics and stage tolerance validation
- custom Newton inner convergence
- custom Newton line-search inner convergence
- trust-region inner convergence
- PETSc Newton inner convergence
- PETSc trust-region inner convergence
- manual/injected `EquationSolver` convergence-manager path

Next design: OR-based global or per-block convergence for nonlinear solves and staggered exit

Goal

Change the stopping rule so both inner nonlinear solves and `CoupledSystemSolver` early exit use:

- global scalar convergence OR
- all-block per-block convergence

That is, convergence should mean:

- `global_converged || block_converged`

Desired semantics

- Global criterion stays as it is today:
  - `global_norm <= max(abs_tol, rel_tol * initial_global_norm)`
- Per-block criterion stays as it is today:
  - block `i` passes when `||r_i|| <= max(abs_i, rel_i * initial_block_norm_i)`
  - the block path passes only when all blocks pass
- Combined criterion:
  - overall convergence is true when either the global path passes or the all-blocks path passes
- Initial references:
  - global relative tolerances remain relative to the initial global residual norm
  - per-block relative tolerances remain relative to each block's own initial residual norm

Current gap

- No small targeted end-to-end test yet for the real `CoupledSystemSolver` outer staggered OR exit path
- Broader regression suites using the new behavior have not all been run yet
- PETSc convergence reasons/monitor output still report OR-path success using the scalar fnorm reason code

Recommended design

Introduce one Smith-owned convergence evaluator used everywhere. It should evaluate:

- global scalar convergence
- per-block convergence
- combined OR result

This should be the single source of truth for:

- `NonlinearBlockSolver`
- `CoupledSystemSolver`
- custom Newton
- trust-region
- PETSc SNES callback path

Recommended API shape

Add a small status type, for example:

```cpp
struct ConvergenceStatus {
  bool global_converged;
  bool block_converged;
  bool converged;
  double global_norm;
  double global_goal;
  std::vector<double> block_norms;
  std::vector<double> block_goals;
};
```

Also add companion state for the stored initial norms:

- initial global residual norm
- initial residual norm for each block

Implementation status

1. Shared convergence helper

Status: done

- Add one reusable evaluator that takes:
  - current residual blocks
  - scalar tolerances
  - effective per-block tolerances
  - stored initial norms
- It returns `ConvergenceStatus`
- Keep a bool wrapper if convenient, but use the richer status internally
- Keep ownership outside the helper:
  - inner solves use an `EquationSolver`-owned convergence context
  - staggered outer checks use `CoupledSystemSolver` per-stage convergence contexts

2. `NonlinearBlockSolver`

Status: done

- Extend it to track both:
  - initial global residual norm
  - initial per-block residual norms
- Update its convergence API so it can report:
  - global result
  - block result
  - combined OR result
- Preserve scalar-only behavior when no block tolerances are configured:
  - empty solver-local `block_tolerances` means per-block goals default to `0`
  - this leaves the scalar path as the only practical stop condition unless a block residual is exactly zero

3. `CoupledSystemSolver` early exit

Status: implemented, but still missing one targeted real-path regression test

- Switch stage convergence checks from block-only to OR semantics
- For each stage, compute:
  - stage global norm
  - stage block norms
- Define stage global norm as the L2 norm of the concatenated stage residual blocks
- Early-exit only when every stage satisfies `global || all_blocks`
- Keep stage-local `block_tolerances` as outer-only overrides on the block side of the OR rule
- Allow stage-local block overrides even when the inner solver has no solver-local block tolerances configured

4. Custom Newton integration

Status: done

- Replace the direct scalar stop test in `NewtonSolver::Mult(...)` with the shared convergence evaluator
- Pass block offsets from the differentiable solver layer into `EquationSolver`
- Keep iteration accounting and `GetConverged()` behavior consistent with today's interface

5. Trust-region integration

Status: done

- Replace the direct scalar stop tests with the shared OR rule
- Near-term recommendation:
  - keep trust-region inner CG tolerance based on the global goal only
  - change only the outer trust-region stopping rule to OR semantics
- Revisit later only if block-driven convergence needs tighter inner model solves

6. PETSc SNES integration

Status: done for stop behavior; reason-code cleanup still optional follow-up

- Add a Smith-owned SNES convergence callback that uses the same shared evaluator
- The callback needs access to:
  - block offsets
  - stored initial global norm
  - stored initial block norms
- Keep PETSc scalar tolerance setup as default/fallback, but the Smith callback should become authoritative when block tolerances are active

7. Inner/outer tolerance ownership

Status: done

- Solver-local `NonlinearSolverOptions.block_tolerances` should participate in the inner nonlinear stop test
- Stage-local `CoupledSystemSolver::Stage::block_tolerances` should remain outer-only overrides for staggered early exit
- Do not let the outer staging layer silently mutate the solver's internal stopping behavior
- Keep convergence state ownership split by layer:
  - `EquationSolver` owns the inner-solve convergence context
  - `CoupledSystemSolver` owns a per-stage outer convergence context
- Remove the current pattern where outer code seeds initial norms by calling a fake convergence check with a huge multiplier

8. Revisit the 0.6 inner tolerance factor

Status: done for solver-local per-block tolerances

- Keep scaling the inner solver's scalar tolerances as today
- Likely also scale solver-local per-block tolerances for consistency
- Do not scale stage-local outer overrides

Tests needed

- Remaining:
  - `CoupledSystemSolver` real-path targeted test:
    - stage exits when global passes even if a block fails
    - stage exits when all blocks pass even if global fails
    - no exit when neither path passes
  - broader regression sweep:
    - full `equationsolver` suite
    - full `equationsolver_petsc` suite
    - selected multiphysics/staggered problem tests

- Already covered:
  - inner nonlinear solve helper semantics:
    - global passes, block fails -> converged
    - global fails, all blocks pass -> converged
    - global fails, one block fails -> not converged
    - scalar-only configuration unchanged
  - backend coverage:
    - custom Newton
    - custom Newton line-search
    - trust-region
    - PETSc nonlinear solver when enabled
  - manual/prebuilt `EquationSolver` convergence-manager path

Likely file touch points

- `src/smith/differentiable_numerics/nonlinear_block_solver.hpp`
- `src/smith/differentiable_numerics/nonlinear_block_solver.cpp`
- `src/smith/differentiable_numerics/coupled_system_solver.cpp`
- `src/smith/numerics/equation_solver.hpp`
- `src/smith/numerics/equation_solver.cpp`
- `src/smith/numerics/petsc_solvers.hpp`
- `src/smith/numerics/petsc_solvers.cpp`
- tests under `src/smith/differentiable_numerics/tests` and `src/smith/numerics/tests`

Remaining risk

- Lowest remaining risk:
  - add targeted outer staggered OR-path tests
- Medium remaining risk:
  - broader regression coverage across existing nonlinear solve suites
- Low-priority cleanup:
  - PETSc convergence reason/reporting polish

Future work: callback-based convergence

Per-block residual tolerances are still a limited built-in policy. Future work should allow user-provided convergence callbacks so applications can stop on solution changes, energy, work, mixed residual/state criteria, or other problem-specific signals. That future API should be separate for inner nonlinear solves and outer staggered convergence, since those are different convergence problems.

Recommended future shape:

- keep scalar and per-block tolerances as the default built-in path
- add a convergence context object rather than a long callback signature
- let callbacks explicitly replace the built-in rule when requested

Example sketch:

```cpp
struct BlockConvergenceContext {
  std::vector<const mfem::Vector*> residuals;
  std::vector<const mfem::Vector*> increments;
  std::vector<const FiniteElementState*> states;
  std::vector<double> residual_norms;
  std::vector<double> initial_residual_norms;
  int iteration;
};

struct ConvergenceDecision {
  bool converged;
};
```

This is deferred because the current MFEM/PETSc-backed nonlinear flow does not yet expose all needed state at one clean Smith-owned convergence hook.
