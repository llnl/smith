# dFEM Integration Plan

This plan tracks the remaining work to make the dFEM path usable as a normal
Smith weak-form backend. Older investigation notes were removed after the
assembled Jacobian, reverse-mode VJP, and first solid examples landed.

## Current State

### mfem branch

dfem-multiple-outputs

### Working in Smith

- `smith::DfemWeakForm` supports:
  - residual evaluation through `mfem::future::DifferentiableOperator::Mult`
  - serial assembled Jacobian construction through `GetDerivative(...)->Assemble(...)`
  - `jvp()` through derivative action
  - true reverse-mode `vjp()` for domain integrals
- `smith::DfemSolidWeakForm` supports:
  - solid stress-divergence domain integrals
  - time-dependent reference body forces through `setBodyForce`
  - a `setTraction` API shape, currently blocked by MFEM boundary integration bugs
- Tests/examples in tree:
  - `src/smith/differentiable_numerics/tests/test_dfem_solid_static.cpp`
    converges a dFEM Neo-Hookean static block extension.
  - `src/smith/differentiable_numerics/tests/test_dfem_shallow_arch_buckling.cpp`
    mirrors the functional shallow-arch setup as closely as possible, but uses
    a body-force surrogate for the missing traction path.
  - `src/smith/differentiable_numerics/tests/test_dfem_bugs.cpp`
    documents and reproduces the MFEM dFEM bugs Smith depends on.

### Vendored MFEM Patch Status

The current local patch is:

- `mfem/0001-Some-attempt-at-patching-dfem-for-smith-use-cases.patch`

That patch effectively contains the earlier `dfem_patches` series needed to get
the non-boundary cached derivative path usable:

- fixed broken dFEM relative includes
- guarded Enzyme-only helper templates with `MFEM_USE_ENZYME`
- set `DifferentiableOperator::height`
- stopped leaking/reallocating `SparseMatrix` on repeated `Assemble()`
- changed LocalQF derivative loops from compile-time `nfields` to runtime
  `ctx.unionfds.size()` where the operator union can be larger than one
  integrator's referenced fields
- standardized cached Jacobian layout across setup, apply, transpose apply, and
  assembly for multi-output/multi-field nonlinear operators
- added MFEM-side Neo-Hookean QoI/reverse tests and timing miniapp scaffolding

Keep this patch carried until those changes are either upstreamed or replaced
by a clean vendored MFEM commit.

## Priority Order

Start with system integration. Boundary integration, material state, and
deployment remain important, but they should be driven by a real `dfem_system`
path instead of more standalone tests.

1. Integrate `DfemSolidWeakForm` into the high-level system solver path.
2. Resolve boundary integration bugs 8 and 9.
3. Support history-dependent and rate-dependent materials.
4. Clean up build/toolchain/deployment and MPI verification.

## Cross-Cutting Blocker: Physical Integration Scaling

The most important non-API issue is integration measure ownership.

`mfem::future::Weight` currently gives the reference integration-rule weight,
not `det(dX/dxi) * w`. Smith dFEM q-functions therefore must multiply by the
physical measure explicitly whenever they integrate over physical volume or
surface:

- domain: `det(dX_dxi) * weight`
- boundary: `weight(dX_dxi) * weight`

Current solid stress and body-force qfunctions already do this explicitly.
Before adding more user-facing APIs, audit every dFEM qfunction and test for
the same convention so we do not reintroduce the old `1024x` mesh-refinement
scaling error.

Open decision:

- either keep explicit measure multiplication as the Smith convention and
  document it clearly
- or fix/wrap `Weight` upstream so physical measure is automatic, then update
  all qfunctions once

Do not mix both conventions.

## Main Remaining Blockers

### 1. System solver integration / `dfem_system`

The next implementation target is a real dFEM system path. The immediate goal
is not full feature parity with `SolidMechanicsSystem`; it is a clean,
user-facing path that solves through the existing `SystemBase` /
`SystemSolver` stack with assembled dFEM residuals and Jacobians.

Preferred direction:

- reuse `SystemBase`, `SystemSolver`, `FieldStore`, reaction registration, and
  existing Dirichlet BC handling
- add a dFEM-specific builder first, e.g. `buildDfemSolidMechanicsSystem(...)`
  or `buildSolidMechanicsSystemDfem(...)`
- only introduce a `DfemSolidMechanicsSystem` type if the current
  `SolidMechanicsSystem` typed `FunctionalWeakForm` members make reuse awkward
- keep the first system narrow:
  - quasi-static or simplest 4-state solid layout
  - one displacement weak form
  - no stress projection parity initially
  - no cycle-zero transient solve initially unless the chosen time rule forces it
- do not clone all of `SolidMechanicsSystem` behavior unless the shared path is
  clearly worse

Key questions to answer while implementing:

- can `SolidMechanicsSystem` hold a base `std::shared_ptr<WeakForm>` for the
  main solid residual while keeping functional-only conveniences optional?
- should material/body-force/traction setup live on the dFEM weak form, on a
  small dFEM system wrapper, or in builder helper functions?
- how should the high-level options/YAML path select functional vs dFEM backend?
- does the current serial-only `DfemWeakForm::jacobian()` block the intended
  first solver test, or can the first acceptance test remain serial?

Acceptance target:

- a dFEM solid system builds from registered fields and solves through
  `SystemBase::solve()`
- displacement Dirichlet BCs use the normal `FieldStore` /
  `DirichletBoundaryConditions` path
- the test/example does not manually wire weak forms outside the public builder
- backend selection has an obvious route into options/config, even if the first
  patch only exposes the C++ builder

### 2. Boundary integration

`DfemSolidWeakForm::setTraction` is wired on the Smith side, but cannot be used
for real solves until MFEM dFEM handles boundary entities correctly.

Known upstream bugs:

- `LocalQFBackend::Action` is hardcoded to `Entity::Element`
- `DifferentiableOperator::Mult` restricts, accumulates, and scatters all
  callbacks as volume-element callbacks
- derivative callback storage is entity-blind, so boundary Jacobians and
  boundary reverse-mode are not representable yet
- `GlobalQFBackend` is not an interim fallback for Smith qfunctions because it
  requires batched `tensor_array` qfunction signatures, not Smith's per-QP
  scalar/tensor style

Acceptance target:

- re-enable and pass
  `DfemSolidTraction.DISABLED_ConstantTractionMatchesSurfaceIntegral`
- a constant traction over selected boundary attributes must match analytic
  surface measure times traction

### 3. History-dependent and rate-dependent materials

`StressDivQFunction::operator()` still hardcodes `dt = 1.0`. This blocks
viscoelastic, transient thermal-stress, plasticity, damage, and other material
models where the response depends on the actual time step or evolving
quadrature-point state.

Needed:

- expose `dt` from `DfemWeakForm` to qfunctions, analogous to
  `currentTimePtr()`
- update `StressDivQFunction` to pass the real step size to
  `material.pkStress`
- define quadrature-state ownership:
  - read old internal variables at the start of a solve/time step
  - update/commit new internal variables only after accepted solves
  - support restartable storage for plastic strain, backstress, damage, etc.
- keep differentiated FE fields separate from nondifferentiated history state
  so material-state bookkeeping does not pollute the weak-form API

### 4. Build toolchain and deployment

Users need a reliable way to compile custom dFEM qfunctions.

Needed:

- document Enzyme/LLVM plugin requirements for user-facing builds
- verify non-Enzyme builds either exclude dFEM cleanly or compile all safe
  headers without Enzyme symbols
- remove or upstream the local MFEM patch stack
- validate MPI behavior after Julian's upstream parallel functional fixes and
  after dFEM assembled Jacobian construction is no longer serial-only

### 5. Cached derivative and transpose confidence

The carried MFEM patch fixes the non-boundary cached derivative bugs that were
blocking assembled Jacobian use. Keep the regression coverage alive because
this area is fragile:

- `Bug12_JacobianMismatch`
- `Bug12_MultipleFields`
- `Bug12_NonlinearJacobianAndTranspose`
- `Bug13_SingleInputTranspose`
- `Bug13_ValueOnlyTranspose`

Next verification should be on the real arch workload after integration scaling
and body-force loading are confirmed. If the dFEM arch case still diverges with
the functional-reference solver stack, isolate whether the problem is:

- an actual assembled-Jacobian defect
- stiffness/conditioning different from the functional reference
- remaining qfunction scaling/sign convention mismatch
- solver/preconditioner mismatch

## Next Steps

### Step 1: Build the first `dfem_system` path

Goal: make dFEM solids enter the same high-level solve path as functional
solids, even if the first version is deliberately narrow.

Work:

- inspect `SolidMechanicsSystem` for the smallest change that lets the main
  solid residual be held as `std::shared_ptr<WeakForm>`
- add a dFEM builder that registers/reuses the standard solid fields:
  - `displacement_solve_state`
  - `displacement`
  - `velocity`
  - `acceleration`
  - optional parameter/coupling fields only if needed by the first test
- construct `DfemSolidWeakForm` with the displacement test space and matching
  input spaces
- attach it to `SystemBase::weak_forms`
- reuse the normal displacement BC manager from `FieldStore`
- add one minimal serial solve test through `SystemBase::solve()`
- keep the current standalone dFEM tests as lower-level coverage, but stop using
  them as the only user-facing API

Deliverable:

- `buildDfemSolidMechanicsSystem(...)` / `buildSolidMechanicsSystemDfem(...)`
  plus one test proving the returned system solves through the normal solver
  stack with Dirichlet BCs.

Design guardrail:

- if adapting `SolidMechanicsSystem` creates noisy functional/dFEM branching,
  create a small `DfemSolidMechanicsSystem` wrapper instead. It should still
  derive from `SystemBase` and share field registration/BC conventions; it
  should not fork solver infrastructure.

### Step 2: Stabilize the body-force arch surrogate

Goal: make the current dFEM shallow-arch test a meaningful body-force benchmark
before using it for timing or derivative conclusions.

Work:

- verify every qfunction in the arch path multiplies by physical measure exactly
  once
- inspect ParaView output for `paraview_dfem_shallow_arch_buckling` and
  `paraview_dfem_solid_static`
- compare total applied load in the dFEM body-force surrogate against the
  functional top-face traction history
- log nonlinear residuals and linear iterations for the first failing load step
- decide whether the current assertion should remain output-only or become a
  real body-force snap-through check

Deliverable:

- either a converged body-force arch regression, or a narrowed bug report with a
  residual/Jacobian mismatch reproducer from the arch setup.

### Step 3: Land MFEM boundary forward action

Goal: make `AddBoundaryIntegrator<LocalQFBackend>` produce correct residuals.

MFEM files in scope:

- `mfem/fem/dfem/backends/local_qf/action.hpp`
- `mfem/fem/dfem/backends/local_qf/prelude.hpp`
- `mfem/fem/dfem/doperator.hpp`

Implementation outline:

- thread `entity_t` through `LocalQFBackend::MakeAction`
- template `LocalQFImpl::Action` on `entity_t`
- use entity geometry:
  - `mesh.GetTypicalElementGeometry()` for `Entity::Element`
  - `mesh.GetTypicalFaceGeometry()` for `Entity::BoundaryElement`
- build DofToQuad maps, shmem layout, and output restrictions with `entity_t`
- tag action callbacks by entity kind at registration
- in `DifferentiableOperator::Mult`, dispatch callbacks by entity kind:
  - prolongate true dofs to local dofs once
  - restrict to element or boundary element vectors per group
  - accumulate each group's residual back to the shared local residual
  - run one final local-to-true transpose prolongation

MFEM tests to add:

- scalar boundary mass with `LocalQFBackend`, compared to classical MFEM
  boundary mass
- vector H1 constant traction on selected boundary attributes, compared to
  analytic surface integral
- mixed body plus boundary residual on the same `DifferentiableOperator`

Smith re-entry:

- re-enable `DfemSolidTraction.ConstantTractionMatchesSurfaceIntegral`
- switch the shallow arch test from body force back to top-face traction

### Step 4: Add boundary derivative support

Goal: make follower loads and boundary sensitivities possible.

Work:

- thread `entity_t` through LocalQF derivative factories:
  - `MakeDerivativeAction`
  - `MakeDerivativeSetup`
  - `MakeDerivativeApply`
  - `MakeDerivativeApplyTranspose`
  - `MakeDerivativeAssemble`
  - `MakeDerivativeAssembleDiagonal`
- store derivative callbacks with entity tags
- assemble and apply boundary derivative blocks with boundary restrictions
- add reverse-mode boundary gradient support in the Smith/MFEM reverse helper

Tests:

- finite-difference a position-dependent boundary load residual against
  `GetDerivative(...)->Mult(...)`
- compare assembled boundary derivative action against finite differences
- add a VJP/reverse-mode test for a traction-loaded QoI once the reverse helper
  exists

### Step 5: Pass `dt` and add material state support

Current `StressDivQFunction` hardcodes `dt = 1.0`. That blocks
history-dependent and rate-dependent materials.

Work:

- expose `dt` from `DfemWeakForm` to qfunctions the same way current time is
  exposed (`currentTimePtr()` already exists)
- update `StressDivQFunction` to pass the real step size to `material.pkStress`
- define a quadrature-state API for internal variables such as plastic strain,
  backstress, damage, or history-dependent thermal variables
- keep the first version explicit about ownership:
  - FE state fields are differentiated inputs
  - quadrature history is read/write step state
  - material parameters can be FE fields or constants

Tests:

- a material test proving `dt` changes the residual/tangent as expected
- one minimal history-variable update test with restartable state

### Step 6: Toolchain and deployment cleanup

Work:

- document the Enzyme/LLVM requirements for user qfunctions
- verify a non-Enzyme build either excludes dFEM cleanly or compiles all
  non-Enzyme-safe headers
- run the dFEM tests in MPI once the serial assembled-matrix limitation is
  removed
- upstream or replace the carried MFEM patch so users do not have to maintain a
  loose patch stack

## Deferred Work

- matrix-free Newton/Krylov integration
- full transient parity with the functional solid system
- stress/strain projection output parity
- multiphysics dFEM builders
- generic boundary/internal-face helper APIs beyond the solid traction use case

## References

- `mfem/0001-Some-attempt-at-patching-dfem-for-smith-use-cases.patch`
- `mfem/dfem_current_bugs.md`
- `dfem_bugs.md`
- `mfem/dfem_index.md`
- `src/smith/differentiable_numerics/tests/test_dfem_bugs.cpp`
