# dFEM Bugs on the `dfem-multiple-outputs` Branch

This document catalogues bugs that exist in the MFEM dFEM implementation on the
`dfem-multiple-outputs` branch (commit `1fa7888706`, "fixes for transpose
functional computation") as shipped to smith. These bugs were identified
through smith integration testing and inspection of the cached
`LocalQFBackend` derivative path. Each bug is described in terms of root cause,
observable symptom, and affected files.

To address these issues, a series of five patches (`0002` through `0006`) has been developed to resolve critical C++ memory leaks, operator dimensioning mismatches, and out-of-bounds segfaults in the LocalQFBackend. Crucially, these patches standardize the cached Jacobian tensor layout to a column-major order across the setup, apply, and assembly steps to guarantee complete numerical consistency for multi-field and nonlinear operators. Together, they ensure the dFEM backend operates reliably and matches finite-difference evaluations to machine precision, resolving all non-boundary element bugs identified during Smith integration.

---


## Bug 1: Cache index ordering wrong for multiple outputs

**Files:**
`fem/dfem/backends/local_qf/derivative_setup.hpp`,
`fem/dfem/backends/local_qf/derivative_apply.hpp`,
`fem/dfem/backends/local_qf/derivative_apply_transpose.hpp`

**Root cause:** The per-quadrature-point Jacobian cache stores a 5-dimensional
tensor `J(i, k, j, m, q)` where `(i,k)` are test-space indices and `(j,m)` are
trial-space indices. The code that writes to this cache (in `DerivativeSetup`)
and the code that reads from it (in `DerivativeApply` and
`DerivativeApplyTranspose`) use different linearisation conventions.
`DerivativeSetup` originally wrote the cache in a row-major (test-outer)
layout:

```
cache_idx = i * test_op_dim * trial_vdim * total_trial_op_dim
          + k * trial_vdim * total_trial_op_dim
          + j * total_trial_op_dim
          + (m + m_offset);
```

This ordering is not inherently wrong for single-output integrators because the
`ik_offset` term is zero. However, for multiple-output integrators (where there
is more than one output field operator tuple entry), the `ik_offset` must
correctly stride over each output block's slice of the cache. The original code
computes `ik_offset` inconsistently between setup, apply, and apply-transpose,
causing silent corruption of Jacobian entries when `noutputs > 1`.

Additionally, the apply path wrote its per-output results using a division-based
flat offset rather than an explicit `res_offset` accumulator, which made the
output-offset logic fragile and incorrect in certain configurations.

**Symptom:** For integrators with a single output (the common case for solid
mechanics with `Gradient<U> -> Gradient<V>`), this bug is masked because
`ik_offset = 0`. When multiple outputs are used, the cached Jacobian action
returns numerically wrong results. In smith's solid mechanics path, this
manifested as the assembled Jacobian disagreeing with finite differences by
approximately `O(10)` in norm on a `Gradient<U> -> Gradient<U>` nonlinear
residual.

**Tested by:** `Bug12_JacobianMismatch`, `Bug12_NonlinearJacobianAndTranspose`,
`Bug13_SingleInputTranspose`, and `Bug13_ValueOnlyTranspose` in
`src/smith/differentiable_numerics/tests/test_dfem_bugs.cpp`. All four tests
compare the cached Jacobian action against finite differences; each fails on
the unpatched branch due to the cache stride mismatch.

**Fix:** Patch `0005` in `mfem_patches/`.

---

## Bug 2: Loop bounds use compile-time `nfields` instead of runtime `ctx.unionfds.size()`

**Files:**
`fem/dfem/backends/local_qf/derivative_action.hpp` (4 loops),
`fem/dfem/backends/local_qf/derivative_assemble.hpp` (1 loop),
`fem/dfem/backends/local_qf/derivative_assemble_diagonal.hpp` (1 loop)

**Root cause:** The `LocalQFImpl` derivative classes compute `nfields` as a
compile-time constant: the number of unique field ids referenced by the field
operators in a single integrator's input/output tuples. However,
`ctx.unionfds` is the runtime union of ALL field descriptors registered on the
parent `DifferentiableOperator` across all integrators. When some field
descriptors (e.g., coordinates at field id 2) appear in the operator's union but
are not referenced by a particular integrator's field operators, `nfields <
ctx.unionfds.size()`.

Several loops in these files iterate `for (size_t uf = 0; uf < nfields; uf++)`
to search for trial/test field indices or to build `union_to_infd` mappings.
When the target field's index in `unionfds` is beyond `nfields`, the search
fails silently. In Release builds where `MFEM_ASSERT` is compiled out, this
results in `trial_field_idx = -1` or `test_field_idx = -1`, leading to
out-of-bounds access of `ctx.unionfds[-1]` (undefined behaviour).

**Symptom:** In `derivative_assemble_diagonal.hpp`, the constructor segfaults
(signal 11) when a `DifferentiableOperator` has more registered field
descriptors than a given integrator references. The crash dereferences address
`0x1` (garbage read from before the vector's data buffer). In
`derivative_action.hpp`, the loops that build `union_to_infd` and
`dummy_fields` may silently leave entries uninitialised when `nfields <
ctx.unionfds.size()`. This was specifically triggered by the thermal test where
field id 2 (coordinates) exists in the operator's union but is not used by the
material integrator.

**Tested by:** `Bug12_MultipleFields` in
`src/smith/differentiable_numerics/tests/test_dfem_bugs.cpp`. This test uses
two scalar input fields plus one output field, so `unionfds` has 3 entries and
the `nfields` vs `ctx.unionfds.size()` discrepancy triggers incorrect field
index lookups. Additionally, `test_dfem_thermal_static` in
`src/smith/differentiable_numerics/tests/` segfaults on the unpatched branch
due to the `derivative_assemble_diagonal.hpp` instance of this bug.

**Fix:** Patch `0004` in `mfem_patches/`.

---

## Bug 3: `SparseMatrix` re-allocated (and leaked) on every `Assemble()` call

**Files:**
`fem/dfem/backends/local_qf/derivative_assemble.hpp`

**Root cause:** The `DerivativeAssemble::operator()` method unconditionally
executes `A = new SparseMatrix(...)` at the start of every call, regardless of
whether `A` already points to a valid, correctly-sized matrix from a previous
call. In a Newton iteration loop where `Assemble()` is called repeatedly with
the same `SparseMatrix*&` reference, this allocates a new matrix and overwrites
the pointer each iteration without deleting the prior allocation.

**Symptom:** Memory leak proportional to the number of Newton iterations times
the Jacobian matrix size. For a solid mechanics problem on a moderately refined
mesh with typical Newton iteration counts, this leaks megabytes per solve step.
The matrix itself is numerically correct on each call; only the allocation
behaviour is wrong.

**Tested by:** No dedicated runtime test. This is a memory leak with no
numerical-correctness impact, so it cannot be caught by a value-comparison
assertion. It was confirmed by code inspection: the unconditional
`A = new SparseMatrix(...)` at the top of `DerivativeAssemble::operator()`
is visible in the unpatched source. A valgrind or ASan run of any test that
calls `Assemble()` more than once (e.g., a Newton loop) would report the leak.

**Fix:** Patch `0003` in `mfem_patches/`.

---

## Bug 4: `Operator::height` never set in `DifferentiableOperator::AddIntegrator`

**Files:**
`fem/dfem/doperator.hpp`

**Root cause:** The `DifferentiableOperator` class inherits from `mfem::Operator`
which requires `height` and `width` to be set for correct dimension reporting.
The `AddIntegrator` template function sets `width = GetTrueVSize(infds[0])` but
never computes or assigns `height`. Since `Operator::Operator()` initialises
both to zero, `dop.Height()` returns 0 after integrators are added.

**Symptom:** Any caller that uses `Operator::Height()` for size checks
(iterative solvers, block operators, `MFEM_ASSERT` guards in `Mult`) gets zero
and either asserts or silently misallocates vectors. This blocks use of the
`DifferentiableOperator` in any solver framework that validates operator
dimensions.

**Tested by:** No dedicated runtime test. The bug is confirmed by code
inspection: `AddIntegrator` in `doperator.hpp` sets `width` but never assigns
`height`. Every test in `test_dfem_bugs.cpp` exercises a `DifferentiableOperator`
and would fail with dimension-checking solvers if `height` were still zero,
but the tests call `Mult` directly and do not go through a solver that checks
`Height()`.

**Fix:** Patch `0002` in `mfem_patches/`.

---

## Bug 5: Enzyme-specific templates exposed unconditionally

**Files:**
`fem/dfem/backends/util.hpp`

**Root cause:** The `enzyme_detail` namespace in `util.hpp` contains template
functions (`do_enzyme_call`, `process_inputs`) that reference Enzyme-specific
types and intrinsics (`__enzyme_fwddiff`, `enzyme_dup`, etc.). These templates
are defined outside any `#ifdef MFEM_USE_ENZYME` guard, so they are parsed and
instantiated by the compiler regardless of whether Enzyme is available.

**Symptom:** Build failure when compiling MFEM without Enzyme support. The error
manifests as undefined references to Enzyme intrinsics during template
instantiation. This prevents using the dFEM code in any build configuration that
does not have Enzyme enabled, even though the primary functionality (cached
derivative path) does not require Enzyme at runtime.

**Tested by:** Compile-time bug — no runtime test is applicable. The bug is
confirmed by attempting to build the dFEM headers in a non-Enzyme configuration:
the compiler emits undefined-reference errors for `__enzyme_fwddiff` and related
intrinsics. The fix (wrapping in `#ifdef MFEM_USE_ENZYME`) is verified by the
fact that the smith build succeeds with Enzyme enabled; a non-Enzyme build would
verify the guard.

**Fix:** Patch `0001` in `mfem_patches/`.

---

## Bug 6: Broken relative include paths in three headers

**Files:**
`fem/dfem/backends/util.hpp`,
`fem/dfem/backends/global_qf/derivative_action_enzyme.hpp`,
`fem/dfem/doperator.hpp`

**Root cause:** Several `#include` directives use incorrect relative paths that
do not resolve from the file's location in the source tree:

- `fem/dfem/backends/util.hpp` includes `"../fem/quadinterpolator.hpp"` — should
  be `"../../quadinterpolator.hpp"` (one more level up).
- `fem/dfem/backends/global_qf/derivative_action_enzyme.hpp` includes
  `"../fem/quadinterpolator.hpp"` — should be `"../../../quadinterpolator.hpp"`.
- `fem/dfem/doperator.hpp` includes `"../linalg/multivector.hpp"` — should be
  `"../../linalg/multivector.hpp"`.

**Symptom:** Compilation fails with `file not found` errors on these includes.
The headers cannot be used at all until the paths are corrected.

**Tested by:** Compile-time bug — no runtime test is applicable. The incorrect
paths cause immediate `#include` failures when the headers are parsed. The fix
is verified by the fact that `test_dfem_bugs.cpp` (which transitively includes
all three headers via `doperator.hpp` and `prelude.hpp`) compiles successfully
after the paths are corrected.

**Fix:** Patch `0001` in `mfem_patches/`.

---

## Bug 7: `AddBoundaryIntegrator` does not forward `backend_t` template parameter

**Files:**
`fem/dfem/doperator.hpp`

**Root cause:** The `AddBoundaryIntegrator` member function template lacks the
`backend_t` template parameter that `AddDomainIntegrator` has. The function
signature is:

```cpp
template <typename qfunc_t, typename input_t, typename output_t, typename derivative_ids_t>
void AddBoundaryIntegrator(...)
{
   AddIntegrator<Entity::BoundaryElement>(...);
}
```

Without `backend_t`, the internal call to `AddIntegrator` cannot dispatch to
`LocalQFBackend` (or any specific backend), and users calling
`dop.AddBoundaryIntegrator<LocalQFBackend>(...)` get a compilation error
because the first template argument is consumed as `qfunc_t`.

**Symptom:** Compilation error when attempting to register a boundary integrator
with an explicit backend specification. The error message is a type mismatch on
the first template argument.

**Tested by:** Compile-time bug — no runtime test is applicable. The
`Bug1_2_BoundaryIntegrator` test in `test_dfem_bugs.cpp` calls
`dop.AddBoundaryIntegrator<mfem::future::LocalQFBackend>(...)`, which would
fail to compile without this fix. The fact that the test compiles confirms the
`backend_t` parameter is correctly forwarded.

**Fix:** Patch `0002` in `mfem_patches/`.

---

## Bug 8: `LocalQFBackend` boundary integration is not entity-aware

**Files:**
`fem/dfem/backends/local_qf/action.hpp`,
`fem/dfem/backends/local_qf/prelude.hpp`

**Root cause:** `LocalQFImpl::Action` is templated on
`qfunc_t / inputs_t / outputs_t` but NOT on `entity_t`. Every entity-dependent
operation inside the class is hardcoded to `Entity::Element`:

- `Element::TypeFromGeometry(ctx.mesh.GetTypicalElementGeometry())` uses the
  volume geometry to decide `use_sum_factorization`, not the face geometry.
- `GetDofToQuad<Entity::Element>(...)` builds volume DofToQuad maps that get
  paired with the boundary `IntegrationRule` (inconsistent dof counts).
- `create_dtq_maps<Entity::Element>(...)` lays out memory for volume entities.
- `get_restriction<Entity::Element>(...)` scatters per-QP outputs into volume
  element dof slots rather than boundary face dof slots.

The dispatch path drops `entity_t` at `backend_t::MakeAction(ctx, qfunc, inputs,
outputs)`: `LocalQFBackend::MakeAction` has no `entity_t` template parameter and
always constructs the `Entity::Element`-hardcoded `Action`.

**Symptom:** When a boundary integrator is registered with `LocalQFBackend`, the
residual sum for a constant function integrated over a boundary does not equal
the boundary's geometric measure. For example, integrating `u*v` with `u=1` over
the bottom edge (length 1.0) of the unit square produces ~0.789 instead of 1.0.
The volume DofToQuad maps are paired with a 1D boundary integration rule,
producing garbage interpolation/integration operators.

**Tested by:** `Bug1_2_BoundaryIntegrator` in
`src/smith/differentiable_numerics/tests/test_dfem_bugs.cpp`. The test registers
a boundary mass form on the bottom edge (attribute 1) of the unit square and
asserts `residual.Sum() == 1.0`. It currently fails with ~0.789 on both the
unpatched branch and with all current patches applied.

**Fix:** UNFIXED. Requires threading `entity_t` through the entire
`LocalQFBackend` template chain (`MakeAction`, `Action`, all derivative
factories in `prelude.hpp`) and using face geometry / boundary restrictions
instead of volume ones.

---

## Bug 9: `DifferentiableOperator::Mult` is hardcoded to volume entities

**Files:**
`fem/dfem/doperator.hpp`

**Root cause:** The `Mult` method in `DifferentiableOperator` runs all
registered `action_callbacks` against a single set of element-restricted buffers:

```cpp
restriction<Entity::Element>(infds, infields_l, infields_e);
prepare_residual<Entity::Element>(outfds, residual_e);
for (auto &cb : action_callbacks) { cb(infields_e, residual_e); }
restriction_transpose<Entity::Element>(outfds, residual_e, residual_l);
```

There is no per-entity-kind grouping of callbacks. Boundary integrator callbacks
receive volume-restricted input fields and write into volume-element-sized
residual buffers, then the volume element restriction transpose scatters these
contributions back. The boundary restriction (which would use face DOF counts and
face-to-DOF mappings) is never invoked.

**Symptom:** Even if Bug 8 were fixed (making `LocalQFImpl::Action`
entity-aware), boundary contributions would still be wrong because they operate
on volume-restricted data. The residual from boundary integrals is scattered to
wrong DOF locations. This bug compounds with Bug 8 to produce completely
incorrect boundary residuals.

**Tested by:** `Bug1_2_BoundaryIntegrator` in
`src/smith/differentiable_numerics/tests/test_dfem_bugs.cpp` (same test as
Bug 8). Bugs 8 and 9 compound to produce the wrong residual. Even if Bug 8
were fixed independently, the `restriction<Entity::Element>` calls in `Mult`
would still produce incorrect boundary contributions.

**Fix:** UNFIXED. Requires tagging each `action_callback` with its `entity_t`
at `AddIntegrator` time, grouping callbacks by entity kind in `Mult`, and
running `restriction<entity_t>` / `prepare_residual<entity_t>` /
`restriction_transpose<entity_t>` per group.

---

## Bug 10: `GlobalQFBackend` is not a drop-in replacement for per-QP qfunction style

**Files:**
`fem/dfem/backends/global_qf/action.hpp`,
`fem/dfem/backends/util.hpp`

**Root cause:** `GlobalQFImpl::Action::operator()` contains a `static_assert`
requiring `supports_tensor_array_qfunc<qfunc_t, inputs_t, outputs_t>::value`.
This requires every qfunction parameter to be a `tensor_array<scalar, sizes...>`
(a batched type spanning all quadrature points simultaneously).
Smith qfunctions and the canonical dFEM unit tests use the per-QP style: scalar
or `tensor` arguments with either a tuple return or output-parameter convention.

**Symptom:** Switching from `LocalQFBackend` to `GlobalQFBackend` (for example,
as a potential workaround for Bug 8 on boundary integrals) fails at compile time
with a `static_assert` about unsupported qfunction parameter types. This means
`GlobalQFBackend` cannot serve as an interim fallback for boundary integrals
written in the per-QP style.

**Tested by:** `Bug10_GlobalQFBackendIncompatible` in
`src/smith/differentiable_numerics/tests/test_dfem_bugs.cpp`. The test
evaluates the `detail::supports_tensor_array_qfunc` trait (the compile-time
gate in `GlobalQFImpl::Action`) against both a scalar per-QP qfunction
(`BoundaryMassQf`) and a vector-gradient per-QP qfunction
(`VectorGradientQf`). Both correctly return `false`, confirming that
`GlobalQFBackend` would reject these qfunctions with a `static_assert` if
instantiated.

**Fix:** UNFIXED. Either (a) add a per-QP qfunction adapter to
`GlobalQFBackend` so the same qfunction body works for both backends, or
(b) document that the two backends require different qfunction conventions.

---

## Bug 12: Cached derivative `Mult` and `Assemble` incorrect for nonlinear residuals

**Files:**
`fem/dfem/backends/local_qf/derivative_setup.hpp`,
`fem/dfem/backends/local_qf/derivative_apply.hpp`,
`fem/dfem/backends/local_qf/derivative_assemble.hpp`

**Root cause:** This is a consequence of Bug 1. For the specific case of
nonlinear `Gradient<U> -> Gradient<U>` residuals (the solid mechanics case), the
row-major cache indexing in `DerivativeSetup` writes Jacobian entries in one
order, while `DerivativeApply` reads them in a different order. The effect is
that the cached forward `Mult` operation computes a numerically wrong
matrix-vector product.

Additionally, `DerivativeApply` originally processed only the first output's
dimensions (`out_vdim_local[0]`, `out_op_dim_local[0]`) in its contraction loop
rather than iterating over all outputs via `for_constexpr<noutputs>`. This means
the residual result buffer was only partially filled for multi-output cases, and
the output offset was never advanced past the first output.

The `DerivativeAssemble` sparse-matrix assembly code uses the same cache and
therefore produces a numerically identical (but wrong) sparse matrix.

**Symptom:** The cached `GetDerivative(U, X)->Mult(dir, out)` disagrees with
the finite-difference residual perturbation by `O(10)` in norm for nonlinear
solid-mechanics qfunctions. The assembled Jacobian has the same error. The
raw Enzyme `fwddiff` of the qfunction itself is correct (verified by disabling
the cached path), confirming the bug is in cache write/read indexing.

**Tested by:** `Bug12_JacobianMismatch` (linear case),
`Bug12_MultipleFields` (multi-field case), and
`Bug12_NonlinearJacobianAndTranspose` (nonlinear case) in
`src/smith/differentiable_numerics/tests/test_dfem_bugs.cpp`. All three
compare cached `Mult` and assembled Jacobian action against finite differences
and fail on the unpatched branch.

**Fix:** Patch `0005` in `mfem_patches/`.

---

## Bug 13: Cached `MultTranspose` incorrect for multi-dependent-input residuals

**Files:**
`fem/dfem/backends/local_qf/derivative_apply_transpose.hpp`

**Root cause:** When multiple field operators reference the same field id as
the derivative direction (e.g., both `Value<0>` and `Gradient<0>` depend on the
same field), the `DerivativeApplyTranspose` implementation must sum the
transpose contribution from each dependent input and integrate each to the
correct trial-space DOFs independently, because each dependent input may have a
different DOF-to-quadrature map (e.g., B for Value vs G for Gradient).

The original implementation attempts to compute the full contraction
`result(j,m,q) = sum_{i,k} J^T(i,k,j,m,q) * dir(i+k*vdim,q)` into a single
flat buffer and then slice that buffer per-input for the DOF mapping step. This
approach has two defects:

1. The cache index formula uses a row-major layout inconsistent with
   `DerivativeSetup`'s column-major write ordering (same root cause as Bug 1).
2. For multi-dependent-input cases, after computing the full contraction, the
   code attempts to reshape the flat buffer per-input for
   `map_quadrature_data_to_fields`. But the `Reshape` call reads from
   overlapping memory regions because the buffer was dimensioned for the total
   trial dimension, not partitioned per-input. The `map_quadrature_data_to_fields`
   function reads `input_vdim * trial_op_dim * num_qp` consecutive entries, so
   slicing the flat buffer with an advancing pointer corrupts the integration.

**Symptom:** For a nonlinear vector residual where both `Value<0>` and
`Gradient<0>` are dependent inputs, `MultTranspose` disagrees with
`SparseMatrix::MultTranspose` (from the correct assembled Jacobian) by ~1.9 in
absolute norm. The adjoint identity `<J*t, v> = <t, J^T*v>` fails (e.g.,
1.179 vs 0.952). The forward `Mult` and assembled paths are self-consistent
and correct, isolating the bug to the transpose application.

For single-dependent-input cases, the transpose bug still exists due to the
row-major/column-major indexing mismatch (Bug 1 applied to the transpose path),
but the DOF-mapping slicing issue does not manifest because there is only one
slice.

**Tested by:** `Bug12_NonlinearJacobianAndTranspose` (multi-dependent-input:
both `Value<0>` and `Gradient<0>`), `Bug13_SingleInputTranspose`
(single-dependent-input: only `Gradient<0>`), and `Bug13_ValueOnlyTranspose`
(single-dependent-input: only `Value<0>`) in
`src/smith/differentiable_numerics/tests/test_dfem_bugs.cpp`. All three check
`MultTranspose` against assembled `J^T` and the adjoint identity
`<J*t, v> == <t, J^T*v>`; all three fail on the unpatched branch.

**Fix:** Patch `0006` in `mfem_patches/`.
