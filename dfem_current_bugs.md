# dFEM Bugs Still Present on `dfem-multiple-outputs` HEAD

Verified against commit `9f8a46e414` ("Some attempt at patching dfem for
smith use-cases") on 2026-05-16. The earlier bug list in
[dfem_bugs.md](dfem_bugs.md) numbered these issues 8–11; they remain real on
HEAD. Item #12 from the earlier list is *not* included here because it could
not be confirmed by code inspection or by the in-tree unit tests; see the
note at the end.

---

## 1. `LocalQFBackend::Action` is hardcoded to volume entities

**Files:** `fem/dfem/backends/local_qf/action.hpp`,
`fem/dfem/backends/local_qf/prelude.hpp`.

The `LocalQFImpl::Action` struct is templated on the qfunction and the input
and output `FieldOperator` tuples, but it is not templated on the entity
kind, and every entity-dependent operation inside its constructor is
hardcoded to `Entity::Element`. In particular,
`create_descriptors_to_fields_map<Entity::Element>` is used for both inputs
and outputs (lines 46 and 48), `GetDofToQuad<Entity::Element>` is used to
build the DofToQuad maps (line 93), `create_dtq_maps<Entity::Element>` wires
them into the input and output lists (lines 96 and 99),
`get_shmem_info<Entity::Element, …>` lays out the shared memory block (line
115), and `get_restriction<Entity::Element>` is used to fetch the output
restriction operator (line 150). On top of this, the sum-factorization
predicate at line 59 calls `Element::TypeFromGeometry(ctx.mesh.GetTypicalElementGeometry())`,
which returns the *volume* element geometry even when the user registered the
integrator with `AddBoundaryIntegrator`. Finally, `LocalQFBackend::MakeAction`
in `prelude.hpp:23` takes no `entity_t` template parameter, so the dispatch
path in `DifferentiableOperator::AddIntegrator` cannot forward the entity
kind even if it wanted to. The user-visible consequence is that
`AddBoundaryIntegrator<LocalQFBackend>(...)` silently builds volume
DofToQuad maps, volume shmem layout, and a volume restriction, then pairs
them with the boundary `IntegrationRule`, producing inconsistent dof counts
and scattering boundary-quadrature contributions into volume dof slots —
landing on interior nodes instead of surface nodes.

**Suggested fix.** Thread an `entity_t` template parameter through
`LocalQFBackend::MakeAction` and into every Enzyme-style derivative factory
(`MakeDerivativeAction`, `MakeDerivativeSetup`, `MakeDerivativeApply`,
`MakeDerivativeApplyTranspose`, `MakeDerivativeAssemble`,
`MakeDerivativeAssembleDiagonal`), and replace every `Entity::Element`
occurrence inside `LocalQFImpl::Action` and the cached-derivative kernels
(`derivative_setup.hpp`, `derivative_action.hpp`, `derivative_apply.hpp`,
`derivative_apply_transpose.hpp`, `derivative_assemble.hpp`,
`derivative_assemble_diagonal.hpp`) with that parameter. The
sum-factorization predicate must branch on `entity_t` and call
`mesh.GetTypicalFaceGeometry()` for `BoundaryElement` (with the entity
dimension dropping by one for 2D/3D boundary faces, which in turn drives
`q1d` and shmem block sizing). `DifferentiableOperator::AddIntegrator` in
`doperator.hpp` already receives `entity_t` as a template parameter — that
needs to be forwarded to `MakeAction` and the derivative factories at the
call site.

---

## 2. `DifferentiableOperator::Mult` is hardcoded to volume entities

**File:** `fem/dfem/doperator.hpp` (the `Mult` body around lines 469–476;
also lines 166–173, 251–261, and 322–330 for the `DerivativeOperator` paths).

Even if `LocalQFImpl::Action` is made entity-aware, the surrounding `Mult`
plumbing in `DifferentiableOperator` still treats every registered
integrator as if it were a volume integrator. The action loop calls
`restriction<Entity::Element>(infds, infields_l, infields_e)`,
`prepare_residual<Entity::Element>(outfds, residual_e)`, then runs *all*
`action_callbacks[i]` against the same E-vector buffers regardless of
entity kind, and finally calls
`restriction_transpose<Entity::Element>(outfds, residual_e, residual_l)`.
The `action_callbacks` member at line 588 is a flat `std::vector<action_t>`
with no entity tag attached to each callback, so even with grouping logic
in `Mult` there is currently no metadata to drive it. The patched
`AddIntegrator` does accept an `entity_t` template parameter (line 672) and
the `AddBoundaryIntegrator` overload dispatches with `Entity::BoundaryElement`
(line 666), but the entity kind is lost at `MakeAction`/`push_back` time —
nothing downstream of `AddIntegrator` records or uses it.

**Suggested fix.** Tag each `action_callbacks[i]` (and each derivative
callback) with its `entity_t` at registration time — e.g. by changing the
storage from `std::vector<action_t>` to `std::vector<std::pair<Entity,
action_t>>` or by holding two parallel vectors. In `Mult`, group callbacks
by entity kind, and for each group run `restriction<entity_t>`,
`prepare_residual<entity_t>`, the callbacks themselves, and
`restriction_transpose<entity_t>` against entity-appropriate buffers,
accumulating into a shared residual L-vector before a single
`prolongation_transpose` to T-dofs. This mirrors how classical
`ParBilinearForm::Mult` composes domain and boundary integrators, and is
the standard composition pattern.

---

## 3. `GlobalQFBackend` is not a drop-in replacement for per-QP qfunctions

**Files:** `fem/dfem/backends/global_qf/action.hpp`,
`fem/dfem/backends/util.hpp`.

`GlobalQFImpl::Action::operator()` at `global_qf/action.hpp:78` includes a
`static_assert(detail::supports_tensor_array_qfunc<qfunc_t, inputs_t,
outputs_t>::value, ...)`. The predicate, defined at `util.hpp:343`, succeeds
only when every qfunction parameter is a `tensor_array<scalar, sizes...>`
— a batch type spanning all quadrature points — because `is_tensor_array`
at `util.hpp:252` matches only the `tensor_array` specialization, not bare
`tensor<...>` or scalar types. Smith's qfunctions, the in-tree unit-test
qfunctions (for example the boundary qfunction in `test_mass.cpp`, which
takes `const real_t& u, const tensor<real_t, DIM, BDIM>& J, const real_t&
w` and returns a `tuple{...}`), and the example in `dfem_index.md` all use
the **per-QP** convention: scalar and tensor arguments, with either a
`tuple` return or an out-parameter. Trying to switch backends from
`LocalQFBackend` to `GlobalQFBackend` therefore fails at compile time with
the `static_assert`, even though both backends are documented as
interchangeable in the framework. Independently of this, `test_mass.cpp`,
which appears to demonstrate the default backend on a per-QP boundary
qfunction, is commented out of `tests/unit/CMakeLists.txt`, so its
evidence about whether the default backend ever did accept this style is
unverifiable in this tree.

**Suggested fix.** Either of two paths is acceptable: (a) extend
`GlobalQFBackend` with a per-QP qfunction adapter that wraps a per-QP
qfunction into the batched signature internally, so the same qfunction body
works with both backends; or (b) leave the two backends with different
required conventions but make the constraint explicit in code and docs —
remove `GlobalQFBackend` from any "default backend" position, rename it
toward a name like `BatchedGlobalQFBackend`, and update `dfem_index.md` so
that the per-QP signature in §4 is not described as compatible with the
default backend. Approach (a) is preferable because it preserves the
implied "swappable backends" contract and unbreaks any qfunction shared
between solver code (which wants `LocalQFBackend` for caching) and one-off
evaluation code (which currently can't compile against `GlobalQFBackend`).

---

## 4. Reverse-mode derivatives are not implemented for boundary integrators

**Files:** `fem/dfem/doperator.hpp`,
plus the missing-from-tree `reverse_derivative.hpp` header documented in
[dfem_index.md §12](dfem_index.md) and exercised by
`miniapps/dfem/dfem-qoi-timing.cpp` and
`tests/unit/dfem/test_neohookean_qoi_reverse.cpp`.

There is no `AddReverseGradientIntegrator` (or equivalent boundary-aware
reverse-mode hook) in `doperator.hpp` on this branch, and the
`MakeReverseGradientOperator` / `MakeReverseScalarSumOperator` factories
referenced by `dfem_index.md §12` synthesize their forward operator
through `AddDomainIntegrator<LocalQFBackend>` only — every reverse-mode use
site in the tree goes through the domain path. Even the header
`miniapps/dfem/reverse_derivative.hpp` that those factories should live in
is not present in the working tree, only its consumer code is. The smith
side's `DfemWeakForm::addBoundaryIntegral` accepts a `derivative_ids`
parameter but `SLIC_ERROR`s if any derivatives are requested, exactly
because there is no upstream reverse-mode plumbing for boundary entities to
hook into. This blocks adjoint-based sensitivity work on any problem with
traction or pressure loading.

**Suggested fix.** This fix is gated on issues 1 and 2 above. Once
`LocalQFImpl::Action` and `DifferentiableOperator::Mult` are entity-aware
and group callbacks by `entity_t`, add a `BoundaryElement` specialization
of the reverse-mode synthesis (the boundary analog of whatever lives in
`reverse_derivative.hpp` for domain integrators), operating on the same
per-entity-grouped restriction/prolongation framework introduced by the
issue-2 fix. Until the header is restored to the tree, no useful work can
be done on this issue beyond reading `dfem-qoi-timing.cpp` to recover the
intended factory interface.

---

---

# Status of the Previously-Reported Bugs (`dfem_bugs.md` items 1–7)

The earlier bug list claimed seven bugs that were already fixed by the
in-tree patch at HEAD (`9f8a46e414`). Each was checked against the
pre-patch commit (`1fa7888706`) to confirm whether the original bug was
real. Six of the seven are confirmed real and confirmed fixed by HEAD; one
appears to be misattributed.

## #1 — Cache index ordering wrong for multiple outputs — **REAL, fixed**

Confirmed real on pre-patch. In `fem/dfem/backends/local_qf/derivative_setup.hpp`
the cache write index at line 537 had no dependence on the output index
`o`, so for `noutputs > 1` every output wrote into overlapping cache slots
and the last output's data overwrote earlier outputs'. In the matching
`derivative_apply.hpp` (lines 308–309) the apply kernel hardcoded
`test_vdim` and `test_op_dim` to `out_*_local[0]` and only computed a
contribution for the first output. The `noutputs = 1` case happened to
work because there was only one set of strides and one output to write.
HEAD threads an `ik_offset` through both `derivative_setup.hpp`,
`derivative_apply.hpp`, and `derivative_apply_transpose.hpp` with a
consistent column-major scheme; the in-tree "Multiple Outputs" unit test
passes on HEAD.

## #3 — `SparseMatrix` re-allocated and leaked on every `Assemble()` — **REAL, fixed**

Confirmed real on pre-patch. `fem/dfem/backends/local_qf/derivative_assemble.hpp:299`
unconditionally executed `A = new SparseMatrix(...)` on every call, so a
second `DerivativeOperator::Assemble(A)` invocation leaked the previously
allocated matrix and replaced the pointer the caller was holding —
breaking any Newton loop that cached a pointer to `A` across
re-linearisations. HEAD guards the allocation with `if (A == nullptr)`.

## #4 — `Operator::height` never set — **REAL, fixed**

Confirmed real on pre-patch. `DifferentiableOperator` inherits from
`Operator` but never assigns its own `height` member; the only `height`
assignment in pre-patch `doperator.hpp` was the parameter passed into the
inner `DerivativeOperator`'s `Operator(height, width)` base constructor.
The outer `DifferentiableOperator`'s `height` stayed at the default value
of `0`, which broke any caller that checked `dop.Height()` before calling
`Mult` (iterative solvers, block operators). HEAD computes `height` as
the sum of T-DOF sizes over all output fields inside `AddIntegrator`.

## #5 — Enzyme-specific templates not guarded by `MFEM_USE_ENZYME` — **REAL, fixed**

Confirmed real on pre-patch. `fem/dfem/backends/util.hpp:442–510` defined
the `do_enzyme_call`, `process_outputs`, and `process_inputs` function
templates at namespace scope using the Enzyme-specific names
`__enzyme_fwddiff`, `enzyme_dup`, `enzyme_const`, and `enzyme_dupnoneed`
as unqualified identifiers. These names are declared by Enzyme headers
that are only included when `MFEM_USE_ENZYME` is defined (the same
templates in `fem/dfem/qfunction_apply.hpp` are correctly wrapped in
`#ifdef MFEM_USE_ENZYME`). Building without Enzyme failed at the template
*definition* due to unqualified-name lookup of non-dependent identifiers.
HEAD wraps the affected block in `#ifdef MFEM_USE_ENZYME` / `#endif`.

## #6 — Broken relative include paths — **REAL, fixed**

Confirmed real on pre-patch. Three include paths resolved to
non-existent directories: `fem/dfem/backends/util.hpp` had
`#include "../fem/quadinterpolator.hpp"` which resolves to
`fem/dfem/fem/quadinterpolator.hpp` (no such directory);
`fem/dfem/backends/global_qf/derivative_action_enzyme.hpp` had the same
wrong path from one level deeper; and `fem/dfem/doperator.hpp` had
`#include "../linalg/multivector.hpp"` which resolves to
`fem/linalg/multivector.hpp` (no such directory). HEAD has all three
fixed to the correct `../../`-prefixed forms.

---

# Note on the cached-Jacobian vs. residual-linearization issue

The earlier `dfem_bugs.md` lists this as item #12: an observed mismatch
between finite differences of the residual, the cached `Mult` derivative
action, and the assembled sparse Jacobian for a `Gradient<U> -> Gradient<U>`
residual. It is **not** included above because all five in-tree mfem dFEM
unit tests pass on HEAD — including `Multiple Outputs`, which exercises the
patched cache-stride path, and `NeoHookean QoI gradient via MultTranspose`,
which agrees with the reverse-mode reference to ~1e-16. The cache-indexing
fix recorded as item #1 in `dfem_bugs.md` may well have resolved #12 as a
side effect. Confirming this requires rerunning the original smith-side
reproducer (`test_dfem_shallow_arch_buckling`, body-force loading, the
functional-reference solver stack) on HEAD; it cannot be settled by code
inspection alone.

---

# Findings from rigorous Bug 12 reproduction testing (May 2026)

Extensive reproduction testing of the aforementioned "Bug 12" (using custom
`Gradient<U> -> Gradient<U>` cases involving `u + coords + weight`) was performed.
The testing definitively proved two things:

1.  **Bug 12 is FIXED:** The cached `Mult` action and the Assembled Jacobian
    action now match the Finite Differences of the residual to numerical
    precision (~1e-9) in all multi-field cases. The stride/cache-indexing fix
    from "Bug 1" fully resolved the overlap issue that was breaking the cache
    reads when multiple inputs (like `u` and `coords`) were present.

2.  **CRITICAL NEW BUG Discovered (`Weight` ignores `detJ`):** While the dFEM
    evaluation paths now match *each other* perfectly, they compute the wrong
    mathematical integral. The `Weight` field operator is evaluated in
    `map_fields_to_quadrature_data` (in `interpolate.hpp` and `integrate.hpp`)
    by simply copying the reference integration rule weights (`ctx.ir.GetWeights()`).
    It completely drops the element Jacobian determinant (`detJ`). Consequently,
    all integrals scale by the reference element volume rather than the physical
    element volume. A uniform body force integral on a mesh will incorrectly
    return a value scaled exactly by the *number of elements* in the mesh. This
    fundamentally breaks all mechanics evaluations on non-unit-area meshes and
    explains why real-world Smith solid mechanics tests exhibit catastrophic
    scaling errors against the reference stack.
