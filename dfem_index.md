# dFEM Interface Index

Quick reference for the differentiable FEM framework in `mfem::future`.
All symbols live in `namespace mfem::future` unless noted.

---

## Related documents

- [reverse_plan.md](reverse_plan.md) — design plan for the true reverse-mode
  (adjoint) path for scalar QoI sensitivities
- [dfem_bugs.md](dfem_bugs.md) — bugs found and fixed on the
  `dfem-multiple-outputs` branch (cache indexing, loop bounds, height, include
  paths, diagonal assembly, Enzyme guard)

---

## Table of Contents

1. [Namespace & Required Includes](#1-namespace--required-includes)
2. [Field IDs and FieldDescriptor](#2-field-ids-and-fielddescriptor)
3. [FieldOperators](#3-fieldoperators)
4. [Quadrature Functions](#4-quadrature-functions)
5. [DifferentiableOperator — construction](#5-differentiableoperator--construction)
6. [DifferentiableOperator — Mult](#6-differentiableoperator--mult)
7. [MultiVector](#7-multivector)
8. [DerivativeOperator — JVP, VJP, Assemble](#8-derivativeoperator--jvp-vjp-assemble)
9. [ParameterSpace](#9-parameterspace)
10. [Backends: GlobalQFBackend vs LocalQFBackend](#10-backends-globalqfbackend-vs-localqfbackend)
11. [Scalar QoI Sensitivity Pattern](#11-scalar-qoi-sensitivity-pattern)
12. [Reverse-mode AD — `MakeReverseGradientOperator` / `MakeReverseScalarSumOperator`](#12-reverse-mode-ad--makereversegradientoperator--makereversescalarsumoperator)

---

## 1. Namespace & Required Includes

```cpp
#include "fem/dfem/doperator.hpp"            // DifferentiableOperator, DerivativeOperator
#include "fem/dfem/fielddescriptor.hpp"      // FieldDescriptor
#include "fem/dfem/fieldoperator.hpp"        // Value, Gradient, Weight, Identity, Sum
#include "fem/dfem/parameterspace.hpp"       // ParameterSpace, UniformParameterSpace, ParameterFunction
#include "fem/dfem/backends/local_qf/prelude.hpp"   // LocalQFBackend
#include "fem/dfem/backends/global_qf/prelude.hpp"  // GlobalQFBackend (default)
#include "linalg/multivector.hpp"            // MultiVector (namespace mfem, not mfem::future)

using namespace mfem::future;
using mfem::future::tensor;
```

Requires MPI (`MFEM_USE_MPI`). Enzyme derivatives require `MFEM_USE_ENZYME`.

---

## 2. Field IDs and FieldDescriptor

**Header:** `fem/dfem/fielddescriptor.hpp`

Each field is identified by a compile-time integer ID. A `FieldDescriptor` pairs an ID
with its space (FES, QuadratureFunction, or ParameterSpace).

```cpp
// Declare field IDs as constexpr ints (or an enum)
static constexpr int U = 0;           // solution field
static constexpr int COORDS = 1;      // mesh coordinates
static constexpr int QDATA = 2;       // quadrature-point data
static constexpr int PARAM = 3;       // scalar parameter

// Build descriptor lists: infds = all inputs, outfds = all outputs
ParGridFunction *nodes = static_cast<ParGridFunction*>(mesh.GetNodes());

std::vector<FieldDescriptor> infds {
    {U,      &fes},
    {COORDS, nodes->ParFESpace()},
    {QDATA,  &qfunc},          // QuadratureFunction*
    {PARAM,  &param_space},    // ParameterSpace*
};
std::vector<FieldDescriptor> outfds {
    {V, &fes},
};
```

Supported `data` variants (see `FieldDescriptor::data_variant_t`):
- `const FiniteElementSpace *`
- `const ParFiniteElementSpace *`
- `const QuadratureFunction *`
- `const ParameterSpace *`

> **Note:** Every field ID used in the `inputs`/`outputs` tuples of `AddDomainIntegrator`
> must appear in either `infds` or `outfds`. The union must match exactly.

---

## 3. FieldOperators

**Header:** `fem/dfem/fieldoperator.hpp`

FieldOperators appear in the `inputs` and `outputs` tuples passed to `AddDomainIntegrator`.
They specify what quantity of a field is interpolated to quadrature points.

| Operator | Template param | Description |
|---|---|---|
| `Value<ID>{}` | field ID | Pointwise field values at QPs |
| `Gradient<ID>{}` | field ID | Reference-coordinate gradient at QPs |
| `Identity<ID>{}` | field ID | Pass-through for ParameterSpace fields |
| `Weight{}` | — | Integration rule weights |
| `Sum<ID>{}` | field ID | Sum operator on output |

```cpp
// Typical input/output tuples for a diffusion-like residual:
auto inputs = tuple{
    Gradient<U>{},        // du/dxi at each QP
    Gradient<COORDS>{},   // Jacobian J at each QP
    Weight{},             // integration weights
};
auto outputs = tuple{
    Gradient<V>{},        // test function gradient (output field)
};
```

The quadrature function receives arguments in the same order as the inputs tuple,
followed by outputs tuple. The output arguments are written to (passed by non-const ref).

---

## 4. Quadrature Functions

The quadrature function is the pointwise kernel. It operates on per-QP data.
Two writing styles are supported depending on the backend:

### GlobalQFBackend style — returns tuple, operates per QP

```cpp
// Takes one value per QP as tensors, returns tuple of outputs.
// Enzyme differentiates this function directly.
struct MyQFunc {
    MFEM_HOST_DEVICE inline
    auto operator()(
        const tensor<real_t, DIM> &dudxi,    // Gradient<U>
        const tensor<real_t, DIM, DIM> &J,   // Gradient<COORDS>
        const real_t &w) const               // Weight
    {
        const auto invJ = inv(J);
        const auto dudx = dudxi * invJ;
        return tuple{ dudx * transpose(invJ) * det(J) * w };
    }
};
```

### LocalQFBackend style — loops over QPs via tensor_array

```cpp
// Requires: #include "linalg/tensor_arrays.hpp"
// Takes tensor_array for each input and output, loops explicitly.
struct MyQFuncLocal {
    MFEM_HOST_DEVICE inline
    void operator()(
        const tensor<real_t, DIM> &dudxi,    // single-QP: local backend
        const tensor<real_t, DIM, DIM> &J,
        const real_t &w,
        tensor<real_t, DIM> &out) const      // output written in place
    {
        const auto invJ = inv(J);
        out = dudxi * invJ * transpose(invJ) * det(J) * w;
    }
};
```

> **Rule:** Return `tuple{...}` for GlobalQFBackend; write to output args (void return)
> for LocalQFBackend.

---

## 5. DifferentiableOperator — construction

**Header:** `fem/dfem/doperator.hpp`

```cpp
DifferentiableOperator dop(infds, outfds, pmesh);
```

Add an integrator (domain or boundary):

```cpp
// Specify which input field IDs to differentiate w.r.t.
// The index_sequence must use the *field ID* values, not positions.
auto deriv_ids = std::integer_sequence<size_t, U>{};   // want dR/dU

MyQFunc qfunc;

// Default backend is GlobalQFBackend (forward-mode Enzyme per call)
dop.AddDomainIntegrator(
    qfunc,
    inputs,          // tuple of FieldOperators for inputs
    outputs,         // tuple of FieldOperators for outputs
    ir,              // IntegrationRule
    domain_attrs,    // Array<int> attribute marker
    deriv_ids);      // which derivatives to enable (needs MFEM_USE_ENZYME)

// Use LocalQFBackend explicitly (caches Jacobian at QPs for reuse):
dop.AddDomainIntegrator<LocalQFBackend>(
    qfunc, inputs, outputs, ir, domain_attrs, deriv_ids);

// Boundary integrator (no backend template param supported yet):
dop.AddBoundaryIntegrator(
    qfunc, inputs, outputs, ir, bdr_attrs, deriv_ids);
```

Optional configuration:
```cpp
dop.DisableTensorProductStructure();  // force non-tensor (generic) assembly
dop.SetMultLevel(DifferentiableOperator::LVECTOR); // skip T<->L, operate on L-vecs
```

---

## 6. DifferentiableOperator — Mult

Input and output vectors are passed as `MultiVector` (preferred) or `BlockVector`.
Each block corresponds to a `FieldDescriptor` in order.

```cpp
// Build input MultiVector referencing existing T-vectors / GridFunctions
Vector u_t, nodes_t;
x.GetTrueDofs(u_t);
nodes->GetTrueDofs(nodes_t);

MultiVector X{u_t, nodes_t};   // blocks ordered to match infds

// Build output MultiVector
Vector res_t(fes.GetTrueVSize());
MultiVector Y{res_t};

dop.Mult(X, Y);
// res_t now contains the assembled residual T-vector
```

Access blocks: `Y[0]`, `Y[1]`, … (zero-indexed, returns `Vector&`).

---

## 7. MultiVector

**Header:** `linalg/multivector.hpp` (namespace `mfem`, not `mfem::future`)

Holds an array of `Vector`s with independent sizes. Blocks can be owned or reference
external objects.

```cpp
// Construct referencing existing vectors (no copy):
MultiVector mv{vec_a, vec_b, vec_c};

// Construct with owned storage sized by array:
Array<int> sizes = {n0, n1, n2};
MultiVector mv(sizes);

// Access:
mv[0] = 1.0;
mv.NumBlocks();   // 3

// Make a block reference into a monolithic vector at an offset:
mv.MakeRef(1, base_vec, offset, size);

// Make all blocks reference a list of vectors:
mv.MakeRef(vec_a, vec_b);
```

---

## 8. DerivativeOperator — JVP, VJP, Assemble

**Header:** `fem/dfem/doperator.hpp` (returned by `DifferentiableOperator::GetDerivative`)

Obtain the derivative operator at a linearization point `X`:

```cpp
// X is the MultiVector (or BlockVector) of all input field T-vectors
auto dop_du = dop.GetDerivative(U, X);
// dop_du is std::shared_ptr<DerivativeOperator>
```

`GetDerivative` triggers `DerivativeSetup` (caches J at QPs) if using LocalQFBackend.

### Jacobian-Vector Product (JVP / forward mode)

```cpp
// direction and result are T-vectors for the derivative field
// result_mv is MultiVector aligned with outfds
Vector direction(fes.GetTrueVSize());
MultiVector result_mv{...};

dop_du->Mult(direction, result_mv);
// computes J * direction, stores in result_mv[0], result_mv[1], ...
```

### Vector-Jacobian Product (VJP / adjoint / transpose)

```cpp
// seed is a T-vector in the OUTPUT space (e.g. residual space)
// adjoint_result is a T-vector in the INPUT space (field U space)
MultiVector seed_mv{seed_vec};   // or BlockVector
Vector adjoint_result(fes.GetTrueVSize());

dop_du->MultTranspose(seed_mv, adjoint_result);
// computes J^T * seed, stores in adjoint_result
```

> **Note:** `MultTranspose` uses the transpose of the *forward-mode* Jacobian cached at
> QPs. It is NOT reverse-mode AD. For a scalar QoI, seed with a ones vector.

### Sparse Matrix Assembly

```cpp
SparseMatrix *A = nullptr;
dop_du->Assemble(A);      // allocates and fills A

HypreParMatrix *pA = nullptr;
dop_du->Assemble(pA);     // parallel version
```

### Diagonal Assembly

```cpp
Vector diag(fes.GetTrueVSize());
dop_du->AssembleDiagonal(diag);
// Requires single output field; useful for Jacobi preconditioners
```

---

## 9. ParameterSpace

**Header:** `fem/dfem/parameterspace.hpp`

Use when a field is not a finite element function but a uniform coefficient or
quadrature-point data that lives outside any FES.

### UniformParameterSpace — coefficient uniform over all QPs

```cpp
// vdim scalar values per QP, tensor-product layout
UniformParameterSpace param_space(mesh, ir, /*vdim=*/1);

ParameterFunction param_func(param_space);
param_func = 3.14;   // set uniform value

FieldDescriptor param_fd{PARAM, &param_space};
```

### Custom ParameterSpace

Subclass and implement:
```cpp
class MyParamSpace : public ParameterSpace {
public:
    int GetTrueVSize() const override { return ...; }
    int GetVSize()     const override { return ...; }
    const Operator* GetB()  const override { ... }  // values-to-QPs
    const Operator* GetBt() const override { ... }  // QPs-to-values (transpose)
};
```

`GetB`/`GetBt` act as the interpolation operator to/from quadrature points.
Default `GetProlongationMatrix` and `GetElementRestriction` return identity operators.

---

## 10. Backends: GlobalQFBackend vs LocalQFBackend

| | GlobalQFBackend (default) | LocalQFBackend |
|---|---|---|
| **Header** | `backends/global_qf/prelude.hpp` | `backends/local_qf/prelude.hpp` |
| `has_cached_derivative` | `false` | `true` |
| **Forward action** | Enzyme JVP per call | Explicit loop over QPs |
| **Derivative setup** | None | Caches J at all QPs on `GetDerivative` |
| **JVP** | Re-runs Enzyme each call | Uses cached J |
| **VJP / MultTranspose** | — | Uses cached J^T |
| **Assemble / Diagonal** | — | Uses cached J |
| **Qfunc signature** | Returns `tuple{...}`, single QP | `void`, writes output args, single QP |
| **Best for** | One-off residual evals | Newton loops, multiple J applications |

Select at `AddDomainIntegrator`:
```cpp
dop.AddDomainIntegrator<LocalQFBackend>(...);   // explicit
dop.AddDomainIntegrator<GlobalQFBackend>(...);  // explicit (same as default)
dop.AddDomainIntegrator(...);                   // GlobalQFBackend by default
```

---

## 11. Scalar QoI Sensitivity Pattern

To compute `dQoI/d(field U)` where QoI is a scalar integral over the domain:

**Step 1 — Define QoI as an output field with a scalar FES (or Sum output):**

```cpp
// Option A: output to an H1 space and sum externally
// Option B: use Sum<V>{} output operator — integrates to a single scalar
static constexpr int U = 0, COORDS = 1, V = 2;

std::vector<FieldDescriptor> infds{{U, &fes}, {COORDS, nodes->ParFESpace()}};
std::vector<FieldDescriptor> outfds{{V, &fes}};

DifferentiableOperator qoi_op(infds, outfds, pmesh);

struct QoIQFunc {
    MFEM_HOST_DEVICE inline
    auto operator()(const tensor<real_t> &u,
                    const tensor<real_t, DIM, DIM> &J,
                    const real_t &w) const {
        return tuple{ u * det(J) * w };   // e.g. integral of u
    }
};

QoIQFunc qfunc;
auto deriv_ids = std::integer_sequence<size_t, U>{};
qoi_op.AddDomainIntegrator<LocalQFBackend>(
    qfunc,
    tuple{Value<U>{}, Gradient<COORDS>{}, Weight{}},
    tuple{Value<V>{}},
    ir, domain_attrs, deriv_ids);
```

**Step 2 — Get derivative operator at linearization point:**

```cpp
MultiVector X{u_t, nodes_t};
auto dqoi_du = qoi_op.GetDerivative(U, X);
```

**Step 3 — Apply J^T with seed = ones to get gradient:**

```cpp
// Seed = 1 in the output (residual) space = total derivative of scalar QoI
Vector seed(fes.GetTrueVSize());
seed = 1.0;
MultiVector seed_mv{seed};

Vector dqoi_du_vec(fes.GetTrueVSize());
dqoi_du->MultTranspose(seed_mv, dqoi_du_vec);
// dqoi_du_vec[i] = dQoI / d(u[i])
```

> **Performance note:** `MultTranspose` here applies `J^T * 1` where `J` is assembled
> from forward-mode Enzyme passes (one pass per local DOF per element in `DerivativeSetup`).
> Cost is O(n_local_dofs_per_element × forward_cost), not O(forward_cost) as true
> reverse-mode would be. Adequate for moderate-size problems; for large-scale optimization
> loops, the reverse-mode path in §12 is ~n_local_dofs× cheaper, and emits gradients
> w.r.t. multiple input fields in a single sweep.

---

## 12. Reverse-mode AD — `MakeReverseGradientOperator` / `MakeReverseScalarSumOperator`

User-space extensions that synthesize a `DifferentiableOperator` whose `Mult`
returns reverse-mode (VJP) sensitivities of a scalar QoI w.r.t. multiple input
fields in a single sweep, driven directly off a **standard dFEM residual
qfunction** (no need to hand-write a scalar density). Built on
`__enzyme_autodiff` with the seed hardcoded to 1.

**Header (single file, header-only):** `mfem/miniapps/dfem/reverse_derivative.hpp`

Planned smith location: `smith/src/smith/differentiable_numerics/reverse_derivative.hpp`.
Nothing here needs to be in `libmfem` — copy the header into smith and update
the relative includes to land it there. Until that move, the file is exercised
by `mfem/miniapps/dfem/dfem-qoi-timing.cpp`.

**Namespace:** `mfem::future::reverse`. **Requires** `MFEM_USE_ENZYME` and `MFEM_USE_MPI`.

### Capabilities at a glance

| Factory | Use when | Physics output FOps | Seed |
|---|---|---|---|
| `MakeReverseGradientOperator<V, DiffIds...>` | QoI = ⟨v, R(u; …)⟩ for some user-supplied seed field `v` (incl. `v ≡ 1` on an FE space, i.e. compliance against a unit load) | `Value<…>`, `Gradient<…>`, … (rebound to `V` and contracted via `dot`/`ddot`) | `v` field |
| `MakeReverseScalarSumOperator<DiffIds...>` | QoI = Σ_qp W(u; …) — a globally summed scalar density (e.g. Neo-Hookean strain energy / compliance, total mass, dissipated power) | all must be `Sum<…>` (`static_assert`) | implicit `1` |

Both factories return `std::unique_ptr<DifferentiableOperator>` and use
`LocalQFBackend` internally. The `Mult` output `MultiVector` has one field per
`DiffIds...`, with the gradient living in the same FE/parameter space as the
matching primal input.

Demo & validation: `dfem-qoi-timing.cpp`.
- `MakeReverseGradientOperator`: gradients agree with the forward-mode
  `GetDerivative + MultTranspose` path to ~1e-16 (machine precision).
- `MakeReverseScalarSumOperator`: gradients agree with central finite
  differences to ~1e-9 (FD truncation floor at δ=1e-5).

### What it does

Given a **standard dFEM residual qfunction** (the same kind you'd register with
`AddDomainIntegrator`) plus a seed field id `V` and a list of input field ids
to differentiate, this synthesizes (at compile time) a `DifferentiableOperator`
whose `Mult` returns the VJP ∂(⟨v, R(u; …)⟩)/∂(field) for every requested input
field in a single Enzyme reverse-mode sweep — no Jacobian cache, no separate
`MultTranspose` call per field, and no need for the user to hand-write a scalar
QoI density.

Internally the factory:
1. Rebinds each physics output FOp to field id `V` (e.g. `Gradient<U>` →
   `Gradient<V>`) and appends them as non-differentiated inputs.
2. Wraps the physics qf so it computes its outputs and then a scalar density
   = Σ_k ⟨output_k, v_dual_k⟩ using `dot`/`ddot` per output rank.
3. Reverse-differentiates that synthesized density with output seed = 1.

Compared to the `GetDerivative(..)->MultTranspose(seed)` path in §8:
- One sweep, not one-per-input-field.
- No `DerivativeSetup` cost (no cached per-QP Jacobian).
- Faster per call when you want many input-field gradients, especially as the
  number of differentiable inputs grows.

### Usage

The user writes a **standard residual qfunction** — exactly the form you'd pass
to `DifferentiableOperator::AddDomainIntegrator`:

```cpp
struct NeoHookeanResidualQF {
    MFEM_HOST_DEVICE inline
    void operator()(
        const tensor<real_t, DIM, DIM> &dudxi,   // Gradient<U>
        const real_t &E_val,                     // Value<EFLD>
        const tensor<real_t, DIM, DIM> &J,       // Gradient<COORDS>
        const real_t &w,                         // Weight
        tensor<real_t, DIM, DIM> &dvdxi) const   // Gradient<U> output (residual)
    {
        const auto invJ  = inv(J);
        const auto dudx  = dudxi * invJ;
        const auto sigma = E_val * nh_stress(dudx);
        dvdxi = sigma * transpose(invJ) * det(J) * w;
    }
};
```

Then build the reverse operator with one factory call. The first template
argument is the **seed field id** `V`; the rest are the **input field ids** to
differentiate, in the order they should appear on the output `MultiVector`:

```cpp
#include "reverse_derivative.hpp"   // see header path above
using namespace mfem::future;

static constexpr int U = 0, V = 1, EFLD = 2, COORDS = 3;

std::vector<FieldDescriptor> rev_in
{ {U, &vfes}, {V, &vfes}, {EFLD, &efes}, {COORDS, mfes} };

auto rev = mfem::future::reverse::MakeReverseGradientOperator
           <V, U, EFLD>(                           // seed V; diff w.r.t. U,E
               NeoHookeanResidualQF{},
               tuple{Gradient<U>{}, Value<EFLD>{},
                     Gradient<COORDS>{}, Weight{}},  // physics inputs
               tuple{Gradient<U>{}},                  // physics outputs
               rev_in,
               *ir, all_domain_attr, pmesh);

Vector grad_u(vfes.GetTrueVSize()), grad_E(efes.GetTrueVSize());
MultiVector X{u_t, v_t, E_t, nodes_t};
MultiVector Y{grad_u, grad_E};
rev->Mult(X, Y);
// grad_u[i] = ∂⟨v, R⟩/∂u_i,   grad_E[i] = ∂⟨v, R⟩/∂E_i
```

### Rules and constraints

- The physics qfunction is a normal dFEM residual qf: inputs by `const&`,
  outputs by mutable `&`, no scalar density argument needed.
- Each physics output FOp `Op<id>` is automatically rebound to `Op<VFieldId>`
  and added as a non-differentiated input. Only `Value<>`, `Gradient<>`,
  `Sum<>`, `Identity<>` can be rebound (a `Weight` output makes no sense as a
  v-dual).
- Inner-product convention: scalar outputs `*`, rank-1 outputs `dot`, rank-2
  outputs `ddot`. The `V`-field's FE space must therefore be compatible with
  the output's rank.
- Each `DiffFieldId` template arg must correspond to one input `FieldOperator`
  of the physics qf, and must NOT equal `VFieldId`. Output `FieldDescriptor`s
  are looked up by `.id` in `rev_in`.
- Backend is `LocalQFBackend` internally.

### `Sum<>`-output physics: `MakeReverseScalarSumOperator`

When the physics qf's outputs are all `Sum<...>` (a per-QP scalar that is
globally summed to a single number), the "seed" v ≡ 1 is implicit — there's
no V field, no inner product per QP, and the synthesized density just
accumulates the physics outputs into `q_out` directly. Use the sibling
factory:

```cpp
// Per-QP scalar (will be globally summed by the synthesized density).
struct EnergyDensityQF {
    MFEM_HOST_DEVICE inline
    void operator()(const tensor<real_t, DIM, DIM> &dudxi,
                    const real_t &E_val,
                    const tensor<real_t, DIM, DIM> &J,
                    const real_t &w,
                    real_t &W_qp) const { /* ... */ }
};

std::vector<FieldDescriptor> in{{U,&vfes},{EFLD,&efes},{COORDS,mfes}};
auto rev = mfem::future::reverse::MakeReverseScalarSumOperator
           <U, EFLD>(                              // diff w.r.t. U and E
               EnergyDensityQF{},
               tuple{Gradient<U>{}, Value<EFLD>{},
                     Gradient<COORDS>{}, Weight{}},
               tuple{Sum<-1>{}},                    // must all be Sum<>
               in,
               *ir, all_domain_attr, pmesh);

MultiVector Y{grad_u, grad_E};
rev->Mult(MultiVector{u_t, E_t, nodes_t}, Y);
// grad_u[i] = ∂(Σ_qp W)/∂u_i,   grad_E[i] = ∂(Σ_qp W)/∂E_i
```

The `Sum<-1>{}` (or any `Sum<id>{}`) in `physics_output_fops` is consumed
only by a `static_assert` — the wrapper doesn't build a forward Sum<> dop;
it just confirms via the FOp type that the user really wants v ≡ 1
semantics. Output FieldDescriptors for the gradient outputs are taken from
the matching entries of `in` (same machinery as the dual form).

`dfem-qoi-timing.cpp` validates this against central finite differences of
a NeoHookean strain-energy QoI: agreement to ~1e-9 (FD truncation floor at
δ=1e-5).

### When to prefer this over §8's `MultTranspose`

- You want sensitivities w.r.t. multiple input fields at once.
- You don't otherwise need the full Jacobian (no Newton iterations on this
  operator, no sparse-matrix assembly).
- Output is scalar QoI semantics — seed = 1.

If you instead need a true vector residual operator and reuse of its Jacobian
across multiple right-hand sides, stay with `GetDerivative` + `MultTranspose`.

---

## Full Minimal Example

```cpp
#include "mfem.hpp"
#include "fem/dfem/doperator.hpp"
#include "fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;

static constexpr int U = 1, COORDS = 2;

struct DiffusionQF {
    MFEM_HOST_DEVICE inline
    auto operator()(const tensor<real_t, 2> &dudxi,
                    const tensor<real_t, 2, 2> &J,
                    const real_t &w) const {
        const auto invJ = inv(J);
        return tuple{ dudxi * invJ * transpose(invJ) * det(J) * w };
    }
};

// In main():
auto *nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
std::vector<FieldDescriptor> infds{{U, &fes}, {COORDS, nodes->ParFESpace()}};
std::vector<FieldDescriptor> outfds{{U, &fes}};

DifferentiableOperator dop(infds, outfds, pmesh);

DiffusionQF qfunc;
auto deriv_ids = std::integer_sequence<size_t, U>{};
dop.AddDomainIntegrator<LocalQFBackend>(
    qfunc,
    tuple{Gradient<U>{}, Gradient<COORDS>{}, Weight{}},
    tuple{Gradient<U>{}},
    ir, all_attrs, deriv_ids);

// Residual evaluation
Vector u_t, nodes_t, res_t(fes.GetTrueVSize());
u_gf.GetTrueDofs(u_t);
nodes->GetTrueDofs(nodes_t);
MultiVector X{u_t, nodes_t}, Y{res_t};
dop.Mult(X, Y);

// Jacobian-vector product
auto J = dop.GetDerivative(U, X);
Vector dir(fes.GetTrueVSize()), jv(fes.GetTrueVSize());
dir.Randomize();
MultiVector JV{jv};
J->Mult(dir, JV);

// Sparse matrix assembly (needs LocalQFBackend + MFEM_USE_ENZYME)
HypreParMatrix *A = nullptr;
J->Assemble(A);
```
