# Deflation Preconditioner for the Trust-Region CG Solver

## 1. Summary

A two-level preconditioner for the linearized TR-CG problem in
`equation_solver.cpp`. Built around a small, problem-aware coarse space `W`
whose columns span per-rank **affine deformations** (constant + linear-in-each-axis,
per component). For 3D vector fields: 12 columns per rank. Currently shipped as
additive (`M^{-1} r = M_J^{-1} r + W (W^T A W)^{-1} W^T r`); wired through
`Preconditioner::Deflation` in the existing factory.

State: implemented end-to-end, validated on a slender elasticity beam at
1/2/4/6/8 MPI ranks, faster than plain Jacobi and 5–6× faster than HypreAMG at
≥6 ranks on this problem. Bottleneck after Phase 5b sits **outside our code**
(HYPRE's per-iter SpMV in CG); the deflation preconditioner itself is ~12% of
the deflation-CG total at 8 ranks.

## 2. Math & Data Structures

### 2.1 Coarse space `W`

Per MPI rank `p`, per component `c ∈ {x,y,z}`, define `dim+1` modes:

| mode | values on owned tdofs of component `c` |
|------|----------------------------------------|
| 0    | `1` (constant)                          |
| 1..d | `X_axis_k(i)` (k-th nodal coordinate)   |

zero everywhere else. Counts: 3D ⇒ `12·P` global columns; 2D ⇒ `6·P`. Columns
have **disjoint supports** across ranks ⇒ `W` is block-diagonal across ranks ⇒
linearly independent automatically.

These span all rigid-body modes + constant-strain modes restricted to each
partition — exactly the slow modes that Jacobi/AMG struggle with at the
subdomain scale.

Built directly via `mfem::VectorFunctionCoefficient::ProjectCoefficient +
GetTrueDofs`. Ordering-agnostic (byNODES / byVDIM). High-order / curved
meshes: use `fes->GetMesh()->GetNodes()` for coords.

### 2.2 Deflation math (additive two-level)

For SPD `A`:
```
M^{-1} r  =  M_J^{-1} r  +  W (W^T A W)^{-1} W^T r
```
where `M_J` is a HypreSmoother (Jacobi default). Drops cleanly into the
existing `tr_precond` slot. `WᵀAW` is `m × m` (with `m = 12·P` in 3D),
factored once per outer Newton step.

### 2.3 Storage

| object | type | purpose |
|---|---|---|
| `W_local_` | `vector<mfem::Vector>` | canonical per-column storage; source for repacking |
| `W_mat_`   | `SparseMatrix` (`n × mpr`) | per-iter `MultTranspose` / `AddMult` in coarse correction |
| `W_dense_` | `DenseMatrix` (`n × mpr`)  | dense block consumed by `assembleWtAW` in SetOperator |
| `WtAW_`    | `DenseMatrix` (`m × m`), `CholeskyFactors` with LU fallback | dense replicated coarse matrix solve |

Each W column is nonzero on only one component's tdofs (`1/dim` of local rows)
⇒ `W_mat_` has `dim+1` nnz per row.

### 2.4 Files / API

- `src/smith/numerics/deflation.{hpp,cpp}` — `DeflationPreconditioner`.
- `src/smith/numerics/batched_matvec.{hpp,cpp}` — `assembleWtAW` (TripleProduct)
  + `batchedMatvec` (Loop / PackedDense, kept as comparison baselines).
- `src/smith/numerics/tests/test_deflation.cpp` — 4-rank end-to-end tests.
- `src/smith/numerics/tests/test_batched_matvec.cpp` — 4-rank correctness +
  multi-rank wall-time benchmarks for `assembleWtAW`.

```cpp
class DeflationPreconditioner : public mfem::Solver {
  // SetOperator(A): runs assembleWtAW(A, W_dense_, mpr, WtAW_); factors Cholesky if SPD; (re)builds smoother.
  // Mult(r, z):     z = smoother(r) + W (WtAW)^{-1} W^T r   (additive two-level)
  // setEssentialTrueDofs(ess): zeros W rows on ess dofs; invalidates WtAW factorization.
  // coarseSolve(r, z): z = -W (WtAW)^{-1} W^T r              (warm-start helper)
};

// Higher-level entry point for assembling W^T A W when W is column-distributed:
void assembleWtAW(const HypreParMatrix& A, const DenseMatrix& W_local,
                  int modes_per_rank, DenseMatrix& WtAW,
                  AssembleWtAWTimings* timings = nullptr);
```

### 2.5 Wiring through options

- `Preconditioner::Deflation` enum value in `solver_config.hpp`.
- `LinearSolverOptions::deflation_fes` (non-owning `ParFiniteElementSpace*`),
  required when `preconditioner == Deflation`. `buildPreconditioner` in
  `equation_solver.cpp` errors if null.
- `applyEssentialDofMask(ess_tdofs)` must be called by the caller (e.g., the
  beam test) before the first `SetOperator` when `A` comes from
  `FormLinearSystem` (which puts identity rows on essential dofs).

## 3. Completed Work

- **Per-rank affine basis** + projection through `VectorFunctionCoefficient`.
  Ordering-agnostic (byNODES / byVDIM); any FE order / FEC.
- **Parallel `WᵀAW` assembly** via the **triple-product / W-halo** strategy:
  one packed halo exchange of `W` (m/p doubles per halo entry), local DGEMMs
  for diagonal & per-neighbor blocks, single `MPI_Allreduce(m·m)`. Replaces
  what was `m` separate parallel matvecs + Allreduce.
- **Sparse `W_mat_`** for the per-iter coarse correction (`MultTranspose` /
  `AddMult` skip-zero by construction).
- **Inner HypreSmoother** (Jacobi default; type configurable).
- **Essential-DOF masking** for `FormLinearSystem`'s identity-on-constrained-rows
  convention.
- **Coarse-correction mode toggle** (`CoarseMode` enum, `setCoarseMode`).
  Four modes coexist in the codebase:
  - `Additive` (default, shipped) — `z = M_J^{-1} r + Π r`.
  - `AdditiveLocal` — drops the Allgather; uses only the `(p, p)` diagonal
    block of `WᵀAW`. Convergence degrades badly with rank count (§4.3).
  - `AdditiveSchwarz` — K=1 multi-step block-Jacobi-Schwarz refinement
    using one neighbor-only exchange (same trick as Phase 5b). Marginal iter
    improvement over `AdditiveLocal`; per-iter ~11% faster than `Additive`
    but iter count is ~2.65× worse (§4.3).
  - `Multiplicative` — symmetric V-cycle (`Π → smooth → Π`). Symmetric and
    correct, but per-iter cost ~2.8× higher because of the extra `A · z`
    matvecs (§4.3).
- **Wall-time instrumentation** in `DeflationPreconditioner`
  (`setop_matvec_time_`, `setop_factor_time_`, `setop_smoother_time_`,
  `mult_total_time_`, `mult_smoother_time_`, `mult_coarse_time_`,
  `mult_calls_`) gated by `// === TIMING BEGIN/END ===` for easy removal.
- **Factory wiring**: `Preconditioner::Deflation` in `solver_config.hpp`;
  branch in `buildPreconditioner` keyed on `LinearSolverOptions::deflation_fes`.
- **Tests** (all passing on 4 MPI ranks):
  - `Deflation.AffinePatchPureCoarse_byNODES` / `_byVDIM` — pure-coarse identity
    on affine RHS to machine precision.
  - `Deflation.AffinePatch_CG_OneIter_byVDIM` — CG converges in 1 iter on pure
    affine RHS.
  - `Deflation.CantileverBeam_PreconditionerComparison` — slenderer beam
    (~98k dofs); CG iter counts under Jacobi / HypreAMG / Deflation.
  - `BatchedMatvec.LoopMatchesSequentialApply` / `PackedDenseMatchesLoop` /
    `PackedDenseColumnDistributed` — `batchedMatvec` correctness baselines.
  - `BatchedMatvec.TripleProductMatchesReference` — `assembleWtAW` vs reference
    Loop assembly (mpr = 1/3/12 on small cube).
  - `BatchedMatvec.TripleProductElasticityBeam` — same, on slender elasticity
    beam (multi-neighbor halo, real fill pattern).
  - `BatchedMatvec.BeamTimingTripleProduct` — wall-time benchmark with
    internal breakdown.

### 3.1 Triple-product algorithm (load-bearing detail)

`W` is column-distributed: column `W_q` is nonzero only on its owner rank `s`.
The global dot collapses:
```
(W^T A W)(p, q) = W_p^T · (A_diag|_p · W_q|_p + A_offd|_p · W_q|_halo(p))
                  (only rank p contributes)
```
- **Diagonal block (p, p)** = `W_p^T · A_diag|_p · W_p` — fully local.
- **Off-diag block (p, s)** = `W_p^T · A_offd|_p · W_s|_halo_for_p` —
  nonzero only if `p` and `s` are halo-neighbors.

⇒ `WᵀAW` is **block-sparse on the MPI neighbor graph**.

Algorithm:
1. **Halo exchange of `W`**: pack `m/p` doubles per halo entry into one
   `MPI_Isend` per neighbor; one `MPI_Irecv` per neighbor; reuse HYPRE's
   `comm_pkg` send-map.
2. **Diagonal block** computed locally via row × inner-k loop +
   `mfem::MultAtB` (LAPACK) — overlapped with the halo wait.
3. **Per-neighbor off-diag blocks**: single pass through `A_offd`
   dispatches contributions to per-neighbor accumulators (using a precomputed
   `owner_of_offd_col[c]` lookup), then `mfem::MultAtB` for each block.
4. **`MPI_Allreduce(SUM)`** of the `m × m` result.

Per-rank send volume: `m/p × H` (vs `m × H` for the previous m-loop). At
8 ranks this is 12× less data per rank than the predecessor strategies.

## 4. Work Remaining

### 4.1 Per-rank coordinate centering for conditioning

Subtract per-rank, per-component mean coordinate from each linear-mode column:
```
X̃_i = X_i − X̄_p,c,   X̄_p,c = mean(X_i for i ∈ N_p^c)
```
Span is identical (constant + linear-in-X̃ spans the same 2D as constant +
linear-in-X). Conditioning of `WᵀAW` improves substantially when the mesh
bounding box is far from the origin or when `m` grows large.

- Compute once in `buildBasis()` after projecting the
  `VectorFunctionCoefficient`s; hold fixed for the preconditioner's lifetime.
- Centering is geometry-only; `WᵀAW` can rebuild per outer Newton step without
  recomputing centers.
- Hasn't bitten us yet on the beam at 1e-10 abs_tol, but **add it before** using
  on a mesh whose origin is far from the bounding box.

### 4.2 Nonlinear-solver integration test

End-to-end test of TR-CG-with-deflation on a representative nonlinear problem
(`shallow_arch_buckling` is the candidate). Verify the `WᵀAW` refactor cadence
(controlled by `cumulative_cg_iters_from_last_precond_update` at
`equation_solver.cpp` line 714) doesn't dominate the TR step cost.

### 4.3 Coarse-correction variants — experiments tried

Implemented as the `CoarseMode` enum on `DeflationPreconditioner`
(`setCoarseMode`). Default is `Additive`. Two alternatives explored and
benchmarked on the slenderer beam.

#### `AdditiveLocal` — drop the Allgather, use only diagonal block of `WᵀAW`

```
z = M_J^{-1} r + W (WtAW_pp)^{-1} W^T r   (per-rank only, no MPI)
```

`WtAW_pp` is the `(p, p)` mpr×mpr diagonal block; extracted in `SetOperator`
and factored separately. Saves the per-iter Allgather + `m × m` dense solve.

| ranks | Add (iter / t_solve) | AddLocal (iter / t_solve) | per-iter Δ | iter Δ |
|-------|----------------------|---------------------------|------------|--------|
| 1 | 665 / 2.72 s | 665 / 2.73 s | +0% | +0%    (no cross-rank coupling at p=1) |
| 2 | 534 / 1.25 s | 570 / 1.27 s | −5% | +7% |
| 4 | 714 / 1.08 s | 1155 / 1.93 s | −10% | +62% |
| 8 | 321 / 0.53 s | 922 / 1.55 s  | −6% | **+187%** |

Per-iter is mildly faster (Allgather + tiny dense solve isn't free), but iter
count explodes with rank count — dropping the off-diagonal blocks of
`(WᵀAW)^{-1}` discards exactly the cross-rank coupling that's removing the
global low-frequency error. Net: **2.9× slower at 8 ranks**. Kept in the
codebase as the `AdditiveLocal` mode for comparison.

#### `AdditiveSchwarz` — K=1 multi-step block-Jacobi-Schwarz, neighbor-only exchange

Same idea as Phase 5b's TripleProduct trick: replace the global Allgather with
one neighbor-only point-to-point exchange. The math:
```
u = WtAW_pp^{-1} c                              # local
exchange u with neighbors → u_s
alpha = u - WtAW_pp^{-1} Σ_s WtAW_{p,s} u_s    # local + small DGEMVs
```
This is symmetric by construction (matrix form: `M = WtAW_BJ^{-1} (2I − WtAW · WtAW_BJ^{-1})`,
which equals its transpose). Off-diagonal blocks `WtAW_{p,s}` are cached during
`SetOperator` from the assembled `WtAW_`; neighbor list = A's halo neighbors
(reused from HYPRE's `comm_pkg`).

| ranks | Add (iter / t_solve) | AddLocal (iter / t_solve) | AdditiveSchwarz (iter / t_solve) |
|-------|----------------------|---------------------------|----------------------------------|
| 1 | 665 / 2.63 s | 665 / 2.60 s | 665 / 2.59 s   (degenerate; no neighbors)  |
| 2 | 534 / 1.15 s | 570 / 1.23 s | 567 / 1.25 s |
| 4 | 714 / 1.09 s | 1155 / 1.75 s | 1186 / 1.83 s |
| 8 | 321 / 0.56 s | 922 / 1.45 s | 852 / 1.31 s |

Per-iter at 8 ranks: Add 1.74 ms, Schwarz 1.54 ms (~11% saved). So replacing
the Allgather with a neighbor round **does** trim per-iter cost — just
modestly, because at `m = 96` the Allgather was already cheap (~latency-bound,
small payload).

But: iter count is ~2.65× worse than Add at 8 ranks. One Schwarz refinement
isn't enough to reproduce the exact global coarse solve; the additional
neighbor info captures only direct couplings, missing second-neighbor effects
that matter for the bending mode. **Net: 2.3× slower overall.**

To approach Add's iter count one would need several (K ≥ 3?) Schwarz
iterations, with each one adding a neighbor exchange — wiping out the
per-iter savings. The sharper lesson:

> **The Allgather isn't the bottleneck.** At our `m`, the coarse correction's
> 200 μs/iter is dominated by local sparse work + tiny dense back-sub. The
> Add mode's global coarse solve is **cheap and exact** in one shot — hard to
> beat with anything approximate.

Kept in the codebase as the `AdditiveSchwarz` mode for comparison.

#### `Multiplicative` — symmetric V-cycle (coarse → smooth → coarse)

```
z = Π r;                       # coarse pre
z += M_J^{-1} (r - A z);       # smoother in middle
z += Π (r - A z);              # coarse post           (Π = W (WᵀAW)^{-1} W^T)
```

Symmetric by construction (required for PCG). Costs **2 extra parallel `A · z`
matvecs + 1 extra coarse correction** per `Mult` vs Additive.

> **Note**: the simpler asymmetric form `z = M^{-1} r + Π(r − A M^{-1} r)` was
> tried first and **fails to converge** with CG — `M_eff^{-1}` isn't symmetric
> because `Π A M_J^{-1} ≠ M_J^{-1} A Π` in general. The V-cycle above is the
> minimum symmetric multiplicative form that works with PCG.

| ranks | Add (iter / t_solve) | Mult-VCycle (iter / t_solve) | per-iter Δ | iter Δ |
|-------|----------------------|------------------------------|------------|--------|
| 1 | 665 / 2.72 s | 663 / 8.89 s | +227% | −0.3% |
| 2 | 534 / 1.25 s | 524 / 3.39 s | +176% | −2% |
| 4 | 714 / 1.08 s | 683 / 3.04 s | +195% | −4% |
| 8 | 321 / 0.53 s | 298 / 1.37 s | +178% | −7% |

Iter-count reduction is negligible (<7%); per-iter cost grows ~2.8× from the
two extra parallel matvecs. Net: **2.6–3.3× slower everywhere**. Kept in the
codebase as the `Multiplicative` mode for comparison.

#### Take-away

Additive-global is the right choice on this problem class. Two unrelated
reasons the alternatives both lose:
- Local coarse solve discards the global low-frequency coupling — exactly what
  deflation is there to remove.
- Multiplicative V-cycle's per-iter cost doubles+ because each extra `A · z`
  is itself a HYPRE parallel matvec — the same operation that dominates the
  outer CG iteration cost.

The Allgather + dense solve in the additive coarse correction is **cheap**
(small messages, tiny dense back-substitution), and "cheap" is what wins
when the alternative requires several `A · z` matvecs.

### 4.4 Bottleneck to investigate next: HYPRE / MFEM `HypreParMatrix::Mult(A, p)`

After Phase 5b, the dominant remaining cost is the SpMV called once per CG iter
on the search direction (~68% of the deflation total at 8 ranks). This is
outside our code by default. Angles worth probing:
- Is the matvec kernel using HYPRE's GPU/SIMD path, or just the plain CSR
  loop? Check `HYPRE_MEMORY_*` / `HYPRE_EXEC_*` configuration in the build.
- Does `A`'s sparsity have natural dim×dim per-node blocking that a BSR
  variant could exploit? See §4.5 and §5.6 for the analysis — the
  implementation cost is high and the realistic end-to-end win on our problem size is small
  because the SpMV at 8 ranks is comm-latency-bound rather than
  local-compute-bound.
- Is the comm package reused across consecutive matvecs inside the CG loop,
  or does HYPRE rebuild it each time? Cheap to verify.
- Alternative: s-step / pipelined CG variants that overlap or skip matvecs.

The coarse-correction Mult (~12% of total) was investigated in §4.3 and is
**not** a fruitful target — the two alternative variants both lose to the
Additive global baseline.

### 4.5 Block sparse SpMV demo status

Current workspace has a small `BSROperator` demo intended for profiling, not a
production matrix backend:

- `src/smith/numerics/bsr_operator.{hpp,cpp}` wraps a `HypreParMatrix`, converts
  local `diag` and `offd` CSR blocks to a dense-block BSR layout, and implements
  `Mult` / `AddMult`.
- `LinearSolverOptions::use_bsr_spmv` exists, and trust-region Jacobian assembly
  can wrap a monolithic `HypreParMatrix` with `BSROperator`
  (`bsr_block_size = 2` for 2D, `3` for 3D).
- `DeflationPreconditioner::SetOperator` can unwrap `BSROperator` to keep using
  the underlying `HypreParMatrix` for `W^T A W` and the inner Hypre smoother.
- The beam comparison test has a `Deflation_Add_BSR` path and checks the CG
  iteration count stays essentially unchanged.
- `test_bsr_operator.cpp` exercises the production `BSROperator` path on
  both 2D and 3D elasticity matrices and compares full parallel `Mult` /
  `AddMult` against HYPRE.

Current limitations:

- Assumes block-contiguous `byVDIM` scalar true-dof ordering.
- Assumes HYPRE offd column ids are received as complete, contiguous vector
  blocks. This is validated only indirectly by the current 2D/3D tests.
- Programmatic option only; no input-file parser wiring yet.
- Host-only kernel. The `Operator` wrapper shape is intentional so a later GPU
  backend can reuse the same solver wiring.

Simple plan:

1. Keep BSR as an optional `Operator` wrapper around monolithic
   `HypreParMatrix`. Do not thread a new matrix type through assembly.
2. Keep only `b = 2` and `b = 3`; all other block sizes fall back to HYPRE.
3. Add explicit eligibility/fallback checks before broader use: FES dimension,
   `byVDIM`, local divisibility, and offd block-contiguity.
4. Benchmark across the target problem suite with and without BSR. Track setup
   conversion time, solve time, per-iteration time, and residual agreement.
5. Preserve the GPU path: contiguous BSR storage, halo packing isolated from the
   local kernel, and no solver-facing API tied to host-only containers.

## 5. Future Ideas (deferred)

### 5.1 Matrix-free / on-the-fly basis

Don't materialize `W_local_`, `W_mat_`, or `W_dense_`; instead a `wAt(i, j)`
accessor returns each entry from `(dofComponent(i), nodal coord)` in O(1).
Would naturally skip zero rows (`dim` factor flop reduction on the local SpMV).

**Why deferred**: `SetOperator` matvec is already only 4% of total; doubling
its speed is 2% overall. Per-iter coarse correction uses sparse `W_mat_`
already, which captures the skip-zeros. Storage is small (~9 MB / rank at 97k
dofs). Revisit if `m_per_rank` grows substantially, or if memory becomes
tight, or if the diag SpMV is singled out as the next bottleneck.

The same accessor would enable applying centering on-the-fly.

### 5.2 Deflated CG (true Krylov-subspace deflation)

A different beast from the `Multiplicative` mode in §4.3. Instead of changing
the preconditioner, modify `TrustRegion::solveTrustRegionModelProblem` so
CG itself iterates in the `range(W)^⊥_A` subspace: project search directions
by `P_W = I − W(WᵀAW)⁻¹WᵀA` after each preconditioner call. Tighter
convergence bound (m coarse eigenvalues exactly removed from spectrum). Cost:
one extra coarse solve + one `A · W_j` matvec per CG iter (similar to the
Multiplicative V-cycle's per-iter cost penalty).

Not promising on this problem given §4.3's results: the symmetric
Multiplicative V-cycle already adds Π's projection structure inside the
preconditioner and gave no meaningful iter-count win. True deflated CG would
likely follow the same pattern. Revisit only if a problem class shows the
additive iter-count ceiling biting.

### 5.3 Sparse Allreduce of `WtAW`

`WtAW` is block-sparse on the neighbor graph, so `MPI_Allreduce(m·m)` does a
lot of redundant work. Replace with point-to-point gathers on the actual
neighbor graph. Cheap and worthwhile only if `m` (= 12P) grows large. At
`P ≤ 1000` the dense Allreduce is negligible.

### 5.4 Block / multifield problems

Currently `W` is built per displacement component. For systems with additional
fields (pressure, temperature, contact multipliers), deflation acts on the
displacement block only; other blocks pass through Jacobi unmodified. No
structural change needed in `DeflationPreconditioner`; just wire it as the
displacement-block precond in `block_preconditioner.cpp`.

### 5.5 Curved / high-order meshes

`buildBasis` already uses `fes->GetMesh()->GetNodes()` through the coefficient
projection. Affine-in-physical-coordinates is what we want, not
affine-in-reference. No change anticipated.

### 5.6 BSR / block-CSR storage for the system matrix

Standard CSR stores `(I, J, A)` with one column index per nonzero. BSR (Block
Sparse Row) treats the matrix as a sparse arrangement of small dense `b × b`
blocks: `(I[block_rows+1], J[nblocks], A[nblocks · b²])`. For 3D elasticity
with `byVDIM` ordering, every node-node coupling is naturally a 3×3 block,
giving `b = dim`.

**Theoretical wins** on the local SpMV:
- Index overhead per "effective" nnz drops ~9× (1 int per 9 doubles vs 1 int
  per 1 double).
- Inner kernel becomes a 3×3 dense gemv — fully unrollable, compiler
  vectorizes cleanly. 3 input components reused 3× per block.
- Published BSR-vs-CSR speedups on elasticity: 2–4× on local SpMV.

**Why deferred:**
1. **HYPRE owns the representation.** `hypre_ParCSRMatrix` is CSR internally;
   no public BSR mode. Custom local-SpMV path means converting `A_diag`/`A_offd`
   to BSR on each `SetOperator` and writing our own SpMV that the CG framework
   calls. Library swap (PETSc `MATBAIJ`, Trilinos `BlockCrsMatrix`) is a much
   bigger change.
2. **Local SpMV isn't 100% of the per-iter cost.** Rough estimate on the 98k-dof
   beam at 8 ranks: ~600k nnz per rank per SpMV at ~1 GB/s effective bandwidth
   ≈ 5 μs of "pure local compute". The other ~1100 μs of the per-iter ~1.1 ms
   is halo-exchange latency + CG framework overhead. A 3× local-SpMV speedup
   would shave ~10 μs / iter — single-digit % of the deflation total.
3. **byVDIM required.** Our beam already uses byVDIM, but anything `byNODES`
   would need conversion first.

**Cheaper related tweak**: `HYPRE_BoomerAMGSetNumFunctions(dim)` tells AMG
the matrix is a system PDE — improves AMG's coarsening for elasticity. Affects
iter count, not per-iter time. Free to try on the AMG baseline; doesn't help
deflation directly (we use Jacobi smoother).

**When to revisit**: at problem sizes ≥1M dofs per rank, the local SpMV fraction
of per-iter time grows, and BSR becomes more attractive. Re-evaluate after
moving past the toy beam.

### 5.7 Large `P` (`> few thousand`)

Dense `WtAW` is `144·P²` doubles (~1 GB at P=30k). Beyond that switch to a
sparse / redistributed factorization (SuperLU-DIST / STRUMPACK). Not relevant
at current scale.

## 6. `WᵀAW` Assembly: Algorithm Comparison

Three strategies have been implemented in `batched_matvec.{hpp,cpp}` for
forming `WᵀAW`. Only **TripleProduct** is wired into
`DeflationPreconditioner::SetOperator`; the other two stay in the codebase as
correctness baselines (used by the unit tests and the standalone benchmark).

All three start from the same column-distributed `W` (each column owned by one
rank, supported only on that rank's local dofs).

### 6.1 Loop

The original implementation. For each global column `q ∈ [0, m)`:
1. Owner rank fills `Wq` (a full local-tdof vector) with its `W_local_[q_local]`;
   other ranks fill zero.
2. Single parallel matvec `A · Wq → AWq` (one HYPRE halo exchange per `q`).
3. Each rank computes its row-band: `local_block(my_off + i, q) =
   W_local_[i] · AWq` for `i ∈ [0, mpr)` (local dot).

After the loop, one `MPI_Allreduce(SUM, m·m)` gathers the bands.

- **Comm**: `m` halo exchanges + 1 Allreduce(`m²`).
- **Work**: `m` full parallel SpMVs.
- **Wastes**: every halo exchange ships values from every rank even though the
  owner rank is the only one with nonzeros in `Wq`. Most of the wire traffic
  per matvec is zeros.

### 6.2 PackedDense

Collapses the `m` halo exchanges into one. `W` is passed as a single dense
block `X (n_local × m)`; each rank fills only its owned columns.
1. **One** packed `MPI_Isend`/`MPI_Irecv` per neighbor (k = m values per halo
   dof, packed row-major).
2. Local `A_diag · X → Y` SpMV (one wide CSR × dense block; inner-k loop).
3. `MPI_Waitall`, then `A_offd · recv_buf → Y` adds the halo contribution.
4. Caller does the `W^T · Y` dot products + Allreduce.

- **Comm rounds**: 1 halo exchange + 1 Allreduce (vs `m + 1`).
- **Per-rank comm volume**: still `m × H` per rank (we include zeros for the
  `m − m/p` non-owned columns).
- **Wastes**: comm volume not reduced; local SpMV does k-times work per A_diag
  nnz but cache-unfriendly because X is column-major (k stride = n).

Wall-time win over Loop: ~1.5–1.7× at moderate rank counts. Real but not
transformative — the comm-count saving dominates the modest local-work
penalty.

### 6.3 TripleProduct (shipped)

Skips materializing `A · W` entirely. Exploits that `(WᵀAW)(p, q)` requires
only data from rank `p` (since `W_p` is zero off-rank), giving a block-sparse
structure on the MPI neighbor graph. See §3.1 for the math.

1. **One halo exchange of `W` itself** — only `m/p` values per halo dof
   (each rank ships only its owned columns).
2. **Diagonal block (p, p)** = `W_p^T · A_diag|_p · W_p`: hand-rolled
   row × inner-mpr SpMV → `mfem::MultAtB` → small `mpr × mpr` block. Fully
   local; overlapped with the halo wait.
3. **Off-diagonal blocks (p, s)** for each halo neighbor `s`: single pass
   through `A_offd` dispatches contributions to per-neighbor accumulators
   (using a precomputed `owner_of_offd_col[c]` lookup), then `mfem::MultAtB`
   for each `mpr × mpr` block.
4. **`MPI_Allreduce(SUM, m²)`** of the result.

- **Per-rank comm volume**: `m/p × H` — **p× smaller** than the prior two.
- **Local SpMV becomes BLAS3-shaped**: sparse-CSR × dense-block, then a tiny
  dense × dense. Vectorizes and amortizes indexing.
- **No `AW` intermediate** to allocate.
- **Wins more as rank count grows**: per-rank work shrinks with `p`, where
  Loop and PackedDense per-rank work both grow with `m` = `12 · p`.

Wall-time win over Loop: 5.2× at 8 ranks (and growing). This is the strategy
shipped in `assembleWtAW`.

### Side-by-side

| | Loop | PackedDense | TripleProduct |
|---|---|---|---|
| Halo exchanges per SetOperator | `m` | 1 | 1 |
| Per-rank send volume | `m·H` | `m·H` (with zeros) | `m/p · H` |
| Local SpMV shape | `m` × (CSR · vec) | 1 × (CSR · dense block w/ k stride) | 1 × (CSR · dense block, mpr-wide) |
| Output | `AW` (then caller dots) | `AW` (then caller dots) | `WᵀAW` directly |
| 8-rank wall time | 99 ms | 65 ms | **19 ms** |

## 7. Current Timings

Rerun date: 2026-05-17. Build:
`build-Michaels-Mac-mini.local-darwin-sequoia-aarch64-apple-clang@17.0.0-release`.
The coarse solve now tries Cholesky first and falls back to LU if the coarse
matrix is not SPD/factorable. The production beam cases below use the same
additive coarse-solve behavior and keep the prior downselect conclusion.

### 7.1 End-to-end deflation (slenderer beam, Lx=16, 192×12×12, ~98k dofs, abs_tol = 1e-10)

| ranks | iters | t_setop | t_solve | matvec part | mult_total | coarse/iter | smoother/iter |
|-------|-------|---------|---------|-------------|------------|-------------|---------------|
| 1     | 665   | 0.129 s | 2.68 s  | 0.050 s     | 0.367 s    | 308 μs      | 244 μs        |
| 2     | 534   | 0.081 s | 1.34 s  | 0.036 s     | 0.216 s    | 202 μs      | 201 μs        |
| 4     | 714   | 0.049 s | 1.12 s  | 0.024 s     | 0.188 s    | 146 μs      | 117 μs        |
| 6     | 406   | 0.044 s | 0.70 s  | 0.018 s     | 0.129 s    | 209 μs      | 108 μs        |
| 8     | 320   | 0.040 s | 0.49 s  | 0.023 s     | 0.103 s    | 201 μs      | 119 μs        |

Comparison with the other preconditioners at the same tolerance:

| ranks | Jacobi (iters / total) | HypreAMG (iters / total) | **Deflation (iters / total)** |
|-------|------------------------|--------------------------|-------------------------------|
| 1 | 763 / 2.86 s | 481 / 9.80 s | 665 / **2.81 s** |
| 2 | 764 / 1.53 s | 463 / 5.90 s | 534 / **1.42 s** |
| 4 | 747 / **1.06 s** | 481 / 3.73 s | 714 / 1.17 s |
| 6 | 746 / 1.99 s | 486 / 4.82 s | 406 / **0.74 s** |
| 8 | 747 / 1.02 s | 497 / 3.78 s | 320 / **0.53 s** |

Iter counts trend down with rank count for deflation (more per-rank affine
bases ⇒ better tiling of the bending mode). The 4-rank dip is a METIS
partition-shape artifact; not actionable without overriding the partitioner.

Coarse-correction-mode comparison (see §4.3 for analysis):

| ranks | Add (iter / t_solve) | AddLocal (iter / t) | Schwarz (iter / t) | Mult-VCycle (iter / t) |
|-------|----------------------|---------------------|--------------------|------------------------|
| 1 | 665 / 2.68 s | 665 / 2.72 s | 665 / 2.69 s | 663 / 7.60 s |
| 2 | 534 / 1.34 s | 570 / 1.28 s | 567 / 1.26 s | 524 / 3.36 s |
| 4 | 714 / 1.12 s | 1155 / 1.81 s | 1186 / 1.81 s | 683 / 2.94 s |
| 6 | **406 / 0.70 s** | 995 / 1.65 s | 1017 / 1.75 s | 382 / 1.78 s |
| 8 | **320 / 0.49 s** | 921 / 1.39 s | 852 / 1.32 s | 298 / 1.25 s |

`Additive` (default) is the only mode worth carrying forward for target
multi-rank runs. `AdditiveLocal` / `Schwarz` can be slightly cheaper at 2 ranks,
but their iteration counts collapse by 4+ ranks. `Multiplicative` cuts a few CG
iterations but loses badly on total time from the extra `A·z` matvecs.

### 7.2 Bottleneck breakdown (8 ranks)

| component                                   | time     | share of deflation total |
|---------------------------------------------|----------|--------------------------|
| CG framework + A · p (HYPRE SpMV per iter)  | ~0.36 s  | ~68% |
| `Mult` coarse correction (321 calls)        | ~0.065 s | ~12% |
| `Mult` smoother (Jacobi, 321 calls)         | ~0.038 s | ~7%  |
| `SetOperator` matvec (one-shot)             | ~0.023 s | ~4%  |
| `SetOperator` factor + smoother setup       | ~0.000 s | ~0%  |

Dominant cost is **outside our code** (HYPRE SpMV). The next-largest piece
in our scope is the per-iter coarse correction — investigated through the
alternative `CoarseMode` variants in §4.3; they lose to the default Additive
global form at the target rank counts, so no further optimization is warranted
there.

### 7.3 `assembleWtAW` standalone benchmark (same beam, m = 12·ranks)

| ranks | m  | Loop     | PackedDense | TripleProduct | TP vs Loop | TP vs Pack |
|-------|----|----------|-------------|---------------|------------|------------|
| 1     | 12 | 43.8 ms  | 40.2 ms     | 52.7 ms       | 0.83×      | 0.76×      |
| 2     | 24 | 44.2 ms  | 28.9 ms     | 34.1 ms       | 1.30×      | 0.85×      |
| 4     | 48 | 56.8 ms  | 34.7 ms     | 21.9 ms       | 2.59×      | 1.58×      |
| 6     | 72 | 91.2 ms  | 46.9 ms     | 21.8 ms       | 4.18×      | 2.15×      |
| 8     | 96 | 102.8 ms | 68.0 ms     | **19.4 ms**   | **5.30×**  | **3.50×**  |

Internal breakdown of TripleProduct (ms / call):

| ranks | total | halo (wait) | diag | offd | allreduce |
|-------|-------|-------------|------|------|-----------|
| 1     | 52.7  | 0.0         | 52.6 | 0.0  | 0.0       |
| 2     | 34.1  | 0.02        | 27.3 | 6.7  | 0.01      |
| 4     | 21.9  | 0.03        | 14.7 | 3.7  | 3.3       |
| 6     | 21.8  | 2.8         | 12.6 | 5.9  | 0.4       |
| 8     | 19.4  | 1.7         | 9.7  | 4.2  | 3.8       |

Diag SpMV is the largest single cost and shrinks with rank count; halo wait
only becomes significant once many neighbors appear. `Loop` and `PackedDense`
remain in the codebase as comparison baselines + correctness references.

### 7.4 Post-change validation rerun

After adding the Cholesky-first / LU-fallback coarse solve, the full
`test_deflation` suite was rerun at 4 MPI ranks. All four tests passed. The beam
numbers from that final validation run were:

| mode | iters | t_setop | t_solve | mult_total | coarse | smoother |
|------|-------|---------|---------|------------|--------|----------|
| Jacobi | 747 | 0.000 s | 1.03 s | - | - | - |
| HypreAMG | 481 | 0.000 s | 3.74 s | - | - | - |
| Deflation Additive | 714 | 0.047 s | 1.14 s | 0.188 s | 0.106 s | 0.082 s |
| Deflation AddLocal | 1155 | 0.047 s | 1.75 s | 0.274 s | 0.143 s | 0.131 s |
| Deflation Schwarz | 1186 | 0.046 s | 1.80 s | 0.284 s | 0.151 s | 0.133 s |
| Deflation Mult-VCycle | 683 | 0.046 s | 2.96 s | 2.039 s | 1.948 s | 0.091 s |

This matches the broader rerun: exact additive-global remains the only coarse
mode worth carrying into production work.

## 8. Recommended Next Steps

1. **Prepare the PR**
   - Keep `CoarseMode::Additive` and TripleProduct as the production deflation
     paths.
   - Include `BSROperator` as an opt-in demo path, with `test_bsr_operator.cpp`
     covering 2D and 3D correctness against HYPRE.
   - Leave benchmark-only coarse modes and timing instrumentation in place until
     after the problem-suite profiling run, then remove or guard them.

2. **Run the profiling suite**
   - Compare Jacobi, AMG, Deflation, and Deflation+BSR on representative 2D and
     3D problems.
   - Record setup time, solve time, iterations, per-iteration time, and true
     residual.
   - Treat BSR as useful only if end-to-end solve time improves, not just local
     SpMV timing.

3. **Harden BSR only if the data justifies it**
   - Add explicit FES/order/offd-contiguity eligibility checks and clean fallback.
   - Add input-option parser wiring for `use_bsr_spmv` / `bsr_block_size`.
   - Keep the storage/kernel split compatible with a future GPU backend.

4. **Finish nonlinear deflation validation**
   - Measure the existing trust-region integration on `shallow_arch_buckling`.
   - Confirm preconditioner rebuild cadence does not dominate.
   - Keep per-rank coordinate centering and coarse leftmost-direction hooks as
     required pieces for robust nonlinear runs.

5. **Defer until scale demands it**
   - Sparse/distributed `W^T A W` allreduce/factorization.
   - Matrix-free `W` accessors.
   - Pipelined or s-step CG.

## 9. Build / Run Notes

- Build: `cd build-Michaels-Mac-mini.local-darwin-sequoia-aarch64-apple-clang@17.0.0-release && make -j4 [target]`
- Run: `DYLD_LIBRARY_PATH=$(echo /Users/mrtupek/dev/smith-tpls/apple-clang-17.0.0/netlib-lapack-3.12.1-*/lib) mpirun -n 4 ./tests/test_deflation`
- `liblapack.3.dylib` is not in the default rpath; the `DYLD_LIBRARY_PATH`
  prefix is required until env is fixed.
