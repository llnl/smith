# Plan: Energy Mortar Contact Smoothing

C1-smooth 2D energy mortar contact with Enzyme AD for exact Jacobians.

## Key Files

| Purpose | Path |
|---------|------|
| Math writeup | `doc/smoothing_math.tex` |
| Kernel + options (header) | `tribol/src/tribol/physics/EnergyMortar.hpp` |
| Kernel AD wrappers | `tribol/src/tribol/physics/EnergyMortar.cpp` |
| Shared enums | `tribol/src/tribol/physics/EnergyMortarTypes.hpp` |
| Adapter (assembly) | `tribol/src/tribol/physics/EnergyMortarAdapter.cpp` |
| Public MFEM interface | `tribol/src/tribol/interface/mfem_tribol.hpp` |
| Contact search params | `tribol/src/tribol/common/Parameters.hpp` |
| Coupling scheme (binning) | `tribol/src/tribol/mesh/CouplingScheme.cpp` |
| Interface pair finder | `tribol/src/tribol/search/InterfacePairFinder.cpp` |
| Smith-side caller | `src/smith/physics/contact/contact_interaction.cpp` |
| Smith contact config | `src/smith/physics/contact/contact_config.hpp` |

## Architecture

- `EnergyMortarOptions` -- unified options struct (penalty, smoothing, quadrature).
- All kernel math is inline in `energy_mortar_kernel_detail` namespace in the header.
- `EnergyMortarAdapter` receives pairs via `setInterfacePairs()` each Newton iteration.
- Public API: `setMfemEnergyMortarOptions(cs_id, SmoothingType, PenaltySmoothing, del)`.

## Completed Work (summary)

1. **Integration bounds smoothing** -- C1 ramp (Hermite/Quadratic) clamping to [-0.5, 0.5].
2. **Penalty ramp smoothing** -- centered quadratic C1 ramp over [-del/2, +del/2].
3. **Area regularization + row elimination** -- `A_reg = A + 1e-10`; rows with `A <= 0` eliminated.
4. **Enzyme Hessian symmetry fix** -- inline free function for varying-quad kernel, fixed indexing.
5. **Normal opposition smoothing** -- `eta = (dot < 0) ? -dot*dot : 0.0` on g_tilde only. Eta on area reverted (cancellation in g_tilde/A ratio).
6. **Header/cpp dedup + unified options** -- eliminated ~650 lines of duplication.
7. **Unit tests** -- smooth_bounds, penalty_ramp, Enzyme kernels, vanishing overlap.
8. **Issue E fix** -- unified residual/Jacobian eta path (single source of truth via `gap_area_kernel`).
9. **Diagnostics** -- Hessian asymmetry check on rejected trust-region steps; per-call contact diagnostics that confirmed Issue F as the root cause of convergence stalling.
10. **Issue G investigation** -- tried removing `A <= 0` row elimination; reverted (reintroduced spurious forces). Toggle was upstream in search, not in adapter.

## Issue F: active-set toggling (CONFIRMED ROOT CAUSE of convergence stall)

Diagnostics confirmed a deterministic 2-state limit cycle: `cand` count toggles
96<->95 on bit-noise trial steps, one pair flips across the geomFilter cutoff,
producing identical `rho = -0.387905` every rejection. Trust region collapses to
~1e-18 and churns for 200+ iterations.

### F1: relax normal cutoff (quick fix)

Relax `cos(100deg) = -0.1736` to `cos(90deg) = 0.0` in `InterfacePairFinder.hpp:69`.
Moves the threshold away from typical contact configurations.

### F2: baseline union current pair set (architectural fix)

**Goal**: prevent loss of contact interactions during Newton iterations within a
load step, without accumulating spurious pairs from bad trial steps.

**Concept**: maintain two pair sets:
- **baseline**: snapshot from the first search of the load step (seeded from
  converged previous load step geometry; initial configuration on first load step).
- **current**: fresh search results each Newton iteration.

Adapter receives `baseline U current` each iteration. Baseline pairs can never be
lost mid-load-step. Pairs from bad trial steps that aren't in baseline or current
are naturally discarded.

**Why not pure monotonic**: pure union-of-all-iterations accumulates spurious pairs
from rejected trial steps. Baseline union current is self-correcting -- spurious
pairs vanish when the next search doesn't find them.

**Why baseline must be sticky**: penalty forces may not fully resolve contact in
one step. Without baseline, a partial step can move geometry enough that the search
drops a pair, its penalty vanishes, and the contact surface leaks.

**Data flow** (current, to be changed):
```
performBinning():  wipes m_interface_pairs, rebuilds spatial index, runs geomFilter
apply():           moves m_interface_pairs into adapter (no history retained)
```

**New design**:
```
load_step_begin:
  cs.performBinning()        // fills m_interface_pairs from search
  cs.snapshotBaseline()      // m_baseline_pairs = copy of m_interface_pairs

newton_iter k = 0, 1, 2, ...:
  update geometry
  cs.performBinning()        // overwrites m_interface_pairs (current set)
  cs.apply()                 // passes baseline U current to adapter
```

**Changes required:**

| # | File | Change |
|---|------|--------|
| 1 | `CouplingScheme.hpp` | Add `m_baseline_pairs`, `m_use_baseline_union`. API: `snapshotBaseline()`, `setUseBaselineUnion(bool)`. |
| 2 | `CouplingScheme.cpp` `performBinning()` | No change -- still overwrites `m_interface_pairs`. |
| 3 | `InterfacePairFinder.cpp` | No change -- search writes into `m_interface_pairs` as before. |
| 4 | `CouplingScheme.cpp` `apply()` | When flag on: merge `baseline U current` (dedup by `(e1,e2)`), pass to adapter. |
| 5 | `mfem_tribol.hpp/.cpp` | Add `setMfemUseBaselineUnion(cs_id, bool)` and `snapshotMfemBaselinePairs()`. |
| 6 | `contact_data.cpp` | Add `beginLoadStep()` calling `snapshotMfemBaselinePairs()`. |
| 7 | smith QS solver outer loop | Call `beginLoadStep()` once per load step. Re-call on bisection/retry. |
| 8 | `Parameters.hpp` | Optional: `bool use_baseline_union = false;` default. |

**Merge**: dedup key is `(e1<<32)|e2` in `unordered_set`. O(N_baseline + N_current),
N in the hundreds -- negligible cost.

**Edge cases:**
- First load step: baseline from initial configuration search (trustworthy).
- Load-step bisection: `beginLoadStep()` re-snapshots from converged geometry.
- Redecomp: invalidates element numbering; tie `snapshotBaseline()` to redecomp boundaries.
- Adjoint: doesn't call `performBinning`; inherits forward pair set. Verify no change needed.

**Implementation order:**
1. Add storage + API to `CouplingScheme`. Compile.
2. Add merge in `apply()`, gated by flag. Compile.
3. Add MFEM facade + wire smith side.
4. Enable for ironing_2D, verify diagnostics.
5. Run Hertzian + patch test regression.

### F3: trust-region floor (defensive)

When trust region collapses below meaningful progress, bail out instead of
churning. `tr_floor = max(min_tr_size_abs, rel_floor * ||X||)`.

- File: `equation_solver.cpp`. Existing `min_tr_size = 1e-13` may not be wired
  into the rejection loop as a bailout -- confirm and add if missing.
- Add check after `tr_size *= t1` shrink. On hit, break inner loop, surface
  failure to outer driver (which already handles "Newton failed -> bisect").
- Orthogonal to F2: some problems will always collapse the trust region.

## Remaining Nonsmoothness Issues (lower priority)

### Issue B/D (unified): projection bound singularity near perpendicular

`get_projections` calls `find_intersection(A0, A1, B_endpoints, nB)` where
`det = tA x nB -> 0` as A perp B. Projection bounds fly off, get hard-clamped
to [-0.5, 0.5] by `std::min/max` -- creating derivative kinks.

**Fix**: softclamp projection bounds with smooth min/max:
`smin(a,b) = 0.5*(a+b - sqrt((a-b)^2 + eps^2))`. Derivatives go to zero before
hitting the wall. Note: `gn` itself does NOT diverge inside the kernel (`det = 1`
by construction in the quadrature-loop call).

### Issue C: hard bailout in find_intersection

`if (|det| < 1e-12) { return p; }` -- C0 discontinuity. Landmine at extreme
perpendicular angles. Fix with smooth blend: `w = det^2/(det^2+eps^2)`.
Low priority.

### Issue H: penalty-ramp stiffness from tiny del

Default `del = 1e-5` gives `H'' = -1e5`. With high k, enormous Jacobian curvature
in a microscopic band. Already configurable; document the tradeoff (too-narrow ->
curvature spike; too-wide -> permitted penetration -> geometry inversion).

### Issue A: area nonsmoothness near perpendicular

Eta on area reverted (cancellation with g_tilde). Remaining ideas: different
powers on g_tilde vs area (A2), post-assembly smooth cutoff (A3), or independent
area smoothing via integration bounds (A4). Not currently blocking.

## Known Bugs

- **Bus error on 3rd timestep of ironing_2D**: crash at `compute_gtilde_and_area`
  with invalid address alignment. Suspect stale pointers after redecomp or heap
  corruption. Pre-existing. Needs ASAN investigation.

## TODO

- [ ] **F1**: relax normal cutoff `cos(100deg) -> cos(90deg)`
- [ ] **F3**: trust-region floor bailout in `equation_solver.cpp`
- [ ] **F2**: baseline U current pair set (architectural)
- [ ] B/D: softclamp projection bounds
- [ ] H: document `penalty_smoothing_del` tradeoff per-problem
- [ ] C: smooth `find_intersection` bailout (low priority)
- [ ] Expose `binning_proximity_scale` through smith `ContactOptions`
- [ ] Diagnose bus error (ASAN)
- [ ] Run patch tests with both smoothing types
- [ ] Run Hertzian example, compare Hermite vs Quadratic convergence
