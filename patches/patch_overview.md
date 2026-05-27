# Smooth Mortar Patch Overview

This directory contains a split patch series for testing contact robustness changes independently. Patch `0003` is intentionally absent: the planned area-regularization / zero-area handling change was already present on `develop`, so no patch is needed for it.

Apply order:

```bash
git am patches/smooth_mortar_split/0001-*.patch \
       patches/smooth_mortar_split/0002-*.patch \
       patches/smooth_mortar_split/0004-*.patch \
       patches/smooth_mortar_split/0005-*.patch \
       patches/smooth_mortar_split/0006-*.patch \
       patches/smooth_mortar_split/0007-*.patch
```

For the current reduced experiment, apply only `0001`, `0002`, and `0004`.

## 0001 - Share energy mortar kernels with Enzyme

Purpose: make the Energy Mortar element evaluation path and Enzyme derivative path use the same kernel implementation.

Main changes:
- Moves shared geometry, projection, smoothing, quadrature, gap, and area kernels into inline implementations in `EnergyMortar.hpp`.
- Adds `EnergyMortarTypes.hpp` for shared `SmoothingType` and `PenaltySmoothing` definitions.
- Refactors `EnergyMortar.cpp` to call the shared `gap_area_kernel()` for element-level contact data.
- Updates finite-difference helpers/tests to use the renamed options storage.

Expected effect:
- Reduces residual/Jacobian inconsistency risk from duplicated kernel logic.
- Keeps the normal-alignment factor unchanged: `eta = dot` for opposing normals.

Explicitly excluded:
- No `ENERGY_AREA_PENALTY`.
- No baseline-union mode.
- No diagnostics.

## 0002 - Relax energy mortar normal pair filtering

Purpose: retain more plausible contact candidates before formulation evaluation.

Main changes:
- Changes the geometric normal filter from `cos(100 deg) = -0.173648177` to `0.0`.
- This keeps pairs unless their normals point into the same hemisphere.

Expected effect:
- Reduces active-set loss for pairs hovering near the old normal-angle threshold.

Explicitly excluded:
- The binning proximity multiplier remains `4.0`; the previous `8.0` experiment is not included.
- No baseline-union mode.

## 0004 - Use typed energy mortar smoothing options

Purpose: cleanly plumb smoothing options through the MFEM/Tribol interface using typed enums and a single options struct.

Main changes:
- Changes `setMfemEnergyMortarOptions()` to accept `SmoothingType` and `PenaltySmoothing` instead of raw ints.
- Stores typed smoothing options in `MfemMeshData`.
- Passes a unified `EnergyMortarOptions` object into `EnergyMortarAdapter`.
- Simplifies `ContactFormulationFactory` option extraction for Energy Mortar.

Expected effect:
- Makes bounds smoothing and penalty smoothing configuration less error-prone.
- Keeps the existing smooth penalty behavior available through typed options.

Explicitly excluded:
- No `ENERGY_AREA_PENALTY`.
- No baseline-union mode.
- No diagnostics.

## 0005 - Smooth energy mortar normal alignment factor

Purpose: make the normal-alignment weighting smoother near perpendicular configurations.

Main changes:
- Changes the Energy Mortar alignment factor from:

```cpp
eta = ( dot < 0.0 ) ? dot : 0.0;
```

to:

```cpp
eta = ( dot < 0.0 ) ? -dot * dot : 0.0;
```

- Applies the same factor in both the shared kernel and the diagnostic/FD helper.

Expected effect:
- Smooths the transition as opposing normals approach perpendicularity.
- This is a real formulation change and should be tested separately from patches `0001`, `0002`, and `0004`.

## 0006 - Add energy area penalty formulation

Purpose: add a separate area-integrated energy penalty contact method.

Main changes:
- Adds `ENERGY_AREA_PENALTY` to `ContactMethod`.
- Adds `EnergyAreaPenalty` calculator and kernel:

```text
E = 0.5 * integral H(g)^2 dA
```

- Uses Enzyme AD to compute force (`dE/dx`) and Jacobian (`d2E/dx2`).
- Adds `EnergyAreaPenaltyAdapter` for MFEM force/Jacobian integration.
- Registers the new formulation in `ContactFormulationFactory`.
- Adds CMake entries and finite-difference tests for energy gradient, Hessian, and force consistency.

Expected effect:
- Provides an alternate penalty formulation that avoids nodal `g_tilde / A` assembly as the primary energy definition.

Dependencies:
- Built on top of the shared Energy Mortar kernels from `0001`.
- Uses smoothing option plumbing from `0004`.

## 0007 - Add baseline union contact pair mode

Purpose: reduce active-set churn during nonlinear iterations by keeping load-step baseline pairs active alongside current search pairs.

Main changes:
- Adds `CouplingScheme::setUseBaselineUnion()`.
- Adds `CouplingScheme::resetBaselineForLoadStep()`.
- Changes `CouplingScheme::apply()` to pass `baseline pairs union current pairs` when enabled.
- Adds MFEM API:

```cpp
tribol::setMfemUseBaselineUnion(cs_id, true);
tribol::snapshotMfemBaselinePairs();
```

Expected effect:
- Prevents contact interactions present at load-step start from disappearing during Newton iterations solely because the fresh search set drops them.

Testing note:
- This changes active-set policy, not the element kernel. It should be tested separately from pair-filter and kernel consistency changes.
