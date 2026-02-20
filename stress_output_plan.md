# Stress Output Visualization Plan

## Goal
Implement support for derived field visualization, specifically stress output, in `SolidStaticsWithL2StateSystem`. This allows projecting quadrature-point quantities (like stress) onto an L2 finite element field for visualization in Paraview. The computation should be differentiable so it is tracked by the `gretl` graph, enabling sensitivity analysis of visualized quantities.

## Current Status (Completed)
*   **Implemented**: `SolidStaticsWithL2StateSystem` now includes a `stress_predicted` field and a `stress_weak_form` for L2 projection.
*   **Integrated**: `addStressProjection` method added to the system, allowing easy setup of the projection from a material model.
*   **Verified**: `test_solid_static_with_state.cpp` updated to enable stress projection. Successfully verified that `coupled_system_stress` (6 components for 3D) is present in the ParaView output.
*   **Differentiable**: The projection is implemented as a `WeakForm`, meaning it is solved by the `DifferentiableBlockSolver` and tracked by `gretl`.

## Design

### 1. Differentiable L2 Projection via WeakForm
The L2 projection is implemented as a `TimeDiscretizedWeakForm`. The residual for an L2 projection of a quantity $\sigma_{qp}$ onto a field $\sigma_h$ is:
$$R(\sigma_h) = \int_{\Omega} (\sigma_h - \sigma_{qp}) \cdot v_h d\Omega = 0$$
where $v_h$ is the test function in the L2 space.

This approach is naturally differentiable when using `smith::solve` and `DifferentiableBlockSolver`.

### 2. System Integration
`SolidStaticsWithL2StateSystem` has been extended with:
*   A new `FieldState` for the projected stress (`stress_predicted`, `stress`).
*   A `StressWeakFormType` and a corresponding `stress_weak_form` member.
*   A method `addStressProjection` to define how stress is calculated from displacement and internal states.

### 3. Implementation Details
*   **Symmetric Stress**: In 3D, the projection uses 6 components (xx, yy, zz, xy, yz, xz). In 2D, it uses 3 components.
*   **Material Coupling**: The `addStressProjection` method takes the same material model used for the solid mechanics solve, ensuring consistency between the internal forces and the visualized stress.

## Future Enhancements
*   **Cauchy vs PK Stress**: Currently, the material's return value (often PK stress or small-strain stress) is projected. Adding an option to project Cauchy stress $\sigma = \frac{1}{J} P F^T$ would be useful for finite strain problems.
*   **Other Derived Quantities**1: The same pattern can be used for other quantities like equivalent plastic strain, damage energy release rate, etc.
