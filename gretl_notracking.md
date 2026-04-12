# Gretl Graph Tracking Toggle (Phase 5)

## Overview
This document captures the design decisions for the ability to temporarily disable computational graph tracking in `gretl`. This is primarily useful for computing output quantities (like stress visualization) or running preconditioners where we need `gretl::State` for the forward evaluation syntax, but we do not want to compute adjoints or propagate sensitivities through these operations.

## The "Stop-Gradient" Plan (Finalized)

Instead of completely bypassing graph registration (which breaks `step` indexing, memory management, and prevents mixing), we register states created in an "untracked" context normally in the graph, but replace their Vector-Jacobian Product (VJP) function with a **no-op**. 

They effectively act as "Stop-Gradient" or `.detach()` nodes in the computational graph.

### Mechanism
- **`DataStore::is_tracking()`**: A boolean flag on the `DataStore` that controls the behavior of newly created states.
- **`ScopedGraphDisable`**: An RAII helper that sets `is_tracking() = false` on construction and restores the previous value on destruction.

### Properties of "Untracked" (Stop-Gradient) States

1. **Graph Presence**: When `create_state` is called while tracking is disabled, the state *is* added to the graph. It receives a valid `step` index, and it is added to `states_` and `upstreamSteps_`.
2. **Forward Evaluation & Checkpointing**: The forward pass executes exactly the same way. The state participates in dynamic checkpointing logic if necessary.
3. **VJP Skipping (Stop-Gradient)**: The core optimization is that we assign a **no-op VJP closure** to these states during creation. When back-propagation reaches this state, the no-op VJP is called, which simply returns immediately without accumulating any sensitivities into its upstream dependencies.
4. **Duals & Mixing**: Because they have a `step`, a tracked downstream can safely use them as an input. The downstream's VJP will accumulate derivatives into this untracked state's dual, which is perfectly safe. The dual exists and accumulates, but the sensitivity chain dies there because the untracked state's own VJP is a no-op.

### Advantages
- **Simplicity**: No changes to `DataStore` memory management, dynamic checkpointing, or primal/dual indexing logic. 
- **Mixing Support**: Tracked states and untracked states can be mixed seamlessly.
- **Performance**: Achieves the primary user goal—skipping the costly back-propagation of operations they don't care about (e.g., preconditioners, output processing).

### Subtleties
Because the forward pass and dynamic checkpointing remain essentially unchanged, operations inside a `ScopedGraphDisable` block will still consume memory in the checkpointer's active working set and will still be recomputed if evicted. The savings are strictly in the backward pass.
