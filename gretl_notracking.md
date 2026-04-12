# Gretl Stop-Gradient Feature (Phase 5)

## Overview
This document captures the design decisions for the ability to compute output quantities (like stress visualization) or run preconditioners using `gretl::State` for the forward evaluation syntax, without computing adjoints or propagating sensitivities backward through these operations.

## The "Stop-Gradient" Plan (Finalized)

To achieve this without breaking the graph's internal indexing or dynamic checkpointing logic, we register states normally in the graph, but replace their Vector-Jacobian Product (VJP) function with a **no-op**. 

They effectively act as "Stop-Gradient" or `.detach()` nodes in the computational graph.

### Mechanism
- **`DataStore::gradients_enabled()`**: A boolean flag on the `DataStore` that controls whether newly created states record their actual VJP closure or a no-op.
- **`DataStore::set_gradients_enabled(bool)`**: A simple setter to toggle this behavior on or off.

### Example Usage
```cpp
data_store.set_gradients_enabled(false);

// This state is added to the graph, but its VJP is replaced with a no-op
auto s_out = create_state<double, double>(
    [](const double&) { return 0.0; },
    [](const double& a, const double& b) { return a * b; },
    [](const double&, const double&, const double&, double&, double&, const double&) {
        // This will never be called during back_prop()
    },
    s1, s2);

data_store.set_gradients_enabled(true);
```

### Properties of "Stop-Gradient" States

1. **Graph Presence**: The state *is* added to the graph normally. It receives a valid `step` index, and it is added to `states_` and `upstreamSteps_`.
2. **Forward Evaluation & Checkpointing**: The forward pass executes exactly the same way. The state participates in dynamic checkpointing logic if necessary.
3. **VJP Skipping (Stop-Gradient)**: The core optimization is that we assign a **no-op VJP closure** to these states during creation. When back-propagation reaches this state, the no-op VJP is called, which simply returns immediately without accumulating any sensitivities into its upstream dependencies.
4. **Duals & Mixing**: Because they have a `step`, a tracked downstream can safely use them as an input. The downstream's VJP will accumulate derivatives into this stop-gradient state's dual, which is perfectly safe. The dual exists and accumulates, but the sensitivity chain dies there because the stop-gradient state's own VJP is a no-op.

### Advantages
- **Simplicity**: No changes to `DataStore` memory management, dynamic checkpointing, or primal/dual indexing logic. 
- **Mixing Support**: Tracked states and stop-gradient states can be mixed seamlessly.
- **Performance**: Achieves the primary user goal—skipping the costly back-propagation of operations they don't care about (e.g., preconditioners, output processing).

### Subtleties
Because the forward pass and dynamic checkpointing remain essentially unchanged, operations between `set_gradients_enabled(false)` and `set_gradients_enabled(true)` will still consume memory in the checkpointer's active working set and will still be recomputed if evicted. The savings are strictly in the backward pass.
