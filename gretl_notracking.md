# Gretl Stop-Gradient Feature (Phase 5)

## Overview
This document captures the design decisions for the ability to compute output quantities (like stress visualization) or run preconditioners/iterative solvers using `gretl::State` for the forward evaluation syntax, without computing adjoints or propagating sensitivities backward through these operations.

## The "Stop-Gradient" Plan (Finalized)

To achieve this without breaking the graph's internal indexing or dynamic checkpointing logic, we register states normally in the graph, but replace their Vector-Jacobian Product (VJP) function with a **no-op**. 

They effectively act as "Stop-Gradient" or `.detach()` nodes in the computational graph.

### Mechanism
- **`DataStore::gradients_enabled()`**: A boolean flag on the `DataStore` that controls whether newly created states record their actual VJP closure or a no-op.
- **`DataStore::set_gradients_enabled(bool)`**: A simple setter to toggle this behavior on or off.

### Important Optimization for Checkpointing
If gradients are disabled, `gretl` skips calling `fetch_state_data()` during the backprop pass. This prevents costly dynamic recomputations of evicted states just to feed a no-op VJP. 

### Example Usage: Picard Iteration
A key use-case is performing intermediate iterative solves without tracking every step, then performing one final tracked step to capture parameter sensitivities.

```cpp
data_store.set_gradients_enabled(false);

// Iterate without tracking gradients (stop-gradient nodes)
for (int i = 0; i < 10; ++i) {
    x = create_state<double, double>(..., x, p);
}

// Re-enable gradients for the final step
data_store.set_gradients_enabled(true);

// One final iteration to connect the parameter sensitivity
auto x_final = create_state<double, double>(..., x, p);
```

During `back_prop()`, `x_final` will correctly pass gradient information to `p`. However, the gradient passed to the 10th intermediate step `x` will hit a no-op VJP and stop, saving us from recomputing or propagating derivatives through the 10 loop iterations.

### Properties of "Stop-Gradient" States

1. **Graph Presence**: The state *is* added to the graph normally. It receives a valid `step` index, and it is added to `states_` and `upstreamSteps_`.
2. **Forward Evaluation & Checkpointing**: The forward pass executes exactly the same way. The state participates in dynamic checkpointing logic if necessary.
3. **VJP Skipping (Stop-Gradient)**: The core optimization is that we assign a **no-op VJP closure** to these states during creation. When back-propagation reaches this state, the no-op VJP is skipped, preventing recomputations.
4. **Duals & Mixing**: Because they have a `step`, a tracked downstream can safely use them as an input. The downstream's VJP will accumulate derivatives into this stop-gradient state's dual, which is perfectly safe. The dual exists and accumulates, but the sensitivity chain dies there because the stop-gradient state's own VJP is a no-op.

### Advantages
- **Simplicity**: No changes to `DataStore` memory management, dynamic checkpointing, or primal/dual indexing logic. 
- **Mixing Support**: Tracked states and stop-gradient states can be mixed seamlessly.
- **Performance**: Achieves the primary user goal—skipping the costly back-propagation (and checkpointer recomputations) of operations they don't care about.
