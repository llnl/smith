# Gretl Graph Tracking Toggle (Phase 5)

## Overview
This document captures the design decisions for the ability to temporarily disable computational graph tracking in `gretl`. This is primarily useful for computing output quantities (like stress visualization) or running preconditioners where we need `gretl::State` for the forward evaluation syntax, but we do not want to record the operations in the autodiff graph or compute adjoints.

## The "Graph-Resident but Ephemeral" Plan (Current Direction)

Instead of completely bypassing graph registration (which breaks `step` indexing and prevents mixing), the new plan is to register "untracked" states in the graph so they receive a valid `step` index, but treat them as **"Stop-Gradient" / Ephemeral states**.

### Mechanism
- **`DataStore::is_tracking()`**: A boolean flag on the `DataStore` that controls the behavior of newly created states.
- **`ScopedGraphDisable`**: An RAII helper that sets `is_tracking() = false` on construction and restores the previous value on destruction.

### Properties of Untracked States

1. **Graph Presence**: When `create_state` is called while tracking is disabled, the state *is* added to the graph. It receives a valid `step` index, and it is added to `states_` and `upstreamSteps_`.
2. **VJP Skipping (Stop-Gradient)**: The core optimization is that we **never** call the `vjp` function for these untracked states during back-propagation. We can either register a no-op VJP or explicitly skip them in the reverse pass loop. This saves all the computational cost of propagating sensitivities through the untracked region.
3. **Duals**: Because they have a valid `step`, their duals can be safely allocated in `duals_`. If a tracked downstream depends on an untracked upstream, the downstream's VJP can safely add into the untracked state's dual. The dual accumulates, but is simply ignored (since the untracked state's VJP is never called to read it).
4. **Checkpointing & Memory Management**: These states are intentionally excluded from the dynamic checkpointing strategy. They are not scheduled for recomputation or swap-to-disk. 
    - Their primal value memory is tied directly to their external usage.
    - They will be instantly ejected from memory (primal freed) when there are no outside handles left (i.e., when the `shared_ptr<StateData>`'s `use_count()` drops, tracked via `wild_count()`).

## Open Design Decisions & Subtleties

### The "Primal Availability" Catch
If an untracked state is purely an output (like stress visualization), ejecting its primal when the user's handle goes out of scope is perfectly safe. 

However, if we **mix untracked upstreams with tracked downstreams**, a subtlety arises during back-propagation:
- The tracked downstream's VJP will likely need to read the untracked state's *primal* value to compute the derivative (`inputs[i].get()`).
- If the untracked state is excluded from checkpointing/recomputation, and the user has let their external `State` handle go out of scope, the primal will have been eagerly freed. Calling `get()` during the reverse pass will crash.

**How do we handle the memory lifetime of an untracked upstream that has a tracked downstream?**
* **Option A (Implicit Promotion/Pinning):** When an untracked state is added as an upstream to a *tracked* state, the graph increments an internal `tracked_usage_count`. Eager freeing is only allowed if `wild_count() == 0` AND `tracked_usage_count == 0`. This effectively pins the untracked primal in memory for the duration of the backprop.
* **Option B (User Responsibility):** Rely on the user to hold the `State` handle alive until `back_prop()` is called. If they let it go out of scope, it's an error.
* **Option C (Fallback Checkpointing):** Do not completely remove them from the checkpointing view. Just flag them as "no-vjp". The checkpointer would treat them like constant parameters that need to be kept alive if downstreams need them.

Option A or C seems the most robust, as it maintains the `gretl` philosophy of the graph managing data availability invisibly to the user. Option A is essentially treating the untracked state as a persistent parameter if it crosses the boundary back into the tracked graph.
