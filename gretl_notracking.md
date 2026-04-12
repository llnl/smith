# Gretl Graph Tracking Toggle (Phase 5)

## Overview
This document captures the subtleties and design decisions surrounding the ability to temporarily disable computational graph tracking in `gretl`. This is primarily useful for computing output quantities (like stress visualization) or running preconditioners where we need `gretl::State` for the forward evaluation syntax, but we do not want to record the operations in the autodiff graph or compute adjoints.

## Mechanism
- **`DataStore::is_tracking()`**: A boolean flag on the `DataStore` that controls whether new states are registered.
- **`ScopedGraphDisable`**: An RAII helper that sets `is_tracking() = false` on construction and restores the previous value on destruction.

## Subtleties of Untracked States

1. **Step Index**: When `create_state` is called while tracking is disabled, the returned `State` has its `step` index set to `std::numeric_limits<Int>::max()`. This sentinel value indicates the state is untracked.
2. **Primal Storage**: For tracked states, the primal value is stored in the `DataStore` (either directly or flushed to disk via checkpointer) and accessed via `data_store().get_primal(step)`. For an untracked state, `step` is invalid, so the primal value is kept alive purely by the `std::shared_ptr<std::any>` inside the `State` object itself (`primal_` in `StateBase`).
3. **Dual Values**: Untracked states do not have dual values. Calling `get_dual()` or `set_dual()` on an untracked state will trigger a `gretl_assert`.
4. **Evaluation and VJP**: Untracked states do not record `eval` or `vjp` closures in the `DataStore`. They evaluate their forward value immediately during `create_state` and never participate in `back_prop()`.

## Open Decisions / Considerations

### 1. Mixing Untracked Upstreams with Tracked Downstreams
**Current Behavior:** `create_state` asserts if tracking is enabled but any of the provided upstream states are untracked (step == max).
**Why:** A tracked state needs to compute its VJP during backprop. The VJP closure generally expects all upstream states to have duals so it can add sensitivities to them. If an upstream state is untracked, it has no dual, and `get_dual()` would assert.
**Decision to make:** Should we allow untracked states to be used as inputs to tracked states?
- *Option A (Current):* Strictly forbid it. If a user needs to use an untracked value in a tracked computation, they must extract the raw primal value (`state.get()`) and either capture it by value in a custom lambda, or create a new tracked parameter state from it.
- *Option B (Treat as Constants):* Allow it, treating the untracked state as a constant. The challenge is the VJP signature: `vjp(..., upstream_dual...)`. If an upstream is untracked, we would need to pass a dummy dual that absorbs the sensitivity without writing to the `DataStore`. This complicates the `create_state` variadic template metaprogramming.

### 2. UpstreamState Wrapper
**Current Behavior:** `UpstreamState` wraps a `DataStore*` and a `step`. For untracked states, we had to add a branch in `UpstreamState::get()` to access `dataStore_->any_primal(step_)`. However, `any_primal` assumes the step is valid.
**Issue:** `UpstreamState` doesn't currently hold a reference to the `shared_ptr<std::any>` primal. So if an untracked state is passed to `UpstreamStates`, it loses access to its data.
**Decision to make:** How should `UpstreamState` handle untracked states?
- *Option A:* If we stick to Decision 1 Option A (forbid mixing), `UpstreamState` will *never* see an untracked state, because they are only created inside tracked `create_state` calls. If `create_state` is untracked, it executes the raw `eval` lambda immediately using the `State` objects directly, not the `UpstreamState` wrapper. Thus, `UpstreamState` doesn't need to support untracked states.
- *Option B:* If we want `UpstreamState` to support untracked states, it must either hold the `shared_ptr<std::any>` or we must store untracked primals somewhere in the `DataStore` (e.g., a separate list of untracked values).

### 3. Cloning vs Creating
When tracking is disabled, `clone_state` also evaluates the forward pass immediately. It requires the `eval` function to be passed in.

## Recommendation
For now, the strictest path (forbidding untracked states as inputs to tracked states) is the safest and requires the least amount of invasive plumbing in the metaprogramming of `create_state`. If a user is in a "tracking disabled" region, all intermediate states they create are untracked. They cannot "re-enter" the tracked graph by passing those untracked states back into tracked states. If they must, they should extract the primal values and inject them as new tracked parameters.
