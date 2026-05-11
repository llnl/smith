# Refactoring Plan: Compile Time Reduction & Interface Unification

## 1. Remove `DependsOn` from WeakForm Interfaces [COMPLETED]
- **Status:** **DONE**. Removed `DependsOn` overloads and template parameters from `FunctionalWeakForm`, `FunctionalObjective`, `SolidMechanicsSystem`, and `ThermalSystem`. This dramatically reduced the number of AST instantiations required for AD derivatives.
- **Action Taken:** By default, all integrals now depend on the full parameter pack (`DependsOn<0, 1, ..., N>{}`). The explicit template instantiation burden is completely removed from the user API.

## 2. Eliminate `cycle_zero_weak_form` [COMPLETED]
- **Status:** **DONE**. Deleted `cycle_zero_solid_weak_form` entirely from the SolidMechanics system.
- **Action Taken:** The `MultiphysicsTimeIntegrator` and `SystemBase` now evaluate the cycle-zero acceleration residual directly through the main `solid_weak_form` by properly packing the state variables as `(u, u, v, a)` so the primary time-integration rule interpolates them correctly as initial conditions.

---

## 3. Simplify Deep Type-Level Rewriting (Coupling Types)
- **Problem:** Types like `AppendCouplingToParams`, `TimeRuleParamsHelper`, and operations in `combined_system.hpp` rely on deep, recursive template metaprogramming which bogs down the compiler's frontend.
- **Goal:** Move away from recursive template structs (`struct Helper<T, U> { ... }`) towards flat parameter pack expansions and `decltype(std::tuple_cat(...))` which are significantly faster for the compiler to evaluate.
- **Proposed Architecture:** We can replace much of `coupling_params.hpp` logic by capturing foreign physics fields directly into a flattened `std::tuple` of descriptors at the system construction phase, completely avoiding the need to re-parse the types during weak form evaluations.

## 4. Unified Time Discretization for Coupled Physics
- **Problem:** Currently, self-fields are beautifully interpolated (e.g., `u_new, u_old, v_old, a_old` becomes `u, v, a` inside the user lambda), but coupled fields are passed raw (e.g., `temp_new, temp_old` are passed directly in `auto... params`). 
- **Action:** Transform the time discretization for *all* coupled physics before calling the user closure. If a thermal system is coupled to a solid mechanics system, the solid system's lambdas should see `(t, t_dot)` rather than `(t_new, t_old)`.

### Analysis & Example C++ Code for Steps 3 & 4
To solve both Step 3 and Step 4 simultaneously without introducing massive compile-time overhead, we can create a lightweight, flat **Coupled State Evaluator**. Instead of recursively tearing apart type-lists, we will store a tuple of the active Time Rules for all coupled physics. 

When a weak form evaluates, it receives a flat `auto... raw_args` representing the raw FE states (new, old, old_v, etc.) for *all* physics concatenated together. We will use a compile-time index sequence to partition this flat pack, feed the partitions into their respective time rules, and then flat-unpack the interpolated results into the user's lambda.

**Example implementation approach:**

```cpp
#include <tuple>
#include <utility>

// Helper to extract a sub-pack from a tuple of arguments
template <std::size_t Offset, std::size_t... Is, typename Tuple>
constexpr auto extract_args_impl(const Tuple& t, std::index_sequence<Is...>) {
    return std::make_tuple(std::get<Offset + Is>(t)...);
}

// Given a flat tuple of raw arguments, extract `Count` arguments starting at `Offset`
template <std::size_t Offset, std::size_t Count, typename Tuple>
constexpr auto extract_args(const Tuple& t) {
    return extract_args_impl<Offset>(t, std::make_index_sequence<Count>{});
}

// The Unified Evaluator
template <typename... TimeRules>
struct MultiphysicsEvaluator {
    std::tuple<TimeRules...> rules;

    template <typename UserLambda, typename TInfo, typename XType, typename... RawArgs>
    auto evaluate(UserLambda&& user_fn, const TInfo& t_info, const XType& X, const RawArgs&... raw_args) {
        auto flat_raw_args = std::forward_as_tuple(raw_args...);
        
        // This is pseudo-code. In practice, we would use a compile-time fold over the rules
        // to automatically compute the offsets based on `TimeRule::num_states`.
        
        // E.g., Physics 0 (Solid) takes 4 states (u, u_old, v_old, a_old)
        auto solid_raw = extract_args<0, 4>(flat_raw_args);
        auto solid_interp = std::apply([&](auto... args){ 
            return std::get<0>(rules)->interpolate(t_info, args...); 
        }, solid_raw);

        // E.g., Physics 1 (Thermal) takes 2 states (T, T_old)
        auto thermal_raw = extract_args<4, 2>(flat_raw_args);
        auto thermal_interp = std::apply([&](auto... args){ 
            return std::get<1>(rules)->interpolate(t_info, args...); 
        }, thermal_raw);

        // Finally, expand both interpolated tuples into the user lambda
        // User sees: lambda(t_info, X, u, v, a, T, T_dot)
        return std::apply([&](auto... all_interp_args) {
            return user_fn(t_info, X, all_interp_args...);
        }, std::tuple_cat(solid_interp, thermal_interp));
    }
};
```

**Why this reduces compile time:**
1. We rely on `std::tuple` and `std::apply`, which modern compilers highly optimize via built-in intrinsics.
2. The flat `raw_args` list is passed exactly as it is received from `FunctionalWeakForm`. We do not rewrite the `Parameters<...>` type recursively; we only slice the tuple at evaluation time.
3. AD instantiation stays confined to the `evaluate` wrapper, meaning the user's `UserLambda` is only instantiated once with the fully interpolated, unified types.