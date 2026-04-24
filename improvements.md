1. Done. Removed brittle state-index assumptions in [composable_solid_mechanics.cpp](/usr/WS2/tupek2/dev/smith/examples/solid_mechanics/composable_solid_mechanics.cpp).
2. Done. Put `TimeInfo` class on separate line.
3. Done. State ordering now comes from field names, not hard-coded indices.
4. Done. Removed shims like `outputFields()` and `qoiFields()`.
5. Done. Renamed `TimeInfoYoungsModulusNeoHookean`.
6. Done. Renamed `TimeInfoGreenSaintVenantThermoelasticMaterial`.
7. Done. `outputFields()` layer removed as unneeded.
8. Done. `qoiFields()` layer removed as unneeded.
9. Done. `TimeInfo` now flows through base `FunctionalObjective` and base `FunctionalWeakForm` interfaces; `TimeDiscretizedObjective` and `TimeDiscretizedWeakForm` layers are removed.
10. No item 10.
11. Done. Renamed `custom_solver` to `coupled_solver`; advanced example now uses `(0,1)`.
12. Done. `appendRemappedStages` renamed to `appendStagesWithBlockMapping`; test now uses `combined_solver` and `StaticTimeIntegrationRule`.
13. Done. `TransientFreefallWithConsistentBoundaryConditions` now applies time-dependent freefall displacement BCs and checks cycle-zero acceleration solve against the matching initial acceleration.
13.b. Lets make the test dynamic/transient again to also test the time integration is exactly integrating acceleration.  If there are instabilities, we need to figure those out. Consider how to simplify the new boundary condition changes in the last commit.
14. In `_internal_vars` test, inline `auto [solid, internal_variables] = buildSystems(solid_solver, internal_variable_solver, solid_fields, internal_variable_fields);`. Rename `solid` to `solid_system` and `internal_variables` to `internal_variable_system`.
15. Like solid system, use `thermal_system->setTemperatureBcs`; make interface consistent with plurals.
16. Was `src/smith/differentiable_numerics/tests/test_thermo_mechanics.cpp` replaced by different test?
17. In `StronglyCoupledThreeSystems`, find better way to handle material setup.
18. Is `MonolithicCombinedSystem` still used? Probably delete it.
19. What is `mergeSystems` versus `combineSystems`?
20. Remove `findFieldStore`. This seems overly complex. Just grab first fields field store inline where needed.
21. Review `collectPhysicsFromPack` and `collectParamsFromPack`. What are they doing?
22. Is `TimeRuleParamsWithCoupling` used?
23. In `DifferentiablePhysics::dual(...)`, there is debug print. Change to `SLIC_ERROR` if important.
24. Logic here is confusing. Make clear main exception is acceleration solve.
25. Is all complication in `getBoundaryConditionManagers` needed?
26. Figure out how `outputField` in `fieldStore` is determined and whether it can be simplified.
27. Explain and simplify `MultiphysicsTimeIntegrator::advanceState`.
28. `makeAdvancer` should not take `cycle_zero_system` or `post_solve_systems`; get them off system.
29. Consider elasticity option for `HypreAMG`. Maybe optional configuration. See `NonlinearBlockSolver::completeSetup`.
30. Are `TimeRuleParams` still used? Can `detail::TimeRuleParamsWithCoupling` be renamed back to that and always use coupling?
31. Why is `CycleZeroSolidWeakFormType` adding many new `H1` fields? Does `AppendCouplingToParams` put last arguments first?
32. `DependsOn` for `SolidSystem` (`addTraction`, `addBody`, `addPressure`, etc.) should be indexed starting at displacement, then velocity, and so on. `DependsOn<0>` should only give displacement. `DependsOn<0,1,2,3>` should be displacement, velocity, acceleration, and `param_0`.
33. Make sure `ThermalSystem` stays in sync with this.
34. Return value from all systems should be variable named `physics_system`. Update unit tests, examples, and code comments, such as in `state_variable_system.hpp`.
35. Remove these overloads:

```cpp
template <typename MaterialType>
/// @brief Register an internal-variable material or evolution law on a domain.
void setMaterial(const MaterialType& material, const std::string& domain_name)
{
  addEvolution(domain_name, material);
}

template <typename EvolutionType>
/// @brief Backward-compatible alias for `addEvolution`.
void addStateEvolution(const std::string& domain_name, EvolutionType evolution_law)
{
  addEvolution(domain_name, evolution_law);
}
```

36. Remove `InternalVariableOptions`. Also remove `StateVariableOptions` and constructor using it.
37. There seem to be duplicated `buildInternalVariableSystem`.
38. Explain `appendRemappedStages`. Can it get better name? `Remapped` means something else here, especially with finite-element field remaps.
39. What is `CoupledSolidThermoMechanicsMaterialAdapter` for? Delete it and use intended interface directly.
40. Review `test_linear_solver_none.cpp`. Make sure it still makes sense.
41. Revert changes to `src/smith/physics/materials/green_saint_venant_thermoelastic.hpp`.
