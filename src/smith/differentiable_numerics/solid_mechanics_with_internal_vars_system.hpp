// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_with_internal_vars_system.hpp
 * @brief Defines SolidMechanicsWithInternalVarsSystem and its two-phase factory functions.
 *
 * Two-phase factory (for coupling via combineSystems):
 *   auto info = registerSolidMechanicsWithInternalVarsFields<dim, disp_order, StateSpace>(
 *       field_store, disp_rule, state_rule, params...);
 *   CouplingParams coupling{...};
 *   auto sys = buildSolidMechanicsWithInternalVarsSystemFromStore<...>(
 *       info, solver, opts, coupling);
 *
 * Standalone factory (backwards-compatible, allocates its own FieldStore):
 *   auto sys = buildSolidMechanicsWithInternalVarsSystem<dim, disp_order, StateSpace>(
 *       mesh, solver, disp_rule, state_rule, options);
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/coupling_params.hpp"

namespace smith {

/**
 * @brief System struct for solid mechanics with an additional internal variable (L2 state).
 *
 * Displacement uses a 4-state second-order layout (displacement_solve_state, displacement, velocity, acceleration).
 * Internal variable uses a 2-state first-order layout (state_solve_state, state).
 * Total: 6 state fields.
 *
 * With a non-empty Coupling, coupling fields appear immediately after the hardcoded state fields
 * (after alpha_old for the solid form; after a_old for the state form) and before user parameter fields.
 * setMaterial and addStateEvolution work correctly only when Coupling = CouplingParams<> (default).
 * For coupled systems, register integrands directly on solid_weak_form / state_weak_form.
 *
 * @tparam dim Spatial dimension.
 * @tparam disp_order Polynomial order for displacement field.
 * @tparam StateSpace Finite element space for the internal variable (e.g., L2<order>).
 * @tparam DisplacementTimeRule Time integration rule for displacement (must have num_states == 4).
 * @tparam InternalVarTimeRule Time integration rule for the internal variable (must have num_states == 2).
 * @tparam Coupling CouplingParams listing fields borrowed from other physics (default: no coupling).
 */
template <int dim, int disp_order, typename StateSpace,
          typename DisplacementTimeRule = QuasiStaticSecondOrderTimeIntegrationRule,
          typename InternalVarTimeRule = BackwardEulerFirstOrderTimeIntegrationRule,
          typename Coupling = CouplingParams<>>
struct SolidMechanicsWithInternalVarsSystem : public SystemBase {
  using SystemBase::SystemBase;

  static_assert(DisplacementTimeRule::num_states == 4,
                "SolidMechanicsWithInternalVarsSystem requires a 4-state displacement rule");
  static_assert(InternalVarTimeRule::num_states == 2,
                "SolidMechanicsWithInternalVarsSystem requires a 2-state internal variable rule");

  // Primary weak form: residual for displacement (u).
  // Inputs: u, u_old, v_old, a_old, alpha, alpha_old, coupling_fields..., params...
  using SolidWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<disp_order, dim>,
                              typename detail::AppendCouplingToParams<
                                  Coupling, Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<disp_order, dim>,
                                                       H1<disp_order, dim>, StateSpace, StateSpace>>::type>;

  // State weak form: residual for internal variable (alpha).
  // Inputs: alpha, alpha_old, u, u_old, v_old, a_old, coupling_fields..., params...
  using StateWeakFormType =
      TimeDiscretizedWeakForm<dim, StateSpace,
                              typename detail::AppendCouplingToParams<
                                  Coupling, Parameters<StateSpace, StateSpace, H1<disp_order, dim>, H1<disp_order, dim>,
                                                       H1<disp_order, dim>, H1<disp_order, dim>>>::type>;

  // Cycle-zero weak form: test field = acceleration, inputs: u, v, a, alpha, coupling_fields..., params...
  using CycleZeroSolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      typename detail::AppendCouplingToParams<
          Coupling, Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<disp_order, dim>, StateSpace>>::type>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;  ///< Solid mechanics weak form.
  std::shared_ptr<StateWeakFormType> state_weak_form;  ///< Internal variable weak form.
  std::shared_ptr<CycleZeroSolidWeakFormType>
      cycle_zero_solid_weak_form;                         ///< Typed cycle zero solid mechanics weak form.
  std::shared_ptr<SystemBase> cycle_zero_system;          ///< Cycle-zero system.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;   ///< Displacement boundary conditions.
  std::shared_ptr<DirichletBoundaryConditions> state_bc;  ///< Internal variable boundary conditions.
  std::shared_ptr<DisplacementTimeRule> disp_time_rule;   ///< Time integration for displacement.
  std::shared_ptr<InternalVarTimeRule> state_time_rule;   ///< Time integration for internal variable.

  /**
   * @brief Set the material model for the solid mechanics part.
   *
   * NOTE: works correctly only when Coupling = CouplingParams<> (default).  When coupling is active,
   * register integrands directly on solid_weak_form.
   *
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old,
                                                      auto a_old, auto alpha, auto alpha_old, auto... params) {
      auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
      auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);

      typename MaterialType::State state;
      auto pk_stress = material(state, get<DERIVATIVE>(u_current), get<VALUE>(alpha_current), params...);

      return smith::tuple{get<VALUE>(a_current) * material.density, pk_stress};
    });

    // Cycle-zero: u and v are given, solve for a; alpha at initial condition
    if (cycle_zero_solid_weak_form) {
      cycle_zero_solid_weak_form->addBodyIntegral(
          domain_name, [=](auto /*t_info*/, auto /*X*/, auto u, auto /*v*/, auto a, auto alpha, auto... params) {
            auto alpha_current = alpha;  // at cycle 0, use initial alpha
            typename MaterialType::State state;
            auto pk_stress = material(state, get<DERIVATIVE>(u), get<VALUE>(alpha_current), params...);
            return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
          });
    }
  }

  /**
   * @brief Add a body force to the solid mechanics part (with DependsOn).
   * @param depends_on Selects which primal and parameter fields the contribution depends on.
   * @param domain_name The name of the domain where the body force is applied.
   * @param force_function (t, X, u, v, a, alpha, alpha_dot, params...) -> force vector.
   */
  template <int... active_parameters, typename BodyForceType>
  void addBodyForce(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                    BodyForceType force_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_state_rule = state_time_rule;
    solid_weak_form->addBodySource(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto alpha, auto alpha_old,
            auto... params) {
          auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
          auto [alpha_current, alpha_dot] = captured_state_rule->interpolate(t_info, alpha, alpha_old);
          return force_function(t_info.time(), X, u_current, v_current, a_current, alpha_current, alpha_dot, params...);
        });

    addCycleZeroBodySourceImpl(
        domain_name,
        [=](auto t_info, auto X, auto u, auto v, auto a, auto alpha, auto... params) {
          auto alpha_dot = 0.0 * alpha;
          return force_function(t_info.time(), X, u, v, a, alpha, alpha_dot, params...);
        },
        std::make_index_sequence<4 + Coupling::num_coupling_fields>{});
  }

  /**
   * @brief Add a body force that depends on all state and parameter fields.
   * @param domain_name The name of the domain where the body force is applied.
   * @param force_function (t, X, u, v, a, alpha, alpha_dot, params...) -> force vector.
   */
  template <typename BodyForceType>
  void addBodyForce(const std::string& domain_name, BodyForceType force_function)
  {
    addBodyForceAllParams(domain_name, force_function, std::make_index_sequence<6 + Coupling::num_coupling_fields>{});
  }

  /**
   * @brief Add a surface traction to the solid mechanics part (with DependsOn).
   * @param depends_on Selects which primal and parameter fields the contribution depends on.
   * @param domain_name The name of the boundary where the traction is applied.
   * @param traction_function (t, X, n, u, v, a, alpha, alpha_dot, params...) -> traction vector.
   */
  template <int... active_parameters, typename TractionType>
  void addTraction(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   TractionType traction_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_state_rule = state_time_rule;
    solid_weak_form->addBoundaryFlux(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto u_old, auto v_old, auto a_old, auto alpha, auto alpha_old,
            auto... params) {
          auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
          auto [alpha_current, alpha_dot] = captured_state_rule->interpolate(t_info, alpha, alpha_old);
          return traction_function(t_info.time(), X, n, u_current, v_current, a_current, alpha_current, alpha_dot,
                                   params...);
        });

    addCycleZeroBoundaryFluxImpl(
        domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto v, auto a, auto alpha, auto... params) {
          auto alpha_dot = 0.0 * alpha;
          return traction_function(t_info.time(), X, n, u, v, a, alpha, alpha_dot, params...);
        },
        std::make_index_sequence<4 + Coupling::num_coupling_fields>{});
  }

  /**
   * @brief Add a surface traction that depends on all state and parameter fields.
   * @param domain_name The name of the boundary where the traction is applied.
   * @param traction_function (t, X, n, u, v, a, alpha, alpha_dot, params...) -> traction vector.
   */
  template <typename TractionType>
  void addTraction(const std::string& domain_name, TractionType traction_function)
  {
    addTractionAllParams(domain_name, traction_function, std::make_index_sequence<6 + Coupling::num_coupling_fields>{});
  }

  /**
   * @brief Add a pressure boundary condition (follower force) (with DependsOn).
   * @param depends_on Selects which primal and parameter fields the contribution depends on.
   * @param domain_name The name of the boundary where the pressure is applied.
   * @param pressure_function (t, X, u, v, a, alpha, alpha_dot, params...) -> pressure scalar.
   */
  template <int... active_parameters, typename PressureType>
  void addPressure(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   PressureType pressure_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_state_rule = state_time_rule;
    solid_weak_form->addBoundaryIntegral(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto alpha, auto alpha_old,
            auto... params) {
          auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
          auto [alpha_current, alpha_dot] = captured_state_rule->interpolate(t_info, alpha, alpha_old);

          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), u_current, v_current, a_current,
                                            alpha_current, alpha_dot, get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        });

    addCycleZeroBoundaryIntegralImpl(
        domain_name,
        [=](auto t_info, auto X, auto u, auto v, auto a, auto alpha, auto... params) {
          auto alpha_val = get<VALUE>(alpha);
          auto alpha_dot = 0.0 * alpha_val;

          auto x_current = X + u;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), get<VALUE>(u), get<VALUE>(v), get<VALUE>(a),
                                            alpha_val, alpha_dot, get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        },
        std::make_index_sequence<4 + Coupling::num_coupling_fields>{});
  }

  /**
   * @brief Add a pressure boundary condition that depends on all state and parameter fields.
   * @param domain_name The name of the boundary where the pressure is applied.
   * @param pressure_function (t, X, u, v, a, alpha, alpha_dot, params...) -> pressure scalar.
   */
  template <typename PressureType>
  void addPressure(const std::string& domain_name, PressureType pressure_function)
  {
    addPressureAllParams(domain_name, pressure_function, std::make_index_sequence<6 + Coupling::num_coupling_fields>{});
  }

  /**
   * @brief Add the evolution law for the internal variable.
   *
   * NOTE: works correctly only when Coupling = CouplingParams<> (default).  When coupling is active,
   * register integrands directly on state_weak_form.
   *
   * @tparam EvolutionType The evolution law function type.
   * @param domain_name The name of the domain.
   * @param evolution_law Function (t_info, alpha, alpha_dot, grad_u, params...) returning the ODE residual.
   */
  template <typename EvolutionType>
  void addStateEvolution(const std::string& domain_name, EvolutionType evolution_law)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_state_rule = state_time_rule;

    state_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto alpha, auto alpha_old, auto u,
                                                      auto u_old, auto v_old, auto a_old, auto... params) {
      auto [u_current, v_current, a_current] = captured_disp_rule->interpolate(t_info, u, u_old, v_old, a_old);
      auto [alpha_current, alpha_dot] = captured_state_rule->interpolate(t_info, alpha, alpha_old);

      auto residual_val = evolution_law(t_info, get<VALUE>(alpha_current), get<VALUE>(alpha_dot),
                                        get<DERIVATIVE>(u_current), params...);

      tensor<double, dim> flux{};
      return smith::tuple{residual_val, flux};
    });
  }

 private:
  template <typename BodyForceType, std::size_t... Is>
  void addBodyForceAllParams(const std::string& domain_name, BodyForceType force_function, std::index_sequence<Is...>)
  {
    addBodyForce(DependsOn<static_cast<int>(Is)...>{}, domain_name, force_function);
  }

  template <typename TractionType, std::size_t... Is>
  void addTractionAllParams(const std::string& domain_name, TractionType traction_function, std::index_sequence<Is...>)
  {
    addTraction(DependsOn<static_cast<int>(Is)...>{}, domain_name, traction_function);
  }

  template <typename PressureType, std::size_t... Is>
  void addPressureAllParams(const std::string& domain_name, PressureType pressure_function, std::index_sequence<Is...>)
  {
    addPressure(DependsOn<static_cast<int>(Is)...>{}, domain_name, pressure_function);
  }

  // Cycle-zero helpers: use all-params DependsOn with the 4-state cycle-zero form (u, v, a, alpha)
  template <typename IntegrandType, std::size_t... Is>
  void addCycleZeroBodySourceImpl(const std::string& name, IntegrandType f, std::index_sequence<Is...>)
  {
    if (cycle_zero_solid_weak_form) {
      cycle_zero_solid_weak_form->addBodySource(DependsOn<static_cast<int>(Is)...>{}, name, f);
    }
  }

  template <typename IntegrandType, std::size_t... Is>
  void addCycleZeroBoundaryFluxImpl(const std::string& name, IntegrandType f, std::index_sequence<Is...>)
  {
    if (cycle_zero_solid_weak_form) {
      cycle_zero_solid_weak_form->addBoundaryFlux(DependsOn<static_cast<int>(Is)...>{}, name, f);
    }
  }

  template <typename IntegrandType, std::size_t... Is>
  void addCycleZeroBoundaryIntegralImpl(const std::string& name, IntegrandType f, std::index_sequence<Is...>)
  {
    if (cycle_zero_solid_weak_form) {
      cycle_zero_solid_weak_form->addBoundaryIntegral(DependsOn<static_cast<int>(Is)...>{}, name, f);
    }
  }
};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct SolidMechanicsWithInternalVarsOptions {};

/**
 * @brief Register all solid mechanics with internal vars fields into a FieldStore.
 *
 * Phase 1 of the two-phase initialization. Pass instances of the desired time integration rules
 * so their types are deduced; only `<dim, order, StateSpace>` need be specified explicitly.
 *
 * @return CouplingParams carrying the exported field tokens (for use as coupling input to other systems).
 */
template <int dim, int order, typename StateSpace, typename DisplacementTimeRule, typename InternalVarTimeRule,
          typename... parameter_space>
auto registerSolidMechanicsWithInternalVarsFields(std::shared_ptr<FieldStore> field_store,
                                                  DisplacementTimeRule disp_rule, InternalVarTimeRule state_rule,
                                                  FieldType<parameter_space>... parameter_types)
{
  registerSolidMechanicsFields<dim, order>(field_store, disp_rule);
  registerStateVariableFields<StateSpace>(field_store, state_rule);

  auto prefix_param = [&](auto& pt) {
    pt.name = "param_" + pt.name;
    field_store->addParameter(pt);
  };
  (prefix_param(parameter_types), ...);

  return CouplingParams{FieldType<H1<order, dim>>(field_store->prefix("displacement_solve_state")),
                        FieldType<H1<order, dim>>(field_store->prefix("displacement")),
                        FieldType<H1<order, dim>>(field_store->prefix("velocity")),
                        FieldType<H1<order, dim>>(field_store->prefix("acceleration")),
                        FieldType<StateSpace>(field_store->prefix("state_solve_state")),
                        FieldType<StateSpace>(field_store->prefix("state")),
                        parameter_types...};
}

/**
 * @brief Build a SolidMechanicsWithInternalVarsSystem with coupling, assuming fields are already registered.
 *
 * Phase 2 of the two-phase initialization. Pass the same rule instances used in
 * registerSolidMechanicsWithInternalVarsFields so their types are deduced;
 * only `<dim, order, StateSpace>` need be specified.
 *
 * Returns `{system, cycle_zero_system, end_step_systems}` as a tuple.
 */
template <int dim, int order, typename StateSpace, typename DisplacementTimeRule, typename InternalVarTimeRule,
          typename Coupling>
  requires detail::is_coupling_params_v<Coupling>
auto buildSolidMechanicsWithInternalVarsSystem(std::shared_ptr<FieldStore> field_store,
                                               DisplacementTimeRule /*disp_rule*/, InternalVarTimeRule /*state_rule*/,
                                               const Coupling& coupling, std::shared_ptr<SystemSolver> solver,
                                               const SolidMechanicsWithInternalVarsOptions& /*options*/)
{
  auto disp_time_rule_ptr = std::make_shared<DisplacementTimeRule>();
  auto state_time_rule_ptr = std::make_shared<InternalVarTimeRule>();

  FieldType<H1<1, dim>> shape_disp_type(field_store->prefix("shape_displacement"));
  FieldType<H1<order, dim>> disp_type(field_store->prefix("displacement_solve_state"), true);
  FieldType<H1<order, dim>> disp_old_type(field_store->prefix("displacement"));
  FieldType<H1<order, dim>> velo_old_type(field_store->prefix("velocity"));
  FieldType<H1<order, dim>> accel_old_type(field_store->prefix("acceleration"));

  FieldType<StateSpace> state_type(field_store->prefix("state_solve_state"), true);
  FieldType<StateSpace> state_old_type(field_store->prefix("state"));

  auto disp_bc = field_store->getBoundaryConditions(disp_type.name);
  auto state_bc = field_store->getBoundaryConditions(state_type.name);

  using SystemType =
      SolidMechanicsWithInternalVarsSystem<dim, order, StateSpace, DisplacementTimeRule, InternalVarTimeRule, Coupling>;

  // Solid weak form: (u, u_old, v_old, a_old, alpha, alpha_old, coupling_fields..., params...)
  std::string solid_res_name = field_store->prefix("solid_residual");
  auto solid_weak_form = std::apply(
      [&](auto&... cfs) {
        return std::make_shared<typename SystemType::SolidWeakFormType>(
            solid_res_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
            field_store->createSpaces(solid_res_name, disp_type.name, disp_type, disp_old_type, velo_old_type,
                                      accel_old_type, state_type, state_old_type, cfs...));
      },
      coupling.fields);

  // State weak form: (alpha, alpha_old, u, u_old, v_old, a_old, coupling_fields..., params...)
  std::string state_res_name = field_store->prefix("state_residual");
  auto state_weak_form = std::apply(
      [&](auto&... cfs) {
        return std::make_shared<typename SystemType::StateWeakFormType>(
            state_res_name, field_store->getMesh(), field_store->getField(state_type.name).get()->space(),
            field_store->createSpaces(state_res_name, state_type.name, state_type, state_old_type, disp_type,
                                      disp_old_type, velo_old_type, accel_old_type, cfs...));
      },
      coupling.fields);

  auto sys = std::make_shared<SystemType>(field_store, solver,
                                          std::vector<std::shared_ptr<WeakForm>>{solid_weak_form, state_weak_form});
  sys->disp_bc = disp_bc;
  sys->state_bc = state_bc;
  sys->disp_time_rule = disp_time_rule_ptr;
  sys->state_time_rule = state_time_rule_ptr;
  sys->solid_weak_form = solid_weak_form;
  sys->state_weak_form = state_weak_form;

  std::shared_ptr<SystemBase> cycle_zero_system;
  std::vector<std::shared_ptr<SystemBase>> end_step_systems;

  if (disp_time_rule_ptr->requiresInitialAccelerationSolve()) {
    std::string cycle_zero_name = field_store->prefix("solid_reaction");
    auto accel_as_unknown = accel_old_type;
    accel_as_unknown.is_unknown = true;
    FieldType<H1<order, dim>> disp_cz_input(disp_type.name);
    FieldType<StateSpace> state_cz_input(state_type.name);
    sys->cycle_zero_solid_weak_form = std::apply(
        [&](auto&... cfs) {
          return std::make_shared<typename SystemType::CycleZeroSolidWeakFormType>(
              cycle_zero_name, field_store->getMesh(), field_store->getField(accel_old_type.name).get()->space(),
              field_store->createSpaces(cycle_zero_name, accel_old_type.name, disp_cz_input, velo_old_type,
                                        accel_as_unknown, state_cz_input, cfs...));
        },
        coupling.fields);
    field_store->markWeakFormInternal(cycle_zero_name);
    field_store->shareBoundaryConditions(accel_old_type.name, disp_bc);

    NonlinearSolverOptions cz_nonlin{.nonlin_solver = NonlinearSolver::Newton,
                                     .relative_tol = 1e-14,
                                     .absolute_tol = 1e-14,
                                     .max_iterations = 2,
                                     .print_level = 0};
    LinearSolverOptions cz_lin{.linear_solver = LinearSolver::CG,
                               .preconditioner = Preconditioner::HypreJacobi,
                               .relative_tol = 1e-14,
                               .absolute_tol = 1e-14,
                               .max_iterations = 1000,
                               .print_level = 0};
    auto cycle_zero_solver =
        std::make_shared<SystemSolver>(buildNonlinearBlockSolver(cz_nonlin, cz_lin, *field_store->getMesh()));

    sys->cycle_zero_system = makeSystem(field_store, cycle_zero_solver, {sys->cycle_zero_solid_weak_form});
    cycle_zero_system = sys->cycle_zero_system;
  }

  return std::make_tuple(sys, cycle_zero_system, end_step_systems);
}

/**
 * @brief Build a SolidMechanicsWithInternalVarsSystem without coupling, assuming fields are already registered.
 *
 * Overload for the common case of no inter-physics coupling.
 * Parameters (if any) are wrapped into a CouplingParams so the system sees them.
 */
template <int dim, int order, typename StateSpace, typename DisplacementTimeRule, typename InternalVarTimeRule,
          typename... parameter_space>
auto buildSolidMechanicsWithInternalVarsSystem(std::shared_ptr<FieldStore> field_store, DisplacementTimeRule disp_rule,
                                               InternalVarTimeRule state_rule, std::shared_ptr<SystemSolver> solver,
                                               const SolidMechanicsWithInternalVarsOptions& options,
                                               FieldType<parameter_space>... parameter_types)
{
  auto prefix_param = [&](auto& pt) {
    pt.name = "param_" + pt.name;
    field_store->addParameter(pt);
  };
  (prefix_param(parameter_types), ...);
  return buildSolidMechanicsWithInternalVarsSystem<dim, order, StateSpace>(
      field_store, disp_rule, state_rule, CouplingParams{parameter_types...}, solver, options);
}

}  // namespace smith
