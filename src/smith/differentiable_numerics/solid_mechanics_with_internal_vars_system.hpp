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
 *   CouplingSpec coupling{...};
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
#include "smith/differentiable_numerics/coupling_spec.hpp"

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
 * setMaterial and addStateEvolution work correctly only when Coupling = CouplingSpec<> (default).
 * For coupled systems, register integrands directly on solid_weak_form / state_weak_form.
 *
 * @tparam dim Spatial dimension.
 * @tparam disp_order Polynomial order for displacement field.
 * @tparam StateSpace Finite element space for the internal variable (e.g., L2<order>).
 * @tparam DisplacementTimeRule Time integration rule for displacement (must have num_states == 4).
 * @tparam InternalVarTimeRule Time integration rule for the internal variable (must have num_states == 2).
 * @tparam Coupling CouplingSpec listing fields borrowed from other physics (default: no coupling).
 * @tparam parameter_space Parameter spaces for material properties.
 */
template <int dim, int disp_order, typename StateSpace,
          typename DisplacementTimeRule = QuasiStaticSecondOrderTimeIntegrationRule,
          typename InternalVarTimeRule = BackwardEulerFirstOrderTimeIntegrationRule,
          typename Coupling = CouplingSpec<>, typename... parameter_space>
struct SolidMechanicsWithInternalVarsSystem : public SystemBase {
  using SystemBase::SystemBase;

  static_assert(DisplacementTimeRule::num_states == 4,
                "SolidMechanicsWithInternalVarsSystem requires a 4-state displacement rule");
  static_assert(InternalVarTimeRule::num_states == 2,
                "SolidMechanicsWithInternalVarsSystem requires a 2-state internal variable rule");

  // Primary weak form: residual for displacement (u).
  // Inputs: u, u_old, v_old, a_old, alpha, alpha_old, coupling_fields..., params...
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      typename detail::AppendCouplingToParams<
          Coupling,
          Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<disp_order, dim>, H1<disp_order, dim>, StateSpace,
                     StateSpace>,
          parameter_space...>::type>;

  // State weak form: residual for internal variable (alpha).
  // Inputs: alpha, alpha_old, u, u_old, v_old, a_old, coupling_fields..., params...
  using StateWeakFormType = TimeDiscretizedWeakForm<
      dim, StateSpace,
      typename detail::AppendCouplingToParams<
          Coupling,
          Parameters<StateSpace, StateSpace, H1<disp_order, dim>, H1<disp_order, dim>, H1<disp_order, dim>,
                     H1<disp_order, dim>>,
          parameter_space...>::type>;

  // Cycle-zero weak form: test field = acceleration, inputs: u, v, a, alpha, coupling_fields..., params...
  using CycleZeroSolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      typename detail::AppendCouplingToParams<
          Coupling,
          Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<disp_order, dim>, StateSpace>,
          parameter_space...>::type>;

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
   * NOTE: works correctly only when Coupling = CouplingSpec<> (default).  When coupling is active,
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
        std::make_index_sequence<4 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a body force that depends on all state and parameter fields.
   * @param domain_name The name of the domain where the body force is applied.
   * @param force_function (t, X, u, v, a, alpha, alpha_dot, params...) -> force vector.
   */
  template <typename BodyForceType>
  void addBodyForce(const std::string& domain_name, BodyForceType force_function)
  {
    addBodyForceAllParams(domain_name, force_function,
                          std::make_index_sequence<6 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
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
        std::make_index_sequence<4 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a surface traction that depends on all state and parameter fields.
   * @param domain_name The name of the boundary where the traction is applied.
   * @param traction_function (t, X, n, u, v, a, alpha, alpha_dot, params...) -> traction vector.
   */
  template <typename TractionType>
  void addTraction(const std::string& domain_name, TractionType traction_function)
  {
    addTractionAllParams(domain_name, traction_function,
                         std::make_index_sequence<6 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
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
        std::make_index_sequence<4 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a pressure boundary condition that depends on all state and parameter fields.
   * @param domain_name The name of the boundary where the pressure is applied.
   * @param pressure_function (t, X, u, v, a, alpha, alpha_dot, params...) -> pressure scalar.
   */
  template <typename PressureType>
  void addPressure(const std::string& domain_name, PressureType pressure_function)
  {
    addPressureAllParams(domain_name, pressure_function,
                         std::make_index_sequence<6 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add the evolution law for the internal variable.
   *
   * NOTE: works correctly only when Coupling = CouplingSpec<> (default).  When coupling is active,
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

template <int dim, int disp_order, typename StateSpace, typename DisplacementTimeRule, typename InternalVarTimeRule,
          typename... parameter_space>
struct SolidMechanicsWithInternalVarsOptions {
  std::string prepend_name{};
  std::shared_ptr<FieldStore> shared_field_store{};  ///< Shared store for coupled systems; nullptr = allocate own.
  std::shared_ptr<SystemSolver> cycle_zero_solver{};
};

// ---------------------------------------------------------------------------
// FieldInfo — returned by register; consumed by buildFromStore
// ---------------------------------------------------------------------------

/// @brief Returned by registerSolidMechanicsWithInternalVarsFields.
template <int dim, int order, typename StateSpace, typename DisplacementTimeRule, typename InternalVarTimeRule,
          typename... parameter_space>
struct SolidMechanicsWithInternalVarsFieldInfo {
  std::shared_ptr<FieldStore> field_store;
  FieldType<H1<order, dim>> disp_type;
  FieldType<H1<order, dim>> disp_old_type;
  FieldType<H1<order, dim>> velo_old_type;
  FieldType<H1<order, dim>> accel_old_type;
  FieldType<StateSpace> state_type;
  FieldType<StateSpace> state_old_type;
  std::tuple<FieldType<parameter_space>...> parameter_types;
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;
  std::shared_ptr<DirichletBoundaryConditions> state_bc;
  std::shared_ptr<DisplacementTimeRule> disp_time_rule_ptr;
  std::shared_ptr<InternalVarTimeRule> state_time_rule_ptr;
};

// ---------------------------------------------------------------------------
// Phase 1: register fields
// ---------------------------------------------------------------------------

/**
 * @brief Register all solid-mechanics-with-internal-vars fields into a FieldStore.
 *
 * Phase 1 of the two-phase factory. Safe to call before other physics have registered their
 * fields; weak form construction is deferred to buildSolidMechanicsWithInternalVarsSystemFromStore.
 * When composing coupled systems, call all registerXxxFields functions before any buildXxxFromStore.
 */
template <int dim, int order, typename StateSpace, typename DisplacementTimeRule, typename InternalVarTimeRule,
          typename... parameter_space>
SolidMechanicsWithInternalVarsFieldInfo<dim, order, StateSpace, DisplacementTimeRule, InternalVarTimeRule,
                                        parameter_space...>
registerSolidMechanicsWithInternalVarsFields(std::shared_ptr<FieldStore> field_store, DisplacementTimeRule disp_rule,
                                             InternalVarTimeRule state_rule,
                                             FieldType<parameter_space>... parameter_types)
{
  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  if (!field_store->hasField(shape_disp_type.name)) {
    field_store->addShapeDisp(shape_disp_type);
  }

  // Displacement fields (4-state second-order)
  auto disp_time_rule_ptr = std::make_shared<DisplacementTimeRule>(disp_rule);
  FieldType<H1<order, dim>> disp_type("displacement_solve_state");
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule_ptr);
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VAL, "displacement");
  auto velo_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DOT, "velocity");
  auto accel_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DDOT, "acceleration");

  // Internal variable fields (2-state first-order)
  auto state_time_rule_ptr = std::make_shared<InternalVarTimeRule>(state_rule);
  FieldType<StateSpace> state_type("state_solve_state");
  auto state_bc = field_store->addIndependent(state_type, state_time_rule_ptr);
  auto state_old_type = field_store->addDependent(state_type, FieldStore::TimeDerivative::VAL, "state");

  // Parameters
  auto prefix_param = [&](auto& pt) {
    pt.name = "param_" + pt.name;
    field_store->addParameter(pt);
  };
  (prefix_param(parameter_types), ...);

  return {field_store,  disp_type,    disp_old_type,       velo_old_type,      accel_old_type,
          state_type,   state_old_type, std::make_tuple(parameter_types...), disp_bc,
          state_bc,     disp_time_rule_ptr, state_time_rule_ptr};
}

// ---------------------------------------------------------------------------
// Phase 2: build system from store
// ---------------------------------------------------------------------------

/**
 * @brief Build a SolidMechanicsWithInternalVarsSystem from an already-populated FieldStore.
 *
 * Phase 2 of the two-phase factory.  Constructs all weak forms from fields already registered
 * in the store (including any coupling fields from other physics).
 *
 * @tparam Coupling  CouplingSpec listing fields borrowed from other physics (leading tail positions).
 */
template <int dim, int order, typename StateSpace, typename DisplacementTimeRule, typename InternalVarTimeRule,
          typename Coupling, typename... parameter_space>
std::shared_ptr<SolidMechanicsWithInternalVarsSystem<dim, order, StateSpace, DisplacementTimeRule, InternalVarTimeRule,
                                                     Coupling, parameter_space...>>
buildSolidMechanicsWithInternalVarsSystemFromStore(
    SolidMechanicsWithInternalVarsFieldInfo<dim, order, StateSpace, DisplacementTimeRule, InternalVarTimeRule,
                                            parameter_space...>
        info,
    std::shared_ptr<SystemSolver> solver,
    const SolidMechanicsWithInternalVarsOptions<dim, order, StateSpace, DisplacementTimeRule, InternalVarTimeRule,
                                                parameter_space...>& options,
    const Coupling& coupling)
{
  auto& field_store = info.field_store;
  auto& disp_type = info.disp_type;
  auto& disp_old_type = info.disp_old_type;
  auto& velo_old_type = info.velo_old_type;
  auto& accel_old_type = info.accel_old_type;
  auto& state_type = info.state_type;
  auto& state_old_type = info.state_old_type;
  auto parameter_types = info.parameter_types;
  auto& disp_bc = info.disp_bc;
  auto& state_bc = info.state_bc;
  auto& disp_time_rule_ptr = info.disp_time_rule_ptr;
  auto& state_time_rule_ptr = info.state_time_rule_ptr;

  using SystemType = SolidMechanicsWithInternalVarsSystem<dim, order, StateSpace, DisplacementTimeRule,
                                                          InternalVarTimeRule, Coupling, parameter_space...>;

  // Solid weak form: (u, u_old, v_old, a_old, alpha, alpha_old, coupling_fields..., params...)
  std::string solid_res_name = field_store->prefix("solid_residual");
  auto solid_weak_form = std::apply(
      [&](auto&... params) {
        return std::apply(
            [&](auto&... cfs) {
              return std::make_shared<typename SystemType::SolidWeakFormType>(
                  solid_res_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
                  field_store->createSpaces(solid_res_name, disp_type.name, disp_type, disp_old_type, velo_old_type,
                                            accel_old_type, state_type, state_old_type, cfs..., params...));
            },
            coupling.fields);
      },
      parameter_types);

  // State weak form: (alpha, alpha_old, u, u_old, v_old, a_old, coupling_fields..., params...)
  std::string state_res_name = field_store->prefix("state_residual");
  auto state_weak_form = std::apply(
      [&](auto&... params) {
        return std::apply(
            [&](auto&... cfs) {
              return std::make_shared<typename SystemType::StateWeakFormType>(
                  state_res_name, field_store->getMesh(), field_store->getField(state_type.name).get()->space(),
                  field_store->createSpaces(state_res_name, state_type.name, state_type, state_old_type, disp_type,
                                            disp_old_type, velo_old_type, accel_old_type, cfs..., params...));
            },
            coupling.fields);
      },
      parameter_types);

  auto sys = std::make_shared<SystemType>(field_store, solver,
                                          std::vector<std::shared_ptr<WeakForm>>{solid_weak_form, state_weak_form});
  sys->disp_bc = disp_bc;
  sys->state_bc = state_bc;
  sys->disp_time_rule = disp_time_rule_ptr;
  sys->state_time_rule = state_time_rule_ptr;
  sys->solid_weak_form = solid_weak_form;
  sys->state_weak_form = state_weak_form;

  if (disp_time_rule_ptr->requiresInitialAccelerationSolve()) {
    std::string cycle_zero_name = field_store->prefix("solid_reaction");
    auto accel_as_unknown = accel_old_type;
    accel_as_unknown.is_unknown = true;
    FieldType<H1<order, dim>> disp_cz_input(disp_type.name);
    FieldType<StateSpace> state_cz_input(state_type.name);
    sys->cycle_zero_solid_weak_form = std::apply(
        [&](auto&... params) {
          return std::apply(
              [&](auto&... cfs) {
                return std::make_shared<typename SystemType::CycleZeroSolidWeakFormType>(
                    cycle_zero_name, field_store->getMesh(),
                    field_store->getField(accel_old_type.name).get()->space(),
                    field_store->createSpaces(cycle_zero_name, accel_old_type.name, disp_cz_input, velo_old_type,
                                              accel_as_unknown, state_cz_input, cfs..., params...));
              },
              coupling.fields);
        },
        parameter_types);
    field_store->markWeakFormInternal(cycle_zero_name);
    field_store->shareBoundaryConditions(accel_old_type.name, disp_bc);

    std::shared_ptr<SystemSolver> cycle_zero_solver;
    if (options.cycle_zero_solver) {
      cycle_zero_solver = options.cycle_zero_solver;
    } else {
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
      cycle_zero_solver = std::make_shared<SystemSolver>(
          buildNonlinearBlockSolver(cz_nonlin, cz_lin, *field_store->getMesh()));
    }
    sys->cycle_zero_system = makeSubSystem(field_store, cycle_zero_solver, {sys->cycle_zero_solid_weak_form});
  }

  return sys;
}

// ---------------------------------------------------------------------------
// Standalone factory — thin wrapper, backwards-compatible
// ---------------------------------------------------------------------------

/**
 * @brief Standalone factory — allocates its own FieldStore and builds the full system.
 *
 * Thin wrapper around registerSolidMechanicsWithInternalVarsFields +
 * buildSolidMechanicsWithInternalVarsSystemFromStore.  Existing call sites are unchanged.
 */
template <int dim, int disp_order, typename StateSpace, typename DisplacementTimeRule, typename InternalVarTimeRule,
          typename... parameter_space>
std::shared_ptr<SolidMechanicsWithInternalVarsSystem<dim, disp_order, StateSpace, DisplacementTimeRule,
                                                     InternalVarTimeRule, CouplingSpec<>, parameter_space...>>
buildSolidMechanicsWithInternalVarsSystem(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<SystemSolver> solver, DisplacementTimeRule disp_rule,
    InternalVarTimeRule state_rule,
    SolidMechanicsWithInternalVarsOptions<dim, disp_order, StateSpace, DisplacementTimeRule, InternalVarTimeRule,
                                          parameter_space...>
        options,
    FieldType<parameter_space>... parameter_types)
{
  auto field_store = options.shared_field_store ? options.shared_field_store
                                                : std::make_shared<FieldStore>(mesh, 100, options.prepend_name);
  auto info = registerSolidMechanicsWithInternalVarsFields<dim, disp_order, StateSpace>(
      field_store, disp_rule, state_rule, parameter_types...);
  return buildSolidMechanicsWithInternalVarsSystemFromStore<dim, disp_order, StateSpace, DisplacementTimeRule,
                                                            InternalVarTimeRule, CouplingSpec<>>(
      info, solver, options, CouplingSpec<>{});
}

}  // namespace smith
