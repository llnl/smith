// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_system.hpp
 * @brief Defines the SolidMechanicsSystem struct and its factory function
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/coupling_spec.hpp"

namespace smith {

/**
 * @brief System struct for solid dynamics with configurable time integration.
 *
 * Always uses a 4-state field layout (displacement_solve_state, displacement, velocity, acceleration).
 * Use ImplicitNewmarkSecondOrderTimeIntegrationRule for transient dynamics,
 * or QuasiStaticSecondOrderTimeIntegrationRule for quasi-static problems.
 *
 * @tparam dim Spatial dimension.
 * @tparam order Polynomial order for displacement field.
 * @tparam DisplacementTimeRule Time integration rule type (must have num_states == 4).
 * @tparam Coupling CouplingSpec listing fields borrowed from other physics (default: none).
 *         Coupling fields occupy leading positions in the tail after the 4 time-rule state fields,
 *         before user parameter_space fields.
 * @tparam parameter_space Parameter spaces for material properties.
 */
template <int dim, int order, typename DisplacementTimeRule = ImplicitNewmarkSecondOrderTimeIntegrationRule,
          typename Coupling = CouplingSpec<>, typename... parameter_space>
struct SolidMechanicsSystem : public SystemBase {
  using SystemBase::SystemBase;

  static_assert(DisplacementTimeRule::num_states == 4, "SolidMechanicsSystem requires a 4-state time integration rule");

  /// Main weak form: (u, u_old, v_old, a_old, coupling_fields..., params...)
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<order, dim>,
      typename detail::TimeRuleParamsWithCoupling<DisplacementTimeRule, H1<order, dim>, Coupling,
                                                  parameter_space...>::type>;

  /// Cycle-zero form: (u, v_old, a, coupling_fields..., params...)
  /// 3-state form: u, v, a (no u_old needed; at cycle 0 u and v are given, solve for a)
  using CycleZeroSolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<order, dim>,
      typename detail::AppendCouplingToParams<Coupling, Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>>,
                                              parameter_space...>::type>;

  /// L2 projection weak form for PK1 stress output (dim*dim components).
  /// Args: (stress_unknown, u, u_old, v_old, a_old, coupling_fields..., params...).
  using StressOutputWeakFormType = TimeDiscretizedWeakForm<
      dim, L2<0, dim * dim>,
      typename detail::AppendCouplingToParams<
          Coupling, Parameters<L2<0, dim * dim>, H1<order, dim>, H1<order, dim>, H1<order, dim>, H1<order, dim>>,
          parameter_space...>::type>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;  ///< Solid mechanics weak form.
  std::shared_ptr<CycleZeroSolidWeakFormType>
      cycle_zero_solid_weak_form;                        ///< Typed cycle zero solid mechanics weak form.
  std::shared_ptr<SystemBase> cycle_zero_system;         ///< Cycle-zero system for initial acceleration solve.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;  ///< Displacement boundary conditions.
  std::shared_ptr<DisplacementTimeRule> disp_time_rule;  ///< Time integration rule.

  std::shared_ptr<StressOutputWeakFormType> stress_weak_form;  ///< Stress projection weak form (nullptr if disabled).
  std::shared_ptr<SystemBase> stress_output_system;            ///< Post-solve system for stress projection.

  /**
   * @brief Set the material model for a domain, defining integrals for the solid weak form.
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_rule = disp_time_rule;
    solid_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          auto [u_current, v_current, a_current] = captured_rule->interpolate(t_info, u, u_old, v_old, a_old);

          typename MaterialType::State state;
          auto pk_stress = material(state, get<DERIVATIVE>(u_current), params...);

          return smith::tuple{get<VALUE>(a_current) * material.density, pk_stress};
        });

    // Add to cycle-zero weak form (at cycle 0, u and v are given, solve for a)
    if (cycle_zero_solid_weak_form) {
      cycle_zero_solid_weak_form->addBodyIntegral(
          domain_name, [=](auto /*t_info*/, auto /*X*/, auto u, auto /*v_old*/, auto a, auto... params) {
            typename MaterialType::State state;
            auto pk_stress = material(state, get<DERIVATIVE>(u), params...);

            return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
          });
    }

    // Stress output projection: L2 projection of PK1 stress onto an L2 piecewise-constant field.
    // Residual: ∫ test · (stress_unknown - pk_stress(u)) dx = 0.
    // Args: (stress_unknown, u, u_old, v_old, a_old, params...). stress_unknown is the Jacobian
    // variable so the solver builds the mass matrix against it, and the (- pk_stress) term
    // becomes the RHS.
    if (stress_weak_form) {
      stress_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto stress, auto u, auto u_old,
                                                         auto v_old, auto a_old, auto... params) {
        auto [u_current, v_current, a_current] = captured_rule->interpolate(t_info, u, u_old, v_old, a_old);

        typename MaterialType::State state;
        auto pk_stress = material(state, get<DERIVATIVE>(u_current), params...);

        // Flatten dim x dim stress tensor into dim*dim vector, subtract from current stress unknown
        auto pk_flat = make_tensor<dim * dim>([&](int i) { return pk_stress[i / dim][i % dim]; });
        return smith::tuple{get<VALUE>(stress) - pk_flat, tensor<double, dim * dim, dim>{}};
      });
    }
  }

  /**
   * @brief Add a body force to the system (with DependsOn).
   * @tparam active_parameters Indices of fields this force depends on.
   * @tparam BodyForceType The body force function type.
   * @param depends_on Dependency specification for which input fields to pass.
   * @param domain_name The name of the domain to apply the force to.
   * @param force_function The force function (t, X, u, v, a, params...).
   */
  template <int... active_parameters, typename BodyForceType>
  void addBodyForce(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                    BodyForceType force_function)
  {
    auto captured_rule = disp_time_rule;
    solid_weak_form->addBodySource(
        depends_on, domain_name, [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          auto [u_current, v_current, a_current] = captured_rule->interpolate(t_info, u, u_old, v_old, a_old);
          return force_function(t_info.time(), X, u_current, v_current, a_current, params...);
        });

    addCycleZeroBodySourceImpl(
        domain_name,
        [=](auto t_info, auto X, auto u, auto v_old, auto a, auto... params) {
          return force_function(t_info.time(), X, u, v_old, a, params...);
        },
        std::make_index_sequence<3 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a body force to the system.
   * @tparam BodyForceType The body force function type.
   * @param domain_name The name of the domain to apply the force to.
   * @param force_function The force function (t, X, u, v, a, params...).
   */
  template <typename BodyForceType>
  void addBodyForce(const std::string& domain_name, BodyForceType force_function)
  {
    addBodyForceAllParams(domain_name, force_function, std::make_index_sequence<4 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a surface traction (flux) to the system (with DependsOn).
   * @tparam active_parameters Indices of fields this traction depends on.
   * @tparam TractionType The traction function type.
   * @param depends_on Dependency specification for which input fields to pass.
   * @param domain_name The name of the boundary domain to apply the traction to.
   * @param traction_function The traction function (t, X, n, u, v, a, params...).
   */
  template <int... active_parameters, typename TractionType>
  void addTraction(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   TractionType traction_function)
  {
    auto captured_rule = disp_time_rule;
    solid_weak_form->addBoundaryFlux(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          auto [u_current, v_current, a_current] = captured_rule->interpolate(t_info, u, u_old, v_old, a_old);
          return traction_function(t_info.time(), X, n, u_current, v_current, a_current, params...);
        });

    addCycleZeroBoundaryFluxImpl(
        domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto v_old, auto a, auto... params) {
          return traction_function(t_info.time(), X, n, u, v_old, a, params...);
        },
        std::make_index_sequence<3 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a surface traction (flux) to the system.
   * @tparam TractionType The traction function type.
   * @param domain_name The name of the boundary domain to apply the traction to.
   * @param traction_function The traction function (t, X, n, u, v, a, params...).
   */
  template <typename TractionType>
  void addTraction(const std::string& domain_name, TractionType traction_function)
  {
    addTractionAllParams(domain_name, traction_function, std::make_index_sequence<4 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a pressure boundary condition (follower force) (with DependsOn).
   * @tparam active_parameters Indices of fields this pressure depends on.
   * @tparam PressureType The pressure function type.
   * @param depends_on Dependency specification for which input fields to pass.
   * @param domain_name The name of the boundary domain.
   * @param pressure_function The pressure function (t, X, params...).
   */
  template <int... active_parameters, typename PressureType>
  void addPressure(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   PressureType pressure_function)
  {
    auto captured_rule = disp_time_rule;
    solid_weak_form->addBoundaryIntegral(
        depends_on, domain_name, [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          auto u_current = captured_rule->value(t_info, u, u_old, v_old, a_old);

          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        });

    addCycleZeroBoundaryIntegralImpl(
        domain_name,
        [=](auto t_info, auto X, auto u, auto /*v_old*/, auto /*a*/, auto... params) {
          auto u_current = u;

          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        },
        std::make_index_sequence<3 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a pressure boundary condition (follower force).
   * @tparam PressureType The pressure function type.
   * @param domain_name The name of the boundary domain.
   * @param pressure_function The pressure function (t, X, params...).
   */
  template <typename PressureType>
  void addPressure(const std::string& domain_name, PressureType pressure_function)
  {
    addPressureAllParams(domain_name, pressure_function, std::make_index_sequence<4 + Coupling::num_coupling_fields + sizeof...(parameter_space)>{});
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

  // Cycle-zero helpers: always use all-params DependsOn with the 3-state cycle-zero form count
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

template <int dim, int order, typename DisplacementTimeRule, typename... parameter_space>
struct SolidMechanicsOptions {
  std::string prepend_name{};
  std::shared_ptr<FieldStore> shared_field_store{};  ///< Shared store for coupled systems; nullptr = allocate own.
  std::shared_ptr<SystemSolver> cycle_zero_solver{};
  bool enable_stress_output = false;
  std::shared_ptr<SystemSolver> stress_output_solver{};
};

/// @brief Returned by registerSolidMechanicsFields; holds the FieldType tokens needed by buildSolidMechanicsSystemFromStore.
template <int dim, int order, typename DisplacementTimeRule, typename... parameter_space>
struct SolidMechanicsFieldInfo {
  std::shared_ptr<FieldStore> field_store;
  FieldType<H1<order, dim>> disp_type;
  FieldType<H1<order, dim>> disp_old_type;
  FieldType<H1<order, dim>> velo_old_type;
  FieldType<H1<order, dim>> accel_old_type;
  std::tuple<FieldType<parameter_space>...> parameter_types;
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;
  std::shared_ptr<DisplacementTimeRule> disp_time_rule_ptr;
};

/**
 * @brief Register all solid mechanics fields into a FieldStore.
 *
 * Phase 1 of the two-phase factory. Safe to call before other physics have registered their
 * fields; weak form construction is deferred to buildSolidMechanicsSystemFromStore.
 * When composing coupled systems, call all registerXxxFields functions before any buildXxxFromStore.
 */
template <int dim, int order, typename DisplacementTimeRule, typename... parameter_space>
SolidMechanicsFieldInfo<dim, order, DisplacementTimeRule, parameter_space...> registerSolidMechanicsFields(
    std::shared_ptr<FieldStore> field_store, DisplacementTimeRule disp_time_rule,
    FieldType<parameter_space>... parameter_types)
{
  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  if (!field_store->hasField(shape_disp_type.name)) {
    field_store->addShapeDisp(shape_disp_type);
  }

  auto disp_time_rule_ptr = std::make_shared<DisplacementTimeRule>(disp_time_rule);
  FieldType<H1<order, dim>> disp_type("displacement_solve_state");
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule_ptr);

  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VAL, "displacement");
  auto velo_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DOT, "velocity");
  auto accel_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DDOT, "acceleration");

  auto prefix_param = [&](auto& pt) {
    pt.name = "param_" + pt.name;
    field_store->addParameter(pt);
  };
  (prefix_param(parameter_types), ...);

  return {field_store,    disp_type,    disp_old_type, velo_old_type, accel_old_type,
          std::make_tuple(parameter_types...), disp_bc, disp_time_rule_ptr};
}

/**
 * @brief Build a SolidMechanicsSystem from an already-populated FieldStore.
 *
 * Phase 2 of the two-phase factory. Constructs all weak forms using fields already registered
 * in the store (including any coupling fields from other physics).
 *
 * @tparam Coupling  CouplingSpec listing fields borrowed from other physics (leading tail positions).
 */
template <int dim, int order, typename DisplacementTimeRule, typename Coupling, typename... parameter_space>
std::shared_ptr<SolidMechanicsSystem<dim, order, DisplacementTimeRule, Coupling, parameter_space...>>
buildSolidMechanicsSystemFromStore(
    SolidMechanicsFieldInfo<dim, order, DisplacementTimeRule, parameter_space...> info,
    std::shared_ptr<SystemSolver> solver,
    const SolidMechanicsOptions<dim, order, DisplacementTimeRule, parameter_space...>& options,
    const Coupling& coupling)
{
  auto& field_store = info.field_store;
  auto& disp_type = info.disp_type;
  auto& disp_old_type = info.disp_old_type;
  auto& velo_old_type = info.velo_old_type;
  auto& accel_old_type = info.accel_old_type;
  auto parameter_types = info.parameter_types;
  auto& disp_bc = info.disp_bc;
  auto& disp_time_rule_ptr = info.disp_time_rule_ptr;

  using SystemType = SolidMechanicsSystem<dim, order, DisplacementTimeRule, Coupling, parameter_space...>;

  // Create solid mechanics weak form: (u, u_old, v_old, a_old, coupling_fields..., params...)
  std::string force_name = field_store->prefix("reactions");
  auto solid_weak_form = std::apply(
      [&](auto&... params) {
        return std::apply(
            [&](auto&... cfs) {
              return std::make_shared<typename SystemType::SolidWeakFormType>(
                  force_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
                  field_store->createSpaces(force_name, disp_type.name, disp_type, disp_old_type, velo_old_type,
                                            accel_old_type, cfs..., params...));
            },
            coupling.fields);
      },
      parameter_types);

  auto sys = std::make_shared<SystemType>(field_store, solver, std::vector<std::shared_ptr<WeakForm>>{solid_weak_form});
  sys->disp_bc = disp_bc;
  sys->disp_time_rule = disp_time_rule_ptr;
  sys->solid_weak_form = solid_weak_form;

  if (disp_time_rule_ptr->requiresInitialAccelerationSolve()) {
    std::string cycle_zero_name = field_store->prefix("solid_reaction");
    auto accel_as_unknown = accel_old_type;
    accel_as_unknown.is_unknown = true;
    FieldType<H1<order, dim>> disp_cz_input(disp_type.name);
    sys->cycle_zero_solid_weak_form = std::apply(
        [&](auto&... params) {
          return std::apply(
              [&](auto&... cfs) {
                return std::make_shared<typename SystemType::CycleZeroSolidWeakFormType>(
                    cycle_zero_name, field_store->getMesh(),
                    field_store->getField(accel_old_type.name).get()->space(),
                    field_store->createSpaces(cycle_zero_name, accel_old_type.name, disp_cz_input, velo_old_type,
                                              accel_as_unknown, cfs..., params...));
              },
              coupling.fields);
        },
        parameter_types);
    field_store->markWeakFormInternal(cycle_zero_name);
    field_store->shareBoundaryConditions(accel_old_type.name, disp_bc);

    std::shared_ptr<SystemSolver> cz_solver;
    if (options.cycle_zero_solver) {
      cz_solver = options.cycle_zero_solver;
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
      cz_solver = std::make_shared<SystemSolver>(
          buildNonlinearBlockSolver(cz_nonlin, cz_lin, *field_store->getMesh()));
    }
    sys->cycle_zero_system = makeSubSystem(field_store, cz_solver, {sys->cycle_zero_solid_weak_form});
  }

  if (options.enable_stress_output) {
    auto stress_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
    FieldType<L2<0, dim * dim>> stress_type("stress_solve_state");
    field_store->addIndependent(stress_type, stress_time_rule);
    field_store->addDependent(stress_type, FieldStore::TimeDerivative::VAL, "stress");

    FieldType<H1<order, dim>> disp_as_input(disp_type.name);
    std::string stress_name = field_store->prefix("stress_projection");
    sys->stress_weak_form = std::apply(
        [&](auto&... params) {
          return std::apply(
              [&](auto&... cfs) {
                return std::make_shared<typename SystemType::StressOutputWeakFormType>(
                    stress_name, field_store->getMesh(), field_store->getField(stress_type.name).get()->space(),
                    field_store->createSpaces(stress_name, stress_type.name, stress_type, disp_as_input, disp_old_type,
                                              velo_old_type, accel_old_type, cfs..., params...));
              },
              coupling.fields);
        },
        parameter_types);

    std::shared_ptr<SystemSolver> stress_solver;
    if (options.stress_output_solver) {
      stress_solver = options.stress_output_solver;
    } else {
      NonlinearSolverOptions stress_nonlin{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 1e-14,
                                           .absolute_tol = 1e-14,
                                           .max_iterations = 2,
                                           .print_level = 0};
      LinearSolverOptions stress_lin{.linear_solver = LinearSolver::CG,
                                     .preconditioner = Preconditioner::HypreJacobi,
                                     .relative_tol = 1e-14,
                                     .absolute_tol = 1e-14,
                                     .max_iterations = 1000,
                                     .print_level = 0};
      stress_solver = std::make_shared<SystemSolver>(
          buildNonlinearBlockSolver(stress_nonlin, stress_lin, *field_store->getMesh()));
    }
    sys->stress_output_system = makeSubSystem(field_store, stress_solver, {sys->stress_weak_form});
  }

  return sys;
}

/**
 * @brief Standalone factory — allocates its own FieldStore and builds the full system.
 *
 * Thin wrapper around registerSolidMechanicsFields + buildSolidMechanicsSystemFromStore.
 * Existing call sites are unchanged.
 */
template <int dim, int order, typename DisplacementTimeRule, typename... parameter_space>
std::shared_ptr<SolidMechanicsSystem<dim, order, DisplacementTimeRule, CouplingSpec<>, parameter_space...>>
buildSolidMechanicsSystem(std::shared_ptr<Mesh> mesh, std::shared_ptr<SystemSolver> solver,
                          DisplacementTimeRule disp_time_rule,
                          SolidMechanicsOptions<dim, order, DisplacementTimeRule, parameter_space...> options,
                          FieldType<parameter_space>... parameter_types)
{
  auto field_store = options.shared_field_store
                         ? options.shared_field_store
                         : std::make_shared<FieldStore>(mesh, 100, options.prepend_name);
  auto info = registerSolidMechanicsFields<dim, order>(field_store, disp_time_rule, parameter_types...);
  return buildSolidMechanicsSystemFromStore<dim, order, DisplacementTimeRule, CouplingSpec<>>(info, solver, options,
                                                                                              CouplingSpec<>{});
}

}  // namespace smith
