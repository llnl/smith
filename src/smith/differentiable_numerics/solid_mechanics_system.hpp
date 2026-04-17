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
#include "smith/differentiable_numerics/coupling_params.hpp"

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
 * @tparam Coupling CouplingParams listing fields borrowed from other physics (default: none).
 *         Coupling fields occupy leading positions in the tail after the 4 time-rule state fields,
 *         before user parameter_space fields.
 * @tparam parameter_space Parameter spaces for material properties.
 */
template <int dim, int order, typename DisplacementTimeRule = ImplicitNewmarkSecondOrderTimeIntegrationRule,
          typename Coupling = CouplingParams<>>
struct SolidMechanicsSystem : public SystemBase {
  using SystemBase::SystemBase;

  static_assert(DisplacementTimeRule::num_states == 4, "SolidMechanicsSystem requires a 4-state time integration rule");

  /// Main weak form: (u, u_old, v_old, a_old, coupling_fields..., params...)
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<order, dim>,
      typename detail::TimeRuleParamsWithCoupling<DisplacementTimeRule, H1<order, dim>, Coupling>::type>;

  /// Cycle-zero form: (u, v_old, a, coupling_fields..., params...)
  /// 3-state form: u, v, a (no u_old needed; at cycle 0 u and v are given, solve for a)
  using CycleZeroSolidWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<order, dim>,
                              typename detail::AppendCouplingToParams<
                                  Coupling, Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>>>::type>;

  /// L2 projection weak form for PK1 stress output (dim*dim components).
  /// Args: (stress_unknown, u, u_old, v_old, a_old, coupling_fields..., params...).
  using StressOutputWeakFormType = TimeDiscretizedWeakForm<
      dim, L2<0, dim * dim>,
      typename detail::AppendCouplingToParams<Coupling, Parameters<L2<0, dim * dim>, H1<order, dim>, H1<order, dim>,
                                                                   H1<order, dim>, H1<order, dim>>>::type>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;  ///< Solid mechanics weak form.
  std::shared_ptr<CycleZeroSolidWeakFormType>
      cycle_zero_solid_weak_form;                        ///< Typed cycle zero solid mechanics weak form.
  std::shared_ptr<SystemBase> cycle_zero_system;         ///< Cycle-zero system for initial acceleration solve.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;  ///< Displacement boundary conditions.
  std::shared_ptr<DisplacementTimeRule> disp_time_rule;  ///< Time integration rule.

  std::shared_ptr<StressOutputWeakFormType> stress_weak_form;  ///< Stress projection weak form (nullptr if disabled).
  std::shared_ptr<SystemBase> stress_output_system;            ///< Post-solve system for stress projection.
  bool output_cauchy_stress = false;                           ///< Project Cauchy stress instead of PK1 when true.

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
      bool do_cauchy = this->output_cauchy_stress;
      stress_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto stress, auto u, auto u_old,
                                                         auto v_old, auto a_old, auto... params) {
        auto [u_current, v_current, a_current] = captured_rule->interpolate(t_info, u, u_old, v_old, a_old);

        typename MaterialType::State state;
        auto pk_stress = material(state, get<DERIVATIVE>(u_current), params...);

        // Flatten the chosen stress into a dim*dim vector and subtract from the unknown.
        auto flat_stress = [&]() {
          if (do_cauchy) {
            static constexpr auto I_ = Identity<dim>();
            auto F = get<DERIVATIVE>(u_current) + I_;
            auto J = det(F);
            auto sigma = (1.0 / J) * dot(pk_stress, transpose(F));
            return make_tensor<dim * dim>([&](int i) { return sigma[i / dim][i % dim]; });
          }
          return make_tensor<dim * dim>([&](int i) { return pk_stress[i / dim][i % dim]; });
        }();
        return smith::tuple{get<VALUE>(stress) - flat_stress, tensor<double, dim * dim, dim>{}};
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
        std::make_index_sequence<3 + Coupling::num_coupling_fields>{});
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
    addBodyForceAllParams(domain_name, force_function, std::make_index_sequence<4 + Coupling::num_coupling_fields>{});
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
        std::make_index_sequence<3 + Coupling::num_coupling_fields>{});
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
    addTractionAllParams(domain_name, traction_function, std::make_index_sequence<4 + Coupling::num_coupling_fields>{});
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
        std::make_index_sequence<3 + Coupling::num_coupling_fields>{});
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
    addPressureAllParams(domain_name, pressure_function, std::make_index_sequence<4 + Coupling::num_coupling_fields>{});
  }

  /// Set zero-displacement Dirichlet BC on all components.
  void setDisplacementBC(const Domain& domain) { disp_bc->template setFixedVectorBCs<dim>(domain); }

  /// Set zero-displacement BC on specific components.
  void setDisplacementBC(const Domain& domain, std::vector<int> components)
  {
    disp_bc->template setFixedVectorBCs<dim>(domain, components);
  }

  /// Set displacement BC with a prescribed function.
  template <typename AppliedDisplacementFunction>
  void setDisplacementBC(const Domain& domain, AppliedDisplacementFunction f)
  {
    disp_bc->template setVectorBCs<dim>(domain, f);
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

struct SolidMechanicsOptions {
  bool enable_stress_output = false;
  bool output_cauchy_stress = false;  ///< When true, project Cauchy stress (sigma) instead of PK1 (P).
};

/**
 * @brief Register all solid mechanics fields into a FieldStore.
 *
 * Phase 1 of the two-phase initialization. Pass an instance of the desired time integration rule
 * so its type is deduced; only `<dim, order>` need be specified explicitly.
 *
 * @return PhysicsFields carrying the exported field tokens and time rule type.
 */
template <int dim, int order, typename DisplacementTimeRule>
auto registerSolidMechanicsFields(std::shared_ptr<FieldStore> field_store, DisplacementTimeRule /*rule*/)
{
  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  if (!field_store->hasField(field_store->prefix(shape_disp_type.name))) {
    field_store->addShapeDisp(shape_disp_type);
  }

  auto disp_time_rule_ptr = std::make_shared<DisplacementTimeRule>();
  FieldType<H1<order, dim>> disp_type("displacement_solve_state");
  field_store->addIndependent(disp_type, disp_time_rule_ptr);

  field_store->addDependent(disp_type, FieldStore::TimeDerivative::VAL, "displacement");
  field_store->addDependent(disp_type, FieldStore::TimeDerivative::DOT, "velocity");
  field_store->addDependent(disp_type, FieldStore::TimeDerivative::DDOT, "acceleration");

  return PhysicsFields<DisplacementTimeRule, H1<order, dim>, H1<order, dim>, H1<order, dim>, H1<order, dim>>{
      field_store, FieldType<H1<order, dim>>(field_store->prefix("displacement_solve_state")),
      FieldType<H1<order, dim>>(field_store->prefix("displacement")),
      FieldType<H1<order, dim>>(field_store->prefix("velocity")),
      FieldType<H1<order, dim>>(field_store->prefix("acceleration"))};
}

/**
 * @brief Register all solid mechanics fields (no rule instance — Rule given as explicit template param).
 *
 * Preferred form: Rule is deduced from the PhysicsFields type rather than a runtime instance.
 * Equivalent to the rule-instance overload but avoids requiring a dummy rule object.
 */
template <int dim, int order, typename DisplacementTimeRule>
auto registerSolidMechanicsFields(std::shared_ptr<FieldStore> field_store)
{
  return registerSolidMechanicsFields<dim, order>(field_store, DisplacementTimeRule{});
}

/**
 * @brief Register solid mechanics fields with parameters (no rule instance).
 *
 * @return CouplingParams carrying the exported field tokens (for use as coupling input to other systems).
 */
template <int dim, int order, typename DisplacementTimeRule, typename... parameter_space>
  requires(sizeof...(parameter_space) > 0)
auto registerSolidMechanicsFields(std::shared_ptr<FieldStore> field_store,
                                  FieldType<parameter_space>... parameter_types)
{
  return registerSolidMechanicsFields<dim, order>(field_store, DisplacementTimeRule{}, std::move(parameter_types)...);
}

/**
 * @brief Register all solid mechanics fields into a FieldStore (with parameters).
 *
 * Legacy overload that also registers parameter fields directly.
 * Prefer the no-params overload + registerParameterFields for new code.
 *
 * @return CouplingParams carrying the exported field tokens (for use as coupling input to other systems).
 */
template <int dim, int order, typename DisplacementTimeRule, typename... parameter_space>
  requires(sizeof...(parameter_space) > 0)
auto registerSolidMechanicsFields(std::shared_ptr<FieldStore> field_store, DisplacementTimeRule /*rule*/,
                                  FieldType<parameter_space>... parameter_types)
{
  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  if (!field_store->hasField(field_store->prefix(shape_disp_type.name))) {
    field_store->addShapeDisp(shape_disp_type);
  }

  auto disp_time_rule_ptr = std::make_shared<DisplacementTimeRule>();
  FieldType<H1<order, dim>> disp_type("displacement_solve_state");
  field_store->addIndependent(disp_type, disp_time_rule_ptr);

  field_store->addDependent(disp_type, FieldStore::TimeDerivative::VAL, "displacement");
  field_store->addDependent(disp_type, FieldStore::TimeDerivative::DOT, "velocity");
  field_store->addDependent(disp_type, FieldStore::TimeDerivative::DDOT, "acceleration");

  auto prefix_param = [&](auto& pt) {
    pt.name = "param_" + pt.name;
    field_store->addParameter(pt);
  };
  (prefix_param(parameter_types), ...);

  return CouplingParams{FieldType<H1<order, dim>>(field_store->prefix("displacement_solve_state")),
                        FieldType<H1<order, dim>>(field_store->prefix("displacement")),
                        FieldType<H1<order, dim>>(field_store->prefix("velocity")),
                        FieldType<H1<order, dim>>(field_store->prefix("acceleration")), parameter_types...};
}

/**
 * @brief Build a SolidMechanicsSystem with coupling, assuming fields are already registered.
 *
 * Phase 2 of the two-phase initialization. Pass the same rule instance used in
 * registerSolidMechanicsFields so the type is deduced; only `<dim, order>` need be specified.
 *
 * Returns `{system, cycle_zero_system, end_step_systems}` as a tuple.
 * `cycle_zero_system` is nullptr unless the rule requires an initial acceleration solve.
 * `end_step_systems` contains the stress output system when `enable_stress_output` is set.
 */
template <int dim, int order, typename DisplacementTimeRule, typename Coupling>
  requires detail::is_coupling_params_v<Coupling>
auto buildSolidMechanicsSystem(std::shared_ptr<FieldStore> field_store, DisplacementTimeRule /*rule*/,
                               const Coupling& coupling, std::shared_ptr<SystemSolver> solver,
                               const SolidMechanicsOptions& options)
{
  auto disp_time_rule_ptr = std::make_shared<DisplacementTimeRule>();

  FieldType<H1<1, dim>> shape_disp_type(field_store->prefix("shape_displacement"));
  FieldType<H1<order, dim>> disp_type(field_store->prefix("displacement_solve_state"), true);
  FieldType<H1<order, dim>> disp_old_type(field_store->prefix("displacement"));
  FieldType<H1<order, dim>> velo_old_type(field_store->prefix("velocity"));
  FieldType<H1<order, dim>> accel_old_type(field_store->prefix("acceleration"));

  auto disp_bc = field_store->getBoundaryConditions(disp_type.name);

  using SystemType = SolidMechanicsSystem<dim, order, DisplacementTimeRule, Coupling>;

  std::string force_name = field_store->prefix("reactions");
  auto solid_weak_form = std::apply(
      [&](auto&... cfs) {
        return std::make_shared<typename SystemType::SolidWeakFormType>(
            force_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
            field_store->createSpaces(force_name, disp_type.name, disp_type, disp_old_type, velo_old_type,
                                      accel_old_type, cfs...));
      },
      coupling.fields);

  auto sys = std::make_shared<SystemType>(field_store, solver, std::vector<std::shared_ptr<WeakForm>>{solid_weak_form});
  sys->disp_bc = disp_bc;
  sys->disp_time_rule = disp_time_rule_ptr;
  sys->solid_weak_form = solid_weak_form;
  sys->output_cauchy_stress = options.output_cauchy_stress;

  std::shared_ptr<SystemBase> cycle_zero_system;
  std::vector<std::shared_ptr<SystemBase>> end_step_systems;

  if (disp_time_rule_ptr->requiresInitialAccelerationSolve()) {
    std::string cycle_zero_name = field_store->prefix("solid_reaction");
    auto accel_as_unknown = accel_old_type;
    accel_as_unknown.is_unknown = true;
    FieldType<H1<order, dim>> disp_cz_input(disp_type.name);
    sys->cycle_zero_solid_weak_form = std::apply(
        [&](auto&... cfs) {
          return std::make_shared<typename SystemType::CycleZeroSolidWeakFormType>(
              cycle_zero_name, field_store->getMesh(), field_store->getField(accel_old_type.name).get()->space(),
              field_store->createSpaces(cycle_zero_name, accel_old_type.name, disp_cz_input, velo_old_type,
                                        accel_as_unknown, cfs...));
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

  if (options.enable_stress_output) {
    auto stress_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
    FieldType<L2<0, dim * dim>> stress_type("stress_solve_state");
    field_store->addIndependent(stress_type, stress_time_rule);
    field_store->addDependent(stress_type, FieldStore::TimeDerivative::VAL, "stress");

    FieldType<H1<order, dim>> disp_as_input(disp_type.name);
    std::string stress_name = field_store->prefix("stress_projection");
    sys->stress_weak_form = std::apply(
        [&](auto&... cfs) {
          return std::make_shared<typename SystemType::StressOutputWeakFormType>(
              stress_name, field_store->getMesh(), field_store->getField(stress_type.name).get()->space(),
              field_store->createSpaces(stress_name, stress_type.name, stress_type, disp_as_input, disp_old_type,
                                        velo_old_type, accel_old_type, cfs...));
        },
        coupling.fields);

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
    auto stress_solver =
        std::make_shared<SystemSolver>(buildNonlinearBlockSolver(stress_nonlin, stress_lin, *field_store->getMesh()));

    sys->stress_output_system = makeSystem(field_store, stress_solver, {sys->stress_weak_form});
    end_step_systems.push_back(sys->stress_output_system);
  }

  return std::make_tuple(sys, cycle_zero_system, end_step_systems);
}

/**
 * @brief Build a SolidMechanicsSystem from a coupling pack and optional parameter fields.
 *
 * Preferred API: Rule is given as explicit template param (no rule instance needed).
 * The coupling argument carries the fields borrowed from another physics (or CouplingParams<> for none).
 * Additional parameter_types are registered and appended after the coupling fields.
 *
 * Returns a @c SystemBuildResult so callers can write @c res.system, @c res.cycle_zero_system, etc.
 *
 * Usage:
 * @code
 *   auto res = buildSolidMechanicsSystem<dim, order, DispRule>(
 *       field_store, thermal_fields, solver, opts, youngs_modulus);
 *   auto solid = res.system;
 * @endcode
 */
template <int dim, int order, typename DisplacementTimeRule, typename Coupling, typename... parameter_space>
  requires detail::is_coupling_params_v<Coupling>
auto buildSolidMechanicsSystem(std::shared_ptr<FieldStore> field_store, const Coupling& coupling,
                               std::shared_ptr<SystemSolver> solver, const SolidMechanicsOptions& options,
                               FieldType<parameter_space>... parameter_types)
{
  (
      [&](auto& pt) {
        pt.name = "param_" + pt.name;
        field_store->addParameter(pt);
      }(parameter_types),
      ...);

  auto combined = std::apply([&](auto... cfs) { return CouplingParams{cfs..., parameter_types...}; }, coupling.fields);

  auto [sys, cz, ends] =
      buildSolidMechanicsSystem<dim, order>(field_store, DisplacementTimeRule{}, combined, solver, options);
  using SysType = typename decltype(sys)::element_type;
  return SystemBuildResult<SysType>{std::move(sys), std::move(cz), std::move(ends)};
}

/**
 * @brief Build a SolidMechanicsSystem from variadic field packs.
 *
 * New API: accepts any combination of PhysicsFields and CouplingParams packs.
 * The FieldStore is extracted from the PhysicsFields pack matching DisplacementTimeRule.
 * Non-self packs become coupling fields; CouplingParams packs are registered as parameters.
 *
 * Usage:
 * @code
 *   auto [solid, cz, end] = buildSolidMechanicsSystem<dim, order, DispRule>(
 *       solver, opts, param_fields, solid_fields, thermal_fields);
 * @endcode
 */
template <int dim, int order, typename DisplacementTimeRule, typename... FieldPacks>
  requires(sizeof...(FieldPacks) > 0 && (detail::is_physics_fields_v<FieldPacks> || ...) &&
           !(std::is_same_v<std::decay_t<FieldPacks>, SolidMechanicsOptions> || ...))
auto buildSolidMechanicsSystem(std::shared_ptr<SystemSolver> solver, const SolidMechanicsOptions& options,
                               const FieldPacks&... field_packs)
{
  auto field_store = detail::findFieldStore(field_packs...);
  (detail::registerParamsIfNeeded(field_store, field_packs), ...);
  auto coupling = detail::collectCouplingFields<DisplacementTimeRule>(field_packs...);
  return buildSolidMechanicsSystem<dim, order>(field_store, DisplacementTimeRule{}, coupling, solver, options);
}

}  // namespace smith
