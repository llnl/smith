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
#include "smith/physics/functional_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/coupling_params.hpp"

namespace smith {

namespace detail {

template <typename ValueType>
/// @brief Scale a cycle-zero residual contribution by `dt*dt`.
auto scaleCycleZeroTerm(const TimeInfo& t_info, ValueType value)
{
  return (t_info.dt() * t_info.dt()) * value;
}

template <typename InertiaType, typename StressType>
/// @brief Package scaled inertia and stress terms for the cycle-zero weak form.
auto makeScaledCycleZeroResidual(const TimeInfo& t_info, InertiaType inertia, StressType stress)
{
  return smith::tuple{scaleCycleZeroTerm(t_info, inertia), scaleCycleZeroTerm(t_info, stress)};
}

}  // namespace detail

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
  using SolidWeakFormType =
      FunctionalWeakForm<dim, H1<order, dim>, detail::TimeRuleParams<DisplacementTimeRule, H1<order, dim>, Coupling>>;

  /// Cycle-zero startup form reusing the main solid fields: (u, v, a, coupling_fields..., params...)
  /// At cycle 0 the velocity field holds the initial velocity (no prior step has run), so the
  /// second argument is `v` (initial), not `v_old`. Acceleration is the unknown for this internal solve.
  /// No extra fields are registered for cycle zero.
  using CycleZeroSolidWeakFormType =
      FunctionalWeakForm<dim, H1<order, dim>,
                         typename detail::AppendCouplingToParams<
                             Coupling, Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>>>::type>;

  /// L2 projection weak form for PK1 stress output (dim*dim components).
  /// Args: (stress_unknown, u, u_old, v_old, a_old, coupling_fields..., params...).
  using StressOutputWeakFormType = FunctionalWeakForm<
      dim, L2<0, dim * dim>,
      typename detail::AppendCouplingToParams<Coupling, Parameters<L2<0, dim * dim>, H1<order, dim>, H1<order, dim>,
                                                                   H1<order, dim>, H1<order, dim>>>::type>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;  ///< Solid mechanics weak form.
  std::shared_ptr<CycleZeroSolidWeakFormType>
      cycle_zero_solid_weak_form;                        ///< Typed cycle zero solid mechanics weak form.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;  ///< Displacement boundary conditions.
  std::shared_ptr<DisplacementTimeRule> disp_time_rule;  ///< Time integration rule.

  std::shared_ptr<StressOutputWeakFormType> stress_weak_form;  ///< Stress projection weak form (nullptr if disabled).
  std::shared_ptr<SystemBase> stress_output_system;            ///< Post-solve system for stress projection.
  bool output_cauchy_stress = false;                           ///< Project Cauchy stress instead of PK1 when true.

  /**
   * @brief Set material model for domain.
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
          auto pk_stress = material(t_info, state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current), params...);

          return smith::tuple{get<VALUE>(a_current) * material.density, pk_stress};
        });

    // Add to cycle-zero weak form (at cycle 0, u and v are given, solve for a)
    if (cycle_zero_solid_weak_form) {
      cycle_zero_solid_weak_form->addBodyIntegral(
          domain_name, [=](auto t_info, auto /*X*/, auto u, auto /*v_old*/, auto a, auto... params) {
            typename MaterialType::State state;
            auto pk_stress = material(t_info, state, get<DERIVATIVE>(u), tensor<double, dim, dim>{}, params...);

            return detail::makeScaledCycleZeroResidual(t_info, get<VALUE>(a) * material.density, pk_stress);
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
        auto pk_stress = material(t_info, state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current), params...);

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
    solid_weak_form->template addBodySource<0, 1, 2, 3, (4 + active_parameters)...>(
        DependsOn<0, 1, 2, 3, (4 + active_parameters)...>{}, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          auto [u_current, v_current, a_current] = captured_rule->interpolate(t_info, u, u_old, v_old, a_old);
          return force_function(t_info.time(), X, u_current, v_current, a_current, params...);
        });

    addCycleZeroBodySourceImpl(
        depends_on, domain_name, [=](auto t_info, auto X, auto u, auto v, auto a, auto... params) {
          return detail::scaleCycleZeroTerm(t_info, force_function(t_info.time(), X, u, v, a, params...));
        });
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
    addBodyForceAllParams(domain_name, force_function, std::make_index_sequence<Coupling::num_coupling_fields>{});
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
    solid_weak_form->template addBoundaryFlux<0, 1, 2, 3, (4 + active_parameters)...>(
        DependsOn<0, 1, 2, 3, (4 + active_parameters)...>{}, domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          auto [u_current, v_current, a_current] = captured_rule->interpolate(t_info, u, u_old, v_old, a_old);
          return traction_function(t_info.time(), X, n, u_current, v_current, a_current, params...);
        });

    addCycleZeroBoundaryFluxImpl(
        depends_on, domain_name, [=](auto t_info, auto X, auto n, auto u, auto v, auto a, auto... params) {
          return detail::scaleCycleZeroTerm(t_info, traction_function(t_info.time(), X, n, u, v, a, params...));
        });
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
    addTractionAllParams(domain_name, traction_function, std::make_index_sequence<Coupling::num_coupling_fields>{});
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
    solid_weak_form->template addBoundaryIntegral<0, 1, 2, 3, (4 + active_parameters)...>(
        DependsOn<0, 1, 2, 3, (4 + active_parameters)...>{}, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          auto u_current = captured_rule->value(t_info, u, u_old, v_old, a_old);

          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        });

    addCycleZeroBoundaryIntegralImpl(
        depends_on, domain_name, [=](auto t_info, auto X, auto u, auto /*v_old*/, auto /*a*/, auto... params) {
          auto u_current = u;

          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), get<VALUE>(params)...);

          return detail::scaleCycleZeroTerm(t_info, pressure * n_deformed * (1.0 / n_shape_norm));
        });
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
    addPressureAllParams(domain_name, pressure_function, std::make_index_sequence<Coupling::num_coupling_fields>{});
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

  // Cycle-zero helpers always include the 3 cycle-zero state slots, followed by the selected tail args.
  template <int... active_parameters, typename IntegrandType>
  void addCycleZeroBodySourceImpl(DependsOn<active_parameters...>, const std::string& name, IntegrandType f)
  {
    if (cycle_zero_solid_weak_form) {
      cycle_zero_solid_weak_form->template addBodySource<0, 1, 2, (3 + active_parameters)...>(
          DependsOn<0, 1, 2, (3 + active_parameters)...>{}, name, f);
    }
  }

  template <int... active_parameters, typename IntegrandType>
  void addCycleZeroBoundaryFluxImpl(DependsOn<active_parameters...>, const std::string& name, IntegrandType f)
  {
    if (cycle_zero_solid_weak_form) {
      cycle_zero_solid_weak_form->template addBoundaryFlux<0, 1, 2, (3 + active_parameters)...>(
          DependsOn<0, 1, 2, (3 + active_parameters)...>{}, name, f);
    }
  }

  template <int... active_parameters, typename IntegrandType>
  void addCycleZeroBoundaryIntegralImpl(DependsOn<active_parameters...>, const std::string& name, IntegrandType f)
  {
    if (cycle_zero_solid_weak_form) {
      cycle_zero_solid_weak_form->template addBoundaryIntegral<0, 1, 2, (3 + active_parameters)...>(
          DependsOn<0, 1, 2, (3 + active_parameters)...>{}, name, f);
    }
  }
};

/**
 * @brief Optional auxiliary systems and outputs for solid mechanics.
 */
struct SolidMechanicsOptions {
  bool enable_stress_output = false;  ///< Register stress output fields during phase 1.
  bool output_cauchy_stress = false;  ///< When true, project Cauchy stress (sigma) instead of PK1 (P).
};

/**
 * @brief Register all solid mechanics fields into a FieldStore.
 *
 * Phase 1 of the two-phase initialization.
 *
 * @return PhysicsFields carrying the exported field tokens and time rule type.
 */
template <int dim, int order, typename DisplacementTimeRule>
auto registerSolidMechanicsFields(std::shared_ptr<FieldStore> field_store,
                                  const SolidMechanicsOptions& options = SolidMechanicsOptions{})
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

  auto physics_fields =
      PhysicsFields<DisplacementTimeRule, H1<order, dim>, H1<order, dim>, H1<order, dim>, H1<order, dim>>{
          field_store, FieldType<H1<order, dim>>(field_store->prefix("displacement_solve_state")),
          FieldType<H1<order, dim>>(field_store->prefix("displacement")),
          FieldType<H1<order, dim>>(field_store->prefix("velocity")),
          FieldType<H1<order, dim>>(field_store->prefix("acceleration"))};

  if (options.enable_stress_output) {
    auto stress_time_rule = std::make_shared<StaticTimeIntegrationRule>();
    FieldType<L2<0, dim * dim>> stress_type("stress");
    field_store->addIndependent(stress_type, stress_time_rule);
  }

  return physics_fields;
}
/**
 * @brief Internal solid mechanics builder after coupling fields are assembled.
 *
 * Phase 2 of the two-phase initialization.
 */
namespace detail {

/// @brief Return true when stress output fields were registered during phase 1.
inline bool hasRegisteredStressOutput(const std::shared_ptr<FieldStore>& field_store)
{
  return field_store->hasField(field_store->prefix("stress"));
}

/// @brief Build a cycle-zero solver from the main solver when possible, else use fallback defaults.
inline std::shared_ptr<SystemSolver> makeCycleZeroSolver(std::shared_ptr<SystemSolver> solver, const Mesh& mesh)
{
  if (solver) {
    if (auto derived_solver = solver->singleBlockSolver(0)) {
      return derived_solver;
    }
  }

  NonlinearSolverOptions cycle_zero_nonlin{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 1e-14,
                                           .absolute_tol = 1e-14,
                                           .max_iterations = 2,
                                           .print_level = 0};
  LinearSolverOptions cycle_zero_lin{.linear_solver = LinearSolver::CG,
                                     .preconditioner = Preconditioner::HypreJacobi,
                                     .relative_tol = 1e-14,
                                     .absolute_tol = 1e-14,
                                     .max_iterations = 1000,
                                     .print_level = 0};
  return std::make_shared<SystemSolver>(buildNonlinearBlockSolver(cycle_zero_nonlin, cycle_zero_lin, mesh));
}

/**
 * @brief Internal solid builder after public registration and coupling collection.
 */
template <int dim, int order, typename DisplacementTimeRule, typename Coupling>
  requires detail::is_coupling_params_v<Coupling>
auto buildSolidMechanicsSystemImpl(std::shared_ptr<FieldStore> field_store, const Coupling& coupling,
                                   std::shared_ptr<SystemSolver> solver, const SolidMechanicsOptions& options,
                                   bool has_stress_output)
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
      [&](auto&... coupling_fields) {
        return std::make_shared<typename SystemType::SolidWeakFormType>(
            force_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
            field_store->createSpaces(force_name, disp_type.name, disp_type, disp_old_type, velo_old_type,
                                      accel_old_type, coupling_fields...));
      },
      coupling.fields);

  auto sys = std::make_shared<SystemType>(field_store, solver, std::vector<std::shared_ptr<WeakForm>>{solid_weak_form});
  sys->disp_bc = disp_bc;
  sys->disp_time_rule = disp_time_rule_ptr;
  sys->solid_weak_form = solid_weak_form;
  sys->output_cauchy_stress = options.output_cauchy_stress;

  if (disp_time_rule_ptr->requiresInitialAccelerationSolve()) {
    std::string cycle_zero_name = field_store->prefix("cycle_zero_acceleration_reaction");
    auto accel_as_unknown = accel_old_type;
    accel_as_unknown.is_unknown = true;
    FieldType<H1<order, dim>> disp_cz_input(disp_type.name);
    sys->cycle_zero_solid_weak_form = std::apply(
        [&](auto&... coupling_fields) {
          return std::make_shared<typename SystemType::CycleZeroSolidWeakFormType>(
              cycle_zero_name, field_store->getMesh(), field_store->getField(accel_old_type.name).get()->space(),
              field_store->createSpaces(cycle_zero_name, accel_old_type.name, disp_cz_input, velo_old_type,
                                        accel_as_unknown, coupling_fields...));
        },
        coupling.fields);
    field_store->markWeakFormInternal(cycle_zero_name);
    field_store->shareBoundaryConditions(accel_old_type.name, disp_bc);
    auto cycle_zero_solver = detail::makeCycleZeroSolver(solver, *field_store->getMesh());
    sys->cycle_zero_system = makeSystem(field_store, cycle_zero_solver, {sys->cycle_zero_solid_weak_form});
  }

  if (has_stress_output) {
    FieldType<L2<0, dim * dim>> stress_type(field_store->prefix("stress"), true);
    FieldType<H1<order, dim>> disp_as_input(disp_type.name);
    std::string stress_name = field_store->prefix("stress_projection");
    sys->stress_weak_form = std::apply(
        [&](auto&... coupling_fields) {
          return std::make_shared<typename SystemType::StressOutputWeakFormType>(
              stress_name, field_store->getMesh(), field_store->getField(stress_type.name).get()->space(),
              field_store->createSpaces(stress_name, stress_type.name, stress_type, disp_as_input, disp_old_type,
                                        velo_old_type, accel_old_type, coupling_fields...));
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
    sys->post_solve_systems.push_back(sys->stress_output_system);
  }

  return sys;
}

}  // namespace detail

/**
 * @brief Build a SolidMechanicsSystem from already-registered field packs.
 *
 * Explicit-rule API: rule is given as template param.
 * Additional parameter packs are registered as parameters. Coupling packs are taken from
 * the trailing field-pack arguments.
 *
 * Usage:
 * @code
 *   auto solid_system = buildSolidMechanicsSystem<dim, order, DispRule>(
 *       solver, opts, solid_fields, youngs_modulus, thermal_fields);
 * @endcode
 */
template <int dim, int order, typename DisplacementTimeRule, typename SelfFields, typename... OtherPacks>
  requires(detail::is_physics_fields_v<SelfFields> &&
           std::is_same_v<typename std::decay_t<SelfFields>::time_rule_type, DisplacementTimeRule> &&
           (detail::is_coupling_params_v<OtherPacks> && ...))
auto buildSolidMechanicsSystem(std::shared_ptr<SystemSolver> solver, const SolidMechanicsOptions& options,
                               const SelfFields& self_fields, const OtherPacks&... other_packs)
{
  auto field_store = self_fields.field_store;
  (detail::registerParamsIfNeeded(field_store, other_packs), ...);
  auto coupling = detail::collectCouplingFields<DisplacementTimeRule>(field_store, self_fields, other_packs...);
  bool has_stress_output = detail::hasRegisteredStressOutput(field_store);
  return detail::buildSolidMechanicsSystemImpl<dim, order, DisplacementTimeRule>(field_store, coupling, solver, options,
                                                                                 has_stress_output);
}

/**
 * @brief Build a SolidMechanicsSystem from already-registered field packs.
 *
 * Preferred API: deduce rule from `self_fields`.
 *
 * Usage:
 * @code
 *   auto solid_system = buildSolidMechanicsSystem<dim, order>(
 *       solver, opts, solid_fields, youngs_modulus, thermal_fields);
 * @endcode
 */
template <int dim, int order, typename SelfFields, typename... OtherPacks>
  requires(detail::has_time_rule_v<SelfFields> && (detail::is_coupling_params_v<OtherPacks> && ...))
auto buildSolidMechanicsSystem(std::shared_ptr<SystemSolver> solver, const SolidMechanicsOptions& options,
                               const SelfFields& self_fields, const OtherPacks&... other_packs)
{
  using DisplacementTimeRule = typename std::decay_t<SelfFields>::time_rule_type;
  return buildSolidMechanicsSystem<dim, order, DisplacementTimeRule>(solver, options, self_fields, other_packs...);
}

/**
 * @brief Build a SolidMechanicsSystem from solver options and a FieldStore.
 *
 * Registers the solid field pack, builds a nonlinear block solver from the supplied options,
 * then forwards to the existing field-pack overload.
 *
 * Usage:
 * @code
 *   auto solid_system = buildSolidMechanicsSystem<dim, order, DispRule>(
 *       nonlin_opts, lin_opts, field_store, opts, param_fields, thermal_fields);
 * @endcode
 */
template <int dim, int order, typename DisplacementTimeRule, typename... OtherPacks>
  requires(detail::is_coupling_params_v<OtherPacks> && ...)
auto buildSolidMechanicsSystem(const NonlinearSolverOptions& nonlinear_options,
                               const LinearSolverOptions& linear_options, const SolidMechanicsOptions& options,
                               std::shared_ptr<FieldStore> field_store, const OtherPacks&... other_packs)
{
  auto self_fields = registerSolidMechanicsFields<dim, order, DisplacementTimeRule>(field_store, options);
  auto solver = std::make_shared<SystemSolver>(
      buildNonlinearBlockSolver(nonlinear_options, linear_options, *field_store->getMesh()));
  return buildSolidMechanicsSystem<dim, order, DisplacementTimeRule>(solver, options, self_fields, other_packs...);
}

}  // namespace smith
