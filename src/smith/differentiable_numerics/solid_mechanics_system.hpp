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

#include <algorithm>

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
 * @tparam Coupling Tuple of coupling and parameter packs (default: none).
 *         Coupling fields occupy leading positions in the tail after the 4 time-rule state fields,
 *         before user parameter_space fields.
 * @tparam parameter_space Parameter spaces for material properties.
 */
template <int dim, int order, typename DisplacementTimeRule = ImplicitNewmarkSecondOrderTimeIntegrationRule,
          typename Coupling = std::tuple<>>
struct SolidMechanicsSystem : public SystemBase {
  using SystemBase::SystemBase;

  static_assert(DisplacementTimeRule::num_states == 4, "SolidMechanicsSystem requires a 4-state time integration rule");

  /// Main weak form: (u, u_old, v_old, a_old, coupling_fields..., params...)
  using SolidWeakFormType =
      FunctionalWeakForm<dim, H1<order, dim>, detail::TimeRuleParams<DisplacementTimeRule, H1<order, dim>, Coupling>>;

  /// L2 projection weak form for PK1 stress output (dim*dim components).
  /// Args: (stress_unknown, u, u_old, v_old, a_old, coupling_fields..., params...).
  using StressOutputWeakFormType = FunctionalWeakForm<
      dim, L2<0, dim * dim>,
      typename detail::AppendCouplingToParams<Coupling, Parameters<L2<0, dim * dim>, H1<order, dim>, H1<order, dim>,
                                                                   H1<order, dim>, H1<order, dim>>>::type>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;    ///< Solid mechanics weak form.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;  ///< Displacement boundary conditions.
  std::shared_ptr<DisplacementTimeRule> disp_time_rule;  ///< Time integration rule.
  std::shared_ptr<const Coupling> coupling;              ///< Coupling metadata for callback interpolation.

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
    auto captured_coupling = coupling;
    solid_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto... raw_args) {
      return detail::applyTimeRuleAndCoupling(
          *captured_rule, *captured_coupling, t_info,
          [&](auto u_current, auto v_current, auto a_current, auto... interpolated_params) {
            typename MaterialType::State state;
            auto pk_stress =
                material(t_info, state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current), interpolated_params...);
            return smith::tuple{get<VALUE>(a_current) * material.density, pk_stress};
          },
          raw_args...);
    });

    // Stress output projection: L2 projection of PK1 stress onto an L2 piecewise-constant field.
    // Residual: ∫ test · (stress_unknown - pk_stress(u)) dx = 0.
    // Args: (stress_unknown, u, u_old, v_old, a_old, params...). stress_unknown is the Jacobian
    // variable so the solver builds the mass matrix against it, and the (- pk_stress) term
    // becomes the RHS.
    if (stress_weak_form) {
      bool do_cauchy = this->output_cauchy_stress;
      stress_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto stress, auto... raw_args) {
        return detail::applyTimeRuleAndCoupling(
            *captured_rule, *captured_coupling, t_info,
            [&](auto u_current, auto v_current, auto /*a_current*/, auto... interpolated_params) {
              typename MaterialType::State state;
              auto pk_stress = material(t_info, state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current),
                                        interpolated_params...);

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
            },
            raw_args...);
      });
    }
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
    auto captured_rule = disp_time_rule;
    auto captured_coupling = coupling;
    solid_weak_form->addBodySource(domain_name, [=](auto t_info, auto X, auto... raw_args) {
      return detail::applyTimeRuleAndCoupling(
          *captured_rule, *captured_coupling, t_info,
          [&](auto u_current, auto v_current, auto a_current, auto... interpolated_params) {
            return force_function(t_info.time(), X, u_current, v_current, a_current, interpolated_params...);
          },
          raw_args...);
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
    auto captured_rule = disp_time_rule;
    auto captured_coupling = coupling;
    solid_weak_form->addBoundaryFlux(domain_name, [=](auto t_info, auto X, auto n, auto... raw_args) {
      return detail::applyTimeRuleAndCoupling(
          *captured_rule, *captured_coupling, t_info,
          [&](auto u_current, auto v_current, auto a_current, auto... interpolated_params) {
            return traction_function(t_info.time(), X, n, u_current, v_current, a_current, interpolated_params...);
          },
          raw_args...);
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
    auto captured_rule = disp_time_rule;
    auto captured_coupling = coupling;
    solid_weak_form->addBoundaryIntegral(domain_name, [=](auto t_info, auto X, auto... raw_args) {
      return detail::applyTimeRuleAndCoupling(
          *captured_rule, *captured_coupling, t_info,
          [&](auto u_current, auto /*v_current*/, auto /*a_current*/, auto... interpolated_params) {
            auto x_current = X + u_current;
            auto n_deformed = cross(get<DERIVATIVE>(x_current));
            auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

            auto pressure = pressure_function(t_info.time(), get<VALUE>(X), get<VALUE>(interpolated_params)...);

            return pressure * n_deformed * (1.0 / n_shape_norm);
          },
          raw_args...);
    });
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
      PhysicsFields<dim, order, DisplacementTimeRule, H1<order, dim>, H1<order, dim>, H1<order, dim>, H1<order, dim>>{
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
  requires detail::is_coupling_packs_v<Coupling>
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

  auto coupling_fields_flat = detail::flattenCouplingFields(coupling);
  std::string force_name = field_store->prefix("reactions");
  auto solid_weak_form = detail::buildWeakFormWithCoupling<typename SystemType::SolidWeakFormType>(
      field_store, force_name, disp_type.name, disp_type, disp_old_type, velo_old_type, accel_old_type,
      coupling_fields_flat);

  auto sys = std::make_shared<SystemType>(field_store, solver, std::vector<std::shared_ptr<WeakForm>>{solid_weak_form});
  sys->disp_bc = disp_bc;
  sys->disp_time_rule = disp_time_rule_ptr;
  sys->coupling = std::make_shared<Coupling>(coupling);
  sys->solid_weak_form = solid_weak_form;
  sys->output_cauchy_stress = options.output_cauchy_stress;

  if (disp_time_rule_ptr->requiresInitialAccelerationSolve()) {
    std::string cycle_zero_name = field_store->prefix("cycle_zero_acceleration_reaction");
    field_store->markWeakFormInternal(cycle_zero_name);
    auto cycle_zero_solver = detail::makeCycleZeroSolver(solver, *field_store->getMesh());

    auto cycle_zero_system = makeSystem(field_store, cycle_zero_solver, {sys->solid_weak_form});
    cycle_zero_system->solve_result_field_names = {accel_old_type.name};
    auto cycle_zero_inputs =
        std::vector<std::string>{disp_old_type.name, disp_old_type.name, velo_old_type.name, accel_old_type.name};
    auto append_if_state = [&](const auto& field) {
      const auto& states = field_store->getStateFields();
      if (std::any_of(states.begin(), states.end(),
                      [&](const auto& state) { return state.get()->name() == field.name; })) {
        cycle_zero_inputs.push_back(field.name);
      }
    };
    std::apply([&](const auto&... coupling_fields) { (append_if_state(coupling_fields), ...); }, coupling_fields_flat);
    cycle_zero_system->solve_input_field_names = {cycle_zero_inputs};
    sys->cycle_zero_systems.push_back(cycle_zero_system);
  }

  if (has_stress_output) {
    FieldType<L2<0, dim * dim>> stress_type(field_store->prefix("stress"), true);
    FieldType<H1<order, dim>> disp_as_input(disp_type.name);
    std::string stress_name = field_store->prefix("stress_projection");
    sys->stress_weak_form = detail::buildWeakFormWithCoupling<typename SystemType::StressOutputWeakFormType>(
        field_store, stress_name, stress_type.name, stress_type, disp_as_input, disp_old_type, velo_old_type,
        accel_old_type, coupling_fields_flat);

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
 * Deduces the displacement time rule from `self_fields`.
 *
 * Usage:
 * @code
 *   auto solid_system = buildSolidMechanicsSystem<dim, order>(
 *       solver, opts, solid_fields, couplingFields(thermal_fields), param_fields);
 * @endcode
 */
template <typename SelfFields>
  requires(detail::has_time_rule_v<SelfFields>)
auto buildSolidMechanicsSystem(std::shared_ptr<SystemSolver> solver, const SolidMechanicsOptions& options,
                               const SelfFields& self_fields)
{
  constexpr int dim = SelfFields::dim;
  constexpr int order = SelfFields::order;
  using DisplacementTimeRule = typename std::decay_t<SelfFields>::time_rule_type;
  auto field_store = self_fields.field_store;
  auto coupling = detail::collectCouplingFields();
  bool has_stress_output = detail::hasRegisteredStressOutput(field_store);
  return detail::buildSolidMechanicsSystemImpl<dim, order, DisplacementTimeRule>(field_store, coupling, solver, options,
                                                                                 has_stress_output);
}

/**
 * @brief Build a SolidMechanicsSystem from registered self fields plus coupled physics fields.
 */
template <typename SelfFields, typename... PFs>
  requires(detail::has_time_rule_v<SelfFields>)
auto buildSolidMechanicsSystem(std::shared_ptr<SystemSolver> solver, const SolidMechanicsOptions& options,
                               const SelfFields& self_fields, const CouplingFields<PFs...>& coupled)
{
  constexpr int dim = SelfFields::dim;
  constexpr int order = SelfFields::order;
  using DisplacementTimeRule = typename std::decay_t<SelfFields>::time_rule_type;
  auto field_store = self_fields.field_store;
  auto coupling = detail::collectCouplingFields(coupled);
  bool has_stress_output = detail::hasRegisteredStressOutput(field_store);
  return detail::buildSolidMechanicsSystemImpl<dim, order, DisplacementTimeRule>(field_store, coupling, solver, options,
                                                                                 has_stress_output);
}

/**
 * @brief Build a SolidMechanicsSystem from registered self fields plus registered parameter fields.
 */
template <typename SelfFields, typename... ParamSpaces>
  requires(detail::has_time_rule_v<SelfFields>)
auto buildSolidMechanicsSystem(std::shared_ptr<SystemSolver> solver, const SolidMechanicsOptions& options,
                               const SelfFields& self_fields, const ParamFields<ParamSpaces...>& params)
{
  constexpr int dim = SelfFields::dim;
  constexpr int order = SelfFields::order;
  using DisplacementTimeRule = typename std::decay_t<SelfFields>::time_rule_type;
  auto field_store = self_fields.field_store;
  auto coupling = detail::collectCouplingFields(params);
  bool has_stress_output = detail::hasRegisteredStressOutput(field_store);
  return detail::buildSolidMechanicsSystemImpl<dim, order, DisplacementTimeRule>(field_store, coupling, solver, options,
                                                                                 has_stress_output);
}

/**
 * @brief Build a SolidMechanicsSystem from registered self fields, coupled physics fields, and parameter fields.
 */
template <typename SelfFields, typename... PFs, typename... ParamSpaces>
  requires(detail::has_time_rule_v<SelfFields>)
auto buildSolidMechanicsSystem(std::shared_ptr<SystemSolver> solver, const SolidMechanicsOptions& options,
                               const SelfFields& self_fields, const CouplingFields<PFs...>& coupled,
                               const ParamFields<ParamSpaces...>& params)
{
  constexpr int dim = SelfFields::dim;
  constexpr int order = SelfFields::order;
  using DisplacementTimeRule = typename std::decay_t<SelfFields>::time_rule_type;
  auto field_store = self_fields.field_store;
  auto coupling = detail::collectCouplingFields(coupled, params);
  bool has_stress_output = detail::hasRegisteredStressOutput(field_store);
  return detail::buildSolidMechanicsSystemImpl<dim, order, DisplacementTimeRule>(field_store, coupling, solver, options,
                                                                                 has_stress_output);
}

}  // namespace smith
