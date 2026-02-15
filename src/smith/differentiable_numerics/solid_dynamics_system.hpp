// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_dynamics_system.hpp
 * @brief Defines the SolidDynamicsSystem struct and its factory function
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/solid_mechanics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/system_base.hpp"

namespace smith {

/**
 * @brief System struct for solid dynamics with second-order time integration.
 * @tparam dim Spatial dimension.
 * @tparam order Polynomial order for displacement field.
 * @tparam parameter_space Parameter spaces for material properties.
 */
template <int dim, int order, typename... parameter_space>
struct SolidDynamicsSystem : public SystemBase {
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<order, dim>,
      Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>, H1<order, dim>, parameter_space...>>;

  using CycleZeroWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<order, dim>,
      Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>, H1<order, dim>, parameter_space...>>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;  ///< Solid mechanics weak form.
  std::shared_ptr<CycleZeroWeakFormType>
      cycle_zero_weak_form;                              ///< Cycle-zero weak form for initial acceleration solve.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;  ///< Displacement boundary conditions.
  std::shared_ptr<ImplicitNewmarkSecondOrderTimeIntegrationRule> time_rule;  ///< Time integration rule.

  /**
   * @brief Get the list of all state fields (displacement, displacement_old, velocity_old, acceleration_old).
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    return {field_store->getField(prefix("displacement_predicted")), field_store->getField(prefix("displacement")),
            field_store->getField(prefix("velocity")), field_store->getField(prefix("acceleration"))};
  }

  /**
   * @brief Create a DifferentiablePhysics object for this system.
   * @param physics_name The name of the physics.
   * @return std::shared_ptr<DifferentiablePhysics> The differentiable physics object.
   */
  std::shared_ptr<DifferentiablePhysics> createDifferentiablePhysics(std::string physics_name)
  {
    return std::make_shared<DifferentiablePhysics>(
        field_store->getMesh(), field_store->graph(), field_store->getShapeDisp(), getStateFields(),
        getParameterFields(), advancer, physics_name, std::vector<std::string>{prefix("reactions")});
  }

  /**
   * @brief Set the material model for a domain, defining integrals for the solid weak form.
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    // Add to solid weak form (inputs: u, u_old, v_old, a_old, params...)
    // Manually apply time integration rule to compute current state
    auto captured_rule = time_rule;
    solid_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          // Apply time integration rule to compute current state from history
          auto u_current = captured_rule->value(t_info, u, u_old, v_old, a_old);
          auto a_current = captured_rule->ddot(t_info, u, u_old, v_old, a_old);

          typename MaterialType::State state;
          auto pk_stress = material(state, get<DERIVATIVE>(u_current), params...);

          return smith::tuple{get<VALUE>(a_current) * material.density, pk_stress};
        });

    // Add to cycle-zero weak form (inputs: u, u_old, v_old, a, params...)
    // At cycle 0, we directly use u, v, a (no time integration needed)
    cycle_zero_weak_form->addBodyIntegral(
        domain_name, [=](auto /*t_info*/, auto /*X*/, auto u, auto /*u_old*/, auto /*v_old*/, auto a, auto... params) {
          typename MaterialType::State state;
          auto pk_stress = material(state, get<DERIVATIVE>(u), params...);

          return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
        });
  }

  /**
   * @brief Add a body force to the system (with DependsOn).
   * @tparam active_parameters Indices of fields this force depends on.
   * @tparam BodyForceType The body force function type.
   * @param depends_on Dependency specification for which input fields to pass.
   * @param domain_name The name of the domain to apply the force to.
   * @param force_function The force function (t, X, selected time-integrated inputs...).
   * @note Time integration is applied to the state fields before calling the user function.
   */
  template <int... active_parameters, typename BodyForceType>
  void addBodyForce(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                    BodyForceType force_function)
  {
    auto captured_rule = time_rule;
    solid_weak_form->addBodySource(
        depends_on, domain_name, [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          // Apply time integration rule to get current state
          auto u_current = captured_rule->value(t_info, u, u_old, v_old, a_old);
          auto v_current = captured_rule->dot(t_info, u, u_old, v_old, a_old);
          auto a_current = captured_rule->ddot(t_info, u, u_old, v_old, a_old);

          return force_function(t_info.time(), X, u_current, v_current, a_current, params...);
        });

    cycle_zero_weak_form->addBodySource(
        depends_on, domain_name, [=](auto t_info, auto X, auto u, auto /*u_old*/, auto v_old, auto a, auto... params) {
          return force_function(t_info.time(), X, u, v_old, a, params...);
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
    addBodyForceAllParams(domain_name, force_function, std::make_index_sequence<4 + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a surface traction (flux) to the system (with DependsOn).
   * @tparam active_parameters Indices of fields this traction depends on.
   * @tparam TractionType The traction function type.
   * @param depends_on Dependency specification for which input fields to pass.
   * @param domain_name The name of the boundary domain to apply the traction to.
   * @param traction_function The traction function (t, X, n, selected time-integrated inputs...).
   * @note Time integration is applied to the state fields before calling the user function.
   */
  template <int... active_parameters, typename TractionType>
  void addTraction(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   TractionType traction_function)
  {
    auto captured_rule = time_rule;
    solid_weak_form->addBoundaryFlux(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          // Apply time integration rule to get current state
          auto u_current = captured_rule->value(t_info, u, u_old, v_old, a_old);
          auto v_current = captured_rule->dot(t_info, u, u_old, v_old, a_old);
          auto a_current = captured_rule->ddot(t_info, u, u_old, v_old, a_old);

          return traction_function(t_info.time(), X, n, u_current, v_current, a_current, params...);
        });

    cycle_zero_weak_form->addBoundaryFlux(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto /*u_old*/, auto v_old, auto a, auto... params) {
          return traction_function(t_info.time(), X, n, u, v_old, a, params...);
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
    addTractionAllParams(domain_name, traction_function, std::make_index_sequence<4 + sizeof...(parameter_space)>{});
  }

  /**
   * @brief Add a pressure boundary condition (follower force) (with DependsOn).
   * @tparam active_parameters Indices of fields this pressure depends on.
   * @tparam PressureType The pressure function type.
   * @param depends_on Dependency specification for which input fields to pass.
   * @param domain_name The name of the boundary domain.
   * @param pressure_function The pressure function (t, X, selected time-integrated inputs...).
   * @note Pressure is applied in the current configuration: P * n_deformed.
   * @note Time integration is applied to the state fields before calling the user function.
   */
  template <int... active_parameters, typename PressureType>
  void addPressure(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   PressureType pressure_function)
  {
    auto captured_rule = time_rule;
    solid_weak_form->addBoundaryIntegral(
        depends_on, domain_name, [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          // Apply time integration rule to get current state
          auto u_current = captured_rule->value(t_info, u, u_old, v_old, a_old);

          // Compute deformed normal and apply correction for reference configuration integration
          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        });

    cycle_zero_weak_form->addBoundaryIntegral(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto u, auto /*u_old*/, auto /*v_old*/, auto /*a*/, auto... params) {
          // At cycle 0, u is the current displacement
          auto u_current = u;

          // Compute deformed normal and apply correction for reference configuration integration
          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        });
  }

  /**
   * @brief Add a pressure boundary condition (follower force).
   * @tparam PressureType The pressure function type.
   * @param domain_name The name of the boundary domain.
   * @param pressure_function The pressure function (t, X, params...).
   * @note Pressure is applied in the current configuration: P * n_deformed.
   */
  template <typename PressureType>
  void addPressure(const std::string& domain_name, PressureType pressure_function)
  {
    addPressureAllParams(domain_name, pressure_function, std::make_index_sequence<4 + sizeof...(parameter_space)>{});
  }

 private:
  // Helper functions to forward non-DependsOn calls to DependsOn versions with all parameters
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
};

/**
 * @brief Factory function to build a solid dynamics system with second-order time integration.
 * @tparam dim Spatial dimension.
 * @tparam order Polynomial order for displacement field.
 * @tparam parameter_space Parameter spaces for material properties.
 * @param mesh The mesh.
 * @param solver The differentiable block solver.
 * @param time_rule The time integration rule.
 * @param prepend_name The name of the physics (used as field prefix).
 * @param parameter_types Parameter field types.
 * @return SolidDynamicsSystem with all components initialized.
 */
template <int dim, int order, typename... parameter_space>
SolidDynamicsSystem<dim, order, parameter_space...> buildSolidDynamicsSystem(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<DifferentiableBlockSolver> solver,
    ImplicitNewmarkSecondOrderTimeIntegrationRule time_rule, std::string prepend_name = "",
    FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<FieldStore>(mesh, 100);

  auto prefix = [&](const std::string& name) {
    if (prepend_name.empty()) {
      return name;
    }
    return prepend_name + "_" + name;
  };

  // Add shape displacement
  FieldType<H1<1, dim>> shape_disp_type(prefix("shape_displacement"));
  field_store->addShapeDisp(shape_disp_type);

  // Add displacement as independent (unknown) with time integration rule
  auto time_rule_ptr = std::make_shared<ImplicitNewmarkSecondOrderTimeIntegrationRule>(time_rule);
  FieldType<H1<order, dim>> disp_type(prefix("displacement_predicted"));
  auto disp_bc = field_store->addIndependent(disp_type, time_rule_ptr);

  // Add dependent fields for time integration history
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE, prefix("displacement"));
  auto velo_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DOT, prefix("velocity"));
  auto accel_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DDOT, prefix("acceleration"));

  // Add parameters
  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  // Create solid mechanics weak form (u, u_old, v_old, a_old)
  std::string force_name = prefix("solid_force");
  field_store->addWeakFormTestField(force_name, disp_type.name);
  const mfem::ParFiniteElementSpace& test_space = field_store->getField(disp_type.name).get()->space();
  std::vector<const mfem::ParFiniteElementSpace*> input_spaces;
  createSpaces(force_name, *field_store, input_spaces, 0, disp_type, disp_old_type, velo_old_type, accel_old_type,
               FieldType<parameter_space>(prefix("param_" + parameter_types.name))...);

  auto solid_weak_form =
      std::make_shared<typename SolidDynamicsSystem<dim, order, parameter_space...>::SolidWeakFormType>(
          force_name, field_store->getMesh(), test_space, input_spaces);

  // Create cycle-zero weak form (u, u_old, v_old, a) for initial acceleration solve
  // Note: We solve R(u_0, v_0, a_0) = 0 for a_0.
  // We include history terms to keep trial space indices consistent with solid_weak_form.
  std::string cycle_zero_name = prefix("solid_reaction");
  field_store->addWeakFormTestField(cycle_zero_name, accel_old_type.name);
  const mfem::ParFiniteElementSpace& cycle_zero_test_space = field_store->getField(accel_old_type.name).get()->space();
  std::vector<const mfem::ParFiniteElementSpace*> cycle_zero_input_spaces;
  createSpaces(cycle_zero_name, *field_store, cycle_zero_input_spaces, 0, disp_type, disp_old_type, velo_old_type,
               accel_old_type, FieldType<parameter_space>(prefix("param_" + parameter_types.name))...);

  auto cycle_zero_weak_form =
      std::make_shared<typename SolidDynamicsSystem<dim, order, parameter_space...>::CycleZeroWeakFormType>(
          cycle_zero_name, field_store->getMesh(), cycle_zero_test_space, cycle_zero_input_spaces);

  // Build advancer using SolidMechanicsTimeIntegrator which wraps MultiphysicsTimeIntegrator
  // and handles the initial acceleration solve at cycle=0
  auto advancer =
      std::make_shared<SolidMechanicsTimeIntegrator>(field_store, solid_weak_form, cycle_zero_weak_form, solver);

  return SolidDynamicsSystem<dim, order, parameter_space...>{
      {field_store, solver, advancer, parameter_fields, prepend_name},
      solid_weak_form,
      cycle_zero_weak_form,
      disp_bc,
      time_rule_ptr};
}

/**
 * @brief Factory function to build a solid dynamics system (without physics name).
 * @tparam dim Spatial dimension.
 * @tparam order Polynomial order for displacement field.
 * @tparam parameter_space Parameter spaces for material properties.
 * @param mesh The mesh.
 * @param solver The differentiable block solver.
 * @param time_rule The time integration rule.
 * @param parameter_types Parameter field types.
 * @return SolidDynamicsSystem with all components initialized.
 */
template <int dim, int order, typename... parameter_space>
SolidDynamicsSystem<dim, order, parameter_space...> buildSolidDynamicsSystem(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<DifferentiableBlockSolver> solver,
    ImplicitNewmarkSecondOrderTimeIntegrationRule time_rule, FieldType<parameter_space>... parameter_types)
{
  return buildSolidDynamicsSystem<dim, order, parameter_space...>(mesh, solver, time_rule, "", parameter_types...);
}

}  // namespace smith
