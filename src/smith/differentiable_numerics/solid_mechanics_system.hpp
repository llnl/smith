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
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/physics/weak_form.hpp"

namespace smith {

/**
 * @brief System struct for solid mechanics with second-order time integration.
 * @tparam dim Spatial dimension.
 * @tparam order Polynomial order for displacement field.
 * @tparam parameter_space Parameter spaces for material properties.
 */
template <int dim, int order, typename... parameter_space>
struct SolidMechanicsSystem {
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<order, dim>, Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>, H1<order, dim>, parameter_space...>>;

  using CycleZeroWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<order, dim>, Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>, parameter_space...>>;

  std::shared_ptr<FieldStore> field_store;                 ///< Field store managing the system's fields.
  std::shared_ptr<SolidWeakFormType> solid_weak_form;      ///< Solid mechanics weak form.
  std::shared_ptr<CycleZeroWeakFormType> cycle_zero_weak_form;  ///< Cycle-zero weak form for initial acceleration solve.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;    ///< Displacement boundary conditions.
  std::shared_ptr<DifferentiableBlockSolver> solver;       ///< The solver for the system.
  std::shared_ptr<StateAdvancer> advancer;                 ///< The state advancer.
  std::shared_ptr<ImplicitNewmarkSecondOrderTimeIntegrationRule> time_rule;  ///< Time integration rule.
  std::vector<FieldState> parameter_fields;                                  ///< Optional parameter fields.

  /**
   * @brief Get the list of all state fields (displacement, displacement_old, velocity_old, acceleration_old).
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    return {field_store->getField("displacement"), field_store->getField("displacement_old"),
            field_store->getField("velocity_old"), field_store->getField("acceleration_old")};
  }

  /**
   * @brief Get the list of all parameter fields.
   * @return const std::vector<FieldState>& List of parameter fields.
   */
  const std::vector<FieldState>& getParameterFields() const { return parameter_fields; }

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
        domain_name,
        [=](auto t_info, auto /*X*/, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          // Apply time integration rule to compute current state from history
          auto u_current = captured_rule->value(t_info, u, u_old, v_old, a_old);
          auto a_current = captured_rule->ddot(t_info, u, u_old, v_old, a_old);

          typename MaterialType::State state;
          auto pk_stress = material(state, get<DERIVATIVE>(u_current), params...);

          return smith::tuple{get<VALUE>(a_current) * material.density, pk_stress};
        });

    // Add to cycle-zero weak form (inputs: u, v, a, params...)
    // At cycle 0, we directly use u, v, a (no time integration needed)
    cycle_zero_weak_form->addBodyIntegral(
        domain_name,
        [=](auto /*t_info*/, auto /*X*/, auto u, auto /*v*/, auto a, auto... params) {
          typename MaterialType::State state;
          auto pk_stress = material(state, get<DERIVATIVE>(u), params...);

          return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
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
      auto captured_rule = time_rule;
      solid_weak_form->addBodyIntegral(domain_name,
          [=](auto t_info, auto X, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          // Apply time integration rule to get current state
          auto u_current = captured_rule->value(t_info, u, u_old, v_old, a_old);
          auto v_current = captured_rule->dot(t_info, u, u_old, v_old, a_old);
          auto a_current = captured_rule->ddot(t_info, u, u_old, v_old, a_old);

          return smith::tuple{-force_function(t_info.time(), get<VALUE>(X), get<VALUE>(u_current),
                                              get<VALUE>(v_current), get<VALUE>(a_current),
                                              get<VALUE>(params)...), smith::zero{}};
      });
  }

  /**
   * @brief Add a surface flux (traction) to the system.
   * @tparam SurfaceFluxType The surface flux function type.
   * @param domain_name The name of the boundary domain to apply the flux to.
   * @param flux_function The flux function (t, X, n, u, v, a, params...).
   */
  template <typename SurfaceFluxType>
  void addSurfaceFlux(const std::string& domain_name, SurfaceFluxType flux_function)
  {
      auto captured_rule = time_rule;
      solid_weak_form->addSurfaceFlux(domain_name,
          [=](auto t_info, auto X, auto n, auto u, auto u_old, auto v_old, auto a_old, auto... params) {
          // Apply time integration rule to get current state
          auto u_current = captured_rule->value(t_info, u, u_old, v_old, a_old);
          auto v_current = captured_rule->dot(t_info, u, u_old, v_old, a_old);
          auto a_current = captured_rule->ddot(t_info, u, u_old, v_old, a_old);

          return -flux_function(t_info.time(), get<VALUE>(X), n, get<VALUE>(u_current), get<VALUE>(v_current),
                                get<VALUE>(a_current), get<VALUE>(params)...);
      });
  }
};

/**
 * @brief Factory function to build a solid mechanics system with second-order time integration.
 * @tparam dim Spatial dimension.
 * @tparam order Polynomial order for displacement field.
 * @tparam parameter_space Parameter spaces for material properties.
 * @param mesh The mesh.
 * @param solver The differentiable block solver.
 * @param time_rule The time integration rule.
 * @param parameter_types Parameter field types.
 * @return SolidMechanicsSystem with all components initialized.
 */
template <int dim, int order, typename... parameter_space>
SolidMechanicsSystem<dim, order, parameter_space...> buildSolidMechanicsSystem(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<DifferentiableBlockSolver> solver,
    ImplicitNewmarkSecondOrderTimeIntegrationRule time_rule, FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<FieldStore>(mesh, 100);

  // Add shape displacement
  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  // Add displacement as independent (unknown) with time integration rule
  auto time_rule_ptr = std::make_shared<ImplicitNewmarkSecondOrderTimeIntegrationRule>(time_rule);
  FieldType<H1<order, dim>> disp_type("displacement");
  auto disp_bc = field_store->addIndependent(disp_type, time_rule_ptr);

  // Add dependent fields for time integration history
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE, "displacement_old");
  auto velo_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DOT, "velocity_old");
  auto accel_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::DDOT, "acceleration_old");

  // Add parameters
  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(parameter_types), ...);
  (parameter_fields.push_back(field_store->getField(parameter_types.name)), ...);

  // Create solid mechanics weak form (u, u_old, v_old, a_old)
  field_store->addWeakFormTestField("solid_force", disp_type.name);
  const mfem::ParFiniteElementSpace& test_space = field_store->getField(disp_type.name).get()->space();
  std::vector<const mfem::ParFiniteElementSpace*> input_spaces;
  createSpaces("solid_force", *field_store, input_spaces, 0, disp_type, disp_old_type, velo_old_type, accel_old_type,
               parameter_types...);

  auto solid_weak_form = std::make_shared<typename SolidMechanicsSystem<dim, order, parameter_space...>::SolidWeakFormType>(
      "solid_force", field_store->getMesh(), test_space, input_spaces);

  // Create cycle-zero weak form (u, v, a) for initial acceleration solve
  // The test field is acceleration, which is what we solve for at cycle=0
  auto cycle_zero_weak_form = createWeakForm<dim>("solid_reaction", accel_old_type, *field_store, disp_type,
                                                  velo_old_type, accel_old_type, parameter_types...);

  // Build advancer using SolidMechanicsStateAdvancer which wraps MultiphysicsTimeIntegrator
  // and handles the initial acceleration solve at cycle=0
  auto advancer = std::make_shared<SolidMechanicsStateAdvancer>(field_store, solid_weak_form, cycle_zero_weak_form,
                                                                 solver);

  return SolidMechanicsSystem<dim, order, parameter_space...>{field_store,
                                                               solid_weak_form,
                                                               cycle_zero_weak_form,
                                                               disp_bc,
                                                               solver,
                                                               advancer,
                                                               time_rule_ptr,
                                                               parameter_fields};
}

}  // namespace smith
