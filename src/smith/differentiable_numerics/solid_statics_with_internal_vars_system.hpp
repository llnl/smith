// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_statics_with_L2_state_system.hpp
 * @brief Defines the SolidStaticsWithL2StateSystem struct and its factory function
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
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

namespace smith {

/**
 * @brief System struct for solid statics with an additional L2 state variable.
 * @tparam dim Spatial dimension.
 * @tparam disp_order Polynomial order for displacement field.
 * @tparam StateSpace Finite element space for the state variable (e.g., L2<order>).
 * @tparam parameter_space Parameter spaces for material properties.
 */
template <int dim, int disp_order, typename StateSpace, typename... parameter_space>
struct SolidStaticsWithL2StateSystem : public SystemBase {
  // Primary weak form: residual for displacement (u).
  // Inputs: u, u_old, alpha, alpha_old, params...
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      Parameters<H1<disp_order, dim>, H1<disp_order, dim>, StateSpace, StateSpace, parameter_space...>>;

  // State weak form: residual for state variable (alpha).
  // Inputs: alpha, alpha_old, u, u_old, params...
  using StateWeakFormType = TimeDiscretizedWeakForm<
      dim, StateSpace,
      Parameters<StateSpace, StateSpace, H1<disp_order, dim>, H1<disp_order, dim>, parameter_space...>>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;                           ///< Solid mechanics weak form.
  std::shared_ptr<StateWeakFormType> state_weak_form;                           ///< State variable weak form.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;                         ///< Displacement boundary conditions.
  std::shared_ptr<DirichletBoundaryConditions> state_bc;                        ///< State variable boundary conditions.
  std::shared_ptr<QuasiStaticFirstOrderTimeIntegrationRule> disp_time_rule;     ///< Time integration for displacement.
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule> state_time_rule;  ///< Time integration for state.

  /**
   * @brief Get the list of all state fields.
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    return {field_store->getField(prefix("displacement_predicted")), field_store->getField(prefix("displacement")),
            field_store->getField(prefix("state_predicted")), field_store->getField(prefix("state"))};
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
        getParameterFields(), advancer, physics_name,
        std::vector<std::string>{prefix("solid_residual"), prefix("state_residual")});
  }

  /**
   * @brief Set the material model for the solid mechanics part.
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   * @note The material model should accept (state, deform_grad, alpha, params...).
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto alpha, auto alpha_old, auto... params) {
          // Apply time integration
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);

          typename MaterialType::State state;
          // Material model typically needs deformation gradient and internal state variable
          auto pk_stress = material(state, get<DERIVATIVE>(u_current), get<VALUE>(alpha_current), params...);

          // Return {source, flux}
          // Source is body force (zero here, internal forces only)
          // Flux is stress
          tensor<double, dim> source{};
          return smith::tuple{source, pk_stress};
        });
  }

  /**
   * @brief Add the evolution law for the state variable.
   * @tparam EvolutionType The evolution law function type.
   * @param domain_name The name of the domain.
   * @param evolution_law Function with signature (t_info, alpha, alpha_dot, grad_u, params...) returning
   *        the residual of the ODE: alpha_dot - f(alpha, grad_u, params...).
   *        Time integration is applied so alpha and alpha_dot are the current predicted values.
   */
  template <typename EvolutionType>
  void addStateEvolution(const std::string& domain_name, EvolutionType evolution_law)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_state_rule = state_time_rule;

    state_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto alpha, auto alpha_old, auto u, auto u_old, auto... params) {
          // Apply time integration to get current state and rate
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
          auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

          // The evolution law is in the form: alpha_dot = f(alpha, grad_u, params...)
          // Residual: alpha_dot - f(alpha, grad_u, params...) = 0
          // Pass only the scalar VALUE of alpha/alpha_dot (not the gradient) since this is a
          // local pointwise ODE.  Dual numbers live in the VALUE part so Jacobians are preserved.
          auto residual_val = evolution_law(t_info, get<VALUE>(alpha_current), get<VALUE>(alpha_dot),
                                            get<DERIVATIVE>(u_current), params...);

          // Flux is zero for a local (pointwise) ODE
          tensor<double, dim> flux{};
          return smith::tuple{residual_val, flux};
        });
  }

  // Add other methods (body forces, etc.) as needed...
};

/**
 * @brief Factory function to build a solid statics system with L2 state variable.
 */
template <int dim, int disp_order, typename StateSpace, typename... parameter_space>
SolidStaticsWithL2StateSystem<dim, disp_order, StateSpace, parameter_space...> buildSolidStaticsWithL2StateSystem(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<DifferentiableBlockSolver> solver, std::string prepend_name = "",
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

  // 1. Displacement fields
  auto disp_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
  FieldType<H1<disp_order, dim>> disp_type(prefix("displacement_predicted"));
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE, prefix("displacement"));

  // 2. State variable fields
  auto state_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  FieldType<StateSpace> state_type(prefix("state_predicted"));
  auto state_bc = field_store->addIndependent(state_type, state_time_rule);
  auto state_old_type = field_store->addDependent(state_type, FieldStore::TimeDerivative::VALUE, prefix("state"));

  // 3. Parameters
  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  // 4. Solid Weak Form (Residual for u)
  // Inputs: u, u_old, alpha, alpha_old, params...
  std::string solid_res_name = prefix("solid_residual");
  auto solid_weak_form = std::make_shared<
      typename SolidStaticsWithL2StateSystem<dim, disp_order, StateSpace, parameter_space...>::SolidWeakFormType>(
      solid_res_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
      field_store->createSpaces(solid_res_name, disp_type.name, disp_type, disp_old_type, state_type, state_old_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // 5. State Weak Form (Residual for alpha)
  // Inputs: alpha, alpha_old, u, u_old, params...
  std::string state_res_name = prefix("state_residual");
  auto state_weak_form = std::make_shared<
      typename SolidStaticsWithL2StateSystem<dim, disp_order, StateSpace, parameter_space...>::StateWeakFormType>(
      state_res_name, field_store->getMesh(), field_store->getField(state_type.name).get()->space(),
      field_store->createSpaces(state_res_name, state_type.name, state_type, state_old_type, disp_type, disp_old_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // 6. Solver and Advancer
  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, state_weak_form};
  auto advancer = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, solver);

  return SolidStaticsWithL2StateSystem<dim, disp_order, StateSpace, parameter_space...>{
      {field_store, solver, advancer, parameter_fields, prepend_name},
      solid_weak_form,
      state_weak_form,
      disp_bc,
      state_bc,
      disp_time_rule,
      state_time_rule};
}

}  // namespace smith
