// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file electro_mechanics_system.hpp
 * @brief Defines the ElectroMechanicsystem struct and its factory function
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/system_base.hpp"

namespace smith {
/**
 * @brief Container for a plastic solid mechanics system.
 * @tparam dim Spatial dimension.
 * @tparam disp_order Order of the displacement basis.
 * @tparam parameter_space Finite element spaces for optional parameters.
 * 
 * This system rely on a fixed-point iteration to solve plasticity equivalent to the traditional
 * inner-outer Newton method.
 * n \in [0, N] denotes the timestep and k \in [0, L] denotes the fixed-point iteration count.
 * At every new timestep n+1, first predict new displacement by [u^{n+1,k}, Fp^{n+1,k}, epsilon_p^{n+1,k}] ---> u^{n+1,k+1}.
 * Then update internal state variables [u^{n+1,k+1}, Fp^n, epsilon_p^n] ---> [Fp^{n+1,k+1}, epsilon_p^{n+1,k+1}].
 */
template <int dim, int disp_order, typename... parameter_space>
struct PlasticMechanicsSystem : public SystemBase {
  // Primary weak forms to solve for displacement
  /// @brief using for SolidWeakFormType with inputs [u^{n+1, k+1}, Fp^{n+1, k}, epsilon_p^{n+1, k}]
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      Parameters<H1<disp_order, dim>, L2<disp_order - 1, dim * dim>, L2<disp_order - 1>, parameter_space...>>;

  /// @brief using for PlasticDeformWeakFormType with inputs [Fp^{n+1, k+1}, Fp^n, epsilon_p^{n+1, k+1}, epsilon_p^n, u^{n+1, k+1}]
  using PlasticDeformWeakFormType = TimeDiscretizedWeakForm<
      dim, L2<disp_order - 1, dim * dim>,
      Parameters<L2<disp_order - 1, dim * dim>, L2<disp_order - 1, dim * dim>, L2<disp_order - 1>,
                 L2<disp_order - 1>, H1<disp_order, dim>, parameter_space...>>;

  /// @brief using for PlasticStrainWeakFormType with inputs [epsilon_p^{n+1, k+1}, epsilon_p^n, Fp^{n+1, k+1}, Fp^n, u^{n+1, k+1}]
  using PlasticStrainWeakFormType = TimeDiscretizedWeakForm<
      dim, L2<disp_order - 1>,
      Parameters<L2<disp_order - 1>, L2<disp_order - 1>, L2<disp_order - 1, dim * dim>, L2<disp_order - 1, dim * dim>
                 H1<disp_order, dim>, parameter_space...>>;

  // Primary weak forms
  std::shared_ptr<SolidWeakFormType>         solid_weak_form;               ///< Solid mechanics weak form.

  // Internal variable weak forms to update in staggered solve
  std::shared_ptr<PlasticDeformWeakFormType> plastic_deform_weak_form;       ///< Plastic deformation weak form.
  std::shared_ptr<PlasticStrainWeakFormType> plastic_strain_weak_form;       ///< Plastic deformation weak form.

  // Primary variable bcs
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;                      ///< Displacement boundary conditions.

  // Internal variable bcs
  std::shared_ptr<DirichletBoundaryConditions> plastic_deform_bc;
  std::shared_ptr<DirichletBoundaryConditions> plastic_strain_bc;

  std::shared_ptr<QuasiStaticRule>                            qusistatic_time_rule;
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule> backward_euler_time_rule;

  // Helper functions for plastic deformation gradient bookkeeping
  template <typename T>
  SMITH_HOST_DEVICE tensor<T, dim, dim> recoverTensor(const tensor<T, dim * dim> F_state)
  {
    tensor<T, dim, dim> F_state_tensor{};

    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        F_state_tensor(i, j) = F_state(i * dim + j);
      }
    }

    return F_state_tensor;
  }

  template <typename T>
  SMITH_HOST_DEVICE tensor<T, dim * dim> flattenTensor(const tensor<T, dim, dim> F_state)
  {
    tensor<T, dim * dim> F_state_flattened{};

    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        F_state_flattened(i * dim + j) = F_state(i, j);
      }
    }

    return F_state_flattened;
  }

  /**
   * @brief Get the list of all state fields.
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    std::vector<FieldState> states;
    states.push_back(field_store->getField(prefix("displacement")));

    // Internal states
    states.push_back(field_store->getField(prefix("plastic_defgrad")));
    states.push_back(field_store->getField(prefix("plastic_defgrad_old")));
    states.push_back(field_store->getField(prefix("plastic_strain")));
    states.push_back(field_store->getField(prefix("plastic_strain_old")));

    return states;
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
        std::vector<std::string>{prefix("solid_force")});
  }

  /**
   * @brief Set the material model for a domain, defining integrals for weak form.
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const std::string& domain_name, const MaterialType& mat)
  {
    solid_weak_form->addBodyIntegral(domain_name,
        [&](auto /* t_info */, auto /* X */, auto u, auto Fp, auto epsilon_p, auto... params) {
              auto du_dX = get<DERIVATIVE>(u);
              auto Fp_tensor = this->recoverTensor(get<VALUE>(Fp));
              auto epsilon_pval = get<DERIVATIVE>(epsilon_p);

              auto P = mat.firstPiolaStress(du_dX, Fp_tensor, epsilon_pval, params...);

              return tuple{zero{}, P};
            });
  }

  /**
   * @brief Set the plasticity update model for a domain, defining integrals for weak form.
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setPlasticity(const std::string& domain_name, const MaterialType& mat)
  {
    auto captured_strain_rule = backward_euler_time_rule;

    plastic_deform_weak_form->addBodyIntegral(domain_name,
        [=, this](auto t_info, auto /* X */, auto Fp, auto Fp_old, auto epsilon_p, auto epsilon_p_old, auto u, auto... params) {
              auto du_dX = get<DERIVATIVE>(u);
              auto Fp_tensor = this->recoverTensor(get<VALUE>(Fp));
              auto Fp_old_tensor = this->recoverTensor(get<VALUE>(Fp_old));
              auto [epsilon_current, epsilon_dot] = captured_strain_rule->interpolate(t_info, epsilon_p, epsilon_p_old);
              auto dt = t_info.dt();

              auto Fp_predict_tensor = mat.plasticDeformGrad(dt, Fp_old_tensor, epsilon_dot, du_dX, params...);
              auto Fp_predict = this->flattenTensor(Fp_predict_tensor);

              return tuple(get<VALUE>(Fp) - Fp_predict, zero{});
            });

    plastic_strain_weak_form->addBodyIntegral(domain_name,
        [=, this](auto t_info, auto /* X */, auto epsilon_p, auto epsilon_p_old, auto /* Fp */, auto Fp_old, auto u) {
              auto du_dX = get<DERIVATIVE>(u);
              auto Fp_old_tensor = this->recoverTensor(get<VALUE>(Fp_old));
              auto [epsilon_current, epsilon_dot] = captured_strain_rule->interpolate(t_info, epsilon_p, epsilon_p_old);
              auto dt = t_info.dt();

              return tuple(epsilon_p_dot - mat.plasticStrain(dt, epsilon_current, epsilon_dot, Fp_old_tensor, du_dX), zero{});
            });
  }
};

/**
 * @brief Factory function to build a chemomechanics system with L2 state variables.
 */
template <int dim, int disp_order, typename... parameter_space>
PlasticMechanicsSystem<dim, disp_order, parameter_space...> buildPlasticMechanicsSystem(
  std::shared_ptr<Mesh> mesh, std::shared_ptr<CoupledSystemSolver> solver,
  std::string prepend_name = "", FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<FieldStore>(mesh, 100);

  auto prefix = [&](const std::string& name) {
    if (prepend_name.empty()) {
      return name;
    }
    return prepend_name + "_" + name;
  };

  FieldType<H1<1, dim>> shape_disp_type(prefix("shape_displacement"));
  field_store->addShapeDisp(shape_disp_type);

  auto quasistatic_time_rule = std::make_shared<QuasiStaticRule>();
  auto backward_euler_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();

  // Displacement with quasi-static rule
  FieldType<H1<disp_order, dim>> disp_type(prefix("displacement"));
  auto disp_bc = field_store->addIndependent(disp_type, quasistatic_time_rule);

  // State variable fields
  FieldType<L2<disp_order - 1, dim * dim>> plastic_defgrad_type(prefix("plastic_defgrad"));
  auto plastic_defgrad_bc = field_store->addIndependent(plastic_defgrad_type, backward_euler_time_rule);
  auto plastic_defgrad_old_type =
      field_store->addDependent(plastic_defgrad_type, FieldStore::TimeDerivative::VAL, prefix("plastic_defgrad_old"));

  FieldType<L2<disp_order>> plastic_strain_type(prefix("plastic_strain"));
  auto plastic_strain_bc = field_store->addIndependent(plastic_strain_type, backward_euler_time_rule);
  auto plastic_strain_old_type =
      field_store->addDependent(plastic_strain_type, FieldStore::TimeDerivative::VAL, prefix("plastic_strain_old"));

  // Parameters
  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  using SystemType = PlasticMechanicsSystem<dim, disp_order, parameter_space...>;

  // Main weak forms for mechanics
  std::string solid_res_name = prefix("solid_residual");
  auto solid_weak_form = std::make_shared<typename SystemType::SolidWeakFormType>(
      solid_res_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
      field_store->createSpaces(solid_res_name, disp_type.name, disp_type,
                                plastic_defgrad_type, plastic_strain_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  std::string plastic_defgrad_res_name = prefix("plastic_defgrad_residual");
  auto plastic_defgrad_weak_form = std::make_shared<typename SystemType::PlasticDeformWeakFormType>(
      plastic_defgrad_res_name, field_store->getMesh(), field_store->getField(plastic_defgrad_type.name).get()->space(),
      field_store->createSpaces(plastic_defgrad_res_name, plastic_defgrad_type.name, plastic_defgrad_type,
                                plastic_defgrad_old_type, plastic_strain_type, plastic_strain_old_type, disp_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  std::string plastic_strain_res_name = prefix("plastic_defgrad_residual");
  auto plastic_strain_weak_form = std::make_shared<typename SystemType::PlasticStrainWeakFormType>(
      plastic_strain_res_name, field_store->getMesh(), field_store->getField(plastic_strain_type.name).get()->space(),
      field_store->createSpaces(plastic_strain_res_name, plastic_strain_type.name, plastic_strain_type,
                                plastic_strain_old_type, plastic_defgrad_type, disp_type,
                                FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  // Build solver and advancer
  std::vector<std::shared_ptr<WeakForm>> weak_forms{
    solid_weak_form, plastic_defgrad_weak_form, plastic_strain_weak_form};
  auto advancer = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, solver);

  return SystemType{{field_store, solver, advancer, parameter_fields, prepend_name},
      solid_weak_form, plastic_defgrad_weak_form, plastic_strain_weak_form,
      disp_bc, plastic_defgrad_bc, plastic_strain_bc, quasistatic_time_rule,
      backward_euler_time_rule};
}
} // namespace smith
