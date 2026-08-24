// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file plastic_mechanics_system.hpp
 * @brief Defines the PlasticMechanicsystem struct and its factory function
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/physics/functional_weak_form.hpp"
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
 * n in [0, N] denotes the timestep and k in [0, L] denotes the fixed-point iteration count.
 * At every new timestep n+1, first predict new displacement by [u^{n+1,k}, Fp^{n+1,k}, epsilon_p^{n+1,k}] --->
 * u^{n+1,k+1}. Then update internal state variables [u^{n+1,k+1}, Fp^n, epsilon_p^n] ---> [Fp^{n+1,k+1},
 * epsilon_p^{n+1,k+1}].
 */
template <int dim, int disp_order, typename... parameter_space>
struct PlasticMechanicsSystem : public SystemBase {
  using SystemBase::SystemBase;
  // Primary weak forms to solve for displacement
  /// @brief using for SolidWeakFormType with inputs [u^{n+1, k+1}, Fp^{n+1, k}, epsilon_p^{n+1, k}]
  using SolidWeakFormType = FunctionalWeakForm<
      dim, H1<disp_order, dim>,
      Parameters<H1<disp_order, dim>, L2<disp_order, dim * dim>, L2<disp_order>, parameter_space...>>;

  /// @brief using for PlasticDeformWeakFormType with inputs [Fp^{n+1, k+1}, Fp^n, epsilon_p^{n+1, k+1}, epsilon_p^n,
  /// u^{n+1, k+1}]
  using PlasticDeformWeakFormType =
      FunctionalWeakForm<dim, L2<disp_order, dim * dim>,
                         Parameters<L2<disp_order, dim * dim>, L2<disp_order, dim * dim>, L2<disp_order>,
                                    L2<disp_order>, H1<disp_order, dim>, parameter_space...>>;

  /// @brief using for PlasticStrainWeakFormType with inputs [epsilon_p^{n+1, k+1}, epsilon_p^n, Fp^{n+1, k+1}, Fp^n,
  /// u^{n+1, k+1}]
  using PlasticStrainWeakFormType =
      FunctionalWeakForm<dim, L2<disp_order>,
                         Parameters<L2<disp_order>, L2<disp_order>, L2<disp_order, dim * dim>,
                                    L2<disp_order, dim * dim>, H1<disp_order, dim>, parameter_space...>>;

  // Primary weak forms
  std::shared_ptr<SolidWeakFormType> solid_weak_form;  ///< Solid mechanics weak form.

  // Internal variable weak forms to update in staggered solve
  std::shared_ptr<PlasticDeformWeakFormType> plastic_deform_weak_form;  ///< Plastic deformation weak form.
  std::shared_ptr<PlasticStrainWeakFormType> plastic_strain_weak_form;  ///< Plastic deformation weak form.

  // Primary variable bcs
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;  ///< Displacement boundary conditions.

  // Internal variable bcs
  std::shared_ptr<DirichletBoundaryConditions> plastic_deform_bc;  ///< Plastic deformation gradient conditions.
  std::shared_ptr<DirichletBoundaryConditions> plastic_strain_bc;  ///< Plastic strain conditions

  std::shared_ptr<QuasiStaticRule> quasistatic_time_rule;  ///< Quasistatic time integration rule
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule>
      backward_euler_time_rule;  ///< Backward euler time integration rule

  /**
   * @brief Transform tensor variable stored as [dim * dim, 1] to [dim, dim]
   */
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

  /**
   * @brief Transform tensor variable stored as [dim, dim] to [dim * dim, 1]
   */
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
   * @brief Get the list of physical state fields for visualization.
   * @return std::vector<FieldState> List of physical fields suitable for output.
   */
  std::vector<FieldState> getOutputFieldStates() const
  {
    return {field_store->getField(field_store->prefix("displacement")),
            field_store->getField(field_store->prefix("plastic_defgrad")),
            field_store->getField(field_store->prefix("plastic_strain"))};
  }

  /**
   * @brief Get information about reaction fields for this system.
   * @return List of ReactionInfo structures.
   */
  std::vector<ReactionInfo> getReactionInfos() const
  {
    return {{field_store->prefix("solid_force"),
             &field_store->getField(field_store->prefix("displacement")).get()->space()}};
  }

  /**
   * @brief Set the material model for a domain, defining integrals for weak form.
   * @tparam MaterialType The material model type.
   * @param domain_name The name of the domain to apply the material to.
   * @param mat The material model instance.
   */
  template <typename MaterialType>
  void setMaterial(const std::string& domain_name, const MaterialType& mat)
  {
    solid_weak_form->addBodyIntegral(
        domain_name, [&](auto t_info, auto /* X */, auto u, auto Fp, auto epsilon_p, auto... params) {
          auto du_dX = get<DERIVATIVE>(u);
          auto Fp_tensor = this->recoverTensor(get<VALUE>(Fp));
          auto dt = t_info.dt();

          auto P = mat.firstPiolaStress(dt, du_dX, Fp_tensor, get<VALUE>(epsilon_p), params...);

          return tuple{zero{}, P};
        });
  }

  /**
   * @brief Set the plasticity update model for a domain, defining integrals for weak form.
   * @tparam MaterialType The material model type.
   * @param domain_name The name of the domain to apply the material to.
   * @param mat The material model instance.
   */
  template <typename MaterialType>
  void setPlasticity(const std::string& domain_name, const MaterialType& mat)
  {
    auto captured_strain_rule = backward_euler_time_rule;

    plastic_deform_weak_form->addBodyIntegral(
        domain_name, [=, this](auto t_info, auto /* X */, auto Fp, auto Fp_old, auto epsilon_p, auto epsilon_p_old,
                               auto u, auto... params) {
          auto du_dX = get<DERIVATIVE>(u);
          auto Fp_old_tensor = this->recoverTensor(get<VALUE>(Fp_old));
          auto [epsilon_current, epsilon_dot] = captured_strain_rule->interpolate(t_info, epsilon_p, epsilon_p_old);
          auto dt = t_info.dt();

          auto Fp_predict_tensor = mat.plasticDeformGrad(dt, Fp_old_tensor, get<VALUE>(epsilon_dot), du_dX, params...);
          auto Fp_predict = this->flattenTensor(Fp_predict_tensor);

          return tuple(get<VALUE>(Fp) - Fp_predict, zero{});
        });

    plastic_strain_weak_form->addBodyIntegral(domain_name, [=, this](auto t_info, auto /* X */, auto epsilon_p,
                                                                     auto epsilon_p_old, auto /* Fp */, auto Fp_old,
                                                                     auto u, auto... params) {
      auto du_dX = get<DERIVATIVE>(u);
      auto Fp_old_tensor = this->recoverTensor(get<VALUE>(Fp_old));
      auto [epsilon_current, epsilon_dot] = captured_strain_rule->interpolate(t_info, epsilon_p, epsilon_p_old);
      auto dt = t_info.dt();

      auto epsilon_dot_predict =
          mat.plasticStrain(dt, get<VALUE>(epsilon_current), get<VALUE>(epsilon_dot), Fp_old_tensor, du_dX, params...);
      return tuple(get<VALUE>(epsilon_dot) - epsilon_dot_predict, zero{});
    });
  }
};

/**
 * @brief Factory function to build a chemomechanics system with L2 state variables.
 */
template <int dim, int disp_order, typename... parameter_space>
std::shared_ptr<PlasticMechanicsSystem<dim, disp_order, parameter_space...>> buildPlasticMechanicsSystem(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<SystemSolver> solver, std::string prepend_name = "",
    FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<FieldStore>(mesh, 100, prepend_name);

  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  auto quasistatic_time_rule = std::make_shared<QuasiStaticRule>();
  auto backward_euler_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();

  // Displacement with quasi-static rule
  FieldType<H1<disp_order, dim>> disp_type("displacement");
  auto disp_bc = field_store->addIndependent(disp_type, quasistatic_time_rule);

  // State variable fields
  FieldType<L2<disp_order, dim * dim>> plastic_defgrad_type("plastic_defgrad");
  auto plastic_defgrad_bc = field_store->addIndependent(plastic_defgrad_type, backward_euler_time_rule);
  auto plastic_defgrad_old_type =
      field_store->addDependent(plastic_defgrad_type, FieldStore::TimeDerivative::VAL, "plastic_defgrad_old");

  FieldType<L2<disp_order>> plastic_strain_type("plastic_strain");
  auto plastic_strain_bc = field_store->addIndependent(plastic_strain_type, backward_euler_time_rule);
  auto plastic_strain_old_type =
      field_store->addDependent(plastic_strain_type, FieldStore::TimeDerivative::VAL, "plastic_strain_old");

  // Parameters
  auto register_parameter = [&](auto& parameter_type) {
    parameter_type.name = "param_" + parameter_type.name;
    field_store->addParameter(parameter_type);
  };

  if constexpr (sizeof...(parameter_space) > 0) {
    (register_parameter(parameter_types), ...);
  }

  using SystemType = PlasticMechanicsSystem<dim, disp_order, parameter_space...>;

  // Main weak forms for mechanics
  std::string solid_res_name = field_store->prefix("solid_residual");
  auto solid_weak_form = std::make_shared<typename SystemType::SolidWeakFormType>(
      solid_res_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
      field_store->createSpaces(solid_res_name, {.unknown = disp_type.name, .test = disp_type.name}, disp_type,
                                plastic_defgrad_type, plastic_strain_type, parameter_types...));

  std::string plastic_defgrad_res_name = field_store->prefix("plastic_defgrad_residual");
  auto plastic_defgrad_weak_form = std::make_shared<typename SystemType::PlasticDeformWeakFormType>(
      plastic_defgrad_res_name, field_store->getMesh(), field_store->getField(plastic_defgrad_type.name).get()->space(),
      field_store->createSpaces(plastic_defgrad_res_name,
                                {.unknown = plastic_defgrad_type.name, .test = plastic_defgrad_type.name},
                                plastic_defgrad_type, plastic_defgrad_old_type, plastic_strain_type,
                                plastic_strain_old_type, disp_type, parameter_types...));

  std::string plastic_strain_res_name = field_store->prefix("plastic_strain_residual");
  auto plastic_strain_weak_form = std::make_shared<typename SystemType::PlasticStrainWeakFormType>(
      plastic_strain_res_name, field_store->getMesh(), field_store->getField(plastic_strain_type.name).get()->space(),
      field_store->createSpaces(plastic_strain_res_name,
                                {.unknown = plastic_strain_type.name, .test = plastic_strain_type.name},
                                plastic_strain_type, plastic_strain_old_type, plastic_defgrad_type,
                                plastic_defgrad_old_type, disp_type, parameter_types...));

  // Build solver and advancer
  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, plastic_defgrad_weak_form,
                                                    plastic_strain_weak_form};

  auto sys = std::make_shared<SystemType>(field_store, solver, weak_forms);

  sys->solid_weak_form = solid_weak_form;
  sys->plastic_deform_weak_form = plastic_defgrad_weak_form;
  sys->plastic_strain_weak_form = plastic_strain_weak_form;
  sys->disp_bc = disp_bc;
  sys->plastic_deform_bc = plastic_defgrad_bc;
  sys->plastic_strain_bc = plastic_strain_bc;
  sys->quasistatic_time_rule = quasistatic_time_rule;
  sys->backward_euler_time_rule = backward_euler_time_rule;

  return sys;
}
}  // namespace smith
