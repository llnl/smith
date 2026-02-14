// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermo_mechanics_system.hpp
 * @brief Defines the ThermoMechanicsSystem struct and its factory function
 */

#pragma once

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/physics/weak_form.hpp"

namespace smith {

/**
 * @brief Container for a coupled thermo-mechanical system.
 * @tparam dim Spatial dimension.
 * @tparam disp_order Order of the displacement basis.
 * @tparam temp_order Order of the temperature basis.
 * @tparam parameter_space Finite element spaces for optional parameters.
 */
template <int dim, int disp_order, int temp_order, typename... parameter_space>
struct ThermoMechanicsSystem {
  /// @brief using for SolidWeakFormType
  using SolidWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<disp_order, dim>,
      Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<temp_order>, H1<temp_order>, parameter_space...>>;

  /// @brief using for ThermalWeakFormType
  using ThermalWeakFormType = TimeDiscretizedWeakForm<
      dim, H1<temp_order>,
      Parameters<H1<temp_order>, H1<temp_order>, H1<disp_order, dim>, H1<disp_order, dim>, parameter_space...>>;

  std::shared_ptr<FieldStore> field_store;                      ///< Field store managing the system's fields.
  std::shared_ptr<SolidWeakFormType> solid_weak_form;           ///< Solid mechanics weak form.
  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;       ///< Thermal weak form.
  std::shared_ptr<DirichletBoundaryConditions> disp_bc;         ///< Displacement boundary conditions.
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;  ///< Temperature boundary conditions.
  std::shared_ptr<DifferentiableBlockSolver> solver;            ///< The solver for the coupled system.
  std::shared_ptr<StateAdvancer> advancer;                      ///< The multiphysics state advancer.
  std::shared_ptr<QuasiStaticFirstOrderTimeIntegrationRule> disp_time_rule;  ///< Time integration for displacement.
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule>
      temperature_time_rule;                 ///< Time integration for temperature.
  std::vector<FieldState> parameter_fields;  ///< Optional parameter fields.

  /**
   * @brief Get the list of all state fields (current and old).
   * @return std::vector<FieldState> List of state fields.
   */
  std::vector<FieldState> getStateFields() const
  {
    std::vector<FieldState> states;
    states.push_back(field_store->getField("displacement"));
    states.push_back(field_store->getField("displacement_old"));
    states.push_back(field_store->getField("temperature"));
    states.push_back(field_store->getField("temperature_old"));
    return states;
  }

  /**
   * @brief Get the list of all parameter fields.
   * @return const std::vector<FieldState>& List of parameter fields.
   */
  const std::vector<FieldState>& getParameterFields() const { return parameter_fields; }

  /**
   * @brief Set the material model for a domain, defining integrals for solid and thermal weak forms.
   * @tparam MaterialType The material model type.
   * @param material The material model instance.
   * @param domain_name The name of the domain to apply the material to.
   */
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto dtr = disp_time_rule;
    auto ttr = temperature_time_rule;

    solid_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto disp, auto disp_old,
                                                      auto temperature, auto temperature_old, auto... params) {
      auto u = dtr->value(t_info, disp, disp_old);
      auto v = dtr->dot(t_info, disp, disp_old);
      auto T = ttr->value(t_info, temperature, temperature_old);
      typename MaterialType::State state;
      auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T),
                                        get<DERIVATIVE>(T), params...);
      return smith::tuple{smith::zero{}, pk};
    });

    thermal_weak_form->addBodyIntegral(

        domain_name, [=](auto t_info, auto /*X*/, auto temperature, auto temperature_old, auto disp, auto disp_old,

                         auto... params) {
          typename MaterialType::State state;

          auto u = dtr->value(t_info, disp, disp_old);

          auto v = dtr->dot(t_info, disp, disp_old);

          auto T = ttr->value(t_info, temperature, temperature_old);

          auto T_dot = ttr->dot(t_info, temperature, temperature_old);

          auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T),

                                            get<DERIVATIVE>(T), params...);

          auto dT_dt = get<VALUE>(T_dot);

          return smith::tuple{C_v * dT_dt - s0, -q0};
        });
  }
};

/**
 * @brief Factory function to build a thermo-mechanical system.
 * @tparam dim Spatial dimension.
 * @tparam disp_order Order of the displacement basis.
 * @tparam temp_order Order of the temperature basis.
 * @tparam parameter_space Finite element spaces for optional parameters.
 * @param mesh The mesh.
 * @param solver The differentiable block solver.
 * @param parameter_types Parameter field types.
 * @return ThermoMechanicsSystem with all components initialized.
 */
template <int dim, int disp_order, int temp_order, typename... parameter_space>
ThermoMechanicsSystem<dim, disp_order, temp_order, parameter_space...> buildThermoMechanicsSystem(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<DifferentiableBlockSolver> solver,
    FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<FieldStore>(mesh, 100);

  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  field_store->addShapeDisp(shape_disp_type);

  // Displacement field with quasi-static time integration
  auto disp_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
  FieldType<H1<disp_order, dim>> disp_type("displacement");
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE);

  // Temperature field with backward Euler time integration
  auto temperature_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  FieldType<H1<temp_order>> temperature_type("temperature");
  auto temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule);
  auto temperature_old_type = field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VALUE);

  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(parameter_types), ...);
  (parameter_fields.push_back(field_store->getField(parameter_types.name)), ...);

  // Solid mechanics weak form
  auto solid_weak_form = createWeakForm<dim>("solid_force", disp_type, *field_store, disp_type, disp_old_type,
                                             temperature_type, temperature_old_type, parameter_types...);

  // Thermal weak form
  auto thermal_weak_form = createWeakForm<dim>("thermal_flux", temperature_type, *field_store, temperature_type,
                                               temperature_old_type, disp_type, disp_old_type, parameter_types...);

  // Build solver and advancer
  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, thermal_weak_form};
  auto advancer = std::make_shared<MultiphysicsTimeIntegrator>(field_store, weak_forms, solver);

  return ThermoMechanicsSystem<dim, disp_order, temp_order, parameter_space...>{
      field_store, solid_weak_form, thermal_weak_form,     disp_bc,         temperature_bc, solver,
      advancer,    disp_time_rule,  temperature_time_rule, parameter_fields};
}

}  // namespace smith
