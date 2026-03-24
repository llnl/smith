// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file iterative_extended_thermomechanics.hpp
 * @brief Iterative (non-block) variant of `ExtendedThermoMechanicsSystem`.
 *
 * This example-facing module mirrors `ExtendedThermoMechanicsSystem` but advances the coupled
 * system by solving each equation (solid, thermal, state) individually using separate solvers.
 *
 * The update/order/relaxation strategy is user-defined via a custom update rule callback.
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/physics/weak_form.hpp"

namespace smith {

template <int dim, int disp_order, int temp_order, typename StateSpace, typename... parameter_space>
struct IterativeExtendedThermoMechanicsSystem;

/**
 * @brief State advancer that advances a 3-field thermo-mechanics + state system by sequential solves.
 *
 * The solve strategy (order, number of inner iterations, relaxation, etc.) is provided by a user callback.
 */
class IterativeExtendedThermoMechanicsTimeIntegrator : public StateAdvancer {
 public:
  struct Indices {
    size_t displacement_pred = 0;
    size_t displacement_old = 0;
    size_t temperature_pred = 0;
    size_t temperature_old = 0;
    size_t state_pred = 0;
    size_t state_old = 0;
  };

  struct WeakForms {
    std::shared_ptr<WeakForm> solid;
    std::shared_ptr<WeakForm> thermal;
    std::shared_ptr<WeakForm> state;
  };

  struct Solvers {
    std::shared_ptr<NonlinearBlockSolverBase> solid;
    std::shared_ptr<NonlinearBlockSolverBase> thermal;
    std::shared_ptr<NonlinearBlockSolverBase> state;
  };

  struct Bcs {
    std::shared_ptr<DirichletBoundaryConditions> displacement;
    std::shared_ptr<DirichletBoundaryConditions> temperature;
    std::shared_ptr<DirichletBoundaryConditions> state;
  };

  class Context {
   public:
    Context(const IterativeExtendedThermoMechanicsTimeIntegrator& integrator, const TimeInfo& time_info,
            const FieldState& shape_disp, std::vector<FieldState>& states, const std::vector<FieldState>& params)
        : integrator_(integrator), time_info_(time_info), shape_disp_(shape_disp), states_(states), params_(params)
    {
    }

    const TimeInfo& timeInfo() const { return time_info_; }
    const FieldState& shapeDisp() const { return shape_disp_; }
    std::vector<FieldState>& states() { return states_; }
    const std::vector<FieldState>& params() const { return params_; }

    void solveSolid(double relaxation = 1.0)
    {
      integrator_.solveSolid(time_info_, shape_disp_, states_, params_, relaxation);
    }
    void solveThermal(double relaxation = 1.0)
    {
      integrator_.solveThermal(time_info_, shape_disp_, states_, params_, relaxation);
    }
    void solveState(double relaxation = 1.0)
    {
      integrator_.solveState(time_info_, shape_disp_, states_, params_, relaxation);
    }

   private:
    const IterativeExtendedThermoMechanicsTimeIntegrator& integrator_;
    const TimeInfo& time_info_;
    const FieldState& shape_disp_;
    std::vector<FieldState>& states_;
    const std::vector<FieldState>& params_;
  };

  using UpdateRule = std::function<void(Context&)>;

  IterativeExtendedThermoMechanicsTimeIntegrator(std::shared_ptr<FieldStore> field_store, WeakForms weak_forms,
                                                 Solvers solvers, Bcs bcs, Indices indices, UpdateRule update_rule)
      : field_store_(std::move(field_store)),
        weak_forms_(std::move(weak_forms)),
        solvers_(std::move(solvers)),
        bcs_(std::move(bcs)),
        indices_(indices),
        update_rule_(std::move(update_rule))
  {
    if (!update_rule_) {
      update_rule_ = [](Context& ctx) {
        ctx.solveSolid();
        ctx.solveThermal();
        ctx.solveState();
      };
    }
  }

  std::pair<std::vector<FieldState>, std::vector<ReactionState>> advanceState(
      const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states,
      const std::vector<FieldState>& params) const override
  {
    for (size_t i = 0; i < states.size(); ++i) {
      field_store_->setField(i, states[i]);
    }

    std::vector<FieldState> iter_states = states;
    Context ctx(*this, time_info, shape_disp, iter_states, params);
    update_rule_(ctx);

    // Reactions: use newly updated primary unknowns, but BEFORE time integration updates.
    std::vector<ReactionState> reactions;
    for (auto wf : {weak_forms_.solid, weak_forms_.thermal, weak_forms_.state}) {
      std::vector<FieldState> wf_fields = field_store_->getStatesFromVectors(wf->name(), iter_states, params);
      std::string test_field_name = field_store_->getWeakFormReaction(wf->name());
      size_t test_field_idx = field_store_->getFieldIndex(test_field_name);
      FieldState test_field = iter_states[test_field_idx];
      reactions.push_back(smith::evaluateWeakForm(wf, time_info, shape_disp, wf_fields, test_field));
    }

    // Time integration updates (history + corrected derivatives).
    std::vector<FieldState> new_states = iter_states;
    for (const auto& [rule, mapping] : field_store_->getTimeIntegrationRules()) {
      size_t u_idx = field_store_->getFieldIndex(mapping.primary_name);
      FieldState u_new = iter_states[u_idx];
      new_states[u_idx] = u_new;

      std::vector<FieldState> rule_inputs;
      rule_inputs.push_back(u_new);  // u_{n+1}
      if (rule->num_args() >= 2) {
        rule_inputs.push_back(states[u_idx]);  // u_n
      }

      if (rule->num_args() >= 3 && !mapping.dot_name.empty()) {
        size_t v_idx = field_store_->getFieldIndex(mapping.dot_name);
        rule_inputs.push_back(states[v_idx]);
      }

      if (rule->num_args() >= 4 && !mapping.ddot_name.empty()) {
        size_t a_idx = field_store_->getFieldIndex(mapping.ddot_name);
        rule_inputs.push_back(states[a_idx]);
      }

      if (!mapping.dot_name.empty()) {
        size_t v_idx = field_store_->getFieldIndex(mapping.dot_name);
        new_states[v_idx] = rule->corrected_dot(time_info, rule_inputs);
      }

      if (!mapping.ddot_name.empty()) {
        size_t a_idx = field_store_->getFieldIndex(mapping.ddot_name);
        new_states[a_idx] = rule->corrected_ddot(time_info, rule_inputs);
      }

      if (!mapping.history_name.empty()) {
        size_t hist_idx = field_store_->getFieldIndex(mapping.history_name);
        new_states[hist_idx] = u_new;
      }
    }

    for (size_t i = 0; i < new_states.size(); ++i) {
      field_store_->setField(i, new_states[i]);
    }

    return {new_states, reactions};
  }

 private:
  void solveSolid(const TimeInfo& time_info, const FieldState& shape_disp, std::vector<FieldState>& states,
                  const std::vector<FieldState>& params, double relaxation) const
  {
    std::vector<FieldState> wf_states{states[indices_.displacement_pred], states[indices_.displacement_old],
                                      states[indices_.temperature_pred],  states[indices_.temperature_old],
                                      states[indices_.state_pred],        states[indices_.state_old]};

    FieldState sol =
        smith::solve(*weak_forms_.solid, shape_disp, wf_states, params, time_info, *solvers_.solid, *bcs_.displacement);
    if (relaxation != 1.0) {
      sol = smith::weighted_average(sol, states[indices_.displacement_pred], relaxation);
    }
    states[indices_.displacement_pred] = sol;
  }

  void solveThermal(const TimeInfo& time_info, const FieldState& shape_disp, std::vector<FieldState>& states,
                    const std::vector<FieldState>& params, double relaxation) const
  {
    std::vector<FieldState> wf_states{states[indices_.temperature_pred],  states[indices_.temperature_old],
                                      states[indices_.displacement_pred], states[indices_.displacement_old],
                                      states[indices_.state_pred],        states[indices_.state_old]};

    FieldState sol = smith::solve(*weak_forms_.thermal, shape_disp, wf_states, params, time_info, *solvers_.thermal,
                                  *bcs_.temperature);
    if (relaxation != 1.0) {
      sol = smith::weighted_average(sol, states[indices_.temperature_pred], relaxation);
    }
    states[indices_.temperature_pred] = sol;
  }

  void solveState(const TimeInfo& time_info, const FieldState& shape_disp, std::vector<FieldState>& states,
                  const std::vector<FieldState>& params, double relaxation) const
  {
    std::vector<FieldState> wf_states{states[indices_.state_pred],        states[indices_.state_old],
                                      states[indices_.displacement_pred], states[indices_.displacement_old],
                                      states[indices_.temperature_pred],  states[indices_.temperature_old]};

    FieldState sol =
        smith::solve(*weak_forms_.state, shape_disp, wf_states, params, time_info, *solvers_.state, *bcs_.state);
    if (relaxation != 1.0) {
      sol = smith::weighted_average(sol, states[indices_.state_pred], relaxation);
    }
    states[indices_.state_pred] = sol;
  }

  std::shared_ptr<FieldStore> field_store_;
  WeakForms weak_forms_;
  Solvers solvers_;
  Bcs bcs_;
  Indices indices_;
  UpdateRule update_rule_;
};

/**
 * @brief Container for a coupled thermo-mechanical system with an additional L2 state variable (iterative solve).
 */
template <int dim, int disp_order, int temp_order, typename StateSpace, typename... parameter_space>
struct IterativeExtendedThermoMechanicsSystem : public SystemBase {
  using SolidWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<disp_order, dim>,
                              Parameters<H1<disp_order, dim>, H1<disp_order, dim>, H1<temp_order>, H1<temp_order>,
                                         StateSpace, StateSpace, parameter_space...>>;

  using ThermalWeakFormType =
      TimeDiscretizedWeakForm<dim, H1<temp_order>,
                              Parameters<H1<temp_order>, H1<temp_order>, H1<disp_order, dim>, H1<disp_order, dim>,
                                         StateSpace, StateSpace, parameter_space...>>;

  using StateWeakFormType =
      TimeDiscretizedWeakForm<dim, StateSpace,
                              Parameters<StateSpace, StateSpace, H1<disp_order, dim>, H1<disp_order, dim>,
                                         H1<temp_order>, H1<temp_order>, parameter_space...>>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;
  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;
  std::shared_ptr<StateWeakFormType> state_weak_form;

  std::shared_ptr<DirichletBoundaryConditions> disp_bc;
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc;
  std::shared_ptr<DirichletBoundaryConditions> state_bc;

  std::shared_ptr<QuasiStaticFirstOrderTimeIntegrationRule> disp_time_rule;
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule> temperature_time_rule;
  std::shared_ptr<BackwardEulerFirstOrderTimeIntegrationRule> state_time_rule;

  std::vector<FieldState> getStateFields() const
  {
    return {field_store->getField(prefix("displacement_predicted")), field_store->getField(prefix("displacement")),
            field_store->getField(prefix("temperature_predicted")),  field_store->getField(prefix("temperature")),
            field_store->getField(prefix("state_predicted")),        field_store->getField(prefix("state"))};
  }

  std::shared_ptr<DifferentiablePhysics> createDifferentiablePhysics(std::string physics_name)
  {
    return std::make_shared<DifferentiablePhysics>(
        field_store->getMesh(), field_store->graph(), field_store->getShapeDisp(), getStateFields(),
        getParameterFields(), advancer, std::move(physics_name),
        std::vector<std::string>{prefix("solid_force"), prefix("thermal_flux"), prefix("state_residual")});
  }

  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto temperature, auto temperature_old,
                         auto /*alpha*/, auto alpha_old, auto... params) {
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto v_current = captured_disp_rule->dot(t_info, u, u_old);
          auto T = captured_temp_rule->value(t_info, temperature, temperature_old);

          typename MaterialType::State state;
          auto [pk, C_v, s0, q0, alpha_new] =
              material(t_info.dt(), state, get<DERIVATIVE>(u_current), get<DERIVATIVE>(v_current), get<VALUE>(T),
                       get<DERIVATIVE>(T), get<VALUE>(alpha_old), params...);

          return smith::tuple{0.0 * get<VALUE>(u_current), pk};
        });

    thermal_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto T, auto T_old, auto disp,
                                                        auto disp_old, auto /*alpha*/, auto alpha_old, auto... params) {
      auto T_current = captured_temp_rule->value(t_info, T, T_old);
      auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
      auto u = captured_disp_rule->value(t_info, disp, disp_old);
      auto v = captured_disp_rule->dot(t_info, disp, disp_old);

      typename MaterialType::State state;
      auto [pk, C_v, s0, q0, alpha_new] =
          material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T_current),
                   get<DERIVATIVE>(T_current), get<VALUE>(alpha_old), params...);

      auto dT_dt = get<VALUE>(T_dot);
      return smith::tuple{C_v * dT_dt - s0, -q0};
    });

    state_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto alpha, auto alpha_old, auto disp,
                                                      auto disp_old, auto T, auto T_old, auto... params) {
      auto T_current = captured_temp_rule->value(t_info, T, T_old);
      auto u = captured_disp_rule->value(t_info, disp, disp_old);
      auto v = captured_disp_rule->dot(t_info, disp, disp_old);

      typename MaterialType::State state;
      auto [pk, C_v, s0, q0, alpha_new] =
          material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T_current),
                   get<DERIVATIVE>(T_current), get<VALUE>(alpha_old), params...);

      return smith::tuple{get<VALUE>(alpha) - alpha_new, smith::zero{}};
    });
  }

  template <int... active_parameters, typename BodyForceType>
  void addSolidBodyForce(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                         BodyForceType force_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBodySource(depends_on, domain_name,
                                   [=](auto t_info, auto X, auto u, auto u_old, auto temperature, auto temperature_old,
                                       auto alpha, auto alpha_old, auto... params) {
                                     auto u_current = captured_disp_rule->value(t_info, u, u_old);
                                     auto v_current = captured_disp_rule->dot(t_info, u, u_old);
                                     auto T_current = captured_temp_rule->value(t_info, temperature, temperature_old);
                                     auto T_dot = captured_temp_rule->dot(t_info, temperature, temperature_old);
                                     auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
                                     auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

                                     return force_function(t_info.time(), X, u_current, v_current, T_current, T_dot,
                                                           alpha_current, alpha_dot, params...);
                                   });
  }

  template <typename BodyForceType>
  void addSolidBodyForce(const std::string& domain_name, BodyForceType force_function)
  {
    addSolidBodyForceAllParams(domain_name, force_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename SurfaceFluxType>
  void addSolidTraction(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                        SurfaceFluxType flux_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBoundaryFlux(depends_on, domain_name,
                                     [=](auto t_info, auto X, auto n, auto u, auto u_old, auto temperature,
                                         auto temperature_old, auto alpha, auto alpha_old, auto... params) {
                                       auto u_current = captured_disp_rule->value(t_info, u, u_old);
                                       auto v_current = captured_disp_rule->dot(t_info, u, u_old);
                                       auto T_current = captured_temp_rule->value(t_info, temperature, temperature_old);
                                       auto T_dot = captured_temp_rule->dot(t_info, temperature, temperature_old);
                                       auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
                                       auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

                                       return flux_function(t_info.time(), X, n, u_current, v_current, T_current, T_dot,
                                                            alpha_current, alpha_dot, params...);
                                     });
  }

  template <typename SurfaceFluxType>
  void addSolidTraction(const std::string& domain_name, SurfaceFluxType flux_function)
  {
    addSolidTractionAllParams(domain_name, flux_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename BodySourceType>
  void addThermalHeatSource(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                            BodySourceType source_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    thermal_weak_form->addBodySource(depends_on, domain_name,
                                     [=](auto t_info, auto X, auto T, auto T_old, auto disp, auto disp_old, auto alpha,
                                         auto alpha_old, auto... params) {
                                       auto u_current = captured_disp_rule->value(t_info, disp, disp_old);
                                       auto v_current = captured_disp_rule->dot(t_info, disp, disp_old);
                                       auto T_current = captured_temp_rule->value(t_info, T, T_old);
                                       auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
                                       auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
                                       auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

                                       return source_function(t_info.time(), X, u_current, v_current, T_current, T_dot,
                                                              alpha_current, alpha_dot, params...);
                                     });
  }

  template <typename BodySourceType>
  void addThermalHeatSource(const std::string& domain_name, BodySourceType source_function)
  {
    addThermalHeatSourceAllParams(domain_name, source_function,
                                  std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename SurfaceFluxType>
  void addThermalHeatFlux(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                          SurfaceFluxType flux_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    thermal_weak_form->addBoundaryFlux(depends_on, domain_name,
                                       [=](auto t_info, auto X, auto n, auto T, auto T_old, auto disp, auto disp_old,
                                           auto alpha, auto alpha_old, auto... params) {
                                         auto u_current = captured_disp_rule->value(t_info, disp, disp_old);
                                         auto v_current = captured_disp_rule->dot(t_info, disp, disp_old);
                                         auto T_current = captured_temp_rule->value(t_info, T, T_old);
                                         auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
                                         auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
                                         auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

                                         return -flux_function(t_info.time(), X, n, u_current, v_current, T_current,
                                                               T_dot, alpha_current, alpha_dot, params...);
                                       });
  }

  template <typename SurfaceFluxType>
  void addThermalHeatFlux(const std::string& domain_name, SurfaceFluxType flux_function)
  {
    addThermalHeatFluxAllParams(domain_name, flux_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename PressureType>
  void addPressure(DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   PressureType pressure_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;
    auto captured_state_rule = state_time_rule;

    solid_weak_form->addBoundaryIntegral(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto temperature, auto temperature_old, auto alpha, auto alpha_old,
            auto... params) {
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto v_current = captured_disp_rule->dot(t_info, u, u_old);
          auto T_current = captured_temp_rule->value(t_info, temperature, temperature_old);
          auto T_dot = captured_temp_rule->dot(t_info, temperature, temperature_old);
          auto alpha_current = captured_state_rule->value(t_info, alpha, alpha_old);
          auto alpha_dot = captured_state_rule->dot(t_info, alpha, alpha_old);

          auto x_current = X + u_current;
          auto n_deformed = cross(get<DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<VALUE>(X), u_current, v_current, T_current, T_dot,
                                            alpha_current, alpha_dot, get<VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        });
  }

  template <typename PressureType>
  void addPressure(const std::string& domain_name, PressureType pressure_function)
  {
    addPressureAllParams(domain_name, pressure_function, std::make_index_sequence<6 + sizeof...(parameter_space)>{});
  }

 private:
  template <typename BodyForceType, std::size_t... Is>
  void addSolidBodyForceAllParams(const std::string& domain_name, BodyForceType force_function,
                                  std::index_sequence<Is...>)
  {
    addSolidBodyForce(DependsOn<static_cast<int>(Is)...>{}, domain_name, force_function);
  }

  template <typename SurfaceFluxType, std::size_t... Is>
  void addSolidTractionAllParams(const std::string& domain_name, SurfaceFluxType flux_function,
                                 std::index_sequence<Is...>)
  {
    addSolidTraction(DependsOn<static_cast<int>(Is)...>{}, domain_name, flux_function);
  }

  template <typename PressureType, std::size_t... Is>
  void addPressureAllParams(const std::string& domain_name, PressureType pressure_function, std::index_sequence<Is...>)
  {
    addPressure(DependsOn<static_cast<int>(Is)...>{}, domain_name, pressure_function);
  }

  template <typename BodySourceType, std::size_t... Is>
  void addThermalHeatSourceAllParams(const std::string& domain_name, BodySourceType source_function,
                                     std::index_sequence<Is...>)
  {
    addThermalHeatSource(DependsOn<static_cast<int>(Is)...>{}, domain_name, source_function);
  }

  template <typename SurfaceFluxType, std::size_t... Is>
  void addThermalHeatFluxAllParams(const std::string& domain_name, SurfaceFluxType flux_function,
                                   std::index_sequence<Is...>)
  {
    addThermalHeatFlux(DependsOn<static_cast<int>(Is)...>{}, domain_name, flux_function);
  }
};

template <int dim, int disp_order, int temp_order, typename StateSpace, typename... parameter_space>
IterativeExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace, parameter_space...>
buildIterativeExtendedThermoMechanicsSystem(std::shared_ptr<Mesh> mesh,
                                            std::shared_ptr<NonlinearBlockSolverBase> solid_solver,
                                            std::shared_ptr<NonlinearBlockSolverBase> thermal_solver,
                                            std::shared_ptr<NonlinearBlockSolverBase> state_solver,
                                            IterativeExtendedThermoMechanicsTimeIntegrator::UpdateRule update_rule,
                                            std::string prepend_name = "",
                                            FieldType<parameter_space>... parameter_types)
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

  auto disp_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
  FieldType<H1<disp_order, dim>> disp_type(prefix("displacement_predicted"));
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VAL, prefix("displacement"));

  auto temperature_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  FieldType<H1<temp_order>> temperature_type(prefix("temperature_predicted"));
  auto temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule);
  auto temperature_old_type =
      field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VAL, prefix("temperature"));

  auto state_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  FieldType<StateSpace> state_type(prefix("state_predicted"));
  auto state_bc = field_store->addIndependent(state_type, state_time_rule);
  auto state_old_type = field_store->addDependent(state_type, FieldStore::TimeDerivative::VAL, prefix("state"));

  std::vector<FieldState> parameter_fields;
  (field_store->addParameter(FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  std::string solid_force_name = prefix("solid_force");
  auto solid_weak_form =
      std::make_shared<typename IterativeExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace,
                                                                       parameter_space...>::SolidWeakFormType>(
          solid_force_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
          field_store->createSpaces(solid_force_name, disp_type.name, disp_type, disp_old_type, temperature_type,
                                    temperature_old_type, state_type, state_old_type,
                                    FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  std::string thermal_flux_name = prefix("thermal_flux");
  auto thermal_weak_form =
      std::make_shared<typename IterativeExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace,
                                                                       parameter_space...>::ThermalWeakFormType>(
          thermal_flux_name, field_store->getMesh(), field_store->getField(temperature_type.name).get()->space(),
          field_store->createSpaces(thermal_flux_name, temperature_type.name, temperature_type, temperature_old_type,
                                    disp_type, disp_old_type, state_type, state_old_type,
                                    FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  std::string state_residual_name = prefix("state_residual");
  auto state_weak_form =
      std::make_shared<typename IterativeExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace,
                                                                       parameter_space...>::StateWeakFormType>(
          state_residual_name, field_store->getMesh(), field_store->getField(state_type.name).get()->space(),
          field_store->createSpaces(state_residual_name, state_type.name, state_type, state_old_type, disp_type,
                                    disp_old_type, temperature_type, temperature_old_type,
                                    FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  IterativeExtendedThermoMechanicsTimeIntegrator::Indices indices{
      .displacement_pred = field_store->getFieldIndex(disp_type.name),
      .displacement_old = field_store->getFieldIndex(disp_old_type.name),
      .temperature_pred = field_store->getFieldIndex(temperature_type.name),
      .temperature_old = field_store->getFieldIndex(temperature_old_type.name),
      .state_pred = field_store->getFieldIndex(state_type.name),
      .state_old = field_store->getFieldIndex(state_old_type.name),
  };

  auto advancer = std::make_shared<IterativeExtendedThermoMechanicsTimeIntegrator>(
      field_store,
      IterativeExtendedThermoMechanicsTimeIntegrator::WeakForms{solid_weak_form, thermal_weak_form, state_weak_form},
      IterativeExtendedThermoMechanicsTimeIntegrator::Solvers{std::move(solid_solver), std::move(thermal_solver),
                                                              std::move(state_solver)},
      IterativeExtendedThermoMechanicsTimeIntegrator::Bcs{disp_bc, temperature_bc, state_bc}, indices,
      std::move(update_rule));

  return IterativeExtendedThermoMechanicsSystem<dim, disp_order, temp_order, StateSpace, parameter_space...>{
      {field_store, nullptr, advancer, parameter_fields, prepend_name},
      solid_weak_form,
      thermal_weak_form,
      state_weak_form,
      disp_bc,
      temperature_bc,
      state_bc,
      disp_time_rule,
      temperature_time_rule,
      state_time_rule};
}

}  // namespace smith
