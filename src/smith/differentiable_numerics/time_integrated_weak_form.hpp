// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file time_integrated_weak_form.hpp
 *
 * @brief Wraps a TimeDiscretizedWeakForm to automatically handle time integration of state fields.
 */

#pragma once

#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include <tuple>

namespace smith {

template <int spatial_dim, typename OutputSpace, typename TimeRuleType, typename inputs = Parameters<>>
class TimeIntegratedWeakForm;

/**
 * @brief A weak form wrapper that handles time integration automatically.
 * 
 * It assumes the first N fields (where N is determined by TimeRuleType::num_args) are the state fields
 * (e.g., u, u_old, v_old, a_old) and uses the TimeRuleType to compute the current state (u, v, a).
 * The user-provided integrand is then called with (t_info, X, u, v, a, params...).
 * 
 * @tparam spatial_dim Spatial dimension.
 * @tparam OutputSpace The output space (test function space).
 * @tparam TimeRuleType The time integration rule type.
 * @tparam InputSpaces All input spaces (state fields + parameters).
 */
template <int spatial_dim, typename OutputSpace, typename TimeRuleType, typename... InputSpaces>
class TimeIntegratedWeakForm<spatial_dim, OutputSpace, TimeRuleType, Parameters<InputSpaces...>>
    : public TimeDiscretizedWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>> {
 public:
  using Base = TimeDiscretizedWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>>;
  using Rule = TimeRuleType;

  std::shared_ptr<Rule> rule;

  TimeIntegratedWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh,
                         const mfem::ParFiniteElementSpace& output_mfem_space,
                         const typename Base::WeakFormT::SpacesT& input_mfem_spaces,
                         std::shared_ptr<Rule> rule)
      : Base(physics_name, mesh, output_mfem_space, input_mfem_spaces), rule(rule)
  {
  }

  // Helper implementation to unpack tuple and call rule
  template <typename Integrand, typename XType, typename Tuple, int... StateIdx, int... ParamIdx>
  auto call_integrand(Integrand& integrand, const TimeInfo& t_info, XType X, const Tuple& args,
                      std::integer_sequence<int, StateIdx...>, std::integer_sequence<int, ParamIdx...>)
  {
    // Apply rule to state fields
    auto u = rule->value(t_info, std::get<StateIdx>(args)...);
    auto v = rule->dot(t_info, std::get<StateIdx>(args)...);
    
    if constexpr (Rule::num_args == 4) {
        auto a = rule->ddot(t_info, std::get<StateIdx>(args)...);
        return integrand(t_info, X, u, v, a, std::get<ParamIdx>(args)...);
    } else {
        return integrand(t_info, X, u, v, std::get<ParamIdx>(args)...);
    }
  }

  /// @overload
  template <typename Integrand>
  void addBodyIntegral(std::string body_name, Integrand integrand)
  {
    // Create a wrapper that matches the signature expected by TimeDiscretizedWeakForm
    // (t_info, X, inputs...)
    
    // We need to capture 'this' or 'rule'
    auto captured_rule = rule;
    
    Base::addBodyIntegral(body_name, [captured_rule, integrand](const TimeInfo& t_info, auto X, auto... inputs) {
       constexpr int N = Rule::num_args;
       static_assert(sizeof...(inputs) >= N, "Not enough inputs provided for the time integration rule.");

       auto args_tuple = std::forward_as_tuple(inputs...);
       
       return apply_rule_helper(captured_rule, integrand, t_info, X, args_tuple,
                                std::make_integer_sequence<int, N>{},
                                std::make_integer_sequence<int, sizeof...(inputs) - N>{});
    });
  }
  
  // Static helper to dispatch
  template <typename R, typename Integrand, typename XType, typename Tuple, int... StateIdx, int... ParamIdx>
  static auto apply_rule_helper(std::shared_ptr<R> r, Integrand& integrand, const TimeInfo& t_info, XType X, 
                                const Tuple& args, std::integer_sequence<int, StateIdx...>, 
                                std::integer_sequence<int, ParamIdx...>)
  {
      // Shift ParamIdx by N? No, ParamIdx is 0..M. We need N+ParamIdx.
      // Or ParamIdx can be N, N+1, ...
      
      // Let's assume the indices passed are absolute indices into the tuple.
      // So ParamIdx needs to be shifted.
      
      return apply_rule_helper_indices(r, integrand, t_info, X, args, 
                                       std::integer_sequence<int, StateIdx...>{},
                                       std::integer_sequence<int, (ParamIdx + sizeof...(StateIdx))...>{});
  }
  
  template <typename R, typename Integrand, typename XType, typename Tuple, int... StateIdx, int... ParamIdx>
  static auto apply_rule_helper_indices(std::shared_ptr<R> r, Integrand& integrand, const TimeInfo& t_info, XType X, 
                                const Tuple& args, std::integer_sequence<int, StateIdx...>, 
                                std::integer_sequence<int, ParamIdx...>)
  {
      auto u = r->value(t_info, std::get<StateIdx>(args)...);
      auto v = r->dot(t_info, std::get<StateIdx>(args)...);
      
      if constexpr (R::num_args == 4) {
          auto a = r->ddot(t_info, std::get<StateIdx>(args)...);
          return integrand(t_info, X, u, v, a, std::get<ParamIdx>(args)...);
      } else {
          return integrand(t_info, X, u, v, std::get<ParamIdx>(args)...);
      }
  }

  /// @overload
  template <typename Integrand, int... all_active_parameters>
  void addBodyIntegralImpl(std::string body_name, Integrand integrand,
                           std::integer_sequence<int, all_active_parameters...>)
  {
      // Pass through dependencies.
      // Note: TimeDiscretizedWeakForm::addBodyIntegral takes DependsOn struct.
      // We are calling Base::addBodyIntegralImpl which takes indices.
      
      // We need to wrap the integrand.
      auto captured_rule = rule;
      
      auto wrapped_integrand = [captured_rule, integrand](const TimeInfo& t_info, auto X, auto... inputs) {
           constexpr int N = Rule::num_args;
           auto args_tuple = std::forward_as_tuple(inputs...);
           return apply_rule_helper(captured_rule, integrand, t_info, X, args_tuple,
                                    std::make_integer_sequence<int, N>{},
                                    std::make_integer_sequence<int, sizeof...(inputs) - N>{});
      };
      
      Base::addBodyIntegralImpl(body_name, wrapped_integrand, std::integer_sequence<int, all_active_parameters...>{});
  }
  
  // Also need to support DependsOn overload if Base has it?
  // Base has:
  // void addBodyIntegral(DependsOn<...>, body_name, integrand)
  // void addBodyIntegralImpl(body_name, integrand, sequence)
  // void addBodyIntegral(body_name, integrand) (defaults to all params)

  template <int... active_parameters, typename Integrand>
  void addBodyIntegral(DependsOn<active_parameters...> depends_on, std::string body_name, Integrand integrand)
  {
      auto captured_rule = rule;
      auto wrapped_integrand = [captured_rule, integrand](const TimeInfo& t_info, auto X, auto... inputs) {
           constexpr int N = Rule::num_args;
           auto args_tuple = std::forward_as_tuple(inputs...);
           return apply_rule_helper(captured_rule, integrand, t_info, X, args_tuple,
                                    std::make_integer_sequence<int, N>{},
                                    std::make_integer_sequence<int, sizeof...(inputs) - N>{});
      };
      Base::addBodyIntegral(depends_on, body_name, wrapped_integrand);
  }

  template <int... active_parameters, typename Integrand>
  void addSurfaceFlux(DependsOn<active_parameters...> depends_on, std::string boundary_name, Integrand integrand)
  {
      auto captured_rule = rule;
      auto wrapped_integrand = [captured_rule, integrand](const TimeInfo& t_info, auto X, auto n, auto... inputs) {
           constexpr int N = Rule::num_args;
           auto args_tuple = std::forward_as_tuple(inputs...);
           return apply_rule_helper_boundary(captured_rule, integrand, t_info, X, n, args_tuple,
                                             std::make_integer_sequence<int, N>{},
                                             std::make_integer_sequence<int, sizeof...(inputs) - N>{});
      };
      Base::addBoundaryFlux(depends_on, boundary_name, wrapped_integrand);
  }

  template <int... active_parameters, typename Integrand>
  void addInteriorSurfaceIntegral(DependsOn<active_parameters...> depends_on, std::string interior_name, Integrand integrand)
  {
      auto captured_rule = rule;
      auto wrapped_integrand = [captured_rule, integrand](const TimeInfo& t_info, auto X, auto... inputs) {
           constexpr int N = Rule::num_args;
           // For interior integrals, the first N inputs are tuples of (minus, plus) for state fields.
           // TimeIntegrationRule needs to be applied to both minus and plus.
           auto args_tuple = std::forward_as_tuple(inputs...);
           return apply_rule_helper_interior(captured_rule, integrand, t_info, X, args_tuple,
                                             std::make_integer_sequence<int, N>{},
                                             std::make_integer_sequence<int, sizeof...(inputs) - N>{});
      };
      Base::addInteriorBoundaryIntegral(depends_on, interior_name, wrapped_integrand);
  }

  // Convenience overloads that default to all parameters
  template <typename Integrand>
  void addSurfaceFlux(std::string boundary_name, Integrand integrand)
  {
      constexpr int num_total_inputs = sizeof...(InputSpaces);
      addSurfaceFlux(std::make_integer_sequence<int, num_total_inputs>{}, boundary_name, integrand);
  }

  template <typename Integrand>
  void addInteriorSurfaceIntegral(std::string interior_name, Integrand integrand)
  {
      constexpr int num_total_inputs = sizeof...(InputSpaces);
      addInteriorSurfaceIntegral(std::make_integer_sequence<int, num_total_inputs>{}, interior_name, integrand);
  }

 private:
  template <typename R, typename Integrand, typename XType, typename nType, typename Tuple, int... StateIdx, int... ParamIdx>
  static auto apply_rule_helper_boundary(std::shared_ptr<R> r, Integrand& integrand, const TimeInfo& t_info, XType X, nType n,
                                const Tuple& args, std::integer_sequence<int, StateIdx...>, 
                                std::integer_sequence<int, ParamIdx...>)
  {
      auto u = r->value(t_info, std::get<StateIdx>(args)...);
      auto v = r->dot(t_info, std::get<StateIdx>(args)...);
      
      if constexpr (R::num_args == 4) {
          auto a = r->ddot(t_info, std::get<StateIdx>(args)...);
          return integrand(t_info, X, n, u, v, a, std::get<(ParamIdx + sizeof...(StateIdx))>(args)...);
      } else {
          return integrand(t_info, X, n, u, v, std::get<(ParamIdx + sizeof...(StateIdx))>(args)...);
      }
  }

  template <typename R, typename Integrand, typename XType, typename Tuple, int... StateIdx, int... ParamIdx>
  static auto apply_rule_helper_interior(std::shared_ptr<R> r, Integrand& integrand, const TimeInfo& t_info, XType X,
                                const Tuple& args, std::integer_sequence<int, StateIdx...>, 
                                std::integer_sequence<int, ParamIdx...>)
  {
      // inputs for interior integrals are tuples of (minus, plus)
      auto u_minus = r->value(t_info, std::get<0>(std::get<StateIdx>(args))...);
      auto u_plus = r->value(t_info, std::get<1>(std::get<StateIdx>(args))...);
      auto v_minus = r->dot(t_info, std::get<0>(std::get<StateIdx>(args))...);
      auto v_plus = r->dot(t_info, std::get<1>(std::get<StateIdx>(args))...);
      
      if constexpr (R::num_args == 4) {
          auto a_minus = r->ddot(t_info, std::get<0>(std::get<StateIdx>(args))...);
          auto a_plus = r->ddot(t_info, std::get<1>(std::get<StateIdx>(args))...);
          return integrand(t_info, X, 
                           smith::tuple{u_minus, u_plus}, 
                           smith::tuple{v_minus, v_plus}, 
                           smith::tuple{a_minus, a_plus}, 
                           std::get<(ParamIdx + sizeof...(StateIdx))>(args)...);
      } else {
          return integrand(t_info, X, 
                           smith::tuple{u_minus, u_plus}, 
                           smith::tuple{v_minus, v_plus}, 
                           std::get<(ParamIdx + sizeof...(StateIdx))>(args)...);
      }
  }

};

}  // namespace smith
