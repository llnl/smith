// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file coupling_params.hpp
 * @brief Coupling pack types and helpers for injecting explicit coupled-physics fields into weak form parameter packs.
 *
 * Builders accept at most two optional trailing arguments after `self_fields`:
 *   1. `couplingFields(foreign_physics_fields...)` — foreign physics contributions
 *   2. `param_fields` (a `CouplingParams<...>`) — user parameter fields (must be last)
 *
 * Tail of each material/source closure: `(coupling_fields..., parameter_fields...)`.
 */

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/system_base.hpp"

namespace smith {

/// Sentinel: no time integration rule (used for parameter-only packs).
struct NoTimeRule {
  static constexpr int num_states = 0;
};

/**
 * @brief Fields returned by a physics register function, carrying time rule type information.
 *
 * Doubles as a per-physics coupling segment: when supplied via `couplingFields(...)`, the
 * builder interpolates `TimeRule::num_states` raw arguments before passing the values to
 * the user callback. Parameter packs use `TimeRule = NoTimeRule`.
 */
template <int Dim, int Order, typename TimeRule, typename... Spaces>
struct PhysicsFields {
  using time_rule_type = TimeRule;
  static constexpr int dim = Dim;
  static constexpr int order = Order;
  static constexpr std::size_t num_fields = sizeof...(Spaces);
  std::shared_ptr<FieldStore> field_store;
  std::tuple<FieldType<Spaces>...> fields;

  PhysicsFields(std::shared_ptr<FieldStore> fs, FieldType<Spaces>... f)
      : field_store(std::move(fs)), fields(std::move(f)...)
  {
  }
};

/**
 * @brief Parameter-only field bundle (no time rule, no field store reference).
 *
 * Distinct type so callers can plain-pass it after `couplingFields(...)`. Otherwise has the
 * same shape as a `PhysicsFields` with `NoTimeRule`.
 */
template <typename... Spaces>
struct CouplingParams {
  using time_rule_type = NoTimeRule;
  static constexpr std::size_t num_fields = sizeof...(Spaces);
  std::tuple<FieldType<Spaces>...> fields;
  CouplingParams(FieldType<Spaces>... fs) : fields(std::move(fs)...) {}
};

template <typename... Spaces>
CouplingParams(FieldType<Spaces>...) -> CouplingParams<Spaces...>;

/**
 * @brief Bundle of foreign `PhysicsFields` packs supplied to a builder as a single coupling arg.
 *
 * Order is preserved. Each entry contributes its fields and an interpolation segment governed
 * by its own `time_rule_type`.
 */
template <typename... PFs>
struct CouplingFields {
  std::tuple<PFs...> packs;
};

template <typename... PFs>
auto couplingFields(const PFs&... pfs)
{
  return CouplingFields<PFs...>{std::make_tuple(pfs...)};
}

/**
 * @brief Register parameter fields as type-level tokens.
 */
template <typename... ParamSpaces>
auto registerParameterFields(FieldType<ParamSpaces>... param_types)
{
  return CouplingParams{std::move(param_types)...};
}

namespace detail {

template <typename T>
struct is_physics_fields_impl : std::false_type {};

template <int D, int O, typename R, typename... S>
struct is_physics_fields_impl<PhysicsFields<D, O, R, S...>> : std::true_type {};

template <typename T>
inline constexpr bool is_physics_fields_v = is_physics_fields_impl<std::decay_t<T>>::value;

template <typename T>
struct is_parameter_pack_impl : std::false_type {};

template <typename... S>
struct is_parameter_pack_impl<CouplingParams<S...>> : std::true_type {};

template <typename T>
inline constexpr bool is_parameter_pack_v = is_parameter_pack_impl<std::decay_t<T>>::value;

template <typename T>
struct is_coupling_fields_impl : std::false_type {};

template <typename... PFs>
struct is_coupling_fields_impl<CouplingFields<PFs...>> : std::true_type {};

template <typename T>
inline constexpr bool is_coupling_fields_v = is_coupling_fields_impl<std::decay_t<T>>::value;

/// True for a `std::tuple<Packs...>` returned by `collectCouplingFields`.
template <typename T>
struct is_coupling_packs_impl : std::false_type {};

template <typename... Packs>
struct is_coupling_packs_impl<std::tuple<Packs...>> : std::true_type {};

template <typename T>
inline constexpr bool is_coupling_packs_v = is_coupling_packs_impl<std::decay_t<T>>::value;

template <typename T, typename = void>
inline constexpr bool has_time_rule_v = false;

template <typename T>
inline constexpr bool has_time_rule_v<T, std::enable_if_t<is_physics_fields_v<T>>> =
    !std::is_same_v<typename std::decay_t<T>::time_rule_type, NoTimeRule>;

/// Trailing args must be one of: {}, {CouplingFields}, {CouplingParams}, {CouplingFields, CouplingParams}.
template <typename... Trailing>
inline constexpr bool trailing_coupling_args_valid_v = [] {
  if constexpr (sizeof...(Trailing) == 0) {
    return true;
  } else if constexpr (sizeof...(Trailing) == 1) {
    using T0 = std::tuple_element_t<0, std::tuple<Trailing...>>;
    return is_coupling_fields_v<T0> || is_parameter_pack_v<T0>;
  } else if constexpr (sizeof...(Trailing) == 2) {
    using T0 = std::tuple_element_t<0, std::tuple<Trailing...>>;
    using T1 = std::tuple_element_t<1, std::tuple<Trailing...>>;
    return is_coupling_fields_v<T0> && is_parameter_pack_v<T1>;
  } else {
    return false;
  }
}();

// -------------------------------------------------------------------------
// Trailing-arg extraction
// -------------------------------------------------------------------------

template <typename... Trailing>
constexpr bool hasCouplingFields()
{
  return ((is_coupling_fields_v<Trailing>) || ...);
}

template <typename... Trailing>
constexpr bool hasParamPack()
{
  return ((is_parameter_pack_v<Trailing>) || ...);
}

inline auto extractCouplingPacks() { return std::tuple<>{}; }

template <typename First, typename... Rest>
auto extractCouplingPacks(const First& first, const Rest&... rest)
{
  if constexpr (is_coupling_fields_v<First>) {
    return first.packs;
  } else {
    return extractCouplingPacks(rest...);
  }
}

template <typename... Spaces>
auto qualifyParams(const std::shared_ptr<FieldStore>& fs, const CouplingParams<Spaces...>& pack)
{
  return std::apply(
      [&](auto... pts) {
        auto qualify = [&](auto pt) {
          pt.name = fs->prefix("param_" + pt.name);
          return pt;
        };
        return CouplingParams<Spaces...>{qualify(pts)...};
      },
      pack.fields);
}

/// Register parameter fields from any trailing CouplingParams into the FieldStore.
template <typename... Trailing>
void registerParamsIfNeeded(std::shared_ptr<FieldStore> fs, const Trailing&... trailing)
{
  auto register_one = [&](const auto& pack) {
    using P = std::decay_t<decltype(pack)>;
    if constexpr (is_parameter_pack_v<P>) {
      std::apply(
          [&](auto... pts) {
            auto prefix_and_add = [&](auto pt) {
              pt.name = "param_" + pt.name;
              fs->addParameter(pt);
            };
            (prefix_and_add(pts), ...);
          },
          pack.fields);
    }
  };
  (register_one(trailing), ...);
}

inline auto extractParamPackQualified(const std::shared_ptr<FieldStore>& /*fs*/) { return std::tuple<>{}; }

template <typename First, typename... Rest>
auto extractParamPackQualified(const std::shared_ptr<FieldStore>& fs, const First& first, const Rest&... rest)
{
  if constexpr (is_parameter_pack_v<First>) {
    return std::make_tuple(qualifyParams(fs, first));
  } else {
    return extractParamPackQualified(fs, rest...);
  }
}

/// Concatenate each pack's `.fields` tuple — used to derive trailing weak-form parameter spaces.
template <typename PacksTuple>
auto flattenCouplingFields(const PacksTuple& packs)
{
  return std::apply([](const auto&... pack) { return std::tuple_cat(pack.fields...); }, packs);
}

template <typename... Trailing>
auto collectCouplingFields(const std::shared_ptr<FieldStore>& fs, const Trailing&... trailing)
{
  auto physics_packs = extractCouplingPacks(trailing...);
  auto param_pack = extractParamPackQualified(fs, trailing...);
  return std::tuple_cat(physics_packs, param_pack);
}

// -------------------------------------------------------------------------
// Time-rule interpolation
// -------------------------------------------------------------------------

template <typename Rule, typename TimeInfoT, typename ArgsTuple, typename Callback, std::size_t... StateIs,
          std::size_t... TailIs>
decltype(auto) applyTimeRuleToPrefixImpl(const Rule& rule, const TimeInfoT& t_info, const ArgsTuple& raw_args,
                                         Callback&& callback, std::index_sequence<StateIs...>,
                                         std::index_sequence<TailIs...>)
{
  auto interpolated = rule.interpolate(t_info, std::get<StateIs>(raw_args)...);
  return std::apply(
      [&](auto&&... values) -> decltype(auto) {
        return std::forward<Callback>(callback)(std::forward<decltype(values)>(values)...,
                                                std::get<Rule::num_states + TailIs>(raw_args)...);
      },
      interpolated);
}

template <typename Rule, typename TimeInfoT, typename Callback, typename... RawArgs>
decltype(auto) applyTimeRuleToPrefix(const Rule& rule, const TimeInfoT& t_info, Callback&& callback,
                                     const RawArgs&... raw_args)
{
  static_assert(sizeof...(RawArgs) >= Rule::num_states, "Not enough raw arguments for time-rule interpolation");
  auto raw_tuple = std::forward_as_tuple(raw_args...);
  constexpr std::size_t tail_count = sizeof...(RawArgs) - Rule::num_states;
  return applyTimeRuleToPrefixImpl(rule, t_info, raw_tuple, std::forward<Callback>(callback),
                                   std::make_index_sequence<Rule::num_states>{},
                                   std::make_index_sequence<tail_count>{});
}

template <std::size_t Offset, typename Pack, typename TimeInfoT, typename RawTuple, std::size_t... Is>
auto evaluateCouplingPack(const Pack& /*pack*/, const TimeInfoT& t_info, const RawTuple& raw_args,
                          std::index_sequence<Is...>)
{
  using Rule = typename Pack::time_rule_type;
  if constexpr (std::is_same_v<Rule, NoTimeRule>) {
    return std::forward_as_tuple(std::get<Offset + Is>(raw_args)...);
  } else {
    Rule rule;
    return rule.interpolate(t_info, std::get<Offset + Is>(raw_args)...);
  }
}

template <std::size_t I, std::size_t Offset, typename PacksTuple, typename TimeInfoT, typename RawTuple>
auto evaluateCouplingPacks(const PacksTuple& packs, const TimeInfoT& t_info, const RawTuple& raw_args)
{
  if constexpr (I == std::tuple_size_v<std::decay_t<PacksTuple>>) {
    return std::tuple{};
  } else {
    const auto& pack = std::get<I>(packs);
    using Pack = std::decay_t<decltype(pack)>;
    auto head = evaluateCouplingPack<Offset>(pack, t_info, raw_args, std::make_index_sequence<Pack::num_fields>{});
    auto tail = evaluateCouplingPacks<I + 1, Offset + Pack::num_fields>(packs, t_info, raw_args);
    return std::tuple_cat(head, tail);
  }
}

template <typename PacksTuple, typename TimeInfoT, typename Callback, typename... RawArgs>
decltype(auto) applyCouplingTimeRules(const PacksTuple& packs, const TimeInfoT& t_info, Callback&& callback,
                                      const RawArgs&... raw_args)
{
  auto raw_tuple = std::forward_as_tuple(raw_args...);
  auto interpolated_tail = evaluateCouplingPacks<0, 0>(packs, t_info, raw_tuple);
  return std::apply(std::forward<Callback>(callback), interpolated_tail);
}

/**
 * @brief Interpolate self time-rule states then coupling segments, then invoke callback.
 *
 * Combines `applyTimeRuleToPrefix` with `applyCouplingTimeRules` into a single helper so
 * per-method weak-form bodies stop repeating the same 3-level nested-lambda boilerplate.
 *
 * Callback signature: `(self_states..., interpolated_coupling...)`.
 */
template <typename Rule, typename Coupling, typename TimeInfoT, typename Callback, typename... RawArgs>
decltype(auto) applyTimeRuleAndCoupling(const Rule& rule, const Coupling& coupling, const TimeInfoT& t_info,
                                        Callback&& callback, const RawArgs&... raw_args)
{
  constexpr std::size_t tail_count = sizeof...(RawArgs) - Rule::num_states;
  return applyTimeRuleToPrefix(
      rule, t_info,
      [&](auto... self_states_and_tail) {
        constexpr std::size_t n_self = sizeof...(self_states_and_tail) - tail_count;
        auto all = std::forward_as_tuple(self_states_and_tail...);
        return [&]<std::size_t... Si, std::size_t... Ti>(std::index_sequence<Si...>, std::index_sequence<Ti...>) {
          return applyCouplingTimeRules(
              coupling, t_info,
              [&](auto... interpolated_coupling) {
                return std::forward<Callback>(callback)(std::get<Si>(all)..., interpolated_coupling...);
              },
              std::get<n_self + Ti>(all)...);
        }(std::make_index_sequence<n_self>{}, std::make_index_sequence<tail_count>{});
      },
      raw_args...);
}

// -------------------------------------------------------------------------
// Type-level coupling-space extraction (used by weak-form parameter type computation)
// -------------------------------------------------------------------------

/// Flatten a `tuple<Packs...>` type into `Parameters<all_pack_spaces...>`.
template <typename PacksTuple>
struct FlattenCoupling;

template <typename... Packs>
struct FlattenCoupling<std::tuple<Packs...>> {
 private:
  template <typename Pack>
  struct pack_spaces;
  template <int D, int O, typename R, typename... S>
  struct pack_spaces<PhysicsFields<D, O, R, S...>> {
    using type = Parameters<S...>;
  };
  template <typename... S>
  struct pack_spaces<CouplingParams<S...>> {
    using type = Parameters<S...>;
  };

  template <typename... Ps>
  struct concat;
  template <>
  struct concat<> {
    using type = Parameters<>;
  };
  template <typename... A>
  struct concat<Parameters<A...>> {
    using type = Parameters<A...>;
  };
  template <typename... A, typename... B, typename... Rest>
  struct concat<Parameters<A...>, Parameters<B...>, Rest...> {
    using type = typename concat<Parameters<A..., B...>, Rest...>::type;
  };

 public:
  using parameters = typename concat<typename pack_spaces<std::decay_t<Packs>>::type...>::type;
};

template <typename PacksTuple>
using flatten_coupling_t = typename FlattenCoupling<std::decay_t<PacksTuple>>::parameters;

/// `Parameters<Space, packed_coupling_spaces...>` for a weak form's parameter list.
template <typename Rule, typename Space, typename PacksTuple>
struct TimeRuleParamsImpl;

template <typename Rule, typename Space, typename... CS>
struct TimeRuleParamsImpl<Rule, Space, Parameters<CS...>> {
  using type = smith::TimeRuleParams<Rule, Space, CS...>;
};

template <typename Rule, typename Space, typename PacksTuple>
using TimeRuleParams = typename TimeRuleParamsImpl<Rule, Space, flatten_coupling_t<PacksTuple>>::type;

template <typename PacksTuple, typename FixedParams>
struct AppendCouplingToParams;

template <typename PacksTuple, typename... Fixed>
struct AppendCouplingToParams<PacksTuple, Parameters<Fixed...>> {
 private:
  template <typename P>
  struct expand;
  template <typename... CS>
  struct expand<Parameters<CS...>> {
    using type = Parameters<Fixed..., CS...>;
  };

 public:
  using type = typename expand<flatten_coupling_t<PacksTuple>>::type;
};

}  // namespace detail

}  // namespace smith
