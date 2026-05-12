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

/**
 * @brief Declares the finite element spaces and field names of parameter fields.
 */
template <typename... Spaces>
struct CouplingParams {
  static constexpr std::size_t num_fields = sizeof...(Spaces);
  static constexpr std::size_t num_coupling_fields = sizeof...(Spaces);
  std::tuple<FieldType<Spaces>...> fields;
  CouplingParams(FieldType<Spaces>... fs) : fields(std::move(fs)...) {}
};

template <typename... Spaces>
CouplingParams(FieldType<Spaces>...) -> CouplingParams<Spaces...>;

/// Sentinel: no time integration rule (used for parameter-only packs).
struct NoTimeRule {
  static constexpr int num_states = 0;
};

/**
 * @brief Fields returned by a physics register function, carrying time rule type information.
 *
 * Doubles as a per-physics coupling segment: when supplied via `couplingFields(...)`, the
 * builder interpolates `TimeRule::num_states` raw arguments before passing the values to
 * the user callback.
 */
template <typename TimeRule, typename... Spaces>
struct PhysicsFields {
  using time_rule_type = TimeRule;
  static constexpr std::size_t num_rule_states = TimeRule::num_states;
  static constexpr std::size_t num_fields = sizeof...(Spaces);
  static constexpr std::size_t num_coupling_fields = sizeof...(Spaces);
  std::shared_ptr<FieldStore> field_store;
  std::tuple<FieldType<Spaces>...> fields;

  PhysicsFields(std::shared_ptr<FieldStore> fs, FieldType<Spaces>... f)
      : field_store(std::move(fs)), fields(std::move(f)...)
  {
  }
};

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
 * @brief Aggregate of caller-ordered segments plus a flat field tuple.
 *
 * `segments` holds the foreign `PhysicsFields<...>` in caller order, optionally followed by a
 * single (name-qualified) `CouplingParams<...>`. `fields` is the concatenation of each segment's
 * `fields`, used directly as trailing weak-form parameter spaces.
 */
template <typename FlatParams, typename... Segments>
struct CouplingDescriptor;

template <typename... FlatSpaces, typename... Segments>
struct CouplingDescriptor<Parameters<FlatSpaces...>, Segments...> {
  static constexpr std::size_t num_coupling_fields = sizeof...(FlatSpaces);
  std::tuple<FieldType<FlatSpaces>...> fields;
  std::tuple<Segments...> segments;
};

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

template <typename R, typename... S>
struct is_physics_fields_impl<PhysicsFields<R, S...>> : std::true_type {};

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

template <typename T>
struct is_coupling_descriptor_impl : std::false_type {};

template <typename FlatParams, typename... Segs>
struct is_coupling_descriptor_impl<CouplingDescriptor<FlatParams, Segs...>> : std::true_type {};

template <typename T>
inline constexpr bool is_coupling_params_v = is_coupling_descriptor_impl<std::decay_t<T>>::value;

template <typename T, typename = void>
inline constexpr bool has_time_rule_v = false;

template <typename T>
inline constexpr bool has_time_rule_v<T, std::enable_if_t<is_physics_fields_v<T>>> =
    !std::is_same_v<typename std::decay_t<T>::time_rule_type, NoTimeRule>;

template <typename T>
inline constexpr bool is_field_store_ptr_v = std::is_same_v<std::decay_t<T>, std::shared_ptr<FieldStore>>;

template <typename T>
inline constexpr bool is_mesh_ptr_v = std::is_same_v<std::decay_t<T>, std::shared_ptr<Mesh>>;

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

template <typename T>
auto getOrCreateFieldStore(T source, std::string prefix = "", size_t storage_size = 100)
{
  if constexpr (is_field_store_ptr_v<T>) {
    return source;
  } else {
    return std::make_shared<FieldStore>(source, storage_size, prefix);
  }
}

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

template <typename... Segs>
auto makeCouplingDescriptor(std::tuple<Segs...> segments)
{
  auto flat = std::apply([](const auto&... s) { return std::tuple_cat(s.fields...); }, segments);
  return std::apply(
      [&](auto... field) {
        return CouplingDescriptor<Parameters<typename decltype(field)::space_type...>, Segs...>{flat, segments};
      },
      flat);
}

template <typename... Trailing>
auto collectCouplingFields(const std::shared_ptr<FieldStore>& fs, const Trailing&... trailing)
{
  auto physics_segments = extractCouplingPacks(trailing...);
  auto param_segment = extractParamPackQualified(fs, trailing...);
  return makeCouplingDescriptor(std::tuple_cat(physics_segments, param_segment));
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

template <std::size_t Offset, typename Rule, typename... Spaces, typename TimeInfoT, typename RawTuple,
          std::size_t... Is>
auto evaluateCouplingSegment(const PhysicsFields<Rule, Spaces...>& /*seg*/, const TimeInfoT& t_info,
                             const RawTuple& raw_args, std::index_sequence<Is...>)
{
  Rule rule;
  return rule.interpolate(t_info, std::get<Offset + Is>(raw_args)...);
}

template <std::size_t Offset, typename... Spaces, typename TimeInfoT, typename RawTuple, std::size_t... Is>
auto evaluateCouplingSegment(const CouplingParams<Spaces...>& /*seg*/, const TimeInfoT& /*t_info*/,
                             const RawTuple& raw_args, std::index_sequence<Is...>)
{
  return std::forward_as_tuple(std::get<Offset + Is>(raw_args)...);
}

template <std::size_t I, std::size_t Offset, typename SegmentsTuple, typename TimeInfoT, typename RawTuple>
auto evaluateCouplingSegments(const SegmentsTuple& segments, const TimeInfoT& t_info, const RawTuple& raw_args)
{
  if constexpr (I == std::tuple_size_v<std::decay_t<SegmentsTuple>>) {
    return std::tuple{};
  } else {
    const auto& seg = std::get<I>(segments);
    using Seg = std::decay_t<decltype(seg)>;
    auto head = evaluateCouplingSegment<Offset>(seg, t_info, raw_args, std::make_index_sequence<Seg::num_fields>{});
    auto tail = evaluateCouplingSegments<I + 1, Offset + Seg::num_fields>(segments, t_info, raw_args);
    return std::tuple_cat(head, tail);
  }
}

template <typename Coupling, typename TimeInfoT, typename Callback, typename... RawArgs>
decltype(auto) applyCouplingTimeRules(const Coupling& coupling, const TimeInfoT& t_info, Callback&& callback,
                                      const RawArgs&... raw_args)
{
  auto raw_tuple = std::forward_as_tuple(raw_args...);
  auto interpolated_tail = evaluateCouplingSegments<0, 0>(coupling.segments, t_info, raw_tuple);
  return std::apply(std::forward<Callback>(callback), interpolated_tail);
}

// -------------------------------------------------------------------------
// Type-level coupling-space extraction (used by weak-form parameter type computation)
// -------------------------------------------------------------------------

template <typename Coupling>
struct CouplingSpaces;

template <typename... CS, typename... Segs>
struct CouplingSpaces<CouplingDescriptor<Parameters<CS...>, Segs...>> {
  template <typename Rule, typename Space>
  using time_rule_params = smith::TimeRuleParams<Rule, Space, CS...>;

  template <typename... Fixed>
  using append_to_parameters = Parameters<Fixed..., CS...>;
};

template <typename Rule, typename Space, typename Coupling>
using TimeRuleParams = typename CouplingSpaces<std::decay_t<Coupling>>::template time_rule_params<Rule, Space>;

template <typename Coupling, typename FixedParams>
struct AppendCouplingToParams;

template <typename Coupling, typename... Fixed>
struct AppendCouplingToParams<Coupling, Parameters<Fixed...>> {
  using type = typename CouplingSpaces<std::decay_t<Coupling>>::template append_to_parameters<Fixed...>;
};

}  // namespace detail

}  // namespace smith
