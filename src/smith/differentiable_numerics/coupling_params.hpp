// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file coupling_params.hpp
 * @brief CouplingParams type and helpers for injecting coupled-physics fields into weak form parameter packs.
 *
 * Convention: coupling fields occupy the *leading* positions of the "tail" parameter pack in every
 * weak form constructed with a non-empty CouplingParams.  Concretely, after the time-rule state fields
 * (e.g. u, u_old, v_old, a_old for solid) come the coupling fields in the order declared in
 * CouplingParams::fields, and only then come the user-supplied parameter_space fields.
 *
 * This ordering must be respected in every setMaterial / addBodyForce / addTraction / addPressure
 * closure: the `auto...` tail pack is partitioned as (coupling_fields..., user_params...).
 */

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/system_base.hpp"

namespace smith {

/**
 * @brief Declares the finite element spaces and field names of fields borrowed from another physics.
 *
 * @tparam Spaces  FE space types of the coupling fields (e.g. H1<temp_order>, H1<temp_order>).
 *
 * Usage:
 * @code
 *   CouplingParams solid_coupling{FieldType<H1<temp_order>>("temperature"),
 *                               FieldType<H1<temp_order>>("temperature_old")};
 * @endcode
 *
 * The default CouplingParams<> (empty) leaves weak form parameter packs unchanged.
 */
template <typename... Spaces>
struct CouplingParams {
  static constexpr std::size_t num_coupling_fields = sizeof...(Spaces);  ///< Number of borrowed or parameter fields.
  std::tuple<FieldType<Spaces>...> fields;  ///< Coupling field descriptors in weak-form argument order.
  /// @brief Construct a coupling pack from field descriptors.
  CouplingParams(FieldType<Spaces>... fs) : fields(std::move(fs)...) {}
};

/**
 * @brief Deduction guide for `CouplingParams`.
 *
 * Example:
 * @code
 *   CouplingParams{FieldType<A>("a"), FieldType<B>("b")}
 * @endcode
 * yields `CouplingParams<A, B>`.
 */
template <typename... Spaces>
CouplingParams(FieldType<Spaces>...) -> CouplingParams<Spaces...>;

/// Sentinel: no time integration rule (used for parameter-only packs).
struct NoTimeRule {
  static constexpr int num_states = 0;  ///< Number of time states contributed by this sentinel rule.
};

/**
 * @brief Fields returned by a physics register function, carrying time rule type information.
 *
 * Unlike CouplingParams, PhysicsFields knows which time integration rule governs its fields.
 * This lets variadic build functions deduce which pack is "self" vs coupling, and enables
 * compile-time interpolation of coupling fields in traction/body force wrappers.
 *
 * @tparam TimeRule  The time integration rule type (e.g. QuasiStaticSecondOrderTimeIntegrationRule).
 * @tparam Spaces    FE space types of the fields (e.g. H1<order, dim> repeated num_states times).
 */
template <typename TimeRule, typename... Spaces>
struct PhysicsFields {
  using time_rule_type = TimeRule;  ///< Time integration rule governing these fields.
  static constexpr std::size_t num_rule_states = TimeRule::num_states;   ///< Number of state slots from `TimeRule`.
  static constexpr std::size_t num_fields = sizeof...(Spaces);           ///< Total number of exported fields.
  static constexpr std::size_t num_coupling_fields = sizeof...(Spaces);  ///< Number of fields exposed for coupling.
  std::shared_ptr<FieldStore> field_store;                               ///< Store owning the registered fields.
  std::tuple<FieldType<Spaces>...> fields;  ///< Exported field descriptors in registration order.

  /// @brief Construct a registered-physics field pack.
  PhysicsFields(std::shared_ptr<FieldStore> fs, FieldType<Spaces>... f)
      : field_store(std::move(fs)), fields(std::move(f)...)
  {
  }
};

template <typename TimeRule, typename... Spaces>
struct PhysicsCouplingSegment {
  using time_rule_type = TimeRule;
  static constexpr std::size_t num_fields = sizeof...(Spaces);
  std::tuple<FieldType<Spaces>...> fields;
};

template <typename... Spaces>
struct ParameterCouplingSegment {
  static constexpr std::size_t num_fields = sizeof...(Spaces);
  std::tuple<FieldType<Spaces>...> fields;
};

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
 *
 * Actual FieldStore registration is deferred to the build function.
 * Returns a CouplingParams carrying the parameter field types.
 */
template <typename... ParamSpaces>
auto registerParameterFields(FieldType<ParamSpaces>... param_types)
{
  return CouplingParams{std::move(param_types)...};
}

namespace detail {

/// Type trait: true if T is a CouplingParams<...> or PhysicsFields<...> specialization.
/// Both carry a `fields` tuple and can be used as coupling input to build functions.
template <typename T>
struct is_coupling_params_impl : std::false_type {};

template <typename... Spaces>
struct is_coupling_params_impl<CouplingParams<Spaces...>> : std::true_type {};

template <typename R, typename... Spaces>
struct is_coupling_params_impl<PhysicsFields<R, Spaces...>> : std::true_type {};

template <typename FlatParams, typename... Segments>
struct is_coupling_params_impl<CouplingDescriptor<FlatParams, Segments...>> : std::true_type {};

template <typename T>
inline constexpr bool is_coupling_params_v =
    is_coupling_params_impl<std::decay_t<T>>::value;  ///< True for `CouplingParams` and `PhysicsFields`.

/// Type trait: true if T is a PhysicsFields<...> specialization.
template <typename T>
struct is_physics_fields_impl : std::false_type {};

template <typename R, typename... S>
struct is_physics_fields_impl<PhysicsFields<R, S...>> : std::true_type {};

template <typename T>
inline constexpr bool is_physics_fields_v =
    is_physics_fields_impl<std::decay_t<T>>::value;  ///< True only for `PhysicsFields`.

/// True if T is a PhysicsFields with a real time rule (not NoTimeRule).
template <typename T, typename = void>
inline constexpr bool has_time_rule_v = false;

template <typename T>
inline constexpr bool has_time_rule_v<T, std::enable_if_t<is_physics_fields_v<T>>> =
    !std::is_same_v<typename std::decay_t<T>::time_rule_type, NoTimeRule>;

template <typename T>
inline constexpr bool is_field_store_ptr_v = std::is_same_v<std::decay_t<T>, std::shared_ptr<FieldStore>>;

template <typename T>
inline constexpr bool is_mesh_ptr_v = std::is_same_v<std::decay_t<T>, std::shared_ptr<Mesh>>;

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
// Helpers for variadic build functions
// -------------------------------------------------------------------------

template <typename TargetRule, typename Pack>
auto collectPhysicsSegmentFromPack(const Pack& pack)
{
  if constexpr (is_physics_fields_v<Pack>) {
    if constexpr (std::is_same_v<typename std::decay_t<Pack>::time_rule_type, TargetRule>) {
      return std::tuple{};  // skip self
    } else {
      using Rule = typename std::decay_t<Pack>::time_rule_type;
      return std::apply(
          [](auto... fields) {
            return std::make_tuple(
                PhysicsCouplingSegment<Rule, typename decltype(fields)::space_type...>{std::make_tuple(fields...)});
          },
          pack.fields);
    }
  } else {
    return std::tuple{};  // skip non-physics packs
  }
}

template <typename Pack>
auto collectParamSegmentFromPack(const std::shared_ptr<FieldStore>& fs, const Pack& pack)
{
  if constexpr (is_coupling_params_v<std::decay_t<Pack>> && !is_physics_fields_v<Pack>) {
    return std::apply(
        [&](auto... pts) {
          auto qualify = [&](auto pt) {
            pt.name = fs->prefix("param_" + pt.name);
            return pt;
          };
          return std::make_tuple(
              ParameterCouplingSegment<typename decltype(pts)::space_type...>{std::make_tuple(qualify(pts)...)});
        },
        pack.fields);
  } else {
    return std::tuple{};
  }
}

template <typename... Segments>
auto makeCouplingDescriptor(std::tuple<Segments...> segments)
{
  auto fields = std::apply([](const auto&... segment) { return std::tuple_cat(segment.fields...); }, segments);
  return std::apply(
      [&](auto... field) {
        return CouplingDescriptor<Parameters<typename decltype(field)::space_type...>, Segments...>{fields, segments};
      },
      fields);
}

template <typename TargetRule, typename... Packs>
auto collectCouplingFields(const std::shared_ptr<FieldStore>& fs, const Packs&... packs)
{
  auto physics_segments = std::tuple_cat(collectPhysicsSegmentFromPack<TargetRule>(packs)...);
  auto param_segments = std::tuple_cat(collectParamSegmentFromPack(fs, packs)...);
  return makeCouplingDescriptor(std::tuple_cat(physics_segments, param_segments));
}

/// Register parameter fields from a CouplingParams pack (not PhysicsFields) into a FieldStore.
template <typename Pack>
void registerParamsIfNeeded(std::shared_ptr<FieldStore> fs, const Pack& pack)
{
  if constexpr (is_coupling_params_v<std::decay_t<Pack>> && !is_physics_fields_v<Pack>) {
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
}

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

/// @brief Interpolate the leading Rule::num_states arguments, then pass interpolated values and the raw tail to
/// callback.
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

template <std::size_t Offset, typename Rule, typename TimeInfoT, typename RawTuple, typename... Spaces,
          std::size_t... Is>
auto evaluateCouplingSegment(const PhysicsCouplingSegment<Rule, Spaces...>& /*segment*/, const TimeInfoT& t_info,
                             const RawTuple& raw_args, std::index_sequence<Is...>)
{
  Rule rule;
  return rule.interpolate(t_info, std::get<Offset + Is>(raw_args)...);
}

template <std::size_t Offset, typename TimeInfoT, typename RawTuple, typename... Spaces, std::size_t... Is>
auto evaluateCouplingSegment(const ParameterCouplingSegment<Spaces...>& /*segment*/, const TimeInfoT& /*t_info*/,
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
    const auto& segment = std::get<I>(segments);
    using Segment = std::decay_t<decltype(segment)>;
    auto head =
        evaluateCouplingSegment<Offset>(segment, t_info, raw_args, std::make_index_sequence<Segment::num_fields>{});
    auto tail = evaluateCouplingSegments<I + 1, Offset + Segment::num_fields>(segments, t_info, raw_args);
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

/// @brief Centralized type access for field-pack spaces.
template <typename Pack>
struct CouplingSpaces;

template <typename... CS>
struct CouplingSpaces<CouplingParams<CS...>> {
  template <typename Rule, typename Space>
  using time_rule_params = smith::TimeRuleParams<Rule, Space, CS...>;

  template <typename... Fixed>
  using append_to_parameters = Parameters<Fixed..., CS...>;
};

template <typename R, typename... CS>
struct CouplingSpaces<PhysicsFields<R, CS...>> {
  template <typename Rule, typename Space>
  using time_rule_params = smith::TimeRuleParams<Rule, Space, CS...>;

  template <typename... Fixed>
  using append_to_parameters = Parameters<Fixed..., CS...>;
};

template <typename... CS, typename... Segments>
struct CouplingSpaces<CouplingDescriptor<Parameters<CS...>, Segments...>> {
  template <typename Rule, typename Space>
  using time_rule_params = smith::TimeRuleParams<Rule, Space, CS...>;

  template <typename... Fixed>
  using append_to_parameters = Parameters<Fixed..., CS...>;
};

/// @brief Weak-form parameters for a self time rule followed by the coupling field spaces.
template <typename Rule, typename Space, typename Coupling>
using TimeRuleParams = typename CouplingSpaces<std::decay_t<Coupling>>::template time_rule_params<Rule, Space>;

/// @brief Append coupling spaces onto a base Parameters<Fixed...> type.
template <typename Coupling, typename FixedParams>
struct AppendCouplingToParams;

template <typename Coupling, typename... Fixed>
struct AppendCouplingToParams<Coupling, Parameters<Fixed...>> {
  using type = typename CouplingSpaces<std::decay_t<Coupling>>::template append_to_parameters<Fixed...>;
};

}  // namespace detail

}  // namespace smith
