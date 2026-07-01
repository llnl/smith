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
 *   1. `couplingFields(coupled_physics_fields...)` — coupled physics contributions
 *   2. `param_fields` (a `ParamFields<...>`) — registered user parameter fields (must be last)
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
 * @brief Fields returned by a physics register function, carrying time rule type information.
 *
 * Doubles as a per-physics coupling segment: when supplied via `couplingFields(...)`, the
 * builder interpolates `TimeRule::num_states` raw arguments before passing the values to
 * the user callback.
 */
template <int Dim, int Order, typename TimeRule, typename... Spaces>
struct PhysicsFields {
  using time_rule_type = TimeRule;                              ///< The time integration rule type.
  static constexpr int dim = Dim;                               ///< Spatial dimension.
  static constexpr int order = Order;                           ///< Spatial order.
  static constexpr std::size_t num_fields = sizeof...(Spaces);  ///< Number of fields.
  std::shared_ptr<FieldStore> field_store;                      ///< Pointer to the field store.
  std::tuple<FieldType<Spaces>...> fields;                      ///< The fields.

  /// Constructor
  PhysicsFields(std::shared_ptr<FieldStore> fs, FieldType<Spaces>... f)
      : field_store(std::move(fs)), fields(std::move(f)...)
  {
  }
};

/**
 * @brief Registered parameter-only field bundle.
 */
template <typename... Spaces>
struct ParamFields {
  static constexpr std::size_t num_fields = sizeof...(Spaces);  ///< Number of fields.
  std::tuple<FieldType<Spaces>...> fields;                      ///< The fields.
  /// Constructor
  ParamFields(FieldType<Spaces>... fs) : fields(std::move(fs)...) {}
};

/// Deduction guide for ParamFields
template <typename... Spaces>
ParamFields(FieldType<Spaces>...) -> ParamFields<Spaces...>;

/**
 * @brief Bundle of coupled `PhysicsFields` packs supplied to a builder as a single coupling arg.
 *
 * Order is preserved. Each entry contributes its fields and an interpolation segment governed
 * by its own `time_rule_type`.
 */
template <typename... PFs>
struct CouplingFields {
  std::tuple<PFs...> packs;  ///< The coupling packs.
};

namespace detail {

template <typename T>
struct is_physics_fields_impl : std::false_type {};

template <int D, int O, typename R, typename... S>
struct is_physics_fields_impl<PhysicsFields<D, O, R, S...>> : std::true_type {};

/// True when T is a PhysicsFields pack.
template <typename T>
inline constexpr bool is_physics_fields_v = is_physics_fields_impl<std::decay_t<T>>::value;

}  // namespace detail

/// Helper to construct a CouplingFields bundle
template <typename... PFs>
auto couplingFields(const PFs&... pfs)
{
  static_assert((detail::is_physics_fields_v<PFs> && ...), "couplingFields(...) only accepts PhysicsFields packs");
  return CouplingFields<PFs...>{std::make_tuple(pfs...)};
}

/**
 * @brief Register parameter fields as type-level tokens.
 */
template <typename... ParamSpaces>
auto registerParameterFields(const std::shared_ptr<FieldStore>& field_store, FieldType<ParamSpaces>... param_types)
{
  auto register_one = [&](auto param_type) {
    param_type.name = "param_" + param_type.name;
    field_store->addParameter(param_type);
    return param_type;
  };
  return ParamFields<ParamSpaces...>{register_one(std::move(param_types))...};
}

namespace detail {

template <typename T>
struct is_parameter_pack_impl : std::false_type {};

template <typename... S>
struct is_parameter_pack_impl<ParamFields<S...>> : std::true_type {};

/// @brief True if T is a ParamFields type.
template <typename T>
inline constexpr bool is_parameter_pack_v = is_parameter_pack_impl<std::decay_t<T>>::value;

template <typename T>
struct is_coupling_fields_impl : std::false_type {};

template <typename... PFs>
struct is_coupling_fields_impl<CouplingFields<PFs...>> : std::true_type {};

/// @brief True if T is a CouplingFields type.
template <typename T>
inline constexpr bool is_coupling_fields_v = is_coupling_fields_impl<std::decay_t<T>>::value;

/// True for a `std::tuple<Packs...>` returned by `collectCouplingFields`.
template <typename T>
struct is_coupling_packs_impl : std::false_type {};

template <typename... Packs>
struct is_coupling_packs_impl<std::tuple<Packs...>> : std::true_type {};

/// @brief True if T is a tuple of coupling packs.
template <typename T>
inline constexpr bool is_coupling_packs_v = is_coupling_packs_impl<std::decay_t<T>>::value;

/// @brief Base case: T does not have a time rule.
template <typename T, typename = void>
inline constexpr bool has_time_rule_v = false;

/// @brief True if T is a PhysicsFields type.
template <typename T>
inline constexpr bool has_time_rule_v<T, std::enable_if_t<is_physics_fields_v<T>>> = true;

// -------------------------------------------------------------------------
// Trailing-arg extraction
// -------------------------------------------------------------------------

/// Concatenate each pack's `.fields` tuple — used to derive trailing weak-form parameter spaces.
template <typename PacksTuple>
auto flattenCouplingFields(const PacksTuple& packs)
{
  return std::apply([](const auto&... pack) { return std::tuple_cat(pack.fields...); }, packs);
}

/// @brief Collect no coupling or parameter packs.
inline auto collectCouplingFields() { return std::tuple<>{}; }

template <typename... PFs>
/// @brief Collect only coupled physics packs.
auto collectCouplingFields(const CouplingFields<PFs...>& coupled)
{
  return coupled.packs;
}

template <typename... Spaces>
/// @brief Collect only registered parameter fields.
auto collectCouplingFields(const ParamFields<Spaces...>& params)
{
  return std::make_tuple(params);
}

template <typename... PFs, typename... Spaces>
/// @brief Collect coupled physics packs followed by registered parameter fields.
auto collectCouplingFields(const CouplingFields<PFs...>& coupled, const ParamFields<Spaces...>& params)
{
  return std::tuple_cat(coupled.packs, std::make_tuple(params));
}

// -------------------------------------------------------------------------
// Time-rule interpolation
// -------------------------------------------------------------------------

/// @brief Interpolate the leading time-rule state arguments.
template <typename Rule, typename TimeInfoT, typename ArgsTuple, std::size_t... StateIs>
auto interpolateTimeRulePrefix(const Rule& rule, const TimeInfoT& t_info, const ArgsTuple& raw_args,
                               std::index_sequence<StateIs...>)
{
  return rule.interpolate(t_info, std::get<StateIs>(raw_args)...);
}

/// @brief Select a tuple of raw arguments starting at Offset.
template <std::size_t Offset, typename ArgsTuple, std::size_t... Is>
auto selectArgsFrom(const ArgsTuple& raw_args, std::index_sequence<Is...>)
{
  return std::forward_as_tuple(std::get<Offset + Is>(raw_args)...);
}

/// @brief Evaluate a single coupling pack's time rule.
template <std::size_t Offset, typename Pack, typename TimeInfoT, typename RawTuple, std::size_t... Is>
auto evaluateCouplingPack(const Pack& /*pack*/, const TimeInfoT& t_info, const RawTuple& raw_args,
                          std::index_sequence<Is...>)
{
  if constexpr (is_physics_fields_v<Pack>) {
    using Rule = typename Pack::time_rule_type;
    Rule rule;
    return rule.interpolate(t_info, std::get<Offset + Is>(raw_args)...);
  } else {
    return std::forward_as_tuple(std::get<Offset + Is>(raw_args)...);
  }
}

/// @brief Evaluate all coupling packs over their corresponding raw arguments.
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

/// @brief Interpolate coupling packs and invoke the callback.
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
 * Callback signature: `(self_states..., interpolated_coupling...)`.
 *
 * Raw arguments start with the self time-rule states. Remaining arguments are split by
 * coupling pack, and each pack's time rule interpolates its own segment.
 */
template <typename Rule, typename Coupling, typename TimeInfoT, typename Callback, typename... RawArgs>
decltype(auto) applyTimeRuleAndCoupling(const Rule& rule, const Coupling& coupling, const TimeInfoT& t_info,
                                        Callback&& callback, const RawArgs&... raw_args)
{
  static_assert(sizeof...(RawArgs) >= Rule::num_states, "Not enough raw arguments for time-rule interpolation");
  constexpr std::size_t tail_count = sizeof...(RawArgs) - Rule::num_states;
  auto raw_tuple = std::forward_as_tuple(raw_args...);
  auto self_states = interpolateTimeRulePrefix(rule, t_info, raw_tuple, std::make_index_sequence<Rule::num_states>{});
  auto coupling_raw_args = selectArgsFrom<Rule::num_states>(raw_tuple, std::make_index_sequence<tail_count>{});
  auto coupling_states = evaluateCouplingPacks<0, 0>(coupling, t_info, coupling_raw_args);
  auto callback_args = std::tuple_cat(self_states, coupling_states);
  return std::apply(std::forward<Callback>(callback), callback_args);
}

// -------------------------------------------------------------------------
// Type-level coupling-space extraction (used by weak-form parameter type computation)
// -------------------------------------------------------------------------

/// @brief Flatten a `tuple<Packs...>` type into `Parameters<all_pack_spaces...>`.
template <typename PacksTuple>
struct FlattenCoupling;

/// @brief Converts a `std::tuple<...>` of spaces into `Parameters<...>`.
template <typename Tuple>
struct TupleToParameters;

/// @brief Specialization of TupleToParameters for a tuple of spaces.
template <typename... Spaces>
struct TupleToParameters<std::tuple<Spaces...>> {
  using type = Parameters<Spaces...>;  ///< The converted parameter pack.
};

/// @brief Typedef for converting a tuple of spaces into `Parameters<...>`.
template <typename Tuple>
using tuple_to_parameters_t = typename TupleToParameters<std::decay_t<Tuple>>::type;

/// @brief Maps a coupling pack type to a `std::tuple<...>` of its spaces.
template <typename Pack>
struct pack_tuple;

/// @brief Specialization of pack_tuple for physics coupling fields.
template <int D, int O, typename R, typename... Spaces>
struct pack_tuple<PhysicsFields<D, O, R, Spaces...>> {
  using type = std::tuple<Spaces...>;  ///< The coupling spaces as a tuple.
};

/// @brief Specialization of pack_tuple for parameter-only fields.
template <typename... Spaces>
struct pack_tuple<ParamFields<Spaces...>> {
  using type = std::tuple<Spaces...>;  ///< The parameter spaces as a tuple.
};

/// @brief Typedef for extracting a tuple of spaces from a coupling pack.
template <typename Pack>
using pack_tuple_t = typename pack_tuple<std::decay_t<Pack>>::type;

/// @brief Typedef for concatenating space tuples with `std::tuple_cat`.
template <typename... Tuples>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Tuples>()...));

/// @brief Appends coupling parameter spaces to an existing `Parameters<...>` list.
template <typename CouplingParams, typename FixedParams>
struct AppendParameters;

/// @brief Specialization of AppendParameters for two `Parameters<...>` packs.
template <typename... Coupled, typename... Fixed>
struct AppendParameters<Parameters<Coupled...>, Parameters<Fixed...>> {
  using type = Parameters<Fixed..., Coupled...>;  ///< The appended parameter list.
};

/// @brief Specialization of FlattenCoupling for a tuple of coupling packs.
template <typename... Packs>
struct FlattenCoupling<std::tuple<Packs...>> {
 public:
  using tuple_type = tuple_cat_t<pack_tuple_t<Packs>...>;  ///< The flattened tuple of coupling spaces.
  using parameters = tuple_to_parameters_t<tuple_type>;    ///< Flattened parameter spaces.
};

/// @brief Typedef for flattened coupling parameter spaces.
template <typename PacksTuple>
using flatten_coupling_t = typename FlattenCoupling<std::decay_t<PacksTuple>>::parameters;

/// @brief Type trait to construct `Parameters<Space, packed_coupling_spaces...>` for a weak form's parameter list.
template <typename Rule, typename Space, typename PacksTuple>
struct TimeRuleParamsImpl;

/// @brief Specialization of TimeRuleParamsImpl.
template <typename Rule, typename Space, typename... CS>
struct TimeRuleParamsImpl<Rule, Space, Parameters<CS...>> {
  using type = smith::TimeRuleParams<Rule, Space, CS...>;  ///< The constructed TimeRuleParams type.
};

/// @brief Typedef for TimeRuleParams.
template <typename Rule, typename Space, typename PacksTuple>
using TimeRuleParams = typename TimeRuleParamsImpl<Rule, Space, flatten_coupling_t<PacksTuple>>::type;

/// @brief Type trait to append coupling parameter spaces to fixed parameters.
template <typename PacksTuple, typename FixedParams>
struct AppendCouplingToParams;

/// @brief Specialization of AppendCouplingToParams.
template <typename PacksTuple, typename... Fixed>
struct AppendCouplingToParams<PacksTuple, Parameters<Fixed...>> {
  using type = typename AppendParameters<flatten_coupling_t<PacksTuple>, Parameters<Fixed...>>::type;
  ///< The appended parameter list.
};

}  // namespace detail

}  // namespace smith
