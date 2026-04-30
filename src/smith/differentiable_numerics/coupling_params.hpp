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
/// True if `T` is `PhysicsFields<...>` and its time rule is not `NoTimeRule`.
inline constexpr bool has_time_rule_v<T, std::enable_if_t<is_physics_fields_v<T>>> =
    !std::is_same_v<typename std::decay_t<T>::time_rule_type, NoTimeRule>;

// -------------------------------------------------------------------------
// Helpers for variadic build functions
// -------------------------------------------------------------------------

/**
 * @brief selects foreign `PhysicsFields` (skip self by rule match); keeps field names.
 */
template <typename TargetRule, typename Pack>
auto collectPhysicsFromPack(const Pack& pack)
{
  if constexpr (is_physics_fields_v<Pack>) {
    if constexpr (std::is_same_v<typename std::decay_t<Pack>::time_rule_type, TargetRule>) {
      return std::tuple{};  // skip self
    } else {
      return pack.fields;  // include coupling fields
    }
  } else {
    return std::tuple{};  // skip non-physics packs
  }
}

/**
 * @brief selects pure `CouplingParams` (not `PhysicsFields`); rewrites names to `{prefix}param_{name}`.
 */
template <typename Pack>
auto collectParamsFromPack(const std::shared_ptr<FieldStore>& fs, const Pack& pack)
{
  if constexpr (is_coupling_params_v<std::decay_t<Pack>> && !is_physics_fields_v<Pack>) {
    return std::apply(
        [&](auto... pts) {
          auto qualify = [&](auto pt) {
            pt.name = fs->prefix("param_" + pt.name);
            return pt;
          };
          return std::make_tuple(qualify(pts)...);
        },
        pack.fields);
  } else {
    return std::tuple{};
  }
}

/**
 * @brief concatenates physics then params into a single `CouplingParams`.
 */
template <typename TargetRule, typename... Packs>
auto collectCouplingFields(const std::shared_ptr<FieldStore>& fs, const Packs&... packs)
{
  auto physics = std::tuple_cat(collectPhysicsFromPack<TargetRule>(packs)...);
  auto params = std::tuple_cat(collectParamsFromPack(fs, packs)...);
  auto combined = std::tuple_cat(physics, params);
  return std::apply([](auto&... all) { return CouplingParams{all...}; }, combined);
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

/// @brief Build the weak-form parameter list for a time rule and coupling pack.
///
/// Unpacks Coupling (either CouplingParams<CS...> or PhysicsFields<R, CS...>) and produces
/// TimeRuleParams<Rule, Space, CS...>, i.e. num_states copies of Space followed by the coupling field types.
template <typename Rule, typename Space, typename Coupling>
struct TimeRuleParamsHelper;

/// @brief Maps `CouplingParams` packs to weak-form time-rule parameters.
template <typename Rule, typename Space, typename... CS>
struct TimeRuleParamsHelper<Rule, Space, CouplingParams<CS...>> {
  using type = smith::TimeRuleParams<Rule, Space, CS...>;  ///< Resolved weak-form time-rule parameter list.
};

/// @brief Maps `PhysicsFields` packs to weak-form time-rule parameters.
template <typename Rule, typename Space, typename R, typename... CS>
struct TimeRuleParamsHelper<Rule, Space, PhysicsFields<R, CS...>> {
  using type = smith::TimeRuleParams<Rule, Space, CS...>;  ///< Resolved weak-form time-rule parameter list.
};

/// @brief Convenience alias selecting time-rule parameters for coupling pack type.
template <typename Rule, typename Space, typename Coupling>
using TimeRuleParams = typename TimeRuleParamsHelper<Rule, Space, Coupling>::type;

/**
 * @brief Append coupling spaces (CS...) and Tail... onto a base Parameters<Fixed...> type.
 *
 * Produces Parameters<Fixed..., CS..., Tail...>.
 * Used for weak form types whose leading fields are hardcoded (cycle-zero, stress output).
 */
template <typename Coupling, typename FixedParams, typename... Tail>
struct AppendCouplingToParams;

template <typename... CS, typename... Fixed, typename... Tail>
/// @brief Specialization appending `CouplingParams` field spaces after fixed weak-form parameters.
struct AppendCouplingToParams<CouplingParams<CS...>, Parameters<Fixed...>, Tail...> {
  using type =
      Parameters<Fixed..., CS..., Tail...>;  ///< Base parameter list extended with coupling and trailing parameters.
};

template <typename R, typename... CS, typename... Fixed, typename... Tail>
/// @brief Specialization appending `PhysicsFields` field spaces after fixed weak-form parameters.
struct AppendCouplingToParams<PhysicsFields<R, CS...>, Parameters<Fixed...>, Tail...> {
  using type =
      Parameters<Fixed..., CS..., Tail...>;  ///< Base parameter list extended with coupling and trailing parameters.
};

}  // namespace detail

}  // namespace smith
