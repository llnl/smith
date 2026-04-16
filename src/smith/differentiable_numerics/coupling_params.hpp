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
  static constexpr std::size_t num_coupling_fields = sizeof...(Spaces);
  std::tuple<FieldType<Spaces>...> fields;
  CouplingParams(FieldType<Spaces>... fs) : fields(std::move(fs)...) {}
};

/// Deduction guide: CouplingParams{FieldType<A>("a"), FieldType<B>("b")} -> CouplingParams<A, B>
template <typename... Spaces>
CouplingParams(FieldType<Spaces>...) -> CouplingParams<Spaces...>;

namespace detail {

/// Type trait: true if T is a CouplingParams<...> specialization.
template <typename T>
struct is_coupling_params_impl : std::false_type {};

template <typename... Spaces>
struct is_coupling_params_impl<CouplingParams<Spaces...>> : std::true_type {};

template <typename T>
inline constexpr bool is_coupling_params_v = is_coupling_params_impl<T>::value;

/**
 * @brief Produce TimeRuleParams<Rule, Space, CS..., Tail...> from a CouplingParams<CS...>.
 *
 * Inserts coupling spaces immediately after the num_states copies of Space (the time-rule
 * state fields), and before the user-supplied Tail types (parameter_space...).
 */
template <typename Rule, typename Space, typename Coupling, typename... Tail>
struct TimeRuleParamsWithCoupling;

template <typename Rule, typename Space, typename... CS, typename... Tail>
struct TimeRuleParamsWithCoupling<Rule, Space, CouplingParams<CS...>, Tail...> {
  using type = TimeRuleParams<Rule, Space, CS..., Tail...>;
};

/**
 * @brief Append coupling spaces (CS...) and Tail... onto a base Parameters<Fixed...> type.
 *
 * Produces Parameters<Fixed..., CS..., Tail...>.
 * Used for weak form types whose leading fields are hardcoded (cycle-zero, stress output).
 */
template <typename Coupling, typename FixedParams, typename... Tail>
struct AppendCouplingToParams;

template <typename... CS, typename... Fixed, typename... Tail>
struct AppendCouplingToParams<CouplingParams<CS...>, Parameters<Fixed...>, Tail...> {
  using type = Parameters<Fixed..., CS..., Tail...>;
};

}  // namespace detail

}  // namespace smith
