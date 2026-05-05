// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tuple.hpp
 *
 * @brief Smith tuple compatibility layer over mfem::future::tuple
 */
#pragma once

#include "mfem/fem/dfem/tuple.hpp"

#include "smith/infrastructure/accelerator.hpp"

namespace mfem::future {

/**
 * @brief Smith compatibility extension for the MFEM tuple implementation.
 *
 * MFEM's tuple copy currently stores up to nine elements, while Smith's
 * historical tuple API supports ten and eleven elements.
 */
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9>
struct tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> {
  T0 v0;  ///< The first member of the tuple
  T1 v1;  ///< The second member of the tuple
  T2 v2;  ///< The third member of the tuple
  T3 v3;  ///< The fourth member of the tuple
  T4 v4;  ///< The fifth member of the tuple
  T5 v5;  ///< The sixth member of the tuple
  T6 v6;  ///< The seventh member of the tuple
  T7 v7;  ///< The eighth member of the tuple
  T8 v8;  ///< The ninth member of the tuple
  T9 v9;  ///< The tenth member of the tuple
};

/**
 * @brief Smith compatibility extension for the MFEM tuple implementation.
 */
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10>
struct tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> {
  T0 v0;    ///< The first member of the tuple
  T1 v1;    ///< The second member of the tuple
  T2 v2;    ///< The third member of the tuple
  T3 v3;    ///< The fourth member of the tuple
  T4 v4;    ///< The fifth member of the tuple
  T5 v5;    ///< The sixth member of the tuple
  T6 v6;    ///< The seventh member of the tuple
  T7 v7;    ///< The eighth member of the tuple
  T8 v8;    ///< The ninth member of the tuple
  T9 v9;    ///< The tenth member of the tuple
  T10 v10;  ///< The eleventh member of the tuple
};

namespace detail {

/// @brief Return element @p i from a 10- or 11-element tuple.
template <int i, typename Tuple>
MFEM_HOST_DEVICE constexpr auto& tuple_get_extended(Tuple& values)
{
  if constexpr (i == 0) {
    return values.v0;
  }
  if constexpr (i == 1) {
    return values.v1;
  }
  if constexpr (i == 2) {
    return values.v2;
  }
  if constexpr (i == 3) {
    return values.v3;
  }
  if constexpr (i == 4) {
    return values.v4;
  }
  if constexpr (i == 5) {
    return values.v5;
  }
  if constexpr (i == 6) {
    return values.v6;
  }
  if constexpr (i == 7) {
    return values.v7;
  }
  if constexpr (i == 8) {
    return values.v8;
  }
  if constexpr (i == 9) {
    return values.v9;
  }
  if constexpr (i == 10) {
    return values.v10;
  }
  MFEM_UNREACHABLE();
}

/// @brief Return const element @p i from a 10- or 11-element tuple.
template <int i, typename Tuple>
MFEM_HOST_DEVICE constexpr const auto& tuple_get_extended(const Tuple& values)
{
  if constexpr (i == 0) {
    return values.v0;
  }
  if constexpr (i == 1) {
    return values.v1;
  }
  if constexpr (i == 2) {
    return values.v2;
  }
  if constexpr (i == 3) {
    return values.v3;
  }
  if constexpr (i == 4) {
    return values.v4;
  }
  if constexpr (i == 5) {
    return values.v5;
  }
  if constexpr (i == 6) {
    return values.v6;
  }
  if constexpr (i == 7) {
    return values.v7;
  }
  if constexpr (i == 8) {
    return values.v8;
  }
  if constexpr (i == 9) {
    return values.v9;
  }
  if constexpr (i == 10) {
    return values.v10;
  }
  MFEM_UNREACHABLE();
}

}  // namespace detail

/// @brief Return mutable element @p i from a 10-element tuple.
template <int i, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9>
MFEM_HOST_DEVICE constexpr auto& get(tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& values)
{
  static_assert(i < 10);
  return detail::tuple_get_extended<i>(values);
}

/// @brief Return mutable element @p i from an 11-element tuple.
template <int i, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10>
MFEM_HOST_DEVICE constexpr auto& get(tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>& values)
{
  static_assert(i < 11);
  return detail::tuple_get_extended<i>(values);
}

/// @brief Return const element @p i from a 10-element tuple.
template <int i, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9>
MFEM_HOST_DEVICE constexpr const auto& get(const tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& values)
{
  static_assert(i < 10);
  return detail::tuple_get_extended<i>(values);
}

/// @brief Return const element @p i from an 11-element tuple.
template <int i, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10>
MFEM_HOST_DEVICE constexpr const auto& get(const tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>& values)
{
  static_assert(i < 11);
  return detail::tuple_get_extended<i>(values);
}

/**
 * @brief a function intended to be used for extracting the ith type from a tuple.
 *
 * @note type<i>(my_tuple) returns a value, whereas get<i>(my_tuple) returns a reference
 *
 * @tparam i the index of the tuple to query
 * @tparam T0 The first type stored in the tuple
 * @tparam T1 The second type stored in the tuple
 * @tparam T2 The third type stored in the tuple
 * @tparam T3 The fourth type stored in the tuple
 * @tparam T4 The fifth type stored in the tuple
 * @tparam T5 The sixth type stored in the tuple
 * @tparam T6 The seventh type stored in the tuple
 * @tparam T7 The eighth type stored in the tuple
 * @tparam T8 The ninth type stored in the tuple
 * @tparam T9 The tenth type stored in the tuple
 * @param values the tuple of values
 * @return a copy of the ith entry of the input
 */
template <int i, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9>
MFEM_HOST_DEVICE constexpr auto type(const tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>& values)
{
  static_assert(i < 10);
  return detail::tuple_get_extended<i>(values);
}

/**
 * @brief a function intended to be used for extracting the ith type from a tuple.
 *
 * @note type<i>(my_tuple) returns a value, whereas get<i>(my_tuple) returns a reference
 *
 * @tparam i the index of the tuple to query
 * @tparam T0 The first type stored in the tuple
 * @tparam T1 The second type stored in the tuple
 * @tparam T2 The third type stored in the tuple
 * @tparam T3 The fourth type stored in the tuple
 * @tparam T4 The fifth type stored in the tuple
 * @tparam T5 The sixth type stored in the tuple
 * @tparam T6 The seventh type stored in the tuple
 * @tparam T7 The eighth type stored in the tuple
 * @tparam T8 The ninth type stored in the tuple
 * @tparam T9 The tenth type stored in the tuple
 * @tparam T10 The eleventh type stored in the tuple
 * @param values the tuple of values
 * @return a copy of the ith entry of the input
 */
template <int i, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename T10>
MFEM_HOST_DEVICE constexpr auto type(const tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>& values)
{
  static_assert(i < 11);
  return detail::tuple_get_extended<i>(values);
}

}  // namespace mfem::future

namespace smith {

/// @brief Expose MFEM tuple in the Smith namespace.
template <typename... T>
using tuple = mfem::future::tuple<T...>;

/// @brief Expose MFEM tuple application in the Smith namespace.
using mfem::future::apply;
/// @brief Expose MFEM tuple element access in the Smith namespace.
using mfem::future::get;
/// @brief Expose MFEM tuple construction in the Smith namespace.
using mfem::future::make_tuple;
/// @brief Expose MFEM tuple type selection in the Smith namespace.
using mfem::future::type;

/// @brief Alias for the MFEM tuple size trait.
template <class... Types>
using tuple_size = mfem::future::tuple_size<Types...>;

/// @brief Alias for the MFEM tuple element trait.
template <size_t I, class T>
using tuple_element = mfem::future::tuple_element<I, T>;

/// @brief Alias for the MFEM tuple detection trait.
template <typename T>
using is_tuple = mfem::future::is_tuple<T>;

/// @brief Alias for the MFEM nested tuple detection trait.
template <typename T>
using is_tuple_of_tuples = mfem::future::is_tuple_of_tuples<T>;

}  // namespace smith

#include "smith/numerics/functional/tuple_tensor_dual_functions.hpp"
