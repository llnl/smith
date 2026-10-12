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

#include "mfem.hpp"

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

namespace mfem::future {

namespace detail {

    /// @brief Helper function for concatenating tuples
    template <typename... T, typename... U, size_t... i, size_t... j>
    constexpr auto concatenateTuplesImpl(const mfem::future::tuple<T...>& first,
                                         const mfem::future::tuple<U...>& second,
                                         std::index_sequence<i...>, std::index_sequence<j...>)
    {
        return mfem::future::tuple{mfem::future::get<i>(first)..., mfem::future::get<j>(second)...};
    }

} // namespace detail

/// @overload
///
/// Concatenates two tuples.
template <typename... T, typename... U>
constexpr auto tuple_cat(const mfem::future::tuple<T...>& first,
                                 const mfem::future::tuple<U...>& second)
{
  return detail::concatenateTuplesImpl(first, second, std::index_sequence_for<T...>{}, std::index_sequence_for<U...>{});
}

/// @overload
///
/// Tuple concatenation for a single input. Just returns the input tuple.
template <typename Tuple>
constexpr auto tuple_cat(const Tuple& tuple)
{
  return tuple;
}

/** @brief Concatenate tuples
 *
 * @tparam First Type of the first tuple to concatenate
 * @tparam Second Type of the second tuple to concatenate
 * @tparam Rest... Variadic list of types of any additional tuples to concatenate
 *
 * @param first First tuple to contacenate
 * @param second Second tuple to concatentate
 * @param rest Pack of additional tuples to concatenate
 * @return new tuple that is a concatenation of all input tuples
 */
template <typename First, typename Second, typename... Rest>
constexpr auto tuple_cat(const First& first, const Second& second, const Rest&... rest)
{
  return tuple_cat(tuple_cat(first, second), rest...);
}

} // namespace mfem::future

#include "smith/numerics/functional/tuple_tensor_dual_functions.hpp"
