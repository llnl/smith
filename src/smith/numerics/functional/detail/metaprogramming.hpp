// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file metaprogramming.hpp
 *
 * @brief Utilities for C++ metaprogramming
 */

#pragma once

#include <tuple>
#include <utility>
#include <type_traits>

#include "smith/infrastructure/accelerator.hpp"

/**
 * @brief return the Ith integer in `{n...}`
 */
template <int I, int... n>
constexpr auto get(std::integer_sequence<int, n...>)
{
  constexpr int values[sizeof...(n)] = {n...};
  return values[I];
}

/// @cond
namespace detail {

template <typename T>
struct always_false : std::false_type {};

/**
 * @brief unfortunately std::integral_constant doesn't have __host__ __device__ annotations
 * and we're not using --expt-relaxed-constexpr, so we need to implement something similar
 * to use it in a device context
 *
 * @tparam i the value represented by this struct
 */
template <int i>
struct integral_constant {
  SMITH_HOST_DEVICE constexpr operator int() { return i; }
};

SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda, int... i>
SMITH_HOST_DEVICE constexpr void for_constexpr(const lambda& f, integral_constant<i>... args)
{
  f(args...);
}

SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int... n, typename lambda, typename... arg_types>
SMITH_HOST_DEVICE constexpr void for_constexpr(const lambda& f, std::integer_sequence<int, n...>, arg_types... args)
{
  (detail::for_constexpr(f, args..., integral_constant<n>{}), ...);
}

}  // namespace detail
/// @endcond

/**
 * @brief multidimensional loop tool that evaluates the lambda body inside the innermost loop.
 *
 * @tparam n integer template arguments describing the shape of the iteration space
 * @tparam lambda the type of the functor object to be executed in the loop
 *
 * @note
 * \code{.cpp}
 * for_constexpr< 2, 3 >([](auto i, auto j) { std::cout << i << " " << j << std::endl; }
 * \endcode
 * will print:
 * \code{.cpp}
 * 0 0
 * 0 1
 * 0 2
 * 1 0
 * 1 1
 * 1 2
 * \endcode
 *
 * @note latter integer template parameters correspond to more nested loops
 *
 * @note The lambda function should be a callable object taking sizeof ... (n) arguments.
 * Anything returned from f() will be discarded
 * note: this forces multidimensional loop unrolling, which can be beneficial for
 * runtime performance, but can hurt compile time and executable size as the loop
 * dimensions become larger.
 *
 */
template <int... n, typename lambda>
SMITH_HOST_DEVICE constexpr void for_constexpr(const lambda& f)
{
  detail::for_constexpr(f, std::make_integer_sequence<int, n>{}...);
}
