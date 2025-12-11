// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dual.hpp
 *
 * @brief This file contains the declaration of a dual number class
 */

#pragma once

#include <iostream>
#include <cmath>

#include "smith/infrastructure/accelerator.hpp"

namespace smith {

/**
 * @brief Dual number struct (value plus gradient)
 * @tparam gradient_type The type of the gradient (should support addition, scalar multiplication/division, and unary
 * negation operators)
 */
template <typename gradient_type>
struct dual {
  double value;            ///< the actual numerical value
  gradient_type gradient;  ///< the partial derivatives of value w.r.t. some other quantity

  /**
   * @brief Copy assignment operator
   *
   * @param b rhs to assign to the dual number value
   * @return Resulting dual
   */
  SMITH_HOST_DEVICE constexpr auto& operator=(double b)
  {
    value = b;
    gradient = {};
    return *this;
  }
};

/**
 * @brief class template argument deduction guide for type `dual`.
 *
 * @note this lets users write
 * \code{.cpp} dual something{my_value, my_gradient}; \endcode
 * instead of explicitly writing the template parameter
 * \code{.cpp} dual< decltype(my_gradient) > something{my_value, my_gradient}; \endcode
 */
template <typename T>
dual(double, T) -> dual<T>;

/** @brief addition of a dual number and a non-dual number */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto operator+(dual<gradient_type> a, double b)
{
  return dual{a.value + b, a.gradient};
}

/** @brief addition of a dual number and a non-dual number */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto operator+(double a, dual<gradient_type> b)
{
  return dual{a + b.value, b.gradient};
}

/** @brief addition of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
SMITH_HOST_DEVICE constexpr auto operator+(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value + b.value, a.gradient + b.gradient};
}

/** @brief unary negation of a dual number */
template <typename gradient_type>
constexpr auto operator-(dual<gradient_type> x)
{
  return dual{-x.value, -x.gradient};
}

/** @brief subtraction of a non-dual number from a dual number */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto operator-(dual<gradient_type> a, double b)
{
  return dual{a.value - b, a.gradient};
}

/** @brief subtraction of a dual number from a non-dual number */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto operator-(double a, dual<gradient_type> b)
{
  return dual{a - b.value, -b.gradient};
}

/** @brief subtraction of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
SMITH_HOST_DEVICE constexpr auto operator-(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value - b.value, a.gradient - b.gradient};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto operator*(const dual<gradient_type>& a, double b)
{
  return dual{a.value * b, a.gradient * b};
}

/** @brief multiplication of a dual number and a non-dual number */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto operator*(double a, const dual<gradient_type>& b)
{
  return dual{a * b.value, a * b.gradient};
}

/** @brief multiplication of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
SMITH_HOST_DEVICE constexpr auto operator*(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value * b.value, b.value * a.gradient + a.value * b.gradient};
}

/** @brief division of a dual number by a non-dual number */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto operator/(const dual<gradient_type>& a, double b)
{
  return dual{a.value / b, a.gradient / b};
}

/** @brief division of a non-dual number by a dual number */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto operator/(double a, const dual<gradient_type>& b)
{
  return dual{a / b.value, -(a / (b.value * b.value)) * b.gradient};
}

/** @brief division of two dual numbers */
template <typename gradient_type_a, typename gradient_type_b>
SMITH_HOST_DEVICE constexpr auto operator/(dual<gradient_type_a> a, dual<gradient_type_b> b)
{
  return dual{a.value / b.value, (a.gradient / b.value) - (a.value * b.gradient) / (b.value * b.value)};
}

/**
 * @brief Generates const + non-const overloads for a binary comparison operator
 * Comparisons are conducted against the "value" part of the dual number
 * @param[in] x The comparison operator to overload
 */
#define binary_comparator_overload(x)                                             \
  template <typename T>                                                           \
  SMITH_HOST_DEVICE constexpr bool operator x(const dual<T>& a, double b)         \
  {                                                                               \
    return a.value x b;                                                           \
  }                                                                               \
                                                                                  \
  template <typename T>                                                           \
  SMITH_HOST_DEVICE constexpr bool operator x(double a, const dual<T>& b)         \
  {                                                                               \
    return a x b.value;                                                           \
  };                                                                              \
                                                                                  \
  template <typename T, typename U>                                               \
  SMITH_HOST_DEVICE constexpr bool operator x(const dual<T>& a, const dual<U>& b) \
  {                                                                               \
    return a.value x b.value;                                                     \
  };

binary_comparator_overload(<);   ///< implement operator<  for dual numbers
binary_comparator_overload(<=);  ///< implement operator<= for dual numbers
binary_comparator_overload(==);  ///< implement operator== for dual numbers
binary_comparator_overload(>=);  ///< implement operator>= for dual numbers
binary_comparator_overload(>);   ///< implement operator>  for dual numbers

#undef binary_comparator_overload

/** @brief compound assignment (+) for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto& operator+=(dual<gradient_type>& a, const dual<gradient_type>& b)
{
  a.value += b.value;
  a.gradient += b.gradient;
  return a;
}

/** @brief compound assignment (-) for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto& operator-=(dual<gradient_type>& a, const dual<gradient_type>& b)
{
  a.value -= b.value;
  a.gradient -= b.gradient;
  return a;
}

/** @brief compound assignment (+) for dual numbers with `double` righthand side */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto& operator+=(dual<gradient_type>& a, double b)
{
  a.value += b;
  return a;
}

/** @brief compound assignment (-) for dual numbers with `double` righthand side */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto& operator-=(dual<gradient_type>& a, double b)
{
  a.value -= b;
  return a;
}

/**
 * @brief Implementation of absolute value function for dual numbers
 * @note This is not differentiable at x = 0.0. At that point, the gradient is calculated as the gradient of x.
 */
template <typename gradient_type>
SMITH_HOST_DEVICE auto abs(dual<gradient_type> x)
{
  return (x.value >= 0) ? x : -x;
}

/**
 * @brief Implementation of max for dual numbers
 * @note This is not differentiable at a == b. At that point, the gradient is calculated as the gradient of b.
 */
template <typename gradient_type>
SMITH_HOST_DEVICE auto max(dual<gradient_type> a, double b)
{
  dual<gradient_type> b_dual{b, 0.0 * a.gradient};
  return (a > b_dual) ? a : b_dual;
}

/// @overload
template <typename gradient_type>
SMITH_HOST_DEVICE auto max(double a, dual<gradient_type> b)
{
  dual<gradient_type> a_dual{a, 0.0 * b.gradient};
  return (a_dual > b) ? a_dual : b;
}

/// @overload
template <typename gradient_type>
SMITH_HOST_DEVICE auto max(dual<gradient_type> a, dual<gradient_type> b)
{
  return (a > b) ? a : b;
}

/**
 * @brief Implementation of min for dual numbers
 * @note This is not differentiable at a == b. At that point, the gradient is calculated as the gradient of b.
 */
template <typename gradient_type>
SMITH_HOST_DEVICE auto min(dual<gradient_type> a, double b)
{
  dual<gradient_type> b_dual{b, 0.0 * a.gradient};
  return (a < b_dual) ? a : b_dual;
}

/// @overload
template <typename gradient_type>
SMITH_HOST_DEVICE auto min(double a, dual<gradient_type> b)
{
  dual<gradient_type> a_dual{a, 0.0 * b.gradient};
  return (a_dual < b) ? a_dual : b;
}

/// @overload
template <typename gradient_type>
SMITH_HOST_DEVICE auto min(dual<gradient_type> a, dual<gradient_type> b)
{
  return (a < b) ? a : b;
}

/**
 * @overload
 * @note for zeroth-order tensors (scalars)
 */
template <typename S, typename T>
SMITH_HOST_DEVICE constexpr auto inner(const dual<S>& A, const dual<T>& B)
{
  return A * B;
}

/**
 * @overload
 * @note for zeroth-order tensors (scalars)
 */
template <typename T>
SMITH_HOST_DEVICE constexpr auto inner(double A, const dual<T>& B)
{
  return A * B;
}

/**
 * @overload
 * @note for zeroth-order tensors (scalars)
 */
template <typename S>
SMITH_HOST_DEVICE constexpr auto inner(const dual<S>& A, double B)
{
  return A * B;
}

/** @brief implementation of square root for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto sqrt(dual<gradient_type> x)
{
  using std::sqrt;
  return dual<gradient_type>{sqrt(x.value), x.gradient / (2.0 * sqrt(x.value))};
}

/** @brief implementation of cosine for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto cos(dual<gradient_type> a)
{
  using std::cos, std::sin;
  return dual<gradient_type>{cos(a.value), -a.gradient * sin(a.value)};
}

/** @brief implementation of sine for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto sin(dual<gradient_type> a)
{
  using std::cos, std::sin;
  return dual<gradient_type>{sin(a.value), a.gradient * cos(a.value)};
}

/** @brief implementation of atan for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto atan(dual<gradient_type> a)
{
  using std::atan, std::pow;
  return dual<gradient_type>{atan(a.value), a.gradient / (1.0 + pow(a.value, 2))};
}

/** @brief implementation of atan2 for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto atan2(dual<gradient_type> y, dual<gradient_type> x)
{
  using std::atan2, std::pow;
  return dual<gradient_type>{atan2(y.value, x.value), y.gradient * x.value / (pow(x.value, 2) + pow(y.value, 2)) -
                                                          x.gradient * y.value / (pow(x.value, 2) + pow(y.value, 2))};
}

/** @brief implementation of atan2 for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto atan2(double y, dual<gradient_type> x)
{
  using std::atan2, std::pow;
  return dual<gradient_type>{atan2(y, x.value), -x.gradient * y / (pow(x.value, 2) + pow(y, 2))};
}

/** @brief implementation of atan2 for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto atan2(dual<gradient_type> y, double x)
{
  using std::atan2, std::pow;
  return dual<gradient_type>{atan2(y.value, x), y.gradient * x / (pow(x, 2) + pow(y.value, 2))};
}

/** @brief implementation of asin for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto asin(dual<gradient_type> a)
{
  using std::asin, std::pow, std::sqrt;
  return dual<gradient_type>{asin(a.value), a.gradient / sqrt(1.0 - pow(a.value, 2))};
}

/** @brief implementation of acos for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto acos(dual<gradient_type> a)
{
  using std::acos, std::pow, std::sqrt;
  return dual<gradient_type>{acos(a.value), -a.gradient / sqrt(1.0 - pow(a.value, 2))};
}

/** @brief implementation of exponential function for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto exp(dual<gradient_type> a)
{
  using std::exp;
  return dual<gradient_type>{exp(a.value), exp(a.value) * a.gradient};
}

/** @brief implementation of the natural logarithm function for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto log(dual<gradient_type> a)
{
  using std::log;
  return dual<gradient_type>{log(a.value), a.gradient / a.value};
}

/** @brief implementation of the natural logarithm of one plus the argument function for dual numbers */
template <typename gradient_type>
SMITH_HOST_DEVICE auto log1p(dual<gradient_type> a)
{
  using std::log1p;
  return dual<gradient_type>{log1p(a.value), a.gradient / (1.0 + a.value)};
}

/** @brief implementation of `a` (dual) raised to the `b` (dual) power */
template <typename gradient_type>
SMITH_HOST_DEVICE auto pow(dual<gradient_type> a, dual<gradient_type> b)
{
  using std::pow, std::log;
  double value = pow(a.value, b.value);
  gradient_type grad = pow(a.value, b.value - 1) * (a.gradient * b.value + b.gradient * a.value * log(a.value));
  return dual<gradient_type>{value, grad};
}

/** @brief implementation of `a` (non-dual) raised to the `b` (dual) power */
template <typename gradient_type>
SMITH_HOST_DEVICE auto pow(double a, dual<gradient_type> b)
{
  using std::pow, std::log;
  double value = pow(a, b.value);
  return dual<gradient_type>{value, value * b.gradient * log(a)};
}

/** @brief implementation of `a` (dual) raised to the `b` (non-dual) power */
template <typename gradient_type>
SMITH_HOST_DEVICE auto pow(dual<gradient_type> a, double b)
{
  using std::pow;
  double value = pow(a.value, b);
  gradient_type grad = b * pow(a.value, b - 1) * a.gradient;
  return dual<gradient_type>{value, grad};
}

/** @brief overload of operator<< for `dual` to work with `std::cout` and other `std::ostream`s */
template <typename T, int... n>
auto& operator<<(std::ostream& out, dual<T> A)
{
  out << '(' << A.value << ' ' << A.gradient << ')';
  return out;
}

/** @brief promote a value to a dual number of the appropriate type */
SMITH_HOST_DEVICE constexpr auto make_dual(double x) { return dual{x, 1.0}; }

/** @brief return the "value" part from a given type. For non-dual types, this is just the identity function */
template <typename T>
SMITH_HOST_DEVICE constexpr auto get_value(const T& arg)
{
  return arg;
}

/** @brief return the "value" part from a dual number type */
template <typename T>
SMITH_HOST_DEVICE constexpr auto get_value(dual<T> arg)
{
  return arg.value;
}

/** @brief return the "gradient" part from a dual number type */
template <typename gradient_type>
SMITH_HOST_DEVICE constexpr auto get_gradient(dual<gradient_type> arg)
{
  return arg.gradient;
}

/** @brief class for checking if a type is a dual number or not */
template <typename T>
struct is_dual_number {
  static constexpr bool value = false;  ///< whether or not type T is a dual number
};

/** @brief class for checking if a type is a dual number or not */
template <typename T>
struct is_dual_number<dual<T> > {
  static constexpr bool value = true;  ///< whether or not type T is a dual number
};

}  // namespace smith

#include "smith/numerics/functional/tuple_tensor_dual_functions.hpp"
