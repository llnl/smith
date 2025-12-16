// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tensor.hpp
 *
 * @brief Implementation of the tensor class used by Functional
 */

#pragma once

#include <cmath>

#include "smith/infrastructure/accelerator.hpp"
#include "detail/metaprogramming.hpp"

namespace smith {

/**
 * @brief Arbitrary-rank tensor class
 * @tparam T The type stored at each index
 * @tparam n The dimensions of the tensor
 */
template <typename T, int... n>
struct tensor;

/// @cond
template <typename T, int m, int... n>
struct tensor<T, m, n...> {
  template <typename i_type>
  SMITH_HOST_DEVICE constexpr auto& operator()(i_type i)
  {
    return data[i];
  }
  template <typename i_type>
  SMITH_HOST_DEVICE constexpr auto& operator()(i_type i) const
  {
    return data[i];
  }
  template <typename i_type, typename... jklm_type>
  SMITH_HOST_DEVICE constexpr auto& operator()(i_type i, jklm_type... jklm)
  {
    return data[i](jklm...);
  }
  template <typename i_type, typename... jklm_type>
  SMITH_HOST_DEVICE constexpr auto& operator()(i_type i, jklm_type... jklm) const
  {
    return data[i](jklm...);
  }

  SMITH_HOST_DEVICE constexpr auto& operator[](int i) { return data[i]; }
  SMITH_HOST_DEVICE constexpr const auto& operator[](int i) const { return data[i]; }

  tensor<T, n...> data[m];
};

template <typename T, int m>
struct tensor<T, m> {
  template <typename i_type>
  SMITH_HOST_DEVICE constexpr auto& operator()(i_type i)
  {
    return data[i];
  }
  template <typename i_type>
  SMITH_HOST_DEVICE constexpr auto& operator()(i_type i) const
  {
    return data[i];
  }
  SMITH_HOST_DEVICE constexpr auto& operator[](int i) { return data[i]; }
  SMITH_HOST_DEVICE constexpr const auto& operator[](int i) const { return data[i]; }

  template <int last_dimension = m, typename = typename std::enable_if<last_dimension == 1>::type>
  SMITH_HOST_DEVICE constexpr operator T()
  {
    return data[0];
  }

  template <int last_dimension = m, typename = typename std::enable_if<last_dimension == 1>::type>
  SMITH_HOST_DEVICE constexpr operator T() const
  {
    return data[0];
  }

  T data[m];
};
/// @endcond

/**
 * @brief class template argument deduction guide for type `tensor`.
 *
 * @note this lets users write
 * \code{.cpp} tensor A = {{0.0, 1.0, 2.0}}; \endcode
 * instead of explicitly writing the template parameters
 * \code{.cpp} tensor< double, 3 > A = {{1.0, 2.0, 3.0}}; \endcode
 */
template <typename T, int n1>
tensor(const T (&data)[n1]) -> tensor<T, n1>;

/**
 * @brief class template argument deduction guide for type `tensor`.
 *
 * @note this lets users write
 * \code{.cpp} tensor A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}}; \endcode
 * instead of explicitly writing the template parameters
 * \code{.cpp} tensor< double, 3, 3 > A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}}; \endcode
 */
template <typename T, int n1, int n2>
tensor(const T (&data)[n1][n2]) -> tensor<T, n1, n2>;

using vec2 = tensor<double, 2>;  ///< statically sized vector of 2 doubles
using vec3 = tensor<double, 3>;  ///< statically sized vector of 3 doubles

using mat2 = tensor<double, 2, 2>;  ///< statically sized 2x2 matrix of doubles
using mat3 = tensor<double, 3, 3>;  ///< statically sized 3x3 matrix of doubles

/**
 * @brief A sentinel struct for eliding no-op tensor operations
 */
struct zero {
  /** @brief `zero` is implicitly convertible to double with value 0.0 */
  SMITH_HOST_DEVICE operator double() { return 0.0; }

  /** @brief `zero` is implicitly convertible to a tensor of any shape */
  template <typename T, int... n>
  SMITH_HOST_DEVICE operator tensor<T, n...>()
  {
    return tensor<T, n...>{};
  }

  /** @brief `zero` can be accessed like a multidimensional array */
  template <typename... T>
  SMITH_HOST_DEVICE auto operator()(T...) const
  {
    return zero{};
  }

  /** @brief anything assigned to `zero` does not change its value and returns `zero` */
  template <typename T>
  SMITH_HOST_DEVICE auto operator=(T)
  {
    return zero{};
  }
};

/** @brief checks if a type is `zero` */
template <typename T>
struct is_zero : std::false_type {};

/** @overload */
template <>
struct is_zero<zero> : std::true_type {};

/** @brief the sum of two `zero`s is `zero` */
SMITH_HOST_DEVICE constexpr auto operator+(zero, zero) { return zero{}; }

/** @brief the sum of `zero` with something non-`zero` just returns the other value */
template <typename T>
SMITH_HOST_DEVICE constexpr auto operator+(zero, T other)
{
  return other;
}

/** @brief the sum of `zero` with something non-`zero` just returns the other value */
template <typename T>
SMITH_HOST_DEVICE constexpr auto operator+(T other, zero)
{
  return other;
}

/////////////////////////////////////////////////

/** @brief the unary negation of `zero` is `zero` */
SMITH_HOST_DEVICE constexpr auto operator-(zero) { return zero{}; }

/** @brief the difference of two `zero`s is `zero` */
SMITH_HOST_DEVICE constexpr auto operator-(zero, zero) { return zero{}; }

/** @brief the difference of `zero` with something else is the unary negation of the other thing */
template <typename T>
SMITH_HOST_DEVICE constexpr auto operator-(zero, T other)
{
  return -other;
}

/** @brief the difference of something else with `zero` is the other thing itself */
template <typename T>
SMITH_HOST_DEVICE constexpr auto operator-(T other, zero)
{
  return other;
}

/////////////////////////////////////////////////

/** @brief the product of two `zero`s is `zero` */
SMITH_HOST_DEVICE constexpr auto operator*(zero, zero) { return zero{}; }

/** @brief the product `zero` with something else is also `zero` */
template <typename T>
SMITH_HOST_DEVICE constexpr auto operator*(zero, T /*other*/)
{
  return zero{};
}

/** @brief the product `zero` with something else is also `zero` */
template <typename T>
SMITH_HOST_DEVICE constexpr auto operator*(T /*other*/, zero)
{
  return zero{};
}

/** @brief `zero` divided by something is `zero` */
template <typename T>
SMITH_HOST_DEVICE constexpr auto operator/(zero, T /*other*/)
{
  return zero{};
}

/// @brief Get a human-readable compiler error when you try to divide by zero
template <typename T>
void operator/(T, zero)
{
  static_assert(::detail::always_false<T>{}, "Error: Can't divide by zero!");
}

/** @brief `zero` plus `zero` is `zero` */
SMITH_HOST_DEVICE constexpr auto operator+=(zero, zero) { return zero{}; }

/** @brief `zero` minus `zero` is `zero` */
SMITH_HOST_DEVICE constexpr auto operator-=(zero, zero) { return zero{}; }

/** @brief let `zero` be accessed like a tuple */
template <int i>
SMITH_HOST_DEVICE zero& get(zero& x)
{
  return x;
}

/** @brief let `zero` be accessed like a tuple */
template <int i>
SMITH_HOST_DEVICE zero get(const zero&)
{
  return zero{};
}

/** @brief the dot product of anything with `zero` is `zero` */
template <typename T>
SMITH_HOST_DEVICE constexpr zero dot(const T&, zero)
{
  return zero{};
}

/** @brief the dot product of anything with `zero` is `zero` */
template <typename T>
SMITH_HOST_DEVICE constexpr zero dot(zero, const T&)
{
  return zero{};
}

/**
 * @brief Removes 1s from tensor dimensions
 * For example, a tensor<T, 1, 10> is equivalent to a tensor<T, 10>
 * @tparam T The scalar type of the tensor
 * @tparam n1 The first dimension
 * @tparam n2 The second dimension
 */
template <typename T, int n1, int n2 = 1>
using reduced_tensor = std::conditional_t<
    (n1 == 1 && n2 == 1), double,
    std::conditional_t<n1 == 1, tensor<T, n2>, std::conditional_t<n2 == 1, tensor<T, n1>, tensor<T, n1, n2>>>>;

/**
 * @brief Creates a tensor given the dimensions in a @p std::integer_sequence
 * @see std::integer_sequence
 * @tparam n The parameter pack of integer dimensions
 */
template <typename T, int... n>
SMITH_HOST_DEVICE constexpr auto tensor_with_shape(std::integer_sequence<int, n...>)
{
  return tensor<T, n...>{};
}

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 * Can be thought of as analogous to @p std::transform in that the set of possible
 * indices for dimensions @p n are transformed into the values of the tensor by @a f
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda_type>
SMITH_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f());
  return tensor<T>{f()};
}

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 *
 * @tparam n1 The dimension of the tensor
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 * @pre @a f must accept @p n1 arguments of type @p int
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, typename lambda_type>
SMITH_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f(n1));
  tensor<T, n1> A{};
  for (int i = 0; i < n1; i++) {
    A(i) = f(i);
  }
  return A;
}

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 *
 * @tparam n1 The first dimension of the tensor
 * @tparam n2 The second dimension of the tensor
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 * @pre @a f must accept @p n1 x @p n2 arguments of type @p int
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, typename lambda_type>
SMITH_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f(n1, n2));
  tensor<T, n1, n2> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      A(i, j) = f(i, j);
    }
  }
  return A;
}

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 *
 * @tparam n1 The first dimension of the tensor
 * @tparam n2 The second dimension of the tensor
 * @tparam n3 The third dimension of the tensor
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 * @pre @a f must accept @p n1 x @p n2 x @p n3 arguments of type @p int
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, int n3, typename lambda_type>
SMITH_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f(n1, n2, n3));
  tensor<T, n1, n2, n3> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        A(i, j, k) = f(i, j, k);
      }
    }
  }
  return A;
}

/**
 * @brief Creates a tensor of requested dimension by subsequent calls to a functor
 *
 * @tparam n1 The first dimension of the tensor
 * @tparam n2 The second dimension of the tensor
 * @tparam n3 The third dimension of the tensor
 * @tparam n4 The fourth dimension of the tensor
 * @tparam lambda_type The type of the functor
 * @param[in] f The functor to generate the tensor values from
 * @pre @a f must accept @p n1 x @p n2 x @p n3 x @p n4 arguments of type @p int
 *
 * @note the different cases of 0D, 1D, 2D, 3D, and 4D are implemented separately
 *       to work around a limitation in nvcc involving __host__ __device__ lambdas with `auto` parameters.
 */
SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <int n1, int n2, int n3, int n4, typename lambda_type>
SMITH_HOST_DEVICE constexpr auto make_tensor(lambda_type f)
{
  using T = decltype(f(n1, n2, n3, n4));
  tensor<T, n1, n2, n3, n4> A{};
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        for (int l = 0; l < n4; l++) {
          A(i, j, k, l) = f(i, j, k, l);
        }
      }
    }
  }
  return A;
}

/**
 * @brief return the sum of two tensors
 * @tparam S the underlying type of the lefthand argument
 * @tparam T the underlying type of the righthand argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand operand
 * @param[in] B The righthand operand
 */
template <typename S, typename T, int m, int... n>
SMITH_HOST_DEVICE constexpr auto operator+(const tensor<S, m, n...>& A, const tensor<T, m, n...>& B)
{
  tensor<decltype(S{} + T{}), m, n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = A[i] + B[i];
  }
  return C;
}

/**
 * @brief return the unary negation of a tensor
 * @tparam T the underlying type of the righthand argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor to negate
 */
template <typename T, int m, int... n>
SMITH_HOST_DEVICE constexpr auto operator-(const tensor<T, m, n...>& A)
{
  tensor<T, m, n...> B{};
  for (int i = 0; i < m; i++) {
    B[i] = -A[i];
  }
  return B;
}

/**
 * @brief return the difference of two tensors
 * @tparam S the underlying type of the lefthand argument
 * @tparam T the underlying type of the righthand argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand operand
 * @param[in] B The righthand operand
 */
template <typename S, typename T, int m, int... n>
SMITH_HOST_DEVICE constexpr auto operator-(const tensor<S, m, n...>& A, const tensor<T, m, n...>& B)
{
  tensor<decltype(S{} + T{}), m, n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = A[i] - B[i];
  }
  return C;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int... n>
SMITH_HOST_DEVICE constexpr auto& operator+=(tensor<S, m, n...>& A, const tensor<T, m, n...>& B)
{
  for (int i = 0; i < m; i++) {
    A[i] += B[i];
  }
  return A;
}

#if 0
/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T>
SMITH_HOST_DEVICE constexpr auto& operator+=(tensor<T>& A, const T& B)
{
  return A.data += B;
}
#endif

/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto& operator+=(tensor<T, n, 1>& A, const tensor<T, n>& B)
{
  for (int i = 0; i < n; i++) {
    A.data[i][0] += B[i];
  }
  return A;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto& operator+=(tensor<T, 1, n>& A, const tensor<T, n>& B)
{
  for (int i = 0; i < n; i++) {
    A.data[0][i] += B[i];
  }
  return A;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T>
SMITH_HOST_DEVICE constexpr auto& operator+=(tensor<T, 1>& A, const T& B)
{
  return A.data[0] += B;
}

/**
 * @brief compound assignment (+) on tensors
 * @tparam T the underlying type of the tensor argument
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename T>
SMITH_HOST_DEVICE constexpr auto& operator+=(tensor<T, 1, 1>& A, const T& B)
{
  return A.data[0][0] += B;
}

/**
 * @brief compound assignment (+) between a tensor and zero (no-op)
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 */
template <typename T, int... n>
SMITH_HOST_DEVICE constexpr auto& operator+=(tensor<T, n...>& A, zero)
{
  return A;
}

/**
 * @brief compound assignment (-) on tensors
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int... n>
SMITH_HOST_DEVICE constexpr auto& operator-=(tensor<S, m, n...>& A, const tensor<T, m, n...>& B)
{
  for (int i = 0; i < m; i++) {
    A[i] -= B[i];
  }
  return A;
}

/**
 * @brief compound assignment (-) between a tensor and zero (no-op)
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 */
template <typename T, int... n>
SMITH_HOST_DEVICE constexpr auto& operator-=(tensor<T, n...>& A, zero)
{
  return A;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a scalar, and the right argument is a tensor
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto outer(double A, tensor<T, n> B)
{
  tensor<decltype(double{} * T{}), n> AB{};
  for (int i = 0; i < n; i++) {
    AB[i] = A * B[i];
  }
  return AB;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a tensor, and the right argument is a scalar
 */
template <typename T, int m>
SMITH_HOST_DEVICE constexpr auto outer(const tensor<T, m>& A, double B)
{
  tensor<decltype(T{} * double{}), m> AB{};
  for (int i = 0; i < m; i++) {
    AB[i] = A[i] * B;
  }
  return AB;
}

/**
 * @overload
 * @note this overload implements the case where the left argument is `zero`, and the right argument is a tensor
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto outer(zero, const tensor<T, n>&)
{
  return zero{};
}

/**
 * @overload
 * @note this overload implements the case where the left argument is a tensor, and the right argument is `zero`
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto outer(const tensor<T, n>&, zero)
{
  return zero{};
}

/**
 * @overload
 * @note this overload implements the case where both arguments are vectors
 */
template <typename S, typename T, int m, int n>
SMITH_HOST_DEVICE constexpr auto outer(const tensor<S, m>& A, const tensor<T, n>& B)
{
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i][j] = A[i] * B[j];
    }
  }
  return AB;
}

/**
 * @brief this function contracts over all indices of the two tensor arguments
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam m the number of rows
 * @tparam n the number of columns
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int n>
SMITH_HOST_DEVICE constexpr auto inner(const tensor<S, m, n>& A, const tensor<T, m, n>& B)
{
  decltype(S{} * T{}) sum{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      sum += A[i][j] * B[i][j];
    }
  }
  return sum;
}

/**
 * @overload
 * @note for first order tensors (vectors)
 */
template <typename S, typename T, int m>
SMITH_HOST_DEVICE constexpr auto inner(const tensor<S, m>& A, const tensor<T, m>& B)
{
  decltype(S{} * T{}) sum{};
  for (int i = 0; i < m; i++) {
    sum += A[i] * B[i];
  }
  return sum;
}

/**
 * @overload
 * @note for zeroth-order tensors (scalars)
 */
SMITH_HOST_DEVICE constexpr auto inner(double A, double B) { return A * B; }

/**
 * @brief this function contracts over all indices of the two tensor arguments
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam m the number of rows
 * @tparam n the number of columns
 */
template <typename S, int m, int n>
SMITH_HOST_DEVICE constexpr auto inner(const tensor<S, m, n>&, zero)
{
  return zero{};
}

/**
 * @overload
 * @note for first order tensors (vectors)
 */
template <typename S, int m>
SMITH_HOST_DEVICE constexpr auto inner(const tensor<S, m>&, zero)
{
  return zero{};
}

/**
 * @overload
 * @note for zeroth-order tensors (scalars)
 */
SMITH_HOST_DEVICE constexpr auto inner(double, zero) { return zero{}; }

/**
 * @brief this function contracts over all indices of the two tensor arguments
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam m the number of rows
 * @tparam n the number of columns
 */
template <typename T, int m, int n>
SMITH_HOST_DEVICE constexpr auto inner(zero, const tensor<T, m, n>&)
{
  return zero{};
}

/**
 * @overload
 * @note for first order tensors (vectors)
 */
template <typename T, int m>
SMITH_HOST_DEVICE constexpr auto inner(zero, const tensor<T, m>&)
{
  return zero{};
}

/**
 * @overload
 * @note for zeroth-order tensors (scalars)
 */
SMITH_HOST_DEVICE constexpr auto inner(zero, double) { return zero{}; }

/**
 * @brief this function contracts over the "middle" index of the two tensor arguments
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int n, int p>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n, p>& B)
{
  tensor<decltype(S{} * T{}), m, p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < n; k++) {
        AB[i][j] = AB[i][j] + A[i][k] * B[k][j];
      }
    }
  }
  return AB;
}

/**
 * @overload
 * @note vector . scalar
 */
template <typename T, int m>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<T, m>& A, double B)
{
  return A * B;
}

/**
 * @overload
 * @note scalar * vector
 */
template <typename T, int m>
SMITH_HOST_DEVICE constexpr auto dot(double B, const tensor<T, m>& A)
{
  return B * A;
}

/**
 * @overload
 * @note vector . vector
 */
template <typename S, typename T, int m>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m>& A, const tensor<T, m>& B)
{
  decltype(S{} * T{}) AB{};
  for (int i = 0; i < m; i++) {
    AB = AB + A[i] * B[i];
  }
  return AB;
}

/**
 * @overload
 * @note vector . matrix
 */
template <typename S, typename T, int m, int n>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m>& A, const tensor<T, m, n>& B)
{
  tensor<decltype(S{} * T{}), n> AB{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AB[i] = AB[i] + A[j] * B[j][i];
    }
  }
  return AB;
}

/**
 * @overload
 * @note vector . tensor3D
 */
template <typename S, typename T, int m, int n, int p>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m>& A, const tensor<T, m, n, p>& B)
{
  tensor<decltype(S{} * T{}), n, p> AB{};
  for (int j = 0; j < m; j++) {
    AB = AB + A[j] * B[j];
  }
  return AB;
}

/**
 * @overload
 * @note vector . tensor4D
 *
 * this overload, and others of the form `dot(vector, tensor)`, can be
 * implemented more succinctly as a single variadic function, but for some
 * reason gcc-11 (but not gcc-10 or gcc-12) seemed to break when compiling
 * that compact implementation, so we're manually writing out some of the different
 * dot product overloads in order to support that compiler and version
 */
template <typename S, typename T, int m, int n, int p, int q>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m>& A, const tensor<T, m, n, p, q>& B)
{
  tensor<decltype(S{} * T{}), n, p, q> AB{};
  for (int j = 0; j < m; j++) {
    AB = AB + A[j] * B[j];
  }
  return AB;
}

/**
 * @overload
 * @note matrix . vector
 */
template <typename S, typename T, int m, int n>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n>& B)
{
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i] = AB[i] + A[i][j] * B[j];
    }
  }
  return AB;
}

/**
 * @overload
 * @note matrix . tensor
 */
template <typename S, typename T, int m, int n, int p, int q, int r>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n, p, q, r>& B)
{
  tensor<decltype(S{} * T{}), m, p, q, r> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i] = AB[i] + A[i][j] * B[j];
    }
  }
  return AB;
}

/**
 * @overload
 * @note matrix . tensor
 */
template <typename S, typename T, int m, int n, int p, int q>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m, n>& A, const tensor<T, n, p, q>& B)
{
  tensor<decltype(S{} * T{}), m, p, q> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB[i] = AB[i] + A[i][j] * B[j];
    }
  }
  return AB;
}

/**
 * @overload
 * @note 3rd-order-tensor . vector
 */
template <typename S, typename T, int m, int n, int p>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m, n, p>& A, const tensor<T, p>& B)
{
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        AB[i][j] += A[i][j][k] * B[k];
      }
    }
  }
  return AB;
}

/**
 * @overload
 * @note vector . matrix . vector
 */
template <typename S, typename T, typename U, int m, int n>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m>& u, const tensor<T, m, n>& A, const tensor<U, n>& v)
{
  decltype(S{} * T{} * U{}) uAv{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      uAv += u[i] * A[i][j] * v[j];
    }
  }
  return uAv;
}

/// @overload
template <typename S, typename T, int m, int n, int p, int q>
SMITH_HOST_DEVICE constexpr auto dot(const tensor<S, m, n, p, q>& A, const tensor<T, q>& B)
{
  tensor<decltype(S{} * T{}), m, n, p> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        for (int l = 0; l < q; l++) {
          AB[i][j][k] += A[i][j][k][l] * B[l];
        }
      }
    }
  }
  return AB;
}

/// compute the cross product of the columns of A: A(:,1) x A(:,2)
template <typename T>
auto cross(const tensor<T, 3, 2>& A)
{
  return tensor<T, 3>{A(1, 0) * A(2, 1) - A(2, 0) * A(1, 1), A(2, 0) * A(0, 1) - A(0, 0) * A(2, 1),
                      A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1)};
}

/// return the in-plane components of the cross product of {v[0], v[1], 0} x {0, 0, 1}
template <typename T>
auto cross(const tensor<T, 2, 1>& v)
{
  return tensor<T, 2>{v(1, 0), -v(0, 0)};
}

/// return the in-plane components of the cross product of {v[0], v[1], 0} x {0, 0, 1}
template <typename T>
auto cross(const tensor<T, 2>& v)
{
  return tensor<T, 2>{v[1], -v[0]};
}

/// compute the (right handed) cross product of two 3-vectors
template <typename S, typename T>
auto cross(const tensor<S, 3>& u, const tensor<T, 3>& v)
{
  return tensor<decltype(S{} * T{}), 3>{u(1) * v(2) - u(2) * v(1), u(2) * v(0) - u(0) * v(2),
                                        u(0) * v(1) - u(1) * v(0)};
}

/**
 * @brief double dot product, contracting over the two "middle" indices
 * @tparam S the underlying type of the tensor (lefthand) argument
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam m first dimension of A
 * @tparam n second dimension of A
 * @tparam p third dimension of A, first dimensions of B
 * @tparam q fourth dimension of A, second dimensions of B
 * @param[in] A The lefthand tensor
 * @param[in] B The righthand tensor
 */
template <typename S, typename T, int m, int n, int p, int q>
SMITH_HOST_DEVICE constexpr auto double_dot(const tensor<S, m, n, p, q>& A, const tensor<T, p, q>& B)
{
  tensor<decltype(S{} * T{}), m, n> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        for (int l = 0; l < q; l++) {
          AB[i][j] += A[i][j][k][l] * B[k][l];
        }
      }
    }
  }
  return AB;
}

/**
 * @overload
 * @note 3rd-order-tensor : 2nd-order-tensor
 */
template <typename S, typename T, int m, int n, int p>
SMITH_HOST_DEVICE constexpr auto double_dot(const tensor<S, m, n, p>& A, const tensor<T, n, p>& B)
{
  tensor<decltype(S{} * T{}), m> AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) {
        AB[i] += A[i][j][k] * B[j][k];
      }
    }
  }
  return AB;
}

/**
 * @overload
 * @note 2nd-order-tensor : 2nd-order-tensor, like inner()
 */
template <typename S, typename T, int m, int n>
constexpr auto double_dot(const tensor<S, m, n>& A, const tensor<T, m, n>& B)
{
  decltype(S{} * T{}) AB{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      AB += A[i][j] * B[i][j];
    }
  }
  return AB;
}

/**
 * @brief this is a shorthand for dot(A, B)
 */
template <typename S, typename T, int... m, int... n>
SMITH_HOST_DEVICE constexpr auto operator*(const tensor<S, m...>& A, const tensor<T, n...>& B)
{
  return dot(A, B);
}

/**
 * @brief Returns the squared Frobenius norm of the tensor
 * @param[in] A The tensor to obtain the squared norm from
 */
template <typename T, int m>
SMITH_HOST_DEVICE constexpr auto squared_norm(const tensor<T, m>& A)
{
  T total{};
  for (int i = 0; i < m; i++) {
    total += A[i] * A[i];
  }
  return total;
}

/// @overload
template <typename T, int m, int n>
SMITH_HOST_DEVICE constexpr auto squared_norm(const tensor<T, m, n>& A)
{
  T total{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      total += A[i][j] * A[i][j];
    }
  }
  return total;
}

/// @overload
template <typename T, int... n>
SMITH_HOST_DEVICE constexpr auto squared_norm(const tensor<T, n...>& A)
{
  T total{};
  for_constexpr<n...>([&](auto... i) { total += A(i...) * A(i...); });
  return total;
}

/**
 * @brief Returns the Frobenius norm of the tensor
 * @param[in] A The tensor to obtain the norm from
 */
template <typename T, int... n>
SMITH_HOST_DEVICE auto norm(const tensor<T, n...>& A)
{
  using std::sqrt;
  return sqrt(squared_norm(A));
}

/**
 * @brief overload of Frobenius norm for zero type
 */
SMITH_HOST_DEVICE constexpr auto norm(zero) { return zero{}; }

/**
 * @brief Normalizes the tensor
 * Each element is divided by the Frobenius norm of the tensor, @see norm
 * @param[in] A The tensor to normalize
 */
template <typename T, int... n>
SMITH_HOST_DEVICE auto normalize(const tensor<T, n...>& A)
{
  return A / norm(A);
}

/**
 * @brief promotes a 2x2 matrix to a 3x3 matrix, by populating the upper left block, leaving zeroes in the third row /
 * column
 * @param[in] A the 2x2 matrix to be promoted to a 3x3 matrix
 */
template <typename T>
SMITH_HOST_DEVICE tensor<T, 3, 3> to_3x3(const tensor<T, 2, 2>& A)
{
  tensor<T, 3, 3> output{};
  output[0][0] = A[0][0];
  output[0][1] = A[0][1];
  output[1][0] = A[1][0];
  output[1][1] = A[1][1];
  return output;
}

/**
 * @brief Returns the trace of a square matrix
 * @param[in] A The matrix to compute the trace of
 * @return The sum of the elements on the main diagonal
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto tr(const tensor<T, n, n>& A)
{
  T trA{};
  for (int i = 0; i < n; i++) {
    trA = trA + A[i][i];
  }
  return trA;
}

/**
 * @brief Returns the symmetric part of a square matrix
 * @param[in] A The matrix to obtain the symmetric part of
 * @return (1/2) * (A + A^T)
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto sym(const tensor<T, n, n>& A)
{
  tensor<T, n, n> symA{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      symA[i][j] = 0.5 * (A[i][j] + A[j][i]);
    }
  }
  return symA;
}

/**
 * @brief Returns the antisymmetric part of a square matrix
 * @param[in] A The matrix to obtain the antisymmetric part of
 * @return (1/2) * (A - A^T)
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto antisym(const tensor<T, n, n>& A)
{
  tensor<T, n, n> antisymA{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      antisymA[i][j] = 0.5 * (A[i][j] - A[j][i]);
    }
  }
  return antisymA;
}

/**
 * @brief Calculates the deviator of a matrix (rank-2 tensor)
 * @param[in] A The matrix to calculate the deviator of
 * In the context of stress tensors, the deviator is obtained by
 * subtracting the mean stress (average of main diagonal elements)
 * from each element on the main diagonal
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto dev(const tensor<T, n, n>& A)
{
  auto devA = A;
  auto trA = tr(A);
  for (int i = 0; i < n; i++) {
    devA[i][i] -= trA / n;
  }
  return devA;
}

/**
 * @brief Returns a square matrix (rank-2 tensor) containing the diagonal entries of the input square matrix
 * with zeros in the off-diagonal positions
 * @param[in] A The input square matrix
 * This operation is used to compute a term in the constitutive response of a linear, cubic solid material
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto diagonal_matrix(const tensor<T, n, n>& A)
{
  tensor<T, n, n> D{};
  for (int i = 0; i < n; i++) {
    D[i][i] = A[i][i];
  }
  return D;
}

/**
 * @brief Returns a square diagonal matrix by specifying the diagonal entries
 * @param[in] d a list of diagonal entries
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr tensor<T, n, n> diag(const tensor<T, n>& d)
{
  tensor<T, n, n> D{};
  for (int i = 0; i < n; i++) {
    D[i][i] = d[i];
  }
  return D;
}

/**
 * @brief Returns an array containing the diagonal entries of a square matrix
 * @param[in] D the matrix to extract the diagonal entries from
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr tensor<T, n> diag(const tensor<T, n, n>& D)
{
  tensor<T, n> d{};
  for (int i = 0; i < n; i++) {
    d[i] = D[i][i];
  }
  return d;
}

/**
 * @brief Obtains the identity matrix of the specified dimension
 * @return I_dim
 */
template <int dim>
SMITH_HOST_DEVICE constexpr tensor<double, dim, dim> DenseIdentity()
{
  tensor<double, dim, dim> I{};
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      I[i][j] = (i == j);
    }
  }
  return I;
}

/**
 * @brief Returns the transpose of the matrix
 * @param[in] A The matrix to obtain the transpose of
 */
template <typename T, int m, int n>
SMITH_HOST_DEVICE constexpr auto transpose(const tensor<T, m, n>& A)
{
  tensor<T, n, m> AT{};
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      AT[i][j] = A[j][i];
    }
  }
  return AT;
}

/**
 * @brief Returns the second invariant of a 3x3 matrix
 * @param[in] A The matrix
 */
template <typename T>
SMITH_HOST_DEVICE constexpr auto I2(const tensor<T, 3, 3>& A)
{
  return +A[0][0] * A[1][1] + A[1][1] * A[2][2] + A[2][2] * A[0][0] - A[0][1] * A[1][0] - A[1][2] * A[2][1] -
         A[2][0] * A[0][2];
}

/**
 * @brief Returns the determinant of a matrix
 * @param[in] A The matrix to obtain the determinant of
 */
template <typename T>
SMITH_HOST_DEVICE constexpr auto det(const tensor<T, 2, 2>& A)
{
  return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

/// @overload
template <typename T>
SMITH_HOST_DEVICE constexpr auto det(const tensor<T, 3, 3>& A)
{
  return A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] + A[0][2] * A[1][0] * A[2][1] -
         A[0][0] * A[1][2] * A[2][1] - A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[1][1] * A[2][0];
}

/**
 * @brief computes det(A + I) - 1, where precision is not lost when the entries A_{ij} << 1
 *
 * detApIm1(A) = det(A + I) - 1
 * When the entries of A are small compared to unity, computing
 * det(A + I) - 1 directly will suffer from catastrophic cancellation.
 *
 * @param A Input matrix
 * @return det(A + I) - 1, where I is the identity matrix
 */
template <typename T>
SMITH_HOST_DEVICE constexpr auto detApIm1(const tensor<T, 2, 2>& A)
{
  // From the Cayley-Hamilton theorem, we get that for any N by N matrix A,
  // det(A - I) - 1 = I1(A) + I2(A) + ... + IN(A),
  // where the In are the principal invariants of A.
  // We inline the definitions of the principal invariants to increase computational speed.

  // equivalent to tr(A) + det(A)
  return A(0, 0) - A(0, 1) * A(1, 0) + A(1, 1) + A(0, 0) * A(1, 1);
}

/// @overload
template <typename T>
SMITH_HOST_DEVICE constexpr auto detApIm1(const tensor<T, 3, 3>& A)
{
  // For notes on the implementation, see the 2x2 version.

  // clang-format off
  // equivalent to tr(A) + I2(A) + det(A)
  return A(0, 0) + A(1, 1) + A(2, 2)
       - A(0, 1) * A(1, 0) * (1 + A(2, 2))
       + A(0, 0) * A(1, 1) * (1 + A(2, 2))
       - A(0, 2) * A(2, 0) * (1 + A(1, 1))
       - A(1, 2) * A(2, 1) * (1 + A(0, 0))
       + A(0, 0) * A(2, 2)
       + A(1, 1) * A(2, 2)
       + A(0, 1) * A(1, 2) * A(2, 0)
       + A(0, 2) * A(1, 0) * A(2, 1);
  // clang-format on
}

/**
 * @brief compute the matrix square root of a square, real-valued, symmetric matrix
 *        i.e. given A, find B such that A = dot(B, B)
 *
 * @tparam T the data type stored in each element of the matrix
 * @param A the matrix to compute the square root of
 * @return a square matrix, B, of the same type as A satisfying `dot(B, B) == A`
 */
template <typename T, int dim>
auto matrix_sqrt(const tensor<T, dim, dim>& A)
{
  auto B = A;
  for (int i = 0; i < 15; i++) {
    B = 0.5 * (B + dot(A, inv(B)));
  }
  return B;
}

/**
 * @brief a convenience function that computes a dot product between
 * two tensor, but that allows the user to specify which indices should
 * be summed over. For example:
 *
 * @code{.cpp}
 * tensor< double, 4, 4, 4 > A = ...;
 * tensor< double, 4, 4 > B = ...;
 * tensor< double, 4, 4, 4 > C = contract<1, 0>(A, B);
 *
 * //                 sum over index 1 for A
 * //                          V
 * // C(i, j, k) = \sum_l A(i, l, j) * B(l, k)
 * //                                    ^
 * //                          sum over index 0 for B
 *
 * @endcode
 *
 * @tparam i1 the index of contraction for the left operand
 * @tparam i2 the index of contraction for the right operand
 * @tparam S the datatype stored in the left operand
 * @tparam m leading dimension of the left operand
 * @tparam n the trailing dimensions of the left operand
 * @tparam T the datatype stored in the right operand
 * @tparam p the number of rows in the right operand
 * @tparam q the number of columns in the right operand
 * @param A the left operand
 * @param B the right operand
 */
template <int i1, int i2, typename S, int m, int... n, typename T, int p, int q>
SMITH_HOST_DEVICE auto contract(const tensor<S, m, n...>& A, const tensor<T, p, q>& B)
{
  constexpr int Adims[] = {m, n...};
  constexpr int Bdims[] = {p, q};
  static_assert(sizeof...(n) < 3);
  static_assert(Adims[i1] == Bdims[i2], "error: incompatible tensor dimensions");

  // first, we have to figure out the dimensions of the output tensor
  constexpr int new_dim = (i2 == 0) ? q : p;
  constexpr int d1 = (i1 == 0) ? new_dim : Adims[0];
  constexpr int d2 = (i1 == 1) ? new_dim : Adims[1];
  constexpr int d3 = sizeof...(n) == 1 ? 0 : ((i1 == 2) ? new_dim : Adims[2]);

  // the type of the output tensor is easier to figure out
  using U = decltype(S{} * T{});

  auto C = []() {
    if constexpr (d3 == 0) return tensor<U, d1, d2>{};
    if constexpr (d3 != 0) return tensor<U, d1, d2, d3>{};
  }();

  if constexpr (d3 == 0) {
    for (int i = 0; i < d1; i++) {
      for (int j = 0; j < d2; j++) {
        U sum{};
        for (int k = 0; k < Adims[i1]; k++) {
          if constexpr (i1 == 0 && i2 == 0) sum += A(k, j) * B(k, i);
          if constexpr (i1 == 1 && i2 == 0) sum += A(i, k) * B(k, j);
          if constexpr (i1 == 0 && i2 == 1) sum += A(k, j) * B(i, k);
          if constexpr (i1 == 1 && i2 == 1) sum += A(i, k) * B(j, k);
        }
        C(i, j) = sum;
      }
    }
  } else {
    for (int i = 0; i < d1; i++) {
      for (int j = 0; j < d2; j++) {
        for (int k = 0; k < d3; k++) {
          U sum{};
          for (int l = 0; l < Adims[i1]; l++) {
            if constexpr (i1 == 0 && i2 == 0) sum += A(l, j, k) * B(l, i);
            if constexpr (i1 == 1 && i2 == 0) sum += A(i, l, k) * B(l, j);
            if constexpr (i1 == 2 && i2 == 0) sum += A(i, j, l) * B(l, k);
            if constexpr (i1 == 0 && i2 == 1) sum += A(l, j, k) * B(i, l);
            if constexpr (i1 == 1 && i2 == 1) sum += A(i, l, k) * B(j, l);
            if constexpr (i1 == 2 && i2 == 1) sum += A(i, j, l) * B(k, l);
          }
          C(i, j, k) = sum;
        }
      }
    }
  }

  return C;
}

/// @overload
template <int i1, int i2, typename T>
SMITH_HOST_DEVICE auto contract(const zero&, const T&)
{
  return zero{};
}

/**
 * @brief computes the relative error (in the frobenius norm) between two tensors of the same shape
 *
 * @tparam T the datatype stored in each tensor
 * @tparam n the dimensions of each tensor
 * @param A the left argument
 * @param B the right argument
 * @return norm(A - B) / norm(A)
 */
template <typename T, int... n>
double relative_error(tensor<T, n...> A, tensor<T, n...> B)
{
  return norm(A - B) / norm(A);
}

/**
 * @brief Return whether a square rank 2 tensor is symmetric
 *
 * @tparam n The height of the tensor
 * @param A The square rank 2 tensor
 * @param tolerance The tolerance to check for symmetry
 * @return Whether the square rank 2 tensor (matrix) is symmetric
 */
template <int n>
SMITH_HOST_DEVICE bool is_symmetric(tensor<double, n, n> A, double tolerance = 1.0e-8)
{
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (std::abs(A(i, j) - A(j, i)) > tolerance) {
        return false;
      };
    }
  }
  return true;
}

/**
 * @brief Return whether a matrix is symmetric and positive definite
 * This check uses Sylvester's criterion, checking that each upper left subtensor has a
 * determinant greater than zero.
 *
 * @param A The matrix to test for positive definiteness
 * @return Whether the matrix is positive definite
 */
inline SMITH_HOST_DEVICE bool is_symmetric_and_positive_definite(tensor<double, 2, 2> A)
{
  if (!is_symmetric(A)) {
    return false;
  }
  if (A(0, 0) < 0.0) {
    return false;
  }
  if (det(A) < 0.0) {
    return false;
  }
  return true;
}
/// @overload
inline SMITH_HOST_DEVICE bool is_symmetric_and_positive_definite(tensor<double, 3, 3> A)
{
  if (!is_symmetric(A)) {
    return false;
  }
  if (det(A) < 0.0) {
    return false;
  }
  auto subtensor = make_tensor<2, 2>([A](int i, int j) { return A(i, j); });
  if (!is_symmetric_and_positive_definite(subtensor)) {
    return false;
  }
  return true;
}

/**
 * @brief Representation of an LU factorization
 *
 * The entries of P mean row i of the matrix was exchanged with row P[i].
 */
template <typename T, int n>
struct LuFactorization {
  tensor<int, n> P;   ///< Row permutation indices due to partial pivoting
  tensor<T, n, n> L;  ///< Lower triangular factor. Has ones on diagonal.
  tensor<T, n, n> U;  ///< Upper triangular factor
};

/**
 * @brief Solves a lower triangular system Ly = b
 *
 * L must be lower triangular and normalized such that the
 * diagonal entries are unity. This is not checked in the
 * function, so failure to obey this will produce
 * meaningless results.
 *
 * @param[in] L A lower triangular matrix
 * @param[in] b The right hand side
 * @param[in] P A list of indices to index into b in a permuted fashion.
 *
 * @return y the solution vector
 */
template <typename T, int n, int... m>
SMITH_HOST_DEVICE constexpr auto solve_lower_triangular(const tensor<T, n, n>& L, const tensor<T, n, m...>& b,
                                                        const tensor<int, n>& P)
{
  tensor<T, n, m...> y{};
  for (int i = 0; i < n; i++) {
    auto c = b[P[i]];
    for (int j = 0; j < i; j++) {
      c -= L[i][j] * y[j];
    }
    y[i] = c / L[i][i];
  }
  return y;
}

/**
 * @overload
 * @note For the case when no permutation of the rows is needed.
 */
template <typename T, int n, int... m>
SMITH_HOST_DEVICE constexpr auto solve_lower_triangular(const tensor<T, n, n>& L, const tensor<T, n, m...>& b)
{
  // no permutation provided, so just map each equation to itself
  // TODO make a convienience function for ranges like this
  // BT 05/09/2022
  tensor<int, n> P(make_tensor<n>([](auto i) { return i; }));

  return solve_lower_triangular(L, b, P);
}

/**
 * @brief Solves an upper triangular system Ux = y
 *
 * U must be upper triangular. This is not checked, so
 * failure to obey this will produce meaningless results.
 *
 * @param[in] U An upper triangular matrix
 * @param[in] y The right hand side
 * @return x the solution vector
 */
template <typename T, int n, int... m>
SMITH_HOST_DEVICE constexpr auto solve_upper_triangular(const tensor<T, n, n>& U, const tensor<T, n, m...>& y)
{
  tensor<T, n, m...> x{};
  for (int i = n - 1; i >= 0; i--) {
    auto c = y[i];
    for (int j = i + 1; j < n; j++) {
      c -= U[i][j] * x[j];
    }
    x[i] = c / U[i][i];
  }
  return x;
}

/**
 * @overload
 * @note For use with a matrix that has already been factorized
 */
template <typename S, typename T, int n, int... m>
SMITH_HOST_DEVICE constexpr auto linear_solve(const LuFactorization<S, n>& lu_factors, const tensor<T, n, m...>& b)
{
  // Forward substitution
  // solve Ly = b
  const auto y = solve_lower_triangular(lu_factors.L, b, lu_factors.P);

  // Back substitution
  // Solve Ux = y
  return solve_upper_triangular(lu_factors.U, y);
}

/**
 * @overload
 * @note Shortcut for case of zero rhs
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto linear_solve(const LuFactorization<T, n>& /* lu_factors */, const zero /* b */)
{
  return zero{};
}

/**
 * @brief Inverts a matrix
 * @param[in] A The matrix to invert
 * @note Uses a shortcut for inverting a 2-by-2 matrix
 */
SMITH_HOST_DEVICE constexpr tensor<double, 2, 2> inv(const tensor<double, 2, 2>& A)
{
  double inv_detA(1.0 / det(A));

  tensor<double, 2, 2> invA{};

  invA[0][0] = A[1][1] * inv_detA;
  invA[0][1] = -A[0][1] * inv_detA;
  invA[1][0] = -A[1][0] * inv_detA;
  invA[1][1] = A[0][0] * inv_detA;

  return invA;
}

/**
 * @overload
 * @note Uses a shortcut for inverting a 3-by-3 matrix
 */
SMITH_HOST_DEVICE constexpr tensor<double, 3, 3> inv(const tensor<double, 3, 3>& A)
{
  double inv_detA(1.0 / det(A));

  tensor<double, 3, 3> invA{};

  invA[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_detA;
  invA[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_detA;
  invA[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_detA;
  invA[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_detA;
  invA[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_detA;
  invA[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_detA;
  invA[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_detA;
  invA[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_detA;
  invA[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_detA;

  return invA;
}
/**
 * @overload
 * @note For N-by-N matrices with N > 3, requires Gaussian elimination
 * with partial pivoting
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto inv(const tensor<T, n, n>& A)
{
  auto I = DenseIdentity<n>();
  return linear_solve(A, I);
}

/**
 * @brief recursively serialize the entries in a tensor to an ostream.
 * Output format uses braces and comma separators to mimic C syntax for multidimensional array
 * initialization.
 *
 * @param[in] out the std::ostream to write to (e.g. std::cout or std::ofstream)
 * @param[in] A The tensor to write out
 */
template <typename T, int m, int... n>
auto& operator<<(std::ostream& out, const tensor<T, m, n...>& A)
{
  out << '{' << A[0];
  for (int i = 1; i < m; i++) {
    out << ", " << A[i];
  }
  out << '}';
  return out;
}

/**
 * @brief Write a zero out to an output stream
 *
 * @param[in] out the std::ostream to write to (e.g. std::cout or std::ofstream)
 */
inline auto& operator<<(std::ostream& out, zero)
{
  out << "zero";
  return out;
}

/**
 * @brief print a double using `printf`, so that it is suitable for use inside cuda kernels. (used in final recursion of
 * printf(tensor<...>))
 * @param[in] value The value to write out
 */
inline SMITH_HOST_DEVICE void print(double value) { printf("%f", value); }

/**
 * @brief print a tensor using `printf`, so that it is suitable for use inside cuda kernels.
 * @param[in] A The tensor to write out
 */
template <int m, int... n>
SMITH_HOST_DEVICE void print(const tensor<double, m, n...>& A)
{
  printf("{");
  print(A[0]);
  for (int i = 1; i < m; i++) {
    printf(",");
    print(A[i]);
  }
  printf("}");
}

/**
 * @brief replace all entries in a tensor satisfying |x| < 1.0e-10 by literal zero
 * @param[in] A The tensor to "chop"
 */
template <int n>
SMITH_HOST_DEVICE constexpr auto chop(const tensor<double, n>& A)
{
  auto copy = A;
  for (int i = 0; i < n; i++) {
    if (copy[i] * copy[i] < 1.0e-20) {
      copy[i] = 0.0;
    }
  }
  return copy;
}

/// @overload
template <int m, int n>
SMITH_HOST_DEVICE constexpr auto chop(const tensor<double, m, n>& A)
{
  auto copy = A;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (copy[i][j] * copy[i][j] < 1.0e-20) {
        copy[i][j] = 0.0;
      }
    }
  }
  return copy;
}

/// @cond
namespace detail {

template <typename T1, typename T2>
struct outer_prod;

template <int... m, int... n>
struct outer_prod<tensor<double, m...>, tensor<double, n...>> {
  using type = tensor<double, m..., n...>;
};

template <int... n>
struct outer_prod<double, tensor<double, n...>> {
  using type = tensor<double, n...>;
};

template <int... n>
struct outer_prod<tensor<double, n...>, double> {
  using type = tensor<double, n...>;
};

template <>
struct outer_prod<double, double> {
  using type = tensor<double>;
};

template <typename T>
struct outer_prod<zero, T> {
  using type = zero;
};

template <typename T>
struct outer_prod<T, zero> {
  using type = zero;
};

}  // namespace detail
/// @endcond

/**
 * @brief a type function that returns the tensor type of an outer product of two tensors
 * @tparam T1 the first argument to the outer product
 * @tparam T2 the second argument to the outer product
 */
template <typename T1, typename T2>
using outer_product_t = typename detail::outer_prod<T1, T2>::type;

/**
 * @brief Retrieves the gradient component of a double (which is nothing)
 * @return The sentinel, @see zero
 */
inline SMITH_HOST_DEVICE auto get_gradient(double /* arg */) { return zero{}; }

/**
 * @brief get the gradient of type `tensor` (note: since its stored type is not a dual
 * number, the derivative term is identically zero)
 * @return The sentinel, @see zero
 */
template <int... n>
SMITH_HOST_DEVICE constexpr auto get_gradient(const tensor<double, n...>& /* arg */)
{
  return zero{};
}

/**
 * @brief evaluate the change (to first order) in a function, f, given a small change in the input argument, dx.
 */
SMITH_HOST_DEVICE constexpr auto chain_rule(const zero /* df_dx */, const zero /* dx */) { return zero{}; }

/**
 * @overload
 * @note this overload implements a no-op for the case where the gradient w.r.t. an input argument is identically zero
 */
template <typename T>
SMITH_HOST_DEVICE constexpr auto chain_rule(const zero /* df_dx */, const T /* dx */)
{
  return zero{};
}

/**
 * @overload
 * @note this overload implements a no-op for the case where the small change is indentically zero
 */
template <typename T>
SMITH_HOST_DEVICE constexpr auto chain_rule(const T /* df_dx */, const zero /* dx */)
{
  return zero{};
}

/**
 * @overload
 * @note for a scalar-valued function of a scalar, the chain rule is just multiplication
 */
SMITH_HOST_DEVICE constexpr auto chain_rule(const double df_dx, const double dx) { return df_dx * dx; }

/**
 * @overload
 * @note for a tensor-valued function of a scalar, the chain rule is just scalar multiplication
 */
template <int... n>
SMITH_HOST_DEVICE constexpr auto chain_rule(const tensor<double, n...>& df_dx, const double dx)
{
  return df_dx * dx;
}

/**
 * @overload
 * @note for a scalar-valued function of a tensor, the chain rule is the inner product
 */
template <int... n>
SMITH_HOST_DEVICE constexpr auto chain_rule(const tensor<double, n...>& df_dx, const tensor<double, n...>& dx)
{
  double total{};
  for_constexpr<n...>([&](auto... i) { total += df_dx(i...) * dx(i...); });
  return total;
}

/**
 * @overload
 * @note for a vector-valued function of a tensor, the chain rule contracts over all indices of dx
 */
template <int m, int... n>
SMITH_HOST_DEVICE constexpr auto chain_rule(const tensor<double, m, n...>& df_dx, const tensor<double, n...>& dx)
{
  tensor<double, m> total{};
  for (int i = 0; i < m; i++) {
    total[i] = chain_rule(df_dx[i], dx);
  }
  return total;
}

/**
 * @overload
 * @note for a matrix-valued function of a tensor, the chain rule contracts over all indices of dx
 */
template <int m, int n, int... p>
SMITH_HOST_DEVICE auto chain_rule(const tensor<double, m, n, p...>& df_dx, const tensor<double, p...>& dx)
{
  tensor<double, m, n> total{};
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      total[i][j] = chain_rule(df_dx[i][j], dx);
    }
  }
  return total;
}

/**
 * @brief returns the total number of stored values in a tensor
 *
 * @tparam T the datatype stored in the tensor
 * @tparam n the extents of each dimension
 * @return the total number of values stored in the tensor
 */
template <typename T, int... n>
SMITH_HOST_DEVICE constexpr int size(const tensor<T, n...>&)
{
  return (n * ... * 1);
}

/**
 * @overload
 * @brief overload of size() for `double`, we say a double "stores" 1 value
 */
SMITH_HOST_DEVICE constexpr int size(const double&) { return 1; }

/// @overload
SMITH_HOST_DEVICE constexpr int size(zero) { return 0; }

/**
 * @brief a function for querying the ith dimension of a tensor
 *
 * @tparam i which dimension to query
 * @tparam T the datatype stored in the tensor
 * @tparam n the tensor extents
 * @return the ith dimension
 */
template <int i, typename T, int... n>
SMITH_HOST_DEVICE constexpr int dimension(const tensor<T, n...>&)
{
  constexpr int dimensions[] = {n...};
  return dimensions[i];
}

/**
 * @brief a function for querying the first dimension of a tensor
 *
 * @tparam T the datatype stored in the tensor
 * @tparam m the first dimension of the tensor
 * @tparam n the trailing dimensions of the tensor
 * @return m
 */
template <typename T, int m, int... n>
SMITH_HOST_DEVICE constexpr int leading_dimension(tensor<T, m, n...>)
{
  return m;
}

/// returns `true` if any entry of a tensor is `nan`
template <typename T, int... n>
bool isnan(const tensor<T, n...>& A)
{
  bool found_nan = false;
  for_constexpr<n...>([&](auto... i) { found_nan |= std::isnan(A(i...)); });
  return found_nan;
}

/// @overload
inline bool isnan(const zero&) { return false; }

}  // namespace smith

#if 0

inline float angle_between(const vec < 2 > & a, const vec < 2 > & b) {
  return acos(clip(dot(normalize(a), normalize(b)), -1.0f, 1.0f));
}

inline float angle_between(const vec < 3 > & a, const vec < 3 > & b) {
  return acos(clip(dot(normalize(a), normalize(b)), -1.0f, 1.0f));
}

// angle between proper orthogonal matrices
inline float angle_between(const mat < 3, 3 > & U, const mat < 3, 3 > & V) {
  return acos(0.5f * (tr(dot(U, transpose(V))) - 1.0f));
}

inline mat < 2, 2 > rotation(const float theta) {
  return mat< 2, 2 >{
    {cos(theta), -sin(theta)},
    { sin(theta), cos(theta) }
  };
}

inline mat < 3, 3 > axis_to_rotation(const vec < 3 > & omega) {

  float norm_omega = norm(omega);

  if (fabs(norm_omega) < 0.000001f) {

    return eye< 3 >();

  } else {

    vec3 u = omega / norm_omega;

    float c = cos(norm_omega);
    float s = sin(norm_omega);

    return mat < 3, 3 >{
      {
        u[0]*u[0]*(1.0f - c) + c,
        u[0]*u[1]*(1.0f - c) - u[2]*s,
        u[0]*u[2]*(1.0f - c) + u[1]*s
      },{
        u[1]*u[0]*(1.0f - c) + u[2]*s,
        u[1]*u[1]*(1.0f - c) + c,
        u[1]*u[2]*(1.0f - c) - u[0]*s
      },{
        u[2]*u[0]*(1.0f - c) - u[1]*s,
        u[2]*u[1]*(1.0f - c) + u[0]*s,
        u[2]*u[2]*(1.0f - c) + c
      }
    };

  }

}

// assumes R is a proper-orthogonal matrix
inline vec < 3 > rotation_to_axis(const mat < 3, 3 > & R) {

  float theta = acos(clip(0.5f * (tr(R) - 1.0f), -1.0f, 1.0f));

  float scale;

  // for small angles, prefer series expansion to division by sin(theta) ~ 0
  if (fabs(theta) < 0.00001f) {
    scale = 0.5f + theta * theta / 12.0f;
  }
  else {
    scale = 0.5f * theta / sin(theta);
  }

  return vec3{ R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1) } *scale;

}

inline mat < 3, 3 > look_at(const vec < 3 > & direction, const vec < 3 > & up = vec3{ 0.0f, 0.0f, 1.0f }) {
  vec3 f = normalize(direction);
  vec3 u = normalize(cross(f, cross(up, f)));
  vec3 l = normalize(cross(u, f));

  return mat3{
    {f[0], l[0], u[0]},
    {f[1], l[1], u[1]},
    {f[2], l[2], u[2]}
  };
}

inline mat < 2, 2 > look_at(const vec < 2 > & direction) {
  vec2 f = normalize(direction);
  vec2 l = cross(f);

  return mat2{
    {f[0], l[0]},
    {f[1], l[1]},
  };
}

inline mat < 3, 3 > R3_basis(const vec3 & n) {
  float sign = (n[2] >= 0.0f) ? 1.0f : -1.0f;
  float a = -1.0f / (sign + n[2]);
  float b = n[0] * n[1] * a;

  return mat < 3, 3 >{
    {
      1.0f + sign * n[0] * n[0] * a,
      b,
      n[0],
    },{
      sign * b,
      sign + n[1] * n[1] * a,
      n[1]
    },{
      -sign * n[0],
      -n[1],
      n[2]
    }
  };
}
#endif

#include "smith/numerics/functional/isotropic_tensor.hpp"

#include "smith/numerics/functional/tuple_tensor_dual_functions.hpp"
