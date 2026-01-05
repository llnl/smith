// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "mfem.hpp"

#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/dual.hpp"

namespace smith {

/** @brief class for checking if a type is a tensor of dual numbers or not */
template <typename T>
struct is_tensor_of_dual_number {
  static constexpr bool value = false;  ///< whether or not type T is a dual number
};

/** @brief class for checking if a type is a tensor of dual numbers or not */
template <typename T, int... n>
struct is_tensor_of_dual_number<tensor<dual<T>, n...>> {
  static constexpr bool value = true;  ///< whether or not type T is a dual number
};

/**
 * @brief multiply a tensor by a scalar value
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] scale The scaling factor
 * @param[in] A The tensor to be scaled
 */
template <typename S, typename T, int m, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SMITH_HOST_DEVICE constexpr auto operator*(S scale, const tensor<T, m, n...>& A)
{
  tensor<decltype(S{} * T{}), m, n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = scale * A[i];
  }
  return C;
}

/**
 * @brief multiply a tensor by a scalar value
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor to be scaled
 * @param[in] scale The scaling factor
 */
template <typename S, typename T, int m, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SMITH_HOST_DEVICE constexpr auto operator*(const tensor<T, m, n...>& A, S scale)
{
  tensor<decltype(T{} * S{}), m, n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = A[i] * scale;
  }
  return C;
}

/**
 * @brief divide a scalar by each element in a tensor
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] scale The numerator
 * @param[in] A The tensor of denominators
 */
template <typename S, typename T, int m, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SMITH_HOST_DEVICE constexpr auto operator/(S scale, const tensor<T, m, n...>& A)
{
  tensor<decltype(S{} * T{}), n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = scale / A[i];
  }
  return C;
}

/**
 * @brief divide a tensor by a scalar
 * @tparam S the scalar value type. Must be arithmetic (e.g. float, double, int) or a dual number
 * @tparam T the underlying type of the tensor (righthand) argument
 * @tparam n integers describing the tensor shape
 * @param[in] A The tensor of numerators
 * @param[in] scale The denominator
 */
template <typename S, typename T, int m, int... n,
          typename = std::enable_if_t<std::is_arithmetic_v<S> || is_dual_number<S>::value>>
SMITH_HOST_DEVICE constexpr auto operator/(const tensor<T, m, n...>& A, S scale)
{
  tensor<decltype(T{} * S{}), m, n...> C{};
  for (int i = 0; i < m; i++) {
    C[i] = A[i] / scale;
  }
  return C;
}

/// @cond
template <int i, typename S, typename T>
struct one_hot_helper;

template <int i, int... I, typename T>
struct one_hot_helper<i, std::integer_sequence<int, I...>, T> {
  using type = tuple<std::conditional_t<i == I, T, zero>...>;
};

template <int i, int n, typename T>
struct one_hot : public one_hot_helper<i, std::make_integer_sequence<int, n>, T> {};
/// @endcond

/**
 * @brief a tuple type with n entries, all of which are of type `smith::zero`,
 * except for the i^{th} entry, which is of type T
 *
 *  e.g. one_hot_t< 2, 4, T > == tuple<zero, zero, T, zero>
 */
template <int i, int n, typename T>
using one_hot_t = typename one_hot<i, n, T>::type;

/// @overload
template <int i, int N>
SMITH_HOST_DEVICE constexpr auto make_dual_helper(zero /*arg*/)
{
  return zero{};
}

/**
 * @tparam i the index where the non-`smith::zero` derivative term appears
 * @tparam N how many entries in the gradient type
 *
 * @brief promote a double value to dual number with a one_hot_t< i, N, double > gradient type
 * @param arg the value to be promoted
 */
template <int i, int N>
SMITH_HOST_DEVICE constexpr auto make_dual_helper(double arg)
{
  using gradient_t = one_hot_t<i, N, double>;
  dual<gradient_t> arg_dual{};
  arg_dual.value = arg;
  smith::get<i>(arg_dual.gradient) = 1.0;
  return arg_dual;
}

/**
 * @tparam i the index where the non-`smith::zero` derivative term appears
 * @tparam N how many entries in the gradient type
 *
 * @brief promote a tensor value to dual number with a one_hot_t< i, N, tensor > gradient type
 * @param arg the value to be promoted
 */
template <int i, int N, typename T, int... n>
SMITH_HOST_DEVICE constexpr auto make_dual_helper(const tensor<T, n...>& arg)
{
  using gradient_t = one_hot_t<i, N, tensor<T, n...>>;
  tensor<dual<gradient_t>, n...> arg_dual{};
  for_constexpr<n...>([&](auto... j) {
    arg_dual(j...).value = arg(j...);
    smith::get<i>(arg_dual(j...).gradient)(j...) = 1.0;
  });
  return arg_dual;
}

/**
 * @tparam T0 the first type of the tuple argument
 * @tparam T1 the first type of the tuple argument
 *
 * @brief Promote a tuple of values to their corresponding dual types
 * @param args the values to be promoted
 *
 * example:
 * @code{.cpp}
 * smith::tuple < double, tensor< double, 3 > > f{};
 *
 * smith::tuple <
 *   dual < smith::tuple < double, zero > >
 *   tensor < dual < smith::tuple < zero, tensor< double, 3 > >, 3 >
 * > dual_of_f = make_dual(f);
 * @endcode
 */
template <typename T0, typename T1>
SMITH_HOST_DEVICE constexpr auto make_dual(const tuple<T0, T1>& args)
{
  return tuple{make_dual_helper<0, 2>(get<0>(args)), make_dual_helper<1, 2>(get<1>(args))};
}

/// @overload
template <typename T0, typename T1, typename T2>
SMITH_HOST_DEVICE constexpr auto make_dual(const tuple<T0, T1, T2>& args)
{
  return tuple{make_dual_helper<0, 3>(get<0>(args)), make_dual_helper<1, 3>(get<1>(args)),
               make_dual_helper<2, 3>(get<2>(args))};
}

/**
 * @tparam dualify specify whether or not the value should be made into its dual type
 * @tparam T the type of the value passed in
 *
 * @brief a function that optionally (decided at compile time) converts a value to its dual type
 * @param x the values to be promoted
 */
template <bool dualify, typename T>
SMITH_HOST_DEVICE auto promote_to_dual_when(const T& x)
{
  if constexpr (dualify) {
    return make_dual(x);
  }
  if constexpr (!dualify) {
    return x;
  }
}

/**
 * @brief a function that optionally (decided at compile time) converts a list of values to their dual types
 *
 * @tparam dualify specify whether or not the input should be made into its dual type
 * @tparam T the type of the values passed in
 * @tparam n how many values were passed in
 * @param x the values to be promoted
 */
template <bool dualify, typename T, int n>
SMITH_HOST_DEVICE auto promote_each_to_dual_when(const tensor<T, n>& x)
{
  if constexpr (dualify) {
    using return_type = decltype(make_dual(T{}));
    tensor<return_type, n> output;
    for (int i = 0; i < n; i++) {
      output[i] = make_dual(x[i]);
    }
    return output;
  }
  if constexpr (!dualify) {
    return x;
  }
}

/// @brief layer of indirection required to implement `make_dual_wrt`
template <int n, typename... T, int... i>
SMITH_HOST_DEVICE constexpr auto make_dual_helper(const smith::tuple<T...>& args, std::integer_sequence<int, i...>)
{
  // Sam: it took me longer than I'd like to admit to find this issue, so here's an explanation
  //
  // note: we use smith::make_tuple(...) instead of smith::tuple{...} here because if
  // the first argument passed in is of type `smith::tuple < smith::tuple < T ... > >`
  // then doing something like
  //
  // smith::tuple{smith::get<i>(args)...};
  //
  // will be expand to something like
  //
  // smith::tuple{smith::tuple< T ... >{}};
  //
  // which invokes the copy ctor, returning a `smith::tuple< T ... >`
  // instead of `smith::tuple< smith::tuple < T ... > >`
  //
  // but smith::make_tuple(smith::get<i>(args)...) will never accidentally trigger the copy ctor
  return smith::make_tuple(promote_to_dual_when < i == n > (smith::get<i>(args))...);
}

/**
 * @tparam n the index of the tuple argument to be made into a dual number
 * @tparam T the types of the values in the tuple
 *
 * @brief take a tuple of values, and promote the `n`th one to a one-hot dual number of the appropriate type
 * @param args the values to be promoted
 */
template <int n, typename... T>
constexpr auto make_dual_wrt(const smith::tuple<T...>& args)
{
  return make_dual_helper<n>(args, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>{});
}

/**
 * @brief Extracts all of the values from a tensor of dual numbers
 *
 * @tparam T1 the first type of the tuple stored in the tensor
 * @tparam T2 the second type of the tuple stored in the tensor
 * @tparam n  the number of entries in the input argument
 * @param[in] input The tensor of dual numbers
 * @return the tensor of all of the values
 */
template <typename T1, typename T2, int n>
SMITH_HOST_DEVICE auto get_value(const tensor<tuple<T1, T2>, n>& input)
{
  tensor<decltype(get_value(tuple<T1, T2>{})), n> output{};
  for (int i = 0; i < n; i++) {
    output[i] = get_value(input[i]);
  }
  return output;
}

/**
 * @brief Retrieves the value components of a set of (possibly dual) numbers
 * @param[in] tuple_of_values The tuple of numbers to retrieve values from
 * @pre The tuple must contain only scalars or tensors of @p dual numbers or doubles
 */
template <typename... T>
SMITH_HOST_DEVICE auto get_value(const smith::tuple<T...>& tuple_of_values)
{
  return smith::apply([](const auto&... each_value) { return smith::tuple{get_value(each_value)...}; },
                      tuple_of_values);
}

/**
 * @brief Retrieves the gradient components of a set of dual numbers
 * @param[in] arg The set of numbers to retrieve gradients from
 */
template <typename... T>
SMITH_HOST_DEVICE auto get_gradient(dual<smith::tuple<T...>> arg)
{
  return smith::apply([](auto... each_value) { return smith::tuple{each_value...}; }, arg.gradient);
}

/// @overload
template <typename... T, int... n>
SMITH_HOST_DEVICE auto get_gradient(const tensor<dual<smith::tuple<T...>>, n...>& arg)
{
  smith::tuple<outer_product_t<tensor<double, n...>, T>...> g{};
  for_constexpr<n...>([&](auto... i) {
    for_constexpr<sizeof...(T)>([&](auto j) { smith::get<j>(g)(i...) = smith::get<j>(arg(i...).gradient); });
  });
  return g;
}

/// @overload
template <typename... T>
SMITH_HOST_DEVICE auto get_gradient(smith::tuple<T...> tuple_of_values)
{
  return smith::apply([](auto... each_value) { return smith::tuple{get_gradient(each_value)...}; }, tuple_of_values);
}

/**
 * @brief Constructs a tensor of dual numbers from a tensor of values
 * @param[in] A The tensor of values
 * @note a d-order tensor's gradient will be initialized to the (2*d)-order identity tensor
 */
template <int... n>
SMITH_HOST_DEVICE constexpr auto make_dual(const tensor<double, n...>& A)
{
  tensor<dual<tensor<double, n...>>, n...> A_dual{};
  for_constexpr<n...>([&](auto... i) {
    A_dual(i...).value = A(i...);
    A_dual(i...).gradient(i...) = 1.0;
  });
  return A_dual;
}

/**
 * @brief Compute LU factorization of a matrix with partial pivoting
 *
 * The convention followed is to place ones on the diagonal of the lower
 * triangular factor.
 * @param[in] A The matrix to factorize
 * @return An LuFactorization object
 * @see LuFactorization
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr LuFactorization<T, n> factorize_lu(const tensor<T, n, n>& A)
{
  constexpr auto abs = [](double x) { return (x < 0) ? -x : x; };
  constexpr auto swap = [](auto& x, auto& y) {
    auto tmp = x;
    x = y;
    y = tmp;
  };

  auto U = A;
  // initialize L to Identity
  auto L = tensor<T, n, n>{};
  // This handles the case if T is a dual number
  // TODO - BT: make a dense identity that is templated on type
  for (int i = 0; i < n; i++) {
    if constexpr (is_dual_number<T>::value) {
      L[i][i].value = 1.0;
    } else {
      L[i][i] = 1.0;
    }
  }
  tensor<int, n> P(make_tensor<n>([](auto i) { return i; }));

  for (int i = 0; i < n; i++) {
    // Search for maximum in this column
    double max_val = abs(get_value(U[i][i]));

    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      auto U_ji = get_value(U[j][i]);
      if (abs(U_ji) > max_val) {
        max_val = abs(U_ji);
        max_row = j;
      }
    }

    swap(P[max_row], P[i]);
    swap(U[max_row], U[i]);
  }

  for (int i = 0; i < n; i++) {
    // zero entries below in this column in U
    // and fill in L entries
    for (int j = i + 1; j < n; j++) {
      auto c = U[j][i] / U[i][i];
      L[j][i] = c;
      U[j] -= c * U[i];
      U[j][i] = T{};
    }
  }

  return {P, L, U};
}

/**
 * @brief Solves Ax = b for x using Gaussian elimination with partial pivoting
 * @param[in] A The coefficient matrix A
 * @param[in] b The righthand side vector b
 * @return x The solution vector
 */
template <typename S, typename T, int n, int... m>
SMITH_HOST_DEVICE constexpr auto linear_solve(const tensor<S, n, n>& A, const tensor<T, n, m...>& b)
{
  // We want to avoid accumulating the derivative through the
  // LU factorization, because it is computationally expensive.
  // Instead, we perform the LU factorization on the values of
  // A, and then two backsolves: one to compute the primal (x),
  // and another to compute its derivative (dx).
  // If A is not dual, the second solve is a no-op.

  // Strip off derivatives, if any, and compute only x (ie no derivative)
  auto lu_factors = factorize_lu(get_value(A));
  auto x = linear_solve(lu_factors, get_value(b));

  // Compute directional derivative of x.
  // If both b and A are not dual, the zero type
  // makes these no-ops.
  auto r = get_gradient(b) - dot(get_gradient(A), x);
  auto dx = linear_solve(lu_factors, r);

  if constexpr (is_zero<decltype(dx)>{}) {
    return x;
  } else {
    return make_dual(x, dx);
  }
}

/**
 * @brief Create a tensor of dual numbers with specified seed
 */
template <typename T, int n>
SMITH_HOST_DEVICE constexpr auto make_dual(const tensor<T, n>& x, const tensor<T, n>& dx)
{
  return make_tensor<n>([&](int i) { return dual<T>{x[i], dx[i]}; });
}

/**
 * @brief Create a tensor of dual numbers with specified seed
 */
template <typename T, int m, int n>
SMITH_HOST_DEVICE constexpr auto make_dual(const tensor<T, m, n>& x, const tensor<T, m, n>& dx)
{
  return make_tensor<m, n>([&](int i, int j) { return dual<T>{x[i][j], dx[i][j]}; });
}

/**
 * @overload
 * @note when inverting a tensor of dual numbers,
 * hardcode the analytic derivative of the
 * inverse of a square matrix, rather than
 * apply gauss elimination directly on the dual number types
 *
 * TODO: compare performance of this hardcoded implementation to just using inv() directly
 */
template <typename gradient_type, int n>
SMITH_HOST_DEVICE constexpr auto inv(tensor<dual<gradient_type>, n, n> A)
{
  auto invA = inv(get_value(A));
  return make_tensor<n, n>([&](int i, int j) {
    auto value = invA[i][j];
    gradient_type gradient{};
    for (int k = 0; k < n; k++) {
      for (int l = 0; l < n; l++) {
        gradient -= invA[i][k] * A[k][l].gradient * invA[l][j];
      }
    }
    return dual<gradient_type>{value, gradient};
  });
}

/**
 * @brief Retrieves a value tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <typename T, int... n>
SMITH_HOST_DEVICE auto get_value(const tensor<dual<T>, n...>& arg)
{
  tensor<double, n...> value{};
  for_constexpr<n...>([&](auto... i) { value(i...) = arg(i...).value; });
  return value;
}

/**
 * @brief Retrieves a gradient tensor from a tensor of dual numbers
 * @param[in] arg The tensor of dual numbers
 */
template <int... n>
SMITH_HOST_DEVICE constexpr auto get_gradient(const tensor<dual<double>, n...>& arg)
{
  tensor<double, n...> g{};
  for_constexpr<n...>([&](auto... i) { g(i...) = arg(i...).gradient; });
  return g;
}

/// @overload
template <int... n, int... m>
SMITH_HOST_DEVICE constexpr auto get_gradient(const tensor<dual<tensor<double, m...>>, n...>& arg)
{
  tensor<double, n..., m...> g{};
  for_constexpr<n...>([&](auto... i) { g(i...) = arg(i...).gradient; });
  return g;
}

/**
 * @brief Status and diagnostics of nonlinear equation solvers
 */
struct SolverStatus {
  bool converged;           ///< converged Flag indicating whether solver converged to a solution or aborted.
  unsigned int iterations;  ///< Number of iterations taken
  double residual;          ///< Final value of residual.
};

/**
 * @brief Settings for @p solve_scalar_equation
 */
struct ScalarSolverOptions {
  double xtol;            ///< absolute tolerance on Newton correction
  double rtol;            ///< absolute tolerance on absolute value of residual
  unsigned int max_iter;  ///< maximum allowed number of iterations
};

/// @brief Default options for @p solve_scalar_equation
const ScalarSolverOptions default_solver_options{.xtol = 1e-8, .rtol = 0, .max_iter = 25};

/// @brief Solves a nonlinear scalar-valued equation and gives derivatives of solution to parameters
///
/// @tparam function Function object type for the nonlinear equation to solve
/// @tparam ...ParamTypes Types of the (optional) parameters to the nonlinear function
///
/// @param f Nonlinear function of which a root is sought. Must have the form
/// $f(x, p_1, p_2, ...)$, where $x$ is the independent variable, and the $p_i$ are
/// optional parameters (scalars or tensors of arbitrary order).
/// @param x0 Initial guess of root. If x0 is outside the search interval, the initial
/// guess will be changed to the midpoint of the search interval.
/// @param lower_bound Lower bound of interval to search for root.
/// @param upper_bound Upper bound of interval to search for root.
/// @param options Options controlling behavior of solver.
/// @param ...params Optional parameters to the nonlinear function.
///
/// @return a tuple (@p x, @p status) where @p x is the root, and @p status is a SolverStatus
/// object reporting the status of the solution procedure. If any of the parameters are
/// dual number-valued, @p x will be dual containing the corresponding directional derivative
/// of the root. Otherwise, x will be a @p double containing the root.
/// For example, if one gives the function as $f(x, p)$, where $p$ is a @p dual<double> with
/// @p p.gradient = 1, then the @p x.gradient will be $dx/dp$.
///
/// The solver uses Newton's method, safeguarded by bisection. If the Newton update would take
/// the next iterate out of the search interval, or the absolute value of the residual is not
/// decreasing fast enough, bisection will be used to compute the next iterate. The bounds of the
/// search interval are updated automatically to maintain a bracket around the root. If the sign
/// of the residual is the same at both @p lower_bound and @p upper_bound, the solver aborts.
template <typename function, typename... ParamTypes>
auto solve_scalar_equation(const function& f, double x0, double lower_bound, double upper_bound,
                           ScalarSolverOptions options, ParamTypes... params)
{
  double x, df_dx;
  double fl = f(lower_bound, get_value(params)...);
  double fh = f(upper_bound, get_value(params)...);

  SLIC_ERROR_ROOT_IF(fl * fh > 0, "solve_scalar_equation: root not bracketed by input bounds.");

  unsigned int iterations = 0;
  bool converged = false;

  // handle corner cases where one of the brackets is the root
  if (fl == 0) {
    x = lower_bound;
    converged = true;
  } else if (fh == 0) {
    x = upper_bound;
    converged = true;
  }

  if (converged) {
    df_dx = get_gradient(f(make_dual(x), get_value(params)...));

  } else {
    // orient search so that f(xl) < 0
    double xl = lower_bound;
    double xh = upper_bound;
    if (fl > 0) {
      xl = upper_bound;
      xh = lower_bound;
    }

    // move initial guess if it is not between brackets
    if (x0 < lower_bound || x0 > upper_bound) {
      x0 = 0.5 * (lower_bound + upper_bound);
    }

    x = x0;
    double delta_x_old = std::abs(upper_bound - lower_bound);
    double delta_x = delta_x_old;
    auto R = f(make_dual(x), get_value(params)...);
    auto fval = get_value(R);
    df_dx = get_gradient(R);

    while (!converged) {
      if (iterations == options.max_iter) {
        SLIC_WARNING("solve_scalar_equation failed to converge in allotted iterations.");
        break;
      }

      // use bisection if Newton oversteps brackets or is not decreasing sufficiently
      if ((x - xh) * df_dx - fval > 0 || (x - xl) * df_dx - fval < 0 ||
          std::abs(2. * fval) > std::abs(delta_x_old * df_dx)) {
        delta_x_old = delta_x;
        delta_x = 0.5 * (xh - xl);
        x = xl + delta_x;
        converged = (x == xl);
      } else {  // use Newton step
        delta_x_old = delta_x;
        delta_x = fval / df_dx;
        auto temp = x;
        x -= delta_x;
        converged = (x == temp);
      }

      // function and jacobian evaluation
      R = f(make_dual(x), get_value(params)...);
      fval = get_value(R);
      df_dx = get_gradient(R);

      // convergence check
      converged = converged || (std::abs(delta_x) < options.xtol) || (std::abs(fval) < options.rtol);

      // maintain bracket on root
      if (fval < 0) {
        xl = x;
      } else {
        xh = x;
      }

      ++iterations;
    }
  }

  // Accumulate derivatives so that the user can get derivatives
  // with respect to parameters, subject to constraing that f(x, p) = 0 for all p
  // Conceptually, we're doing the following:
  // [fval, df_dp] = f(get_value(x), p)
  // df = 0
  // for p in params:
  //   df += inner(df_dp, dp)
  // dx = -df / df_dx
  constexpr bool contains_duals =
      (is_dual_number<ParamTypes>::value || ...) || (is_tensor_of_dual_number<ParamTypes>::value || ...);
  if constexpr (contains_duals) {
    auto [fval, df] = f(x, params...);
    auto dx = -df / df_dx;
    SolverStatus status{.converged = converged, .iterations = iterations, .residual = fval};
    return tuple{dual{x, dx}, status};
  }
  if constexpr (!contains_duals) {
    auto fval = f(x, params...);
    SolverStatus status{.converged = converged, .iterations = iterations, .residual = fval};
    return tuple{x, status};
  }
}

/**
 * @brief Finds a root of a vector-valued nonlinear function
 *
 * Uses Newton-Raphson iteration.
 *
 * @tparam function Type for the functor object
 * @tparam n Vector dimension of the equation
 * @param f A callable representing the function of which a root is sought. Must take an n-vector
 * argument and return an n-vector
 * @param x0 Initial guess for root. Must be an n-vector.
 * @return A root of @p f.
 */
template <typename function, int n>
auto find_root(const function& f, tensor<double, n> x0)
{
  static_assert(std::is_same_v<decltype(f(x0)), tensor<double, n>>,
                "error: f(x) must have the same number of equations as unknowns");

  double epsilon = 1.0e-8;
  int max_iterations = 10;

  auto x = x0;

  for (int k = 0; k < max_iterations; k++) {
    auto output = f(make_dual(x));
    auto r = get_value(output);
    if (norm(r) < epsilon) break;
    auto J = get_gradient(output);
    x -= linear_solve(J, r);
  }

  return x;
};

/**
 * @brief compute the eigenvalues of a symmetric matrix A
 *
 * @tparam T either `double` or a `smith::dual` type
 * @tparam size the dimensions of the matrix
 * @param A the matrix
 * @return a vector of the eigenvalues of A (and their derivatives, if A contains dual numbers)
 */
template <typename T, int size>
auto eigenvalues(const smith::tensor<T, size, size>& A)
{
  // put tensor values in an mfem::DenseMatrix
  mfem::DenseMatrix matA(size, size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if constexpr (is_dual_number<T>::value) {
        matA(i, j) = A[i][j].value;
      } else {
        matA(i, j) = A[i][j];
      }
    }
  }

  // compute eigendecomposition
  mfem::DenseMatrixEigensystem eig_sys(matA);
  eig_sys.Eval();

  smith::tensor<T, size> output;

  for (int k = 0; k < size; k++) {
    // extract eigenvalues
    output[k] = eig_sys.Eigenvalue(k);

    // and calculate their derivatives, when appropriate
    if constexpr (is_dual_number<T>::value) {
      tensor<double, size> phi = make_tensor<size>([&](int i) { return eig_sys.Eigenvector(k)[i]; });
      auto dA = make_tensor<size, size>([&](int i, int j) { return A(i, j).gradient; });
      output[k].gradient = dot(phi, dA, phi);
    }
  }

  return output;
}

/**
 * @brief Signum, returns sign of input
 *
 * @param val Input value.
 * @return 0 when input is negative, 0 when input is 0, 1 when input is positive.
 */
template <typename T>
int sgn(T val)
{
  // Should we implement the derivative?
  // It should be NaN when val = 0
  return (T(0) < val) - (val < T(0));
}

/**
 * @brief Find indices that would sort a 3-vector
 *
 * @param v 3-vector to sort.
 * @return 3-vector of indices that would sort \p v in ascending order.
 */
template <typename T>
SMITH_HOST_DEVICE tensor<int, 3> argsort(const tensor<T, 3>& v)
{
  auto swap = [](int& first, int& second) {
    int tmp = first;
    first = second;
    second = tmp;
  };
  tensor<int, 3> order{0, 1, 2};
  if (v[0] > v[1]) swap(order[0], order[1]);
  if (v[order[1]] > v[order[2]]) swap(order[1], order[2]);
  if (v[order[0]] > v[order[1]]) swap(order[0], order[1]);
  return order;
}

/** Eigendecomposition for a symmetric 3x3 matrix
 *
 * @param A Matrix for which the eigendecomposition will be computed. Must be
 * symmetric, this is not checked.
 * @return tuple with the eigenvalues in the first element, and the matrix of
 * eigenvectors (columnwise) in the second element.
 *
 * @note based on "A robust algorithm for finding the eigenvalues and
 * eigenvectors of 3x3 symmetric matrices", by Scherzinger & Dohrmann
 */
inline SMITH_HOST_DEVICE tuple<vec3, mat3> eig_symm(const mat3& A)
{
  // We know of optimizations for this routine. When this becomes the
  // bottleneck, we can revisit. See OptimiSM for details.

  tensor<double, 3> eta{};
  tensor<double, 3, 3> Q = DenseIdentity<3>();

  auto A_dev = dev(A);
  double J2 = 0.5 * inner(A_dev, A_dev);
  double J3 = det(A_dev);

  if (J2 > 0.0) {
    // angle used to find eigenvalues
    double tmp = (0.5 * J3) * std::pow(3.0 / J2, 1.5);
    double alpha = std::acos(fmin(fmax(tmp, -1.0), 1.0)) / 3.0;

    // consider the most distinct eigenvalue first
    if (6.0 * alpha < M_PI) {
      eta[0] = 2 * std::sqrt(J2 / 3.0) * std::cos(alpha);
    } else {
      eta[0] = 2 * std::sqrt(J2 / 3.0) * std::cos(alpha + 2.0 * M_PI / 3.0);
    }

    // find the eigenvector for that eigenvalue
    mat3 r;

    int imax = -1;
    double norm_max = -1.0;

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        r[i][j] = A_dev(j, i) - (i == j) * eta(0);
      }

      double norm_r = norm(r[i]);
      if (norm_max < norm_r) {
        imax = i;
        norm_max = norm_r;
      }
    }

    vec3 s0, s1, t1, t2, v0, v1, v2, w;

    s0 = normalize(r[imax]);
    t1 = r[(imax + 1) % 3] - dot(r[(imax + 1) % 3], s0) * s0;
    t2 = r[(imax + 2) % 3] - dot(r[(imax + 2) % 3], s0) * s0;
    s1 = normalize((norm(t1) > norm(t2)) ? t1 : t2);

    // record the first eigenvector
    v0 = cross(s0, s1);
    for (int i = 0; i < 3; i++) {
      Q[i][0] = v0[i];
    }

    // get the other two eigenvalues by solving the
    // remaining quadratic characteristic polynomial
    auto A_dev_s0 = dot(A_dev, s0);
    auto A_dev_s1 = dot(A_dev, s1);

    double A11 = dot(s0, A_dev_s0);
    double A12 = dot(s0, A_dev_s1);
    double A21 = A12;
    double A22 = dot(s1, A_dev_s1);

    double delta = 0.5 * std::sqrt((A11 - A22) * (A11 - A22) + 4 * A12 * A21);

    eta(1) = 0.5 * (A11 + A22) - delta;
    eta(2) = 0.5 * (A11 + A22) + delta;

    // if the remaining eigenvalues are exactly the same
    // then just use the basis for the orthogonal complement
    // found earlier
    if (fabs(delta) <= 1.0e-15) {
      for (int i = 0; i < 3; i++) {
        Q[i][1] = s0(i);
        Q[i][2] = s1(i);
      }

      // otherwise compute the remaining eigenvectors
    } else {
      t1 = A_dev_s0 - eta(1) * s0;
      t2 = A_dev_s1 - eta(1) * s1;

      w = normalize((norm(t1) > norm(t2)) ? t1 : t2);

      v1 = normalize(cross(w, v0));
      for (int i = 0; i < 3; i++) Q[i][1] = v1(i);

      // define the last eigenvector as
      // the direction perpendicular to the
      // first two directions
      v2 = normalize(cross(v0, v1));
      for (int i = 0; i < 3; i++) Q[i][2] = v2(i);
    }
  }
  // eta are actually eigenvalues of A_dev, so
  // shift them to get eigenvalues of A
  for (int i = 0; i < 3; i++) eta[i] += tr(A) / 3.0;

  // sort eigenvalues into ascending order
  auto order = argsort(eta);
  vec3 eigvals{{eta[order[0]], eta[order[1]], eta[order[2]]}};
  // clang-format off
  mat3 eigvecs{{{Q[0][order[0]], Q[0][order[1]], Q[0][order[2]]},
                {Q[1][order[0]], Q[1][order[1]], Q[1][order[2]]},
                {Q[2][order[0]], Q[2][order[1]], Q[2][order[2]]}}};
  // clang-format on

  return {eigvals, eigvecs};
}

/*
// Should we provide this fallback, or force the author to consider how to
// write a numerically stable version on a case-by-case basis?
// The convenience of this is somewhat undermined by the fact that it would
// only work for functions that already have a dual number overload.
template <typename Function>
double generic_eigenvalue_tangent(double lam1, double lam2, const Function& f)
{
  if (lam1 == lam2) {
    return f(make_dual(lam1));
  } else {
    return (f(lam1) - f(lam2))/(lam1 - lam2);
  }
}
*/

/**
 * @brief Constructs an isotropic tensor-valued function of a symmetric 3x3 tensor from a scalar function
 *
 * This allows one to use a scalar-valued function of a scalar to construct an
 * isotropic tensor-valued function of a symmetric tensor. The scalar function
 * is applied to the principal values of the matrix, and then rotated back into
 * the external coordinate system with the eigenvector matrix.
 *
 * If A = V diag(lambda_0, lambda_1, lambda_2) V^T,
 * then f(A) = V diag(f(lambda_0), f(lambda_1), f(lambda_2)) V^T
 *
 * The function \p g, which we call the eigenvalue secant function, is only used
 * if the derivative of the function is sought by having a dual number input
 * tensor \p A. It must compute
 *                 | df/dx, if lam1 = lam2
 * g(lam1, lam2) = |
 *                 | ( f(lam1) - f(lam2) ) / (lam1 - lam2), otherwise
 *
 * Analytically, this is a continuous function. However, in floating point arithmetic
 * the direct implementation will often suffer from catastrophic cancellation. The
 * presence of the \p g argument gives you a way to write this function in a numerically
 * stable way (and thus preserve accuracy in the derivative of the tensor function).
 *
 * @tparam T The datatype stored in the tensor
 * @tparam Function Type for the functor object representing the scalar function
 * @tparam EigvalSecantFunction Type for the functor object representing the secant eigenvalue function (see below)
 *
 * @param A The matrix to apply the isotropic tensor function to.
 * @param f A scalar-valued function of a scalar. This is applied to the eigenvalues of \p A.
 * @param g The eigenvalue secant function
 * @return The tensor f(A).
 */
template <typename T, typename Function, typename EigvalSecantFunction>
auto symmetric_mat3_function(tensor<T, 3, 3> A, const Function& f, const EigvalSecantFunction& g)
{
  auto [lambda, Q] = eig_symm(get_value(A));
  vec3 y;
  for (int i = 0; i < 3; i++) {
    y[i] = f(lambda[i]);
  }
  auto f_A = dot(Q, dot(diag(y), transpose(Q)));

  if constexpr (!is_dual_number<T>::value) {
    return f_A;
  } else {
    return symmetric_mat3_function_with_derivative(A, f_A, lambda, Q, g);
  }
}

/// Helper function for defining the derivative
template <typename Gradient, typename Function>
SMITH_HOST_DEVICE constexpr auto symmetric_mat3_function_with_derivative(tensor<dual<Gradient>, 3, 3> A,
                                                                         tensor<double, 3, 3> f_A, vec3 lambda, mat3 Q,
                                                                         const Function& g)
{
  return make_tensor<3, 3>([&](int i, int j) {
    auto value = f_A[i][j];
    Gradient gradient{};
    for (int k = 0; k < 3; k++) {
      for (int l = 0; l < 3; l++) {
        for (int a = 0; a < 3; a++) {
          for (int b = 0; b < 3; b++) {
            gradient += g(lambda[a], lambda[b]) * Q[k][a] * Q[l][b] * Q[i][a] * Q[j][b] * A[k][l].gradient;
          }
        }
      }
    }
    return dual<Gradient>{value, gradient};
  });
}

/**
 * @brief Logarithm of a symmetric matrix
 *
 * @param A Matrix to operate on. Must be SPD. This is not checked.
 * @return The logarithmic mapping of \p A.
 */
template <typename T>
auto log_symm(tensor<T, 3, 3> A)
{
  auto g = [](double lam1, double lam2) {
    if (lam1 == lam2) {
      return 1 / lam1;
    } else {
      double y = lam1 / lam2;
      return (std::log(y) / (y - 1.0)) / lam2;
    }
  };
  return symmetric_mat3_function(A, [](double x) { return std::log(x); }, g);
}

/**
 * @brief Exponential of a symmetric matrix
 *
 * @param A Matrix to operate on. Must be symmetric. This is not checked.
 * @return Exponential mapping of \p A.
 */
template <typename T>
auto exp_symm(tensor<T, 3, 3> A)
{
  auto g = [](double lam1, double lam2) {
    if (lam1 == lam2) {
      return std::exp(lam1);
    } else {
      double arg = lam1 - lam2;
      return std::exp(lam2) * std::expm1(arg) / arg;
    }
  };
  return symmetric_mat3_function(A, [](double x) { return std::exp(x); }, g);
}

/**
 * @brief Square root of a symmetric matrix
 *
 * @param A Matrix to operate on. Must be SPD. This is not checked.
 * @return Matrix B such that B*B = A
 */
template <typename T>
auto sqrt_symm(tensor<T, 3, 3> A)
{
  auto g = [](double lam1, double lam2) { return 1.0 / (std::sqrt(lam1) + std::sqrt(lam2)); };
  return symmetric_mat3_function(A, [](double x) { return std::sqrt(x); }, g);
}

}  // namespace smith
