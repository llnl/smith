// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <exception>
#include <cmath>

#include "gtest/gtest.h"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/dual.hpp"
#include "smith/numerics/functional/tensor.hpp"

class SlicErrorException : public std::exception {};

using namespace smith;

TEST(ScalarEquationSolver, ConvergesOnEasyProblem)
{
  double x0 = 2.0;
  double lower = 1e-3;
  double upper = 2.5;
  auto [x_soln, status] =
      solve_scalar_equation([](auto x) { return exp(x) - 2.0; }, x0, lower, upper, default_solver_options);
  double exact = std::log(2.0);
  double error = std::abs((x_soln - exact) / abs(exact));
  EXPECT_LT(error, default_solver_options.xtol);
}

TEST(ScalarEquationSolver, WorksWithScalarParameter)
{
  auto my_sqrt = [](dual<double> p) {
    double x0 = get_value(p);
    double lower = 0;
    double upper = (get_value(p) > 1.0) ? get_value(p) : 1.0;
    auto [x_soln, status] =
        solve_scalar_equation([](auto x, auto a) { return x * x - a; }, x0, lower, upper, default_solver_options, p);
    return x_soln;
  };
  double p = 2.0;
  auto [sqrt_p, dsqrt_p] = my_sqrt(make_dual(p));

  // check value
  double exact_value = std::sqrt(2.0);
  EXPECT_LT(std::abs(sqrt_p - exact_value) / exact_value, 1e-12);

  double exact_derivative = 0.5 / std::sqrt(p);
  EXPECT_LT(std::abs(dsqrt_p - exact_derivative) / std::abs(exact_derivative), 1e-12);
}

TEST(ScalarEquationSolver, AbortsIfRootNotBracketedByCaller)
{
  double x0 = 5.0;
  double lower = 2.0;
  double upper = 10.0;
  EXPECT_THROW(
      {
        [[maybe_unused]] auto result =
            solve_scalar_equation([](auto x) { return x * x - 2.0; }, x0, lower, upper, default_solver_options);
      },
      SlicErrorException);
}

TEST(ScalarEquationSolver, ReturnsImmediatelyIfUpperBoundIsARoot)
{
  double p = 4.0;
  double upper = 2.0;

  double x0 = 1.0;
  double lower = 0.0;

  auto [x_soln, status] =
      solve_scalar_equation([](auto x, auto a) { return x * x - a; }, x0, lower, upper, default_solver_options, p);

  double error = std::abs((x_soln - 2.0)) / 2.0;
  EXPECT_LT(error, default_solver_options.xtol);

  EXPECT_EQ(status.iterations, 0);
}

TEST(ScalarEquationSolver, ReturnsImmediatelyIfLowerBoundIsARoot)
{
  double p = 4.0;
  double lower = 2.0;

  double x0 = 6.0;
  double upper = 8.0;

  auto [x_soln, status] =
      solve_scalar_equation([](auto x, auto a) { return x * x - a; }, x0, lower, upper, default_solver_options, p);

  double error = std::abs((x_soln - 2.0)) / 2.0;
  EXPECT_LT(error, default_solver_options.xtol);

  EXPECT_EQ(status.iterations, 0);
}

TEST(ScalarEquationSolver, ConvergesWithGuessOutsideNewtonBasin)
{
  double x0 = 9.5;
  double lower = -10.0;
  double upper = 10.0;
  auto nasty_function = [](auto x) { return sin(x) + x; };
  auto [x, status] = solve_scalar_equation(nasty_function, x0, lower, upper, default_solver_options);
  double error = std::abs(x);
  EXPECT_LT(error, default_solver_options.xtol);
}

TEST(ScalarEquationSolver, WorksWithTensorParameter)
{
  auto my_norm = [](auto A) {
    double lower = 1e-3;
    double upper = 20.0;
    double x0 = 10.0;
    auto [x_soln, status] = solve_scalar_equation([](auto x, auto P) { return x * x - squared_norm(P); }, x0, lower,
                                                  upper, default_solver_options, A);
    return x_soln;
  };

  tensor<double, 3, 3> A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  auto [normA, dnormA_dA] = my_norm(make_dual(A));
  double exact_value = norm(A);
  EXPECT_LT(std::abs(normA - exact_value) / exact_value, 1e-12);

  tensor<double, 3, 3> exact_derivative = A / norm(A);
  EXPECT_LT(norm(dnormA_dA - exact_derivative) / norm(exact_derivative), 1e-12);
}

TEST(ScalarEquationSolver, CanTakeDirectionalDerivative)
{
  auto my_norm = [](auto A) {
    double lower = 1e-3;
    double upper = 20.0;
    double x0 = 10.0;
    auto [x_soln, status] = solve_scalar_equation([](auto x, auto P) { return x * x - squared_norm(P); }, x0, lower,
                                                  upper, default_solver_options, A);
    return x_soln;
  };

  tensor<double, 3, 3> A = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  tensor<double, 3, 3> dA = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  // this should give d( norm(A) )/dA : dA = (A/norm(A)) : A = norm(A)
  auto [normA, dnormA] = my_norm(make_dual(A, dA));

  double exact_derivative = norm(A);
  EXPECT_LT(std::abs(exact_derivative - dnormA) / exact_derivative, 1e-12);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);

  axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
  axom::slic::setAbortOnError(true);
  axom::slic::setAbortOnWarning(false);
  return RUN_ALL_TESTS();
}
