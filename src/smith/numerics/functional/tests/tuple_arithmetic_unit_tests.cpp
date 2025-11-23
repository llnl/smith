// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/dual.hpp"

using namespace smith;

static constexpr auto I = Identity<3>();
static constexpr double rho = 3.0;
static constexpr double mu = 2.0;

static constexpr double epsilon = 1.0e-6;

TEST(TupleArithmeticUnitTests, StructuredBinding)
{
  smith::tuple x{0, 1.0, 2.0f};
  auto [a, b, c] = x;
  EXPECT_NEAR(a, 0, 1.0e-10);
  EXPECT_NEAR(b, 1.00, 1.0e-10);
  EXPECT_NEAR(c, 2.0f, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, Add)
{
  smith::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
                 make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  smith::tuple b = a + a;
  EXPECT_NEAR(smith::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(smith::get<1>(b)), 10.39230484541326, 1.0e-10);
  EXPECT_NEAR(norm(smith::get<2>(b)), 2.977782431376876, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, Subtract)
{
  smith::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
                 make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  smith::tuple b = a - a;
  EXPECT_NEAR(smith::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(smith::get<1>(b)), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(smith::get<2>(b)), 0.0, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, Multiply)
{
  smith::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
                 make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  smith::tuple b = 2.0 * a;
  EXPECT_NEAR(smith::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(smith::get<1>(b)), 10.39230484541326, 1.0e-10);
  EXPECT_NEAR(norm(smith::get<2>(b)), 2.977782431376876, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, Divide)
{
  smith::tuple a{0.0, make_tensor<3>([](int) { return 3.0; }),
                 make_tensor<5, 3>([](int i, int j) { return 1.0 / (i + j + 1); })};
  smith::tuple b = a / 0.5;
  EXPECT_NEAR(smith::get<0>(b), 0.0, 1.0e-10);
  EXPECT_NEAR(norm(smith::get<1>(b)), 10.39230484541326, 1.0e-10);
  EXPECT_NEAR(norm(smith::get<2>(b)), 2.977782431376876, 1.0e-10);
}

TEST(TupleArithmeticUnitTests, TensorOutputWithTupleInput)
{
  constexpr auto f = [=](auto p, auto v, auto L) { return rho * outer(v, v) * det(I + L) + 2.0 * mu * sym(L) - p * I; };

  [[maybe_unused]] constexpr double p = 3.14;
  [[maybe_unused]] constexpr tensor v = {{1.0, 2.0, 3.0}};
  constexpr tensor<double, 3, 3> L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  constexpr double dp = 1.23;
  constexpr tensor dv = {{2.0, 1.0, 4.0}};
  constexpr tensor<double, 3, 3> dL = {{{3.0, 1.0, 2.0}, {2.0, 7.0, 3.0}, {4.0, 4.0, 3.0}}};

  auto dfdp = get_gradient(f(make_dual(p), v, L));
  auto dfdv = get_gradient(f(p, make_dual(v), L));
  auto dfdL = get_gradient(f(p, v, make_dual(L)));

  auto df0 = (f(p + epsilon * dp, v + epsilon * dv, L + epsilon * dL) -
              f(p - epsilon * dp, v - epsilon * dv, L - epsilon * dL)) /
             (2 * epsilon);

  auto df1 = dfdp * dp + dfdv * dv + double_dot(dfdL, dL);

  EXPECT_NEAR(norm(df1 - df0) / norm(df0), 0.0, 2.0e-8);
}

TEST(TupleArithmeticUnitTests, ReadTheDocsExample)
{
  auto f = [=](auto p, auto v, auto L) {
    auto strain_rate = 0.5 * (L + transpose(L));
    auto stress = -p * I + 2 * mu * strain_rate;
    auto kinetic_energy_density = 0.5 * rho * dot(v, v);
    return tuple{stress, kinetic_energy_density};
  };

  [[maybe_unused]] constexpr double p = 3.14;
  [[maybe_unused]] constexpr tensor v = {{1.0, 2.0, 3.0}};
  constexpr tensor<double, 3, 3> L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  // promote the arguments to dual numbers with make_dual()
  tuple dual_args = make_dual(tuple{p, v, L});

  // then call the function with the dual arguments
  //
  // note: smith::apply is a way to pass an n-tuple to a function that expects n arguments
  //
  // i.e. the two following lines have the same effect
  // f(p, v, L);
  // smith::apply(f, smith::tuple{p, v, L});
  auto outputs = apply(f, dual_args);

  // verify that the derivative types are what we expect
  [[maybe_unused]] tuple<tuple<tensor<double, 3, 3>, zero, tensor<double, 3, 3, 3, 3> >,
                         tuple<zero, tensor<double, 3>, zero> > gradients = get_gradient(outputs);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
