// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <vector>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "serac/gretl/vector_state.hpp"
#include "serac/gretl/data_store.hpp"
#include "serac/gretl/test_utils.hpp"
#include "gtest/gtest.h"

using gretl::print;

// extension ideas
// clarify const correctness of states
// add capability to add persistent states mid-way through
// insert states which were expensive (relative to their memory storage) to compute at higher levels
//   this will allow less recomputes for particularly expensive steps
// detect when leaves of graph are unused?
// provide some default implementations for vector operations
// add ability to clear out dual, and start RE-computing the graph going forward.
//   maybe even a capability for reseting the persistent state to somewhere in the middle?
// refactor to allow for the max memory usage to be directly specified so small entries don't
//   take away essential capacity
// refactor to make the state # quota exact
TEST(Graph, NonlinearGraphGradients)
{
  std::vector<double> dataA = {1.3, 3.5};
  std::vector<double> dataB = {1.7, 1.1};
  std::vector<double> dataZ = {-0.7, 3.1};

  gretl::DataStore dataStore(3);

  auto a = dataStore.create_state(dataA, gretl::vec::initialize_zero_dual);
  auto b = dataStore.create_state(dataB, gretl::vec::initialize_zero_dual);
  auto z = dataStore.create_state(dataZ, gretl::vec::initialize_zero_dual);
  auto c = a + b;      // a + b
  auto d = c + c + b;  // 2a + 3b
  auto e = c + d;      // 3a + 4b
  auto g = d + c;      // 3a + 4b
  auto h = g + c;      // 4a + 5b  // completely unused
  h = a + d;           // 3a + 3b
  g = c + d;           // 3a + 4b
  h = h + a;           // 4a + 3b
  e = e + g;           // 6a + 8b
  h = a + e;           // 7a + 8b
  e = d + e;           // 8a + 11b
  e = z + e;

  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(g.get()[i], 3 * dataA[i] + 4 * dataB[i], 1e-14);
  }

  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(e.get()[i], 8 * dataA[i] + 11 * dataB[i] + dataZ[i], 1e-14);
  }

  auto f = gretl::inner_product(e, g);  // 3a+4b dot (8a+11b+z)
  // da = 3 * (8*a + 11*b + z) + 8 * (3a+4b)
  // db = 4 * (8*a + 11*b + z) + 11 * (3a+4b)
  double fFirstTime = f.get();
  gretl::set_as_objective(f);

  dataStore.back_prop();

  for (size_t i = 0; i < 2; ++i) {
    double A = a.get()[i];
    double B = b.get()[i];
    double Z = z.get()[i];
    ASSERT_NEAR(a.get_dual()[i], 3 * (8 * A + 11 * B + Z) + 8 * (3 * A + 4 * B), 1e-13);
    ASSERT_NEAR(b.get_dual()[i], 4 * (8 * A + 11 * B + Z) + 11 * (3 * A + 4 * B), 1e-13);
    ASSERT_NEAR(z.get_dual()[i], 3 * A + 4 * B, 1e-13);
  }

  ASSERT_EQ(fFirstTime, f.get());
}

TEST(Graph, LinearGraphGradients)
{
  std::vector<double> dataA = {1.2, 3.2};

  gretl::DataStore dataStore(6);

  auto initial = dataStore.create_state(dataA, gretl::vec::initialize_zero_dual);
  auto a = gretl::copy(initial);
  int N = 3;
  for (int i = 0; i < N; ++i) {
    a = gretl::testing_update(a);
  }

  a.set_dual(std::vector<double>{1.0, 0.0});
  dataStore.back_prop();

  EXPECT_EQ(initial.get()[0], dataA[0]);
  EXPECT_EQ(initial.get()[1], dataA[1]);
  EXPECT_NEAR(initial.get_dual()[0], std::pow(1. / 3., N), 1e-14);
  EXPECT_NEAR(initial.get_dual()[1], 0.0, 1e-14);
}

TEST(Graph, LargeNonlinearGraphGradients)
{
  std::vector<double> dataA = {0.3, 0.35};
  std::vector<double> dataB = {0.6, 0.87};
  std::vector<double> dataC = {-0.8, 0.32};

  gretl::DataStore dataStore(3);

  auto a = dataStore.create_state(dataA, gretl::vec::initialize_zero_dual);
  auto b = dataStore.create_state(dataB, gretl::vec::initialize_zero_dual);
  auto c = dataStore.create_state(dataC, gretl::vec::initialize_zero_dual);

  auto g = a * b;
  auto h = a + 3 * c;
  auto f = c * g;
  auto end = f + g;

  int Nj = 5;   // 7;
  int Ni = 32;  // 13;

  for (int j = 0; j < Nj; ++j) {
    for (int i = 0; i < Ni; ++i) {
      auto tmp = h + g;
      g = a * tmp;
      h = f + g;
      h = h * b;
    }
    f = g * h;
    f = f + g;
  }

  auto qoi = gretl::inner_product(end, f);

  gretl::set_as_objective(qoi);
  // dataStore.print_graph();
  dataStore.back_prop();

  double constexpr eps = 1e-7;
  gretl::check_array_gradients(qoi, {a, b, c}, {eps, eps, eps}, {800 * eps, 100 * eps, 100 * eps});
}

auto compute_f(const gretl::State<std::vector<double>>& c, const gretl::State<std::vector<double>>& b)
{
  auto d = c + b;
  d = d + b;
  d = d + b;
  d = d + b;
  return d + b;
}
