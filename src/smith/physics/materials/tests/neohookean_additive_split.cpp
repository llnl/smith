// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file neohookean_additive_split.cpp
 *
 * Test of addivtive split version of neo-Hookean model
 */

#include <cmath>

#include "gtest/gtest.h"

#include "smith/physics/materials/solid_material.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace smith {

TEST(NeoHookeanSplit, zeroPoint)
{
  constexpr int dim = 3;
  double G = 1.0;
  double K = 2.0;
  solid_mechanics::NeoHookeanAdditiveSplit mat{.density = 1.0, .K = K, .G = G};
  solid_mechanics::NeoHookeanAdditiveSplit::State state{};
  tensor<double, dim, dim> H{};
  auto P = mat(state, H);
  EXPECT_LT(norm(P), 1e-10);
};

TEST(NeoHookeanSplit, materialFrameIndifference)
{
  constexpr int dim = 3;
  double G = 1.0;
  double K = 2.0;
  solid_mechanics::NeoHookeanAdditiveSplit mat{.density = 1.0, .K = K, .G = G};
  solid_mechanics::NeoHookeanAdditiveSplit::State state{};

  // clang-format off
    // random rotation matrix
    tensor<double, dim, dim> Q{{{ 0.85293210845696,  0.40557741669729, -0.32865449552427},
                                {-0.51876824611259,  0.72872528356249, -0.44703351990878},
                                { 0.05819214026331,  0.55178475890688,  0.83195387771778}}};
    auto should_be_identity = dot(Q, transpose(Q));
    // make sure it's a rotation
    EXPECT_LT(norm(should_be_identity - Identity<dim>()), 1e-10);
    tensor<double, dim, dim> H{{{0.33749426589249, 0.19423845458191, 0.30783257318134},
                                {0.0901473654803 , 0.6104025179124 , 0.45897891871615},
                                {0.68930932313059, 0.19832140905316, 0.90197331346206}}};
  // clang-format on

  auto P = mat(state, H);
  auto H_star = dot(Q, H + Identity<dim>()) - Identity<dim>();
  auto P_star = mat(state, H_star);
  auto error = P_star - dot(Q, P);
  EXPECT_LT(norm(error), 1e-12);
};

TEST(NeoHookeanSplit, deviatoricPartIsCorrectlySplit)
{
  // When the bulk modulus is zero, the Cauchy stress should be strictly deviatoric

  constexpr int dim = 3;
  double G = 1.0;
  double K = 0.0;
  solid_mechanics::NeoHookeanAdditiveSplit mat{.density = 1.0, .K = K, .G = G};
  solid_mechanics::NeoHookeanAdditiveSplit::State state{};

  // clang-format off
    tensor<double, dim, dim> H{{{0.33749426589249, 0.19423845458191, 0.30783257318134},
                                {0.0901473654803 , 0.6104025179124 , 0.45897891871615},
                                {0.68930932313059, 0.19832140905316, 0.90197331346206}}};
  // clang-format on

  auto P = mat(state, H);
  auto F = H + Identity<dim>();
  auto T = dot(P, transpose(F)) / det(F);  // Cauchy stress
  EXPECT_LT(std::abs(tr(T)), 1e-12);
};

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
