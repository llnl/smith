// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_J2_material.cpp
 */

#include "gtest/gtest.h"

#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/dual.hpp"

namespace smith {

TEST(ParameterizedNonlinearJ2Material, DerivativeWrtYieldStrength)
{
  solid_mechanics::ParameterizedJ2Nonlinear material{.E = 200e6, .nu = 0.25, .density = 1.0};
  solid_mechanics::ParameterizedJ2Nonlinear::State internal_state{};
  tensor<double, 3, 3> H{
      {{0.35490513, 0.60419905, 0.4275843}, {0.23061597, 0.6735498, 0.43953657}, {0.25099766, 0.27730572, 0.7678207}}};
  double sigma_y = 100e3;
  double sigma_sat = 500e3;
  double strain_constant = 0.1;

  auto internal_state_new = internal_state;

  auto stress = material(internal_state_new, H, make_dual(sigma_y), sigma_sat, strain_constant);

  double h = 1e-4 * sigma_y;
  auto internal_state_copy = internal_state;
  auto stress_p = material(internal_state_copy, H, sigma_y + h, sigma_sat, strain_constant);
  internal_state_copy = internal_state;
  auto stress_m = material(internal_state_copy, H, sigma_y - h, sigma_sat, strain_constant);
  auto finite_difference = 1. / (2 * h) * (stress_p - stress_m);

  auto error = get_gradient(stress) - finite_difference;

  EXPECT_LT(norm(error) / norm(finite_difference), 1e-5);
};

TEST(ParameterizedNonlinearJ2Material, DerivativeWrtSaturationStrength)
{
  solid_mechanics::ParameterizedJ2Nonlinear material{.E = 200e6, .nu = 0.25, .density = 1.0};
  solid_mechanics::ParameterizedJ2Nonlinear::State internal_state{};
  tensor<double, 3, 3> H{
      {{0.35490513, 0.60419905, 0.4275843}, {0.23061597, 0.6735498, 0.43953657}, {0.25099766, 0.27730572, 0.7678207}}};
  double sigma_y = 100e3;
  double sigma_sat = 500e3;
  double strain_constant = 0.1;

  auto internal_state_new = internal_state;

  auto stress = material(internal_state_new, H, sigma_y, make_dual(sigma_sat), strain_constant);

  double h = 1e-4 * sigma_sat;
  auto internal_state_copy = internal_state;
  auto stress_p = material(internal_state_copy, H, sigma_y, sigma_sat + h, strain_constant);
  internal_state_copy = internal_state;
  auto stress_m = material(internal_state_copy, H, sigma_y, sigma_sat - h, strain_constant);
  auto finite_difference = (1.0 / (2 * h)) * (stress_p - stress_m);

  auto error = get_gradient(stress) - finite_difference;

  EXPECT_LT(norm(error) / norm(finite_difference), 1e-5);
};

}  // namespace smith

// ------------------------------------------

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
