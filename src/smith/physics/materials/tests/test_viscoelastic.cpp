// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_viscoelastic.cpp
 * @brief Tests of the finite deformation hyper-viscoelastic model
 */

#include "smith/physics/materials/viscoelastic.hpp"

#include "gtest/gtest.h"

#include "smith/numerics/functional/tensor.hpp"
#include "smith/physics/materials/material_verification_tools.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/dual.hpp"
#include "smith/numerics/functional/tuple.hpp"

namespace smith {

TEST(ViscoelasticMaterial, Basic) {
    double K_inf = 3.0e3;
    double G_inf = 500.0;
    double alpha_inf = 0.0;
    double G_0 = 1.5e3;
    double eta_0 = 200.0;

    double theta_r = 350.0;
    double rho_r = 1000.0;

    Viscoelastic material{.K_inf = K_inf, .G_inf = G_inf, .alpha_inf = alpha_inf,
      .G_0 = G_0, .eta_0 = eta_0, .theta_r = theta_r, .rho_r = rho_r};
    
    constexpr double t_max = 10.0;
    size_t num_steps = 100;
    auto strain_cycle = [](double t) {
      if (t < 0.5*t_max) {
        return t;
      } else {
        return t_max - t;
      }
    };
    
    double t = 0;
    const double dt = t_max / double(num_steps - 1);
    tensor<double, 9> internal_state{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    tensor<double, 3, 3> dudX{{{0.015, 0, 0},
                               {0.0, 0.0, 0.0},
                               {0.0, 0.0, 0.0}}};
    double theta = 350.0;

    auto P = material.pkStress(dt, internal_state, dudX, theta);
    std::cout << "P = \n" << P << "\n";
    
}

} // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}