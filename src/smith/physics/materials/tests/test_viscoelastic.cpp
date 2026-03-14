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

#include <iostream>
#include <fstream>

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
    double theta_sf = 350.0;

    double G_0 = 1.5e3;
    double eta_0 = 200.0;

    double theta_r = 350.0;
    double C1 = 15.0;
    double C2 = 50.0;

    double rho_r = 1000.0;

    Viscoelastic material{.K_inf = K_inf, .G_inf = G_inf,
      .alpha_inf = alpha_inf, .theta_sf = theta_sf, .G_0 = G_0, .eta_0 = eta_0,
      .theta_r = theta_r, .C1 = C1, .C2 = C2, .rho_r = rho_r};
    
    constexpr double max_strain = 0.1;
    constexpr double strain_rate = 1.0e-1;
    constexpr double t_max = 2.0*max_strain/strain_rate;
    size_t num_steps = 100;
    auto applied_strain = [](double t) {
      if (t < 0.5*t_max) {
        return strain_rate*t;
      } else {
        return strain_rate*(t_max - t);
      }
    };
    
    tensor<double, 9> internal_state{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double theta = 350.0;
    auto temperature_cycle = [theta](double) { return theta; };

    auto history = uniaxial_stress_test(t_max, num_steps, material, internal_state,
                                        applied_strain, temperature_cycle);
    
    std::ofstream file("viscoelastic_uniaxial.csv");
    for (const auto& timestep : history) {
      double t = get<0>(timestep);
      tensor<double, 3, 3> disp_grad = get<1>(timestep);
      tensor<double, 3, 3> stress = get<2>(timestep);
      tensor<double, 9> isv = get<3>(timestep);
      file << t << " " << disp_grad[0][0] << " " << stress[0][0] << " " << isv[0] << "\n";
    }
    file.close();
}

} // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
