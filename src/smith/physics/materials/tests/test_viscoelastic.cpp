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

TEST(ViscoelasticMaterial, Uniaxial) {
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

    solid_mechanics::Viscoelastic material{.K_inf = K_inf, .G_inf = G_inf,
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
    
    solid_mechanics::Viscoelastic::State internal_state = solid_mechanics::Viscoelastic::initial_internal_state();
    double theta = 350.0;
    tensor<double, 3> grad_theta{};
    auto temperature = make_tuple(theta, grad_theta);
    auto temperature_cycle = [temperature](double) { return temperature; };

    auto history = uniaxial_stress_test2(t_max, num_steps, material, internal_state,
                                         applied_strain, temperature_cycle);
    
    std::ofstream file("viscoelastic_uniaxial.csv");
    for (const auto& timestep : history) {
      double t = get<0>(timestep);
      tensor<double, 3, 3> disp_grad = get<1>(timestep);
      tensor<double, 3, 3> stress = get<2>(timestep);
      solid_mechanics::Viscoelastic::State isv = get<3>(timestep);
      file << t << " " << disp_grad[0][0] << " " << stress[0][0] << " " << isv.Fv[0][0] << "\n";
    }
    file.close();
}

class TestViscoelasticModel : public ::testing::Test {
 protected:
  TestViscoelasticModel(): material{.K_inf = 3.0e3, .G_inf = 500.0,
      .alpha_inf = 45e-6, .theta_sf = 300.0, .G_0 = 1.5e3, .eta_0 = 200.0,
      .theta_r = 350.0, .C1 = 15.0, .C2 = 50.0, .rho_r = 1.1e3} {}

  solid_mechanics::Viscoelastic material;
  static constexpr int dim = 3;
};

TEST_F(TestViscoelasticModel, Symmetry) {
  tensor Fv{{{1.13999899223343, -0.37663886065084, -0.01845954094274},
             {0.23118557813617,  1.25824266214259,  0.36905241011185},
             {0.12569218571529, -0.33628257758523,  0.57289115431496}}};

  // The model expects volume-preserving inelastic distortion,
  // so require this for the test.
  ASSERT_NEAR(det(Fv), 1.0, 1e-14);

  solid_mechanics::Viscoelastic::State internal_state{Fv};

  tensor H{{{0.84410694508235, 0.04541258512666, 0.22498462569285},
            {0.06438560513367, 0.01193894509204, 0.83425129331962},
            {0.62563720341404, 0.76924029241284, 0.85235964292579}}};

  double theta = 330.0;
  tensor<double, 3> grad_theta{};
  auto temperature = make_tuple(theta, grad_theta);
  double dt = 0.1;

  auto stress = material.pkStress(internal_state, dt, make_dual(H), temperature);
  auto C = get_gradient(stress);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < dim; k++) {
        for (int l = 0; l < dim; l++) {
          double rel = std::abs(C[i][j][k][l]);
          rel = rel != 0? rel : 1.0;
          EXPECT_NEAR(C[i][j][k][l], C[k][l][i][j], 1e-12*rel);
        }
      }
    }
  }
}

TEST_F(TestViscoelasticModel, isVariational)
{
  tensor Fv{{{1.13999899223343, -0.37663886065084, -0.01845954094274},
             {0.23118557813617,  1.25824266214259,  0.36905241011185},
             {0.12569218571529, -0.33628257758523,  0.57289115431496}}};

  // The model expects volume-preserving inelastic distortion,
  // so require this for the test.
  ASSERT_NEAR(det(Fv), 1.0, 1e-14);

  solid_mechanics::Viscoelastic::State internal_state{Fv};

  tensor H{{{0.84410694508235, 0.04541258512666, 0.22498462569285},
            {0.06438560513367, 0.01193894509204, 0.83425129331962},
            {0.62563720341404, 0.76924029241284, 0.85235964292579}}};

  double theta = 330.0;
  tensor<double, 3> grad_theta{};
  auto temperature = make_tuple(theta, grad_theta);
  double dt = 0.1;

  auto energy = material.potential(internal_state, dt, make_dual(H), temperature);
  auto P_from_energy = get_gradient(energy);

  auto P = material.pkStress(internal_state, dt, H, temperature);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      double absP = std::abs(P[i][j]);
      double rel = absP != 0? absP : 1.0;
      EXPECT_NEAR(P[i][j], P_from_energy[i][j], 3e-10*rel);
    }
  }
}

} // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
