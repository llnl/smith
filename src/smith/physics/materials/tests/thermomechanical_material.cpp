// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermomechanical_material.cpp
 *
 * @brief unit tests for a thermoelastic material model
 */

#include <complex>

#include "gtest/gtest.h"

#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/dual.hpp"

namespace smith {

static constexpr auto I = Identity<3>();

// An arbitrary rotation tensor for frame indifference tests
static const tensor<double, 3, 3> Q{{{0.852932108456964, 0.405577416697288, -0.328654495524269},
                                     {-0.51876824611259, 0.728725283562492, -0.44703351990878},
                                     {0.058192140263313, 0.551784758906877, 0.831953877717782}}};

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T>
auto greenStrain(const tensor<T, 3, 3>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

/// @brief Green-Saint Venant isotropic thermoelastic model
struct ThermoelasticMaterial {
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C;          ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature
  double k;          ///< thermal conductivity

  struct State { /* this material has no internal variables */
  };

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] grad_theta temperature gradient
   * @param[in] grad_u_old displacement gradient at previous time step
   * @param[in] theta_old temperature at previous time step
   * @param[in] dt time increment
   * @param[out] P Piola stress
   * @param[out] cv0 volumetric heat capacity in ref config
   * @param[out] s0 internal heat supply in ref config
   * @param[out] q0 Piola heat flux
   */
  template <typename T1, typename T2, typename T3>
  auto calculateConstitutiveOutputs(const tensor<T1, 3, 3>& grad_u, T2 theta, const tensor<T3, 3>& grad_theta,
                                    State& /*state*/, const tensor<double, 3, 3>& grad_u_old, double /*theta_old*/,
                                    double dt)
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    auto F = grad_u + I;
    const auto Eg = greenStrain(grad_u);
    const auto trEg = tr(Eg);

    // stress
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    const auto P = F * S;

    // internal heat source
    // use backward difference to estimate rate of green strain
    const auto Eg_old = greenStrain(grad_u_old);
    const double egdot = (trEg - tr(Eg_old)) / dt;
    const double s0 = -3 * K * alpha * theta * egdot;

    // heat flux
    const auto q0 = -k * grad_theta;

    return smith::tuple{P, C, s0, q0};
  }

  /**
   * @brief Return constitutive output for thermal operator
   *
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] grad_theta temperature gradient
   * @param[in] grad_u_old displacement gradient at previous time step
   * @param[in] theta_old temperature at previous time step
   * @param[in] dt time increment
   * @param[out] cv0 volumetric heat capacity in ref config
   * @param[out] s0 internal heat supply in ref config
   * @param[out] q0 Piola heat flux
   */
  auto calculateThermalConstitutiveOutputs(const tensor<double, 3, 3>& grad_u, double theta,
                                           const tensor<double, 3>& grad_theta, State& state,
                                           const tensor<double, 3, 3>& grad_u_old, double theta_old, double dt)
  {
    auto [P, cv0, s0, q0] = calculateConstitutiveOutputs(grad_u, theta, grad_theta, state, grad_u_old, theta_old, dt);
    return smith::tuple{cv0, s0, q0};
  }

  /**
   * @brief Return constitutive output for mechanics operator
   *
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] grad_theta temperature gradient
   * @param[in] grad_u_old displacement gradient at previous time step
   * @param[in] theta_old temperature at previous time step
   * @param[in] dt time increment
   * @param[out] P Piola stress
   */
  auto calculateMechanicalConstitutiveOutputs(const tensor<double, 3, 3>& grad_u, double theta,
                                              const tensor<double, 3>& grad_theta, State& state,
                                              const tensor<double, 3, 3>& grad_u_old, double theta_old, double dt)
  {
    auto [P, cv0, s0, q0] = calculateConstitutiveOutputs(grad_u, theta, grad_theta, state, grad_u_old, theta_old, dt);
    return P;
  }

  /**
   * @brief evaluate free energy density
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   */
  template <typename T1, typename T2>
  auto calculateFreeEnergy(const tensor<T1, 3, 3>& grad_u, T2 theta)
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    auto strain = greenStrain(grad_u);
    auto trE = tr(strain);
    auto psi_1 = G * squared_norm(dev(strain)) + 0.5 * K * trE * trE;
    using std::log;
    auto logT = log(theta / theta_ref);
    auto psi_2 = C * (theta - theta_ref + theta * logT);
    auto psi_3 = -3.0 * K * alpha * (theta - theta_ref) * trE;
    return psi_1 + psi_2 + psi_3;
  }
};

TEST(ThermomechanicalMaterial, FreeEnergyIsZeroInReferenceState)
{
  ThermoelasticMaterial material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  tensor<double, 3, 3> displacement_grad{};
  double temperature = material.theta_ref;
  double free_energy = material.calculateFreeEnergy(displacement_grad, temperature);
  EXPECT_NEAR(free_energy, 0.0, 1e-10);
}

TEST(ThermomechanicalMaterial, StressIsZeroInReferenceState)
{
  ThermoelasticMaterial material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  tensor<double, 3, 3> displacement_grad{};
  double temperature = material.theta_ref;
  tensor<double, 3> temperature_grad{};
  ThermoelasticMaterial::State state{};
  auto displacement_grad_old = displacement_grad;
  double temperature_old = temperature;
  double dt = 1.0;
  auto stress = material.calculateMechanicalConstitutiveOutputs(displacement_grad, temperature, temperature_grad, state,
                                                                displacement_grad_old, temperature_old, dt);
  EXPECT_NEAR(norm(stress), 0.0, 1e-10);
}

TEST(ThermomechanicalMaterial, FreeEnergyAndStressAgree)
{
  ThermoelasticMaterial material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double temperature = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  ThermoelasticMaterial::State state{};
  tensor<double, 3, 3> displacement_grad_old{};
  double temperature_old = temperature;
  double dt = 1.0;
  auto energy_and_stress = material.calculateFreeEnergy(make_dual(displacement_grad), temperature);
  auto stress = material.calculateMechanicalConstitutiveOutputs(displacement_grad, temperature, temperature_grad, state,
                                                                displacement_grad_old, temperature_old, dt);
  auto error = stress - get_gradient(energy_and_stress);
  EXPECT_NEAR(norm(error), 0.0, 1e-12);
}

TEST(ThermomechanicalMaterial, SatisfiesDissipationInequality)
{
  ThermoelasticMaterial material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double temperature = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  double temperature_old = temperature;
  ThermoelasticMaterial::State state{};
  tensor<double, 3, 3> displacement_grad_old{};
  double dt = 1.0;
  auto generalized_fluxes = material.calculateThermalConstitutiveOutputs(
      displacement_grad, temperature, temperature_grad, state, displacement_grad_old, temperature_old, dt);
  auto [heat_capacity, source, heat_flux] = generalized_fluxes;
  auto dissipation = -dot(heat_flux, temperature_grad) / temperature;
  EXPECT_TRUE(dissipation >= 0.0);
}

TEST(ThermomechanicalMaterial, IsFrameIndifferent)
{
  ThermoelasticMaterial material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double temperature = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  double temperature_old = 300;
  ThermoelasticMaterial::State state{};
  tensor<double, 3, 3> displacement_grad_old{};
  double dt = 1.0;

  auto generalized_fluxes = material.calculateConstitutiveOutputs(displacement_grad, temperature, temperature_grad,
                                                                  state, displacement_grad_old, temperature_old, dt);

  auto displacement_grad_transformed = Q * (displacement_grad + I) - I;
  auto displacement_grad_old_transformed = Q * (displacement_grad_old + I) - I;
  auto generalized_fluxes_2 =
      material.calculateConstitutiveOutputs(displacement_grad_transformed, temperature, temperature_grad, state,
                                            displacement_grad_old_transformed, temperature_old, dt);

  auto [piola_stress, heat_capacity, internal_source, heat_flux] = generalized_fluxes;
  auto [piola_stress_2, heat_capacity_2, internal_source_2, heat_flux_2] = generalized_fluxes_2;
  EXPECT_LT(norm(piola_stress - transpose(Q) * piola_stress_2), 1e-12);
  EXPECT_NEAR(heat_capacity, heat_capacity_2, 1e-12);
  EXPECT_NEAR(internal_source, internal_source_2, 1e-12);
  EXPECT_LT(norm(heat_flux - heat_flux_2), 1e-12);
}

TEST(ThermomechanicalMaterial, InternalSourceHasCorrectSign)
{
  ThermoelasticMaterial material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double temperature_old = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  ThermoelasticMaterial::State state{};
  tensor<double, 3, 3> displacement_grad_old{};
  double temperature = temperature_old;
  double dt = 1.0;
  auto [heat_capacity, internal_source, heat_flux] = material.calculateThermalConstitutiveOutputs(
      displacement_grad, temperature, temperature_grad, state, displacement_grad_old, temperature_old, dt);
  // should have same sign as sgn(alpha*trEdot), here negative
  EXPECT_LT(internal_source, 0.0);
}

TEST(ThermomechanicalMaterial, StressHasCorrectSymmetry)
{
  ThermoelasticMaterial material{.E = 100.0, .nu = 0.25, .C = 1.0, .alpha = 1.0e-3, .theta_ref = 300.0, .k = 1.0};
  // clang-format off
  tensor<double, 3, 3>  displacement_grad{{{0.35490513, 0.60419905, 0.4275843},
                                           {0.23061597, 0.6735498,  0.43953657},
                                           {0.25099766, 0.27730572, 0.7678207}}};
  // clang-format on
  double temperature_old = 290.0;
  tensor<double, 3> temperature_grad{0.87241435, 0.11105156, -0.27708054};
  ThermoelasticMaterial::State state{};
  tensor<double, 3, 3> displacement_grad_old{};
  double temperature = temperature_old;
  double dt = 1.0;
  auto piola_stress = material.calculateMechanicalConstitutiveOutputs(
      displacement_grad, temperature, temperature_grad, state, displacement_grad_old, temperature_old, dt);
  auto deformation_grad = displacement_grad + I;
  auto kirchhoff_stress = piola_stress * transpose(deformation_grad);
  double tol = 1e-10;
  EXPECT_TRUE(is_symmetric(kirchhoff_stress, tol));
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
