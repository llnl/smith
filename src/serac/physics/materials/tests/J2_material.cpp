// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_J2_material.cpp
 */

#include "serac/physics/materials/solid_material.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"
#include "serac/physics/materials/material_verification_tools.hpp"

namespace serac {

using namespace serac;

/**
 * @brief a verification problem for the J2 material model, taken from example 2 of
 * R. M. Brannon · S. Leelavanichkul (2009)
 * A multi-stage return algorithm for solving the classical
 * damage component of constitutive models for rocks,
 * ceramics, and other rock-like media
 *
 * @note Sam: the graphs for the analytic stress solutions in the document above
 * are inconsistent with the equations they provide for t > 1. It seems that
 * a couple of the coefficents in the equations appear with the wrong sign,
 * but the implementation below fixes those typos
 */
tensor<double, 3, 3> verification_analytic_soln(double t)
{
  // clang-format off
  using std::sqrt;

  double a = std::exp(12.33 * t);

  tensor< double, 3 > sigma{};

  if (t <= 0.201) {
    sigma = {-474.0 * t, -474.0 * t, 948.0 * t};
  }

  if ((0.201 < t) && (t <= 1.0)) {
    sigma = {-95.26, -95.26, 190.50};
  }

  if (1 < t) {
    tensor < double, 3, 4 > c = {{
      { 189.40,  0.1704, -0.003242, 0.00001712},
      { -76.87, -1.4430,  0.001316, 0.00001712},
      {-112.50,  1.2720,  0.001926, 0.00001712}
    }};

    sigma = {
      (c(0, 0) + c(0, 1) * sqrt(a) + c(0, 2) * a) / (1.0 + c(0, 3) * a),
      (c(1, 0) + c(1, 1) * sqrt(a) + c(1, 2) * a) / (1.0 + c(1, 3) * a),
      (c(2, 0) + c(2, 1) * sqrt(a) + c(2, 2) * a) / (1.0 + c(2, 3) * a)
    };
  }

  return {{
    {sigma[0], 0.0, 0.0}, 
    {0.0, sigma[1], 0.0}, 
    {0.0, 0.0, sigma[2]}
  }};
  // clang-format on
}

TEST(J2SmallStrain, Verification)
{
  double tmax = 2.0;
  size_t num_steps = 64;

  double G = 79000;
  double K = 10 * G;

  double E = 9 * K * G / (3 * K + G);
  double nu = (3 * K - 2 * G) / (2 * (3 * K + G));
  double sigma_y = 165 * std::sqrt(3.0);

  using Hardening = solid_mechanics::LinearHardening;
  using Material = solid_mechanics::J2SmallStrain<Hardening>;

  Hardening hardening{.sigma_y = sigma_y, .Hi = 0.0};
  Material material{.E = E, .nu = nu, .hardening = hardening, .Hk = 0.0, .density = 1.0};
  Material::State initial_state{};

  tensor<double, 3> epsilon[2] = {{-0.0030000, -0.003, 0.0060000}, {-0.0103923, 0.000, 0.0103923}};

  auto du_dX = [=](double t) {
    double a[2] = {(t < 1) ? t : 2.0 - t, (t < 1) ? 0 : t - 1.0};
    return diag(a[0] * epsilon[0] + a[1] * epsilon[1]);
  };

  auto history = single_quadrature_point_test(tmax, num_steps, material, initial_state, du_dX);

  for (auto [t, state, dudx, stress] : history) {
    if (t > 0) {
      double rel_error = norm(stress - verification_analytic_soln(t)) / norm(stress);
      // std::cout << t << ": " << rel_error << std::endl;
      ASSERT_LT(rel_error, 5e-2);
    }

    // for generating a plot like in the paper:
    // std::cout << "{" << t << ", " << stress[0][0] << ", " << stress[1][1] << ", " << stress[2][2] << "}" <<
    // std::endl;
  }
}

TEST(J2, PowerLawHardeningWorksWithDuals)
{
  double sigma_y = 1.0;
  solid_mechanics::PowerLawHardening hardening_law{.sigma_y = sigma_y, .n = 2.0, .eps0 = 0.01};
  double eqps = 0.1;
  auto flow_stress = hardening_law(make_dual(eqps));
  EXPECT_GT(flow_stress.value, sigma_y);
  EXPECT_GT(flow_stress.gradient, 0.0);
};

TEST(J2SmallStrain, SatisfiesConsistency)
{
  // clang-format off
  tensor<double, 3, 3> du_dx{
      {{0.7551559, 0.3129729, 0.12388372},
       {0.548188, 0.8851279, 0.30576992},
       {0.82008433, 0.95633745, 0.3566252}}
  };
  // clang-format on

  using Hardening = solid_mechanics::PowerLawHardening;
  using Material = solid_mechanics::J2SmallStrain<Hardening>;

  Hardening hardening_law{.sigma_y = 0.1, .n = 2.0, .eps0 = 0.01};
  Material material{.E = 1.0, .nu = 0.25, .hardening = hardening_law, .Hk = 0.0, .density = 1.0};
  auto internal_state = Material::State{};
  tensor<double, 3, 3> stress = material(internal_state, du_dx);
  double mises = std::sqrt(1.5) * norm(dev(stress));
  double flow_stress = hardening_law(internal_state.accumulated_plastic_strain);
  EXPECT_NEAR(mises, flow_stress, 1e-9 * mises);

  double twoG = material.E / (1 + material.nu);
  tensor<double, 3, 3> s = twoG * dev(sym(du_dx) - internal_state.plastic_strain);
  EXPECT_LT(norm(s - dev(stress)) / norm(s), 1e-9);
};

TEST(J2SmallStrain, Uniaxial)
{
  using Hardening = solid_mechanics::LinearHardening;
  using Material = solid_mechanics::J2SmallStrain<Hardening>;

  double E = 1.0;
  double nu = 0.25;
  double sigma_y = 0.01;
  double Hi = E / 100.0;

  Hardening hardening{.sigma_y = sigma_y, .Hi = Hi};
  Material material{.E = E, .nu = nu, .hardening = hardening, .Hk = 0.0, .density = 1.0};

  auto internal_state = Material::State{};
  auto strain = [=](double t) { return sigma_y / E * t; };
  auto response_history = uniaxial_stress_test(2.0, 4, material, internal_state, strain);

  auto stress_exact = [=](double epsilon) {
    return epsilon < sigma_y / E ? E * epsilon : E / (E + Hi) * (sigma_y + Hi * epsilon);
  };
  auto plastic_strain_exact = [=](double epsilon) {
    return epsilon < sigma_y / E ? 0.0 : (E * epsilon - sigma_y) / (E + Hi);
  };

  for (auto r : response_history) {
    double e = get<1>(r)[0][0];                  // strain
    double s = get<2>(r)[0][0];                  // stress
    double pe = get<3>(r).plastic_strain[0][0];  // plastic strain
    ASSERT_LE(std::abs(s - stress_exact(e)), 1e-10 * std::abs(stress_exact(e)));
    ASSERT_LE(std::abs(pe - plastic_strain_exact(e)), 1e-10 * std::abs(plastic_strain_exact(e)));
  }
};

TEST(J2, Uniaxial)
{
  /* Log strain J2 plasticity has the nice feature that the exact uniaxial stress solution from
     small strain plasticity are applicable, if you replace the lineasr strain with log strain
     and use the Kirchhoff stress as the output.
  */

  using Hardening = solid_mechanics::LinearHardening;
  using Material = solid_mechanics::J2<Hardening>;

  double E = 1.0;
  double sigma_y = 0.01;
  double Hi = E / 100.0;

  Hardening hardening{.sigma_y = sigma_y, .Hi = Hi};
  Material material{.E = E, .nu = 0.25, .hardening = hardening, .density = 1.0};

  auto internal_state = Material::State{};
  auto strain = [=](double t) { return sigma_y / E * t; };
  auto response_history = uniaxial_stress_test(2.0, 4, material, internal_state, strain);

  auto stress_exact = [=](double epsilon) {
    return epsilon < sigma_y / E ? E * epsilon : E / (E + Hi) * (sigma_y + Hi * epsilon);
  };
  auto plastic_strain_exact = [=](double epsilon) {
    return epsilon < sigma_y / E ? 0.0 : (E * epsilon - sigma_y) / (E + Hi);
  };

  for (auto r : response_history) {
    auto [t, dudx, P, state] = r;
    auto TK = dot(P, transpose(dudx + Identity<3>()));
    double e = std::log1p(get<1>(r)[0][0]);        // log strain
    double s = TK[0][0];                           // Kirchhoff stress
    double pe = -std::log(get<3>(r).Fpinv[0][0]);  // plastic strain
    ASSERT_NEAR(s, stress_exact(e), 1e-6 * std::abs(stress_exact(e)));
    ASSERT_NEAR(pe, plastic_strain_exact(e), 1e-6 * std::abs(plastic_strain_exact(e)));
  }
};

TEST(J2, DerivativeCorrectness)
{
  // This constitutive function is non-differentiable at the yield point,
  // but should be differentiable everywhere else.
  // The elastic response is trivial. We want to check the plastic reponse
  // and make sure the derivative propagates correctly through the nonlinear
  // solve.

  using Hardening = solid_mechanics::PowerLawHardening;
  using Material = solid_mechanics::J2<Hardening>;

  Hardening hardening{.sigma_y = 350e6, .n = 3, .eps0 = 0.00175};
  Material material{.E = 200e9, .nu = 0.25, .hardening = hardening, .density = 1.0};

  // initialize internal state variables
  auto internal_state = Material::State{};

  // clang-format off
  const tensor<double, 3, 3> H{{
    { 0.025, -0.008,  0.005},
    {-0.008, -0.01,   0.003},
    { 0.005,  0.003,  0.0}}};

  tensor< double, 3, 3 > dH = {{
    {0.3, 0.4, 1.6},
    {2.0, 0.2, 0.3},
    {0.1, 1.7, 0.3}
  }};
  // clang-format on

  auto stress_and_tangent = material(internal_state, make_dual(H));
  auto tangent = get_gradient(stress_and_tangent);

  // make sure that this load case is actually yielding
  ASSERT_GT(internal_state.accumulated_plastic_strain, 1e-3);

  const double epsilon = 1.0e-5;

  // finite difference evaluations
  auto internal_state_old_p = Material::State{};
  auto stress_p = material(internal_state_old_p, H + epsilon * dH);

  auto internal_state_old_m = Material::State{};
  auto stress_m = material(internal_state_old_m, H - epsilon * dH);

  // Make sure the finite difference evaluations all took the same branch (yielding).
  ASSERT_GT(internal_state_old_p.accumulated_plastic_strain, 1e-3);
  ASSERT_GT(internal_state_old_m.accumulated_plastic_strain, 1e-3);

  // check AD against finite differences
  tensor<double, 3, 3> dsig[2] = {double_dot(tangent, dH), (stress_p - stress_m) / (2 * epsilon)};

  EXPECT_LT(norm(dsig[0] - dsig[1]), 1e-5 * norm(dsig[1]));
}

TEST(J2, FrameIndifference)
{
  using Hardening = solid_mechanics::VoceHardening;
  using Material = solid_mechanics::J2<Hardening>;

  Hardening hardening{.sigma_y = 350e6, .sigma_sat = 700e6, .strain_constant = 0.01};
  Material material{.E = 200.0e9, .nu = 0.25, .hardening = hardening, .density = 1.0};

  // clang-format off
  const tensor<double, 3, 3> H{{
    { 0.025, -0.008,  0.005},
    {-0.008, -0.01,   0.003},
    { 0.005,  0.003,  0.0}
  }};
  
  // this is a rotation matrix, randomly generated in numpy
  const tensor<double, 3, 3> Q{{
    {-0.928152308749236, -0.091036503308254, -0.360895617636},
    {0.238177386319198, 0.599832274220295, -0.763853896664712},
    {0.28601542687348, -0.794929932679048, -0.535052873762272}
  }};
  //clang-format on

  // before we check frame indifference, make sure Q is a rotation
  constexpr auto I = Identity<3>();
  ASSERT_LT(norm(dot(transpose(Q), Q) - I), 1e-14);

  // initialize internal state variables
  auto internal_state = Material::State{};

  // Piola stress components in original coordinate frame
  auto P = material(internal_state, H);

  // make sure that this load case is actually yielding
  ASSERT_GT(internal_state.accumulated_plastic_strain, 1e-3);

  // transform displacement gradient to new coordinate frame
  // H_star = F_star - I
  //        = QF - I
  //        = Q(H + I) - I
  auto H_star = dot(Q, H + I) - I;

  // Remember to initialize a fresh set of internal variables - the originals have been mutated
  auto internal_state_star = Material::State{};

  // stress in second coordinate frame
  auto P_star = material(internal_state_star, H_star);

  auto error = P_star - dot(Q, P);
  ASSERT_LT(norm(error), 1e-13*norm(P));

  // The plastic distortion Fp has no legs in the observed space and should be invariant
  error = internal_state.Fpinv - internal_state_star.Fpinv;
  ASSERT_LT(norm(error), 1e-13*norm(internal_state.Fpinv));
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  return result;
}
