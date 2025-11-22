// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/physics/materials/green_saint_venant_thermoelastic.hpp"
#include "serac/physics/materials/solid_material.hpp"

/// Thermomechanics helper data types
namespace serac::thermomechanics {
/// @brief Alternative Green-Saint Venant isotropic thermoelastic model
struct AlternativeGreenSaintVenantThermoelasticMaterial {
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity

  using State = Empty;

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the velocity gradient components (number-like)
   * @tparam T3 Type of the temperature (number-like)
   * @tparam T4 Type of the temperature gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] grad_v Velocity gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain(grad_u);
    const auto trEg = tr(Eg);

    // stress
    static constexpr auto I = Identity<dim>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - dim * alpha * (theta - theta_ref)) * I;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);

    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);
    // const auto s0 = -dim * K * alpha* 1000.;

    // heat flux
    const auto q0 = -kappa * grad_theta;

    return serac::tuple{Piola, C_v, s0, q0};
  }
};

///////////////////////////////////////////////////////////////////////////////

struct NeoHookeanThermoelasticMaterial {
  static constexpr int dim = 3;
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity
  double mu;         ///< viscous parameter

  using State = Empty;
  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    using std::log1p;
    // constexpr double eps = 1.0e-12;
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    constexpr auto I = serac::DenseIdentity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(grad_u, transpose(grad_u)) + transpose(grad_u) + grad_u;

    auto logJ = log1p(detApIm1(grad_u));
    // Kirchoff stress, in form that avoids cancellation error when F is near I

    // Pull back to Piola
    auto F = grad_u + I;

    auto L = dot(grad_v, inv(F));
    auto D = sym(L);

    auto TK = lambda * logJ * I + G * B_minus_I + 0.5 * det(F) * mu * D;  // dot(L, inv(transpose(F)));

    // state.F_old = get_value(F);
    const auto S = -1.0 * K * (dim * alpha * (theta - theta_ref)) * I;
    auto Piola = dot(TK, inv(transpose(F))) + dot(F, S);
    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);
    // const auto s0 = -dim * K * alpha * (theta + 273.1);

    // heat flux
    const auto q0 = -kappa * grad_theta;
    return serac::tuple{Piola, C_v, s0, q0};
  }
};


/// PNC thermal stiffening material model
struct ThermalStiffeningMaterial {
  double Km;       ///< matrix bulk modulus, MPa
  double Gm;       ///< matrix shear modulus, MPa
  double betam;    ///< matrix volumetric thermal expansion coefficient
  double rhom0;    ///< matrix initial density
  double etam;     ///< matrix viscosity, MPa-s

  double Ke;       ///< entanglement bulk modulus, MPa
  double Ge;       ///< entanglement shear modulus, MPa
  double betae;    ///< entanglement volumetric thermal expansion coefficient
  double rhoe0;    ///< entanglement (chain) initial density
  double etae;     ///< entanglement viscosity, MPa-s

  double C_v;      ///< net volumetric heat capacity (must account for matrix+chain+particle)
  double kappa;    ///< net thermal conductivity (must account for matrix+chain+particle)

  // E_a and R can be SI units since they cancel out in the exponent
  double Af;       ///< forward (low-high) exponential prefactor, 1/s
  double E_af;     ///< forward (low-high) activation energy, J/mol
  double Ar;       ///< reverse exponential prefactor, 1/s
  double E_ar;     ///< reverse activation energy, J/mol
  double R;        ///< universal gas constant, J/mol/K
  double Tr;       ///< reference temperature, K

  double gw;       ///< particle weight fraction
  double wm;       ///< matrix mass fraction (set to 0.5, not real for now)


  struct State {
    double w_e = 0.0;   //entangled mass fraction
    tensor<double,3,3> Cp{{{1.0, 0.0, 0.0},
                           {0.0, 1.0, 0.0},
                           {0.0, 0.0, 1.0}}}; // previous value of right Cauchy-Green
    tensor<double,3,3> Fesi{{{1.0, 0.0, 0.0},
                             {0.0, 1.0, 0.0},
                             {0.0, 0.0, 1.0}}}; //Inverse of effective mapping tensor F^{es} where F=F^e \dot F^{es}
  };

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the velocity gradient components (number-like)
   * @tparam T3 Type of the temperature (number-like)
   * @tparam T4 Type of the temperature gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] grad_v Velocity gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature 
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */

  // this function calculates the equilibrium low-T mass fraction as a function of temperature
  SERAC_HOST_DEVICE auto equilibrium_xi(double temp) const{
    double Tt = 443.0;
    double k = 36.0;
    return std::exp(-(std::pow(temp/Tt,k)));
  }

  SERAC_HOST_DEVICE auto Gm0(double g) const{
    // low-T shear modulus at reference temperature as a function of particle wt% g
    double junk = g;
    return Gm*junk/g;
  }

  SERAC_HOST_DEVICE auto f1(double T) const{
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return std::exp(-N * (T - Tr));
  }

  SERAC_HOST_DEVICE auto df1(double T) const{
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return -N*std::exp(-N * (T - Tr));
  }

  SERAC_HOST_DEVICE auto Ge0(double g) const{
    // high-T shear modulus at reference temperature as a function of particle wt% g
    double junk = g;
    return Ge*junk/g;
  }

  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double dt, State& state, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    auto wep = state.w_e;     // previous entangled fraction
    auto wfp = 1.0-wep;       // previous free fraction
    auto Cp = state.Cp;       // previous right Cauchy-Green tensor
    auto Fesip = state.Fesi;  // previous inverse of mapping F^{es}

    // get equilibrium wl=xi
    auto xi = equilibrium_xi(theta);
    //std::cout << "wh: " << wh << "\n";

    // get kinematics
    constexpr auto I = Identity<dim>();

    auto F = grad_u + I;
    auto Fe = dot(F,Fesip); // Fe for the extant entangled material, called Fh1 in my notes about the relaxation method
    auto Je = det(Fe);
    //auto Ce = dot(transpose(Fe),Fe);

    auto C = dot(transpose(F), F);
    auto Ci = inv(C);
    //auto Cdot = (C - Cp)/dt;
    //auto CdFi = dot(Cdot, inv(F));
    auto D = 0.5*(grad_v+transpose(grad_v));//dot(inv(transpose(F)),CdFi)*0.5;

    auto B = dot(F, transpose(F));
    auto trB = tr(B);
    auto B_bar = B - (trB / 3.0) * I;
    auto J = det(F);

    // get moduli
    auto Gm_eff = Gm0(gw)*f1(theta);
    auto Ge_eff = Ge0(gw);

    // calculate forward and reverse reaction rate
    auto kf = Af * std::exp(-E_af / (R*theta));
    auto kr = Ar * std::exp(-E_ar / (R*theta));

    // get mass fraction supplies, forward and reverse
    auto dwff = (xi-wfp)*kf*dt/(1.+kf*dt);
    auto dwer = (1.-xi-wep)*kr*dt/(1.+kr*dt);
    // get net mass fraction supply
    auto dwe = -dwff + dwer;

    // if dwh>0, I need to get the new equivalent Fhsi
    if (dwe>0 && wep==0) {
    auto Fesi = inv(F); // initialize Fhsi as the inverse of F at the current time
    Fe = dot(F,Fesi);
    state.Fesi = get_value(Fesi);
    }
    else if (dwe>0) {
    auto Fesi = (wep/(wep+dwe))*Fesip; // update the effective value of Fhsi
    Fe = dot(F,Fesi); // calculate the current elastic deformation of the high-T material
    state.Fesi = get_value(Fesi);
    }
    else {
    auto Fesi = Fesip;
    Fe = dot(F,Fesi);
    state.Fesi = get_value(Fesi);
    }

    // update mass fractions
    auto we = wep + dwe;

    std::cout << "we: " << we << "\n";

  // calculate B_bar, J based on Fh
    auto Be = dot(Fe, transpose(Fe));
    auto trBe = tr(Be);
    auto Be_bar = Be - (trBe / 3.0) * I;

    // calculate kirchoff stress
    auto Tm = Gm_eff * std::pow(J, -2./3.) * B_bar + J * Km * (J - 1. - betam*(theta-Tr)) * I; // + etal * D;
    auto Te = Ge_eff * std::pow(Je, -2./3.) * Be_bar + Je * Ke * (Je - 1.) * I; // + etah * D;

    auto TK = wm * Tm + (1. - wm) * we * Te + 2*((1.-we)*etam+we*etae)*D;
  
    // 1st Piola from Kirchhoff
    const auto Piola = dot(TK, inv(transpose(F)));

    // heat flux
    const auto q0 = -kappa * grad_theta;

    state.w_e = get_value(we);
    state.Cp = get_value(dot(transpose(F),F));
  


    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    // viscous stress
    auto Sv = 2*((1.-we)*etam+we*etae)*dot(Ci,dot(greenStrainRate,Ci));
    // derivative of elastic S with respect to T
    auto dtmdT = Gm0(gw)*df1(theta)*std::pow(J,-2./3)*B_bar-Km*J*betam*I;
    auto dSedT = dot(inv(F),dot(wm*dtmdT,transpose(inv(F))));
    const auto s0 = tr(dot(Sv+theta*dSedT,greenStrainRate));
    //const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);

    return serac::tuple{Piola, C_v, s0, q0};
  }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace serac::thermomechanics