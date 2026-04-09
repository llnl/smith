// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/physics/materials/green_saint_venant_thermoelastic.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/numerics/functional/dual.hpp"

using namespace smith;

/// Thermomechanics helper data types
namespace smith::thermomechanics {
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

    return smith::tuple{Piola, C_v, s0, q0};
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
    constexpr auto I = smith::DenseIdentity<dim>();
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
    return smith::tuple{Piola, C_v, s0, q0};
  }
};
//////////////////////////////////////////////////////////////////////////////
/// Fictitious thermally responsive material (softening when heating)
struct ThermalSofteningDummyMaterial {
  static constexpr int dim = 3;
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity
  double mu;         ///< viscous parameter

  // Bounds for temperature-dependent scaling of mechanical properties
  double theta_low = 293.15;   ///< lower temperature bound for scaling
  double theta_high = 473.15;  ///< upper temperature bound for scaling

  // Maximum softening factor at low temperature
  // E_eff(theta_low)   = scale_factor * E
  // E_eff(theta_high)  = 1 * E
  double scale_factor = 1.0e1;

  using State = Empty;

  template <typename T1, typename T2, typename T3, typename T4, int dim_>
  auto operator()(double, State&, const tensor<T1, dim_, dim_>& grad_u, const tensor<T2, dim_, dim_>& grad_v, T3 theta,
                  const tensor<T4, dim_>& grad_theta) const
  {
    static_assert(dim_ == dim, "Dimension mismatch in ThermalSofteningDummyMaterial");

    using std::log1p;

    // --------------------------------------------------------
    // Temperature dependent scaling factor for mechanical props
    // --------------------------------------------------------
    // We want: at low T -> scale_factor, at high T -> 1
    auto s_mech = theta;  // just to get the AD type; will overwrite immediately

    if (theta <= theta_low) {
      s_mech = scale_factor;
    } else if (theta >= theta_high) {
      s_mech = 1.0;
    } else {
      // r in (0,1)
      auto r = (theta - theta_low) / (theta_high - theta_low);
      // linear from scale_factor (at r=0) down to 1 (at r=1)
      s_mech = scale_factor + (1.0 - scale_factor) * r;
    }

    // Apply scaling to E and mu (promote doubles to AD type via multiplication)
    auto E_eff = s_mech * E;
    auto mu_eff = s_mech * mu;

    auto K = E_eff / (3.0 * (1.0 - 2.0 * nu));
    auto G = 0.5 * E_eff / (1.0 + nu);
    // std::cout<<"... theta = "<<get_value(theta)<<", s_mech = "<<get_value(s_mech)<<", K = "<<get_value(K)<<", G =
    // "<<get_value(G)<<std::endl;
    constexpr auto I = smith::DenseIdentity<dim_>();
    auto lambda = K - (2.0 / 3.0) * G;

    auto B_minus_I = dot(grad_u, transpose(grad_u)) + transpose(grad_u) + grad_u;

    auto logJ = log1p(detApIm1(grad_u));

    // Deformation gradient
    auto F = grad_u + I;

    auto L = dot(grad_v, inv(F));
    auto D = sym(L);

    // Kirchhoff stress with viscous term using temperature dependent mu_eff
    auto TK = lambda * logJ * I + G * B_minus_I + 0.5 * det(F) * mu_eff * D;

    // Thermal stress contribution
    auto S = -K * (dim_ * alpha * (theta - theta_ref)) * I;

    // First Piola stress
    auto Piola = dot(TK, inv(transpose(F))) + dot(F, S);

    // Internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    auto s0 = -dim_ * K * alpha * (theta + 273.1) * tr(greenStrainRate);

    // Heat flux
    auto q0 = -kappa * grad_theta;

    return smith::tuple{Piola, C_v, s0, q0};
  }
};

//////////////////////////////////////////////////////////////////////////////
/// Fictitious thermally responsive material (softening when heating)
struct ThermalStiffeningDummyMaterial {
  static constexpr int dim = 3;
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity
  double mu;         ///< viscous parameter

  // Bounds for temperature-dependent scaling of mechanical properties
  double theta_low = 293.15;   ///< lower temperature bound for scaling
  double theta_high = 473.15;  ///< upper temperature bound for scaling

  // Maximum softening factor at low temperature
  // E_eff(theta_low)   = scale_factor * E
  // E_eff(theta_high)  = 1 * E
  double scale_factor = 1.0e1;

  using State = Empty;

  template <typename T1, typename T2, typename T3, typename T4, int dim_>
  auto operator()(double, State&, const tensor<T1, dim_, dim_>& grad_u, const tensor<T2, dim_, dim_>& grad_v, T3 theta,
                  const tensor<T4, dim_>& grad_theta) const
  {
    static_assert(dim_ == dim, "Dimension mismatch in ThermalSofteningDummyMaterial");

    using std::log1p;

    // --------------------------------------------------------
    // Temperature dependent scaling factor for mechanical props
    // --------------------------------------------------------
    // We want: at low T -> scale_factor, at high T -> 1
    auto s_mech = theta;  // just to get the AD type; will overwrite immediately

    if (theta <= theta_low) {
      s_mech = 1.0;
    } else if (theta >= theta_high) {
      s_mech = scale_factor;
    } else {
      // r in (0,1)
      auto r = (theta - theta_low) / (theta_high - theta_low);
      // linear from scale_factor (at r=0) down to 1 (at r=1)
      s_mech = 1.0 + (scale_factor - 1.0) * r;
    }

    // Apply scaling to E and mu (promote doubles to AD type via multiplication)
    auto E_eff = s_mech * E;
    auto mu_eff = s_mech * mu;

    auto K = E_eff / (3.0 * (1.0 - 2.0 * nu));
    auto G = 0.5 * E_eff / (1.0 + nu);

    constexpr auto I = smith::DenseIdentity<dim_>();
    auto lambda = K - (2.0 / 3.0) * G;

    auto B_minus_I = dot(grad_u, transpose(grad_u)) + transpose(grad_u) + grad_u;

    auto logJ = log1p(detApIm1(grad_u));

    // Deformation gradient
    auto F = grad_u + I;

    auto L = dot(grad_v, inv(F));
    auto D = sym(L);

    // Kirchhoff stress with viscous term using temperature dependent mu_eff
    auto TK = lambda * logJ * I + G * B_minus_I + 0.5 * det(F) * mu_eff * D;

    // Thermal stress contribution
    auto S = -K * (dim_ * alpha * (theta - theta_ref)) * I;

    // First Piola stress
    auto Piola = dot(TK, inv(transpose(F))) + dot(F, S);

    // Internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    auto s0 = -dim_ * K * alpha * (theta + 273.1) * tr(greenStrainRate);

    // Heat flux
    auto q0 = -kappa * grad_theta;

    return smith::tuple{Piola, C_v, s0, q0};
  }
};
///////////////////////////////////////////////////////////////////////////////

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
  
  template<typename scalar>
  SMITH_HOST_DEVICE auto equilibrium_xi(scalar temp) const{
    using std::pow, std::exp;
    auto Tt = 443.0;
    auto k = 36.0;
    return exp(-(pow(temp/Tt,k)));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto Gm0(scalar g) const{
    // low-T shear modulus at reference temperature as a function of particle wt% g
    auto junk = g;
    return Gm*junk/g;
  }
  
  template<typename scalar>
  SMITH_HOST_DEVICE auto f1(scalar T) const{
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return exp(-N * (T - Tr));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto df1(scalar T) const{
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return -N*exp(-N * (T - Tr));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto Ge0(scalar g) const{
    // high-T shear modulus at reference temperature as a function of particle wt% g
    auto junk = g;
    return Ge*junk/g;
  }

  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double dt, State& state, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    using std::pow, std::exp;
    
    // Tr is a double but I need auto to add to theta
    auto tempref = Tr; // 353.0;

    theta=theta+tempref;

    auto wep = state.w_e;     // previous entangled fraction
    auto wfp = 1.0-wep;       // previous free fraction
    auto Fesip = state.Fesi;  // previous inverse of mapping F^{es}

    // get equilibrium wl=xi
    auto xi = equilibrium_xi(theta);
    //std::cout << "wh: " << wh << "\n";

    // get kinematics
    constexpr auto I = Identity<dim>();

    auto F = grad_u + I;
    auto FeIni = dot(F,Fesip); // Fe for the extant entangled material, called Fh1 in my notes about the relaxation method
    auto Je = det(FeIni);

    auto C = dot(transpose(F), F);
    auto Ci = inv(C);
    auto D = 0.5*(grad_v+transpose(grad_v));//dot(inv(transpose(F)),CdFi)*0.5;

    auto B = dot(F, transpose(F));
    auto trB = tr(B);
    auto B_bar = B - (trB / 3.0) * I;
    auto J = det(F);

    // get moduli
    auto Gm_eff = Gm0(gw)*f1(theta);
    auto Ge_eff = Ge0(gw);

    // calculate forward and reverse reaction rate
    auto kf = Af * exp(-E_af / (R*theta));
    auto kr = Ar * exp(-E_ar / (R*theta));

    // get mass fraction supplies, forward and reverse
    auto dwff = (xi-wfp)*kf*dt/(1.+kf*dt);
    auto dwer = (1.-xi-wep)*kr*dt/(1.+kr*dt);
    // get net mass fraction supply
    auto dwe = -dwff + dwer;

    auto aux1 = 0.0, aux2 = 0.0, aux3 = 0.0;
    // if dwh>0, I need to get the new equivalent Fhsi
    if (dwe>0 && wep==0) {
      aux1 = 1.0; // initialize Fhsi as the inverse of F at the current time
    }
    else if (dwe>0) { // calculate the current elastic deformation of the high-T material
      aux2 = 1.0; // update the effective value of Fhsi
    }
    else {
      aux3 = 1.0;
    }

    auto Fesi = aux1 * inv(F) + aux2 * (wep/(wep+dwe))*Fesip + aux3 * Fesip;
    auto Fe = dot(F,Fesi);
    state.Fesi = get_value(Fesi);

    // update mass fractions
    auto we = wep+dwe;//1 + wep + dwe -wep-dwe;

    //std::cout << theta << "," << kf << "," << kr << "," << dwe << "," << we << "," << wep << "\n";

  // calculate B_bar, J based on Fh
    auto Be = dot(Fe, transpose(Fe));
    auto trBe = tr(Be);
    auto Be_bar = Be - (trBe / 3.0) * I;

    // calculate kirchoff stress
    auto Tm = Gm_eff * pow(J, -2./3.) * B_bar + J * Km * (J - 1. - betam*(theta-Tr)) * I; // + etal * D;
    auto Te = Ge_eff * pow(Je, -2./3.) * Be_bar + Je * Ke * (Je - 1.) * I; // + etah * D;

    auto TK = wm * Tm + (1. - wm) * we * Te + 2*((1.-we)*etam+we*etae)*D;
  
    // 1st Piola from Kirchhoff
    const auto Piola = dot(TK, inv(transpose(F)));

    // heat flux
    const auto q0 = -kappa * grad_theta;
    std::cout << dwe+wep << "," << we << "," << wep << "\n";
    state.w_e = get_value(we);
    std::cout << state.w_e << "\n";
    state.Cp = get_value(dot(transpose(F),F));

    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    // viscous stress
    auto Sv = 2*((1.-we)*etam+we*etae)*dot(Ci,dot(greenStrainRate,Ci));
    // derivative of elastic S with respect to T
    auto dtmdT = Gm0(gw)*df1(theta)*pow(J,-2./3)*B_bar-Km*J*betam*I;
    auto dSedT = dot(inv(F),dot(wm*dtmdT,transpose(inv(F))));
    const auto s0 = tr(dot(Sv+theta*dSedT,greenStrainRate));
    //const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);

    return smith::tuple{Piola, C_v, s0, q0};
  }
};


/// Simple, no-ISV PNC thermal stiffening material model
struct SimpleThermalStiffeningMaterial {
  double Km;       ///< matrix bulk modulus, MPa
  double betam;    ///< matrix volumetric thermal expansion coefficient

  double Ke;       ///< entanglement bulk modulus, MPa

  double C_v;      ///< net volumetric heat capacity (must account for matrix+chain+particle)
  double kappa;    ///< net thermal conductivity (must account for matrix+chain+particle)

  double Tr;       ///< reference temperature, K, set to 353

  double gw;       ///< particle weight fraction


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
   * @param[in] grad_theta Temperature 
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */

  // this function calculates equilibrium value of 1-we
  template<typename scalar>
  SMITH_HOST_DEVICE auto equilibrium_xi(scalar temp) const{
    using std::pow, std::exp;
    auto Tt = 443.0;
    auto k = 36.0;
    return exp(-(pow(temp/Tt,k)));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto Gm0(scalar g) const{
    using std::pow;
    // matrix shear modulus at reference temperature as a function of particle wt% g
    auto Gr = 0.017;      //GPa, rigid modulus
    auto Gs0 = 1.7e-6;    //GPa, soft modulus
    auto Xc = 0.21;       //critical percolation volume fraction
    auto rhof = 2.65;     //g/cc, filler density
    auto rhom = 1.06;     //g/cc, matrix density
    auto n = 0.4;         //percolation exponent
    auto gv = g*rhom/(rhof+g*(rhom-rhof));       //convert weight fraction gw to volume fraction gv
    auto peff = gv*pow(150.,3)/pow(100.,3);      //for 100nm particle, 50nm interphase
    auto Gs = Gs0*(1.+2.5*peff+14.1*peff*peff);  //Guth-Gold correction
    auto X = gv*rhom/(rhof+gv*(rhom-rhof));
    auto psi = 0.;
    if (X>Xc) {
      psi = X*pow((X-Xc)/(1.-Xc),n);
    }
    auto Gnum = (1.-2.*psi+psi*X)*Gr*Gs+(1.-X)*psi*Gr*Gr;
    auto Gdenom = (1.-X)*Gr+(X-psi)*Gs;
    auto G = Gnum/Gdenom; // this is in GPa
    return G*1.e9;        // convert to Pa
  }
  
  template<typename scalar>
  SMITH_HOST_DEVICE auto f1(scalar T) const{
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return exp(-N * (T - Tr));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto df1(scalar T) const{
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return -N*exp(-N * (T - Tr));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto Ge0(scalar g) const{
    using std::pow;
    // entanglement shear modulus at reference temperature as a function of particle wt% g
    auto Gr = 0.12;      //GPa, rigid modulus
    auto Gs0 = 6.5e-7;   //GPa, soft modulus
    auto Xc = 0.05;      //critical percolation volume fraction
    auto rhof = 2.65;    //g/cc, filler density
    auto rhom = 1.06;    //g/cc, matrix density
    auto n = 1.2;        // percolation exponent
    auto gv = g*rhom/(rhof+g*(rhom-rhof));       //convert weight fraction gw to volume fraction gv
    auto peff = gv*pow(150.,3)/pow(100.,3);      //effective volume fraction
    auto Gs = Gs0*(1.+2.5*peff+14.1*peff*peff);  //Guth-Gold correction
    auto X = gv*rhom/(rhof+gv*(rhom-rhof));
    auto psi = 0.;
    if (X>Xc) {
      psi = X*pow((X-Xc)/(1.-Xc),n);
    }
    auto Gnum = (1.-2.*psi+psi*X)*Gr*Gs+(1.-X)*psi*Gr*Gr;
    auto Gdenom = (1.-X)*Gr+(X-psi)*Gs;
    auto G = Gnum/Gdenom; // this is in GPa
    return G*1.e9;        //this is Pa
  }

  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    using std::pow, std::exp;
    
    // Tr is a double but I need auto to add to theta
    auto tempref = Tr; // 353.0;

    theta=theta+tempref;

    // get equilibrium we=1-xi
    auto we = 0.0;//1. - equilibrium_xi(theta);

    // get kinematics
    constexpr auto I = Identity<dim>();

    auto F = grad_u + I;

    auto B = dot(F, transpose(F));
    auto trB = tr(B);
    auto B_bar = B - (trB / 3.0) * I;
    auto J = det(F);

    // get moduli
    auto Gm_eff = Gm0(gw)*f1(theta);
    auto Ge_eff = Ge0(gw);

    // calculate B_bar, J based on Fh
    auto Be = dot(F, transpose(F));
    auto trBe = tr(Be);
    auto Be_bar = Be - (trBe / 3.0) * I;

    // calculate kirchoff stress
    auto Tm = Gm_eff * pow(J, -2./3.) * B_bar + J * Km * (J - 1. - betam*(theta-Tr)) * I;
    auto Te = Ge_eff * pow(J, -2./3.) * Be_bar + J * Ke * (J - 1.) * I;

    auto TK = Tm + we * Te;
  
    // 1st Piola from Kirchhoff
    const auto Piola = dot(TK, inv(transpose(F)));

    // heat flux
    const auto q0 = -kappa * grad_theta;

    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    // derivative of elastic S with respect to T
    auto dtmdT = Gm0(gw)*df1(theta)*pow(J,-2./3)*B_bar-Km*J*betam*I;
    auto dSedT = dot(inv(F),dot(dtmdT,transpose(inv(F))));
    const auto s0 = tr(dot(theta*dSedT,greenStrainRate));

    return smith::tuple{Piola, C_v, s0, q0};
  }
};





  /////////////// this stuff might need to be declared in the material?
  /*
  // machinery for tensor inverse
  void ludcmp(double**a, int n, int* indx) {
	int i, imax, j, k;
	double big, dum, sum, temp;
	double* vv;

	const double TINY = 1.0e-20;

	vv = new double[n];

	for (i=0; i<n; ++i)
	{
		big = 0;
		for (j=0; j<n; ++j)
			if ((temp = fabs(a[i][j])) > big) big = temp;

		vv[i] = 1.0 / big;
	}

	for (j=0; j<n; ++j)
	{
		for (i=0; i<j; ++i)
		{
			sum = a[i][j];
			for (k=0; k<i; ++k) sum -= a[i][k]*a[k][j];
			a[i][j] = sum;
		}
		big = 0;
		imax = j;
		for (i=j; i<n; ++i)
		{
			sum = a[i][j];
			for (k=0; k<j; ++k) sum -= a[i][k]*a[k][j];
			a[i][j] = sum;
			if ((dum=vv[i]*fabs(sum)) >= big)
			{
				big = dum;
				imax = i;
			}
		}
		if (j != imax)
		{
			for (k=0; k<n; ++k)
			{
				dum = a[imax][k];
				a[imax][k] = a[j][k];
				a[j][k] = dum;
			}
			vv[imax] = vv[j];
		}
		indx[j] = imax;
		if (a[j][j] == 0.0) 
		{
			a[j][j] = TINY;
		}
		if (j!=n-1)
		{
			dum = 1.0/(a[j][j]);
			for (i=j+1;i<n; ++i) a[i][j] *= dum;
		}
	}

	// clean up
	delete [] vv;
}

void lubksb(double**a, int n, int *indx, double b[])
{
	int i, ii=0, ip, j;
	double sum;

	for (i=0; i<n; ++i)
	{
		ip = indx[i];
		sum = b[ip];
		b[ip] = b[i];
		if (ii != 0)
			for (j=ii-1; j<i; ++j) sum -= a[i][j]*b[j];
		else if (sum != 0.0) ii=i+1;
		b[i] = sum;
	}
	for (i=n-1; i>=0; --i)
	{
		sum = b[i];
		for (j=i+1;j<n;++j) sum -= a[i][j]*b[j];
		b[i] = sum/a[i][i];
	}
}

void invert6x6(const double in[6][6], double out[6][6]) {
    const int n = 6;
    double** a = new double*[n];
    for (int i = 0; i < n; ++i) {
        a[i] = new double[n];
        for (int j = 0; j < n; ++j)
            a[i][j] = in[i][j]; // copy input matrix
    }

    int* indx = new int[n];

    // LU decomposition
    ludcmp(a, n, indx);

    double* col = new double[n];

    // Invert matrix one column at a time
    for (int j = 0; j < n; ++j) {
        // Set up unit vector
        for (int i = 0; i < n; ++i) col[i] = 0.0;
        col[j] = 1.0;

        // Solve a x = col
        lubksb(a, n, indx, col);

        // Copy solution into output
        for (int i = 0; i < n; ++i)
            out[i][j] = col[i];
    }

    // Clean up
    for (int i = 0; i < n; ++i) delete[] a[i];
    delete[] a;
    delete[] indx;
    delete[] col;
}

// Map 4th-order tensor to 6x6 Voigt matrix
void tensor4_to_voigt6x6(const double C[3][3][3][3], Eigen::Matrix<double,6,6>& voigt) {
    int voigt_map[6][2] = { {0,0}, {1,1}, {2,2}, {1,2}, {0,2}, {0,1} };
    double scale[6] = {1,1,1,sqrt(2),sqrt(2),sqrt(2)};
    for(int I=0; I<6; ++I) {
        for(int J=0; J<6; ++J) {
            int i=voigt_map[I][0], j=voigt_map[I][1];
            int k=voigt_map[J][0], l=voigt_map[J][1];
            double s = 1.0/(scale[I]*scale[J]);
            voigt(I,J) = s * C[i][j][k][l];
        }
    }
}

// Map 6x6 Voigt matrix back to 4th-order tensor
void voigt6x6_to_tensor4(const Eigen::Matrix<double,6,6>& voigt, double C[3][3][3][3]) {
    int voigt_map[6][2] = { {0,0}, {1,1}, {2,2}, {1,2}, {0,2}, {0,1} };
    double scale[6] = {1,1,1,sqrt(2),sqrt(2),sqrt(2)};
    for(int I=0; I<6; ++I) {
        for(int J=0; J<6; ++J) {
            int i=voigt_map[I][0], j=voigt_map[I][1];
            int k=voigt_map[J][0], l=voigt_map[J][1];
            double s = scale[I]*scale[J];
            C[i][j][k][l] = s * voigt(I,J);
            C[j][i][k][l] = s * voigt(I,J);
            C[i][j][l][k] = s * voigt(I,J);
            C[j][i][l][k] = s * voigt(I,J);
        }
    }
}
*/

/*
matrix matrix::inverse()
{
	// make sure this is a square matrix
	assert(m_nr == m_nc);

	// make a copy of this matrix
	// since we don't want to change it
	matrix a(*this);

	// do a LU decomposition
	int n = m_nr;
	vector<int> indx(n);
	ludcmp(a, n, &indx[0]);

	// allocate the inverse matrix
	matrix ai(n, n);

	// do a backsubstituation on the columns of a
	vector<double> b; b.assign(n, 0);
	for (int j=0; j<n; ++j)
	{
		b[j] = 1;
		lubksb(a, n, &indx[0], &b[0]);

		for (int i=0; i<n; ++i)
		{
			ai[i][j] = b[i];
			b[i] = 0;
		}
	}

	return ai;
}
*/

  ///////////////////////////////////////////////////////////////////////////////

/// PNC thermal stiffening material model
struct ViscoThermalStiffeningMaterial {
  double Km;       ///< matrix bulk modulus, MPa
  double Gm;       ///< matrix shear modulus, MPa
  double betam;    ///< matrix volumetric thermal expansion coefficient
  double rhom0;    ///< matrix initial density
  double taum;     ///< matrix time constant, s
  double binfm;    ///< matrix visco scaling factor beta^infty

  double Ke;       ///< entanglement bulk modulus, MPa
  double Ge;       ///< entanglement shear modulus, MPa
  double rhoe0;    ///< entanglement (chain) initial density
  double taue;     ///< entanglement time constant, s
  double binfe;    ///< entanglement visco scaling factor beta^infty

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
  //double wm;       ///< matrix mass fraction (set to 0.5, not real for now)


  struct State {
    double w_e = 0.0;   //entangled mass fraction
    double Jes = 1.0;   // volume change term
    double thetap = 1.0; // previous temperature
    tensor<double,3,3> Fes{{{1.0, 0.0, 0.0},
                           {0.0, 1.0, 0.0},
                           {0.0, 0.0, 1.0}}}; // previous value of deformation shift
    tensor<double,3,3> Ce{{{1.0, 0.0, 0.0},
                             {0.0, 1.0, 0.0},
                             {0.0, 0.0, 1.0}}}; // effective entanglement elastic right Cauchy Green
    tensor<double,3,3> Hm{{{1.0, 0.0, 0.0},
                             {0.0, 1.0, 0.0},
                             {0.0, 0.0, 1.0}}};
    tensor<double,3,3> He{{{1.0, 0.0, 0.0},
                             {0.0, 1.0, 0.0},
                             {0.0, 0.0, 1.0}}};
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
  
  template<typename scalar>
  SMITH_HOST_DEVICE auto equilibrium_xi(scalar temp) const{
    using std::pow, std::exp;
    auto Tt = 443.0;
    auto k = 36.0;
    return exp(-(pow(temp/Tt,k)));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto Gm0(scalar g) const{
    // low-T shear modulus at reference temperature as a function of particle wt% g
    auto junk = g;
    return Gm*junk/g;
  }
  
  template<typename scalar>
  SMITH_HOST_DEVICE auto f1(scalar T) const{
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return exp(-N * (T - Tr));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto df1(scalar T) const{
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return -N*exp(-N * (T - Tr));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto Ge0(scalar g) const{
    // high-T shear modulus at reference temperature as a function of particle wt% g
    auto junk = g;
    return Ge*junk/g;
  }

  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double dt, State& state, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    using std::pow, std::exp;
    
    // Tr is a double but I need auto to add to theta
    auto tempref = Tr; // 353.0;

    theta=theta+tempref;
    
    auto thetap = state.thetap;
    auto wep = state.w_e;     // previous entangled fraction
    auto wfp = 1.0-wep;       // previous free fraction
    auto Jesp = state.Jes;
    //auto Cep = state.Ce;  // previous value of Ce
    //auto Uep = matrix_sqrt(Cep); // previous elastic stretch
    auto Fesp = state.Fes;
    auto Hep = state.He;
    auto Hmp = state.Hm;
    // get the previous value of Fes

    // get equilibrium wl=xi
    auto xi = equilibrium_xi(theta);

    // get kinematics
    constexpr auto I = Identity<dim>();
    auto F = grad_u + I;
    auto C = dot(transpose(F), F);
    auto Ci = inv(C);
    auto trC = tr(C);
    auto J = det(F);
    //auto C_hat = C - (trC / 3.0) * I;

    // polar decomposition of F
    auto U = matrix_sqrt(C);
    //auto Rm = dot(F, inv(U));

    // get previous Fes isochoric value
    //auto bFesp = pow(Jesp,-1./3)*Fesp;

    // trial value of bCe (Cbar_e, isochoric elastic right Cauchy Green for entanglements)
    auto Cetr = dot(inv(transpose(Fesp)),dot(C,inv(Fesp)));
    //auto Jetr = det(matrix_sqrt(Cetr));

    // get moduli
    auto Gm_eff = Gm0(gw)*f1(theta);
    auto Ge_eff = Ge0(gw);

    // calculate forward and reverse reaction rate
    auto kf = Af * exp(-E_af / (R*theta));
    auto kr = Ar * exp(-E_ar / (R*theta));

    // get mass fraction supplies, forward and reverse
    auto dwff = (xi-wfp)*kf*dt/(1.+kf*dt);
    auto dwer = (1.-xi-wep)*kr*dt/(1.+kr*dt);
    // get net mass fraction supply
    auto dwe = -dwff + dwer;

    // instead of a while loop, do a one-step Tylor series solution


    /*
    // prepare for the Newton step: this is stress evaluated at trial, constant
    auto bSetr = Ke*Jetr*(Jetr-1.)*inv(Cetr) + Ge_eff*pow(Jetr,-2./3)*(I - inv(Cetr)*(tr(Cetr)/3.0));

    eps = 1e-3;
    norm = 1;
    while (norm>eps) {
      // get the stress evaluated at the trial

    }
    */

    auto aux1 = 0.0, aux2 = 0.0, aux3 = 0.0;
    // if dwe>0, I need to get the new equivalent Fesi
    if (dwe>0 && wep==0) {
      aux1 = 1.0; // initialize Fhsi as the inverse of F at the current time, this is the first formation
    }
    else if (dwe>0) { // need to update the effective value
      aux2 = 1.0; // this is the one that will enter the while loop
    }
    else {
      aux3 = 1.0;
    }

    auto Ce = aux1 * I + aux2 * (wep/(wep+dwe))*Cetr*Jesp/det(matrix_sqrt(Cetr))+ aux3 * Cetr;
    state.Ce = get_value(Ce);
    auto Je = det(matrix_sqrt(Ce));
    // get the Fes related to this value of Ce
    auto Fes = dot(inv(matrix_sqrt(Ce)),U);

    // update mass fractions
    auto we = wep + dwe;

    // std::cout << "we: " << we << "\n";

  // calculate B_bar, J based on Fh
    auto C_bar = I - (trC / 3.0) * Ci;
    auto Ce_bar = I - (tr(Ce) / 3.0) * inv(Ce);

    // calculate second PK stress
    // Cbe is Cbar-entanglement, the new effective value
    // Cei is inverse of new C-entanglement, which is Ce = pow(J/Jes,2./3.)*Cbe
    // Jes is prescribed Jalphas
    auto Smvol = J*(Km*(J-1)-Km*betam*(theta-Tr))*Ci;
    auto Smiso = Gm_eff*pow(J,-2./3)*C_bar;
    auto Sevol = Ke*Je*(Je-1.)*inv(Ce);
    auto Seiso = Ge_eff*pow(Je,-2./3)*Ce_bar;

    // get viscoelastic Q tensors
    auto Qm = Smiso*binfm*exp(-dt/(2.*taum)) + Hmp;
    auto Hm = exp(-dt/(2.*taum))*(Qm*exp(-dt/(2.*taum)) - binfm*Smiso);
    auto Qe = Seiso*binfe*exp(-dt/(2.*taue)) + Hep;
    auto He = exp(-dt/(2.*taue))*(Qe*exp(-dt/(2.*taue)) - binfe*Seiso);
    
    // calculate 2nd PK stress
    auto bSe = Sevol + Seiso + Qe;
    auto S = Smvol + Smiso + Qm + we*(J/Je)*dot(inv(Fes),dot(bSe,inv(transpose(Fes))));

    // get kirchhoff
    auto TK = dot(F,dot(S,transpose(F)));
   
    // Pull back to Piola and store the current H1 as the state H1n
    state.He = get_value(He);
    state.Hm = get_value(Hm);
    state.Fes = get_value(Fes);
    state.thetap = get_value(theta);
    const auto Piola = dot(TK, inv(transpose(F)));

    // heat flux
    const auto q0 = -kappa * grad_theta;

    state.w_e = get_value(we);

    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    // viscous stress
    //auto Sv = 2*((1.-we)*etam+we*etae)*dot(Ci,dot(greenStrainRate,Ci));
    // derivative of elastic S with respect to T
    auto dGmdT = Gm0(gw)*df1(theta);
    auto dSmisodT = dGmdT*pow(J,-2./3)*C_bar;
    auto dQmdT = binfm*exp(-dt/(2*taum))*dSmisodT;
    auto dSmdT = dGmdT*pow(J,-2./3)*C_bar-Km*J*betam*I+dQmdT;
    auto dSdT = dSmdT + (J/Je)*dot(inv(Fes),dot(bSe,inv(transpose(Fes))))*(dwe/(theta-thetap));
    // below I am assuming the viscosity is 1 for both materials, not sure
    // what to do with it since it appears nowhere else
    auto dQterm = tr(dot(Qm-theta*dQmdT,Qm/(2.*taum))) + we*(J/Je)*tr(dot(Qe,Qe/(2.*taue)));
    // now I also need the terms accounting for the dissipation from entanglements breaking
    const auto s0 = tr(dot(theta*dSdT,greenStrainRate))+dQterm;

    return smith::tuple{Piola, C_v, s0, q0};
  }
};

///////////////////////////////////////////////////////////////////////////////

};  // namespace smith::thermomechanics