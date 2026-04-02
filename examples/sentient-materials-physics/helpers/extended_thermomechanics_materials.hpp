
// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "mfem.hpp"
#include <string>
#include <vector>
// #include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
// #include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

namespace extended_thermomechanics_materials {

using smith::cross;
using smith::dev;
using smith::dot;
using smith::get;
using smith::inner;
using smith::norm;
using smith::tr;
using smith::transpose;

template <typename T, int d>
auto greenStrain(const smith::tensor<T, d, d>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

template <typename T1, typename T2, int d>
auto greenStrainRate(const smith::tensor<T1, d, d>& grad_u, const smith::tensor<T2, d, d>& grad_v)
{
  return 0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
}

template <typename T, int d>
auto greenStrainRate(const smith::tensor<T, d, d>& grad_u, const smith::zero&)
{
  return 0.0 * grad_u;
}

template <typename T, int d>
void setIdentity(smith::tensor<T, d, d>& F)
{
  for (size_t i = 0; i < d; i++) {
    for (size_t j = 0; j < d; j++) {
      F(i, j) = static_cast<T>(i == j);
    }
  }
}

struct GreenSaintVenantThermoelasticWithExtendedStateMaterial {
  double density;
  double E0;
  double nu;
  double C_v;
  double alpha_T;
  double theta_ref;
  double kappa;

  using State = smith::Empty;

  template <int d>
  struct SymmetricStatePacking {
    static_assert(d >= 1, "Invalid matrix dimension.");
    static constexpr int sym_size = d * (d + 1) / 2;

    template <int sd, typename ScalarT, typename SymmT>
    static auto pack(const ScalarT& scalar, const smith::tensor<SymmT, d, d>& symm)
    {
      static_assert(sd == 1 + sym_size, "Packed state size mismatch.");
      using PackedValue = decltype(scalar + symm(0, 0));
      smith::tensor<PackedValue, sd> out{};
      out[0] = scalar;
      int k = 1;
      for (int i = 0; i < d; ++i) {
        for (int j = i; j < d; ++j) {
          out[k++] = symm(i, j);
        }
      }
      return out;
    }

    template <typename T, int sd>
    static auto unpack(const smith::tensor<T, sd>& in)
    {
      static_assert(sd == 1 + sym_size, "Packed state size mismatch.");
      T scalar = in[0];
      smith::tensor<T, d, d> symm{};
      int k = 1;
      for (int i = 0; i < d; ++i) {
        for (int j = i; j < d; ++j) {
          symm(i, j) = in[k];
          symm(j, i) = in[k];
          ++k;
        }
      }
      return smith::tuple{scalar, symm};
    }
  };

  template <typename T1, typename T2, typename T3, typename T4, typename T5, int d, int sd>
  auto operator()(double, State&, const smith::tensor<T1, d, d>& grad_u, const T2& grad_v, T3 theta,
                  const smith::tensor<T4, d>& grad_theta, const smith::tensor<T5, sd>& alpha_old) const
  {
    // Calculate Alpha new using the old variables to be used

    auto [w_old, F_old] = SymmetricStatePacking<d>::template unpack<T5, sd>(alpha_old);

    // Extracting 0 index scalar value and calculating rate of change
    auto w_new = w_old;
    auto F_new = F_old;

    // Concatenating results

    auto E = E0;
    const auto K = E / (3.0 * (1.0 - 2.0 * nu));
    const auto G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain<T1, d>(grad_u);
    const auto trEg = tr(Eg);

    static constexpr auto I = smith::Identity<d>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - d * alpha_T * (theta - theta_ref)) * I;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);

    const auto strain_rate = greenStrainRate(grad_u, grad_v);
    const auto s0 = -0.0*d * K * alpha_T * (theta + 273.1) * tr(strain_rate);
    // std::cout << tr(strain_rate) << std::endl;
    const auto q0 = -kappa * grad_theta;

    auto alpha_new = SymmetricStatePacking<d>::template pack<sd>(w_new, F_new);
    return smith::tuple{Piola, C_v, s0, q0, alpha_new};
  }

  static constexpr int numParameters() { return 1; }
};

/// PNC thermal stiffening material model
struct ThermalStiffeningMaterial {
  double Km;     ///< matrix bulk modulus, MPa
  double Gm;     ///< matrix shear modulus, MPa
  double betam;  ///< matrix volumetric thermal expansion coefficient
  double rhom0;  ///< matrix initial density
  double etam;   ///< matrix viscosity, MPa-s

  double Ke;     ///< entanglement bulk modulus, MPa
  double Ge;     ///< entanglement shear modulus, MPa
  double betae;  ///< entanglement volumetric thermal expansion coefficient
  double rhoe0;  ///< entanglement (chain) initial density
  double etae;   ///< entanglement viscosity, MPa-s

  double C_v;    ///< net volumetric heat capacity (must account for matrix+chain+particle)
  double kappa;  ///< net thermal conductivity (must account for matrix+chain+particle)

  // E_a and R can be SI units since they cancel out in the exponent
  double Af;    ///< forward (low-high) exponential prefactor, 1/s
  double E_af;  ///< forward (low-high) activation energy, J/mol
  double Ar;    ///< reverse exponential prefactor, 1/s
  double E_ar;  ///< reverse activation energy, J/mol
  double R;     ///< universal gas constant, J/mol/K
  double Tr;    ///< reference temperature, K

  double gw;  ///< particle weight fraction
  double wm;  ///< matrix mass fraction (set to 0.5, not real for now)

  using State = smith::Empty;

  template <int d>
  struct SymmetricStatePacking {
    static_assert(d >= 1, "Invalid matrix dimension.");
    static constexpr int sym_size = d * (d + 1) / 2;

    template <int sd, typename ScalarT, typename SymmT>
    static auto pack(const ScalarT& scalar, const smith::tensor<SymmT, d, d>& symm)
    {
      static_assert(sd == 1 + sym_size, "Packed state size mismatch.");
      using PackedValue = decltype(scalar + symm(0, 0));
      smith::tensor<PackedValue, sd> out{};
      out[0] = scalar;
      int k = 1;
      for (int i = 0; i < d; ++i) {
        for (int j = i; j < d; ++j) {
          out[k++] = symm(i, j);
        }
      }
      return out;
    }

    template <typename T, int sd>
    static auto unpack(const smith::tensor<T, sd>& in)
    {
      static_assert(sd == 1 + sym_size, "Packed state size mismatch.");
      T scalar = in[0];
      smith::tensor<T, d, d> symm{};
      int k = 1;
      for (int i = 0; i < d; ++i) {
        for (int j = i; j < d; ++j) {
          symm(i, j) = in[k];
          symm(j, i) = in[k];
          ++k;
        }
      }
      return smith::tuple{scalar, symm};
    }
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

  template <typename scalar>
  SMITH_HOST_DEVICE auto equilibrium_xi(scalar temp) const
  {
    using std::pow, std::exp;
    auto Tt = 443.0;
    auto k = 36.0;
    return exp(-(pow(temp / Tt, k)));
  }

  template <typename scalar>
  SMITH_HOST_DEVICE auto Gm0(scalar g) const
  {
    // low-T shear modulus at reference temperature as a function of particle wt% g
    auto junk = g;
    return Gm * junk / g;
  }

  template <typename scalar>
  SMITH_HOST_DEVICE auto f1(scalar T) const
  {
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return exp(-N * (T - Tr));
  }

  template <typename scalar>
  SMITH_HOST_DEVICE auto df1(scalar T) const
  {
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return -N * exp(-N * (T - Tr));
  }

  template <typename scalar>
  SMITH_HOST_DEVICE auto Ge0(scalar g) const
  {
    // high-T shear modulus at reference temperature as a function of particle wt% g
    auto junk = g;
    return Ge * junk / g;
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5, int d, int sd>
  auto operator()(double dt, State&, const smith::tensor<T1, d, d>& grad_u, const T2& grad_v, T3 theta,
                  const smith::tensor<T4, d>& grad_theta, const smith::tensor<T5, sd>& alpha_old) const
  {
    // Calculate Alpha new using the old variables to be used

    auto [w_old, F_old] = SymmetricStatePacking<d>::template unpack<T5, sd>(alpha_old);

    using std::pow, std::exp;

    // Tr is a double but I need auto to add to theta
    auto tempref = Tr;  // 353.0;

    theta = theta + tempref;

    auto wep = w_old;      // previous entangled fraction
    auto wfp = 1.0 - wep;  // previous free fraction
    auto Fesip = F_old;    // previous inverse of mapping F^{es}

    // get equilibrium wl=xi
    auto xi = equilibrium_xi(theta);
    // std::cout << "wh: " << wh << "\n";

    // get kinematics
    static constexpr auto I = smith::Identity<d>();

    auto F = grad_u + I;
    auto FeIni =
        dot(F, Fesip);  // Fe for the extant entangled material, called Fh1 in my notes about the relaxation method
    auto Je = det(FeIni);

    auto C = dot(transpose(F), F);
    auto Ci = inv(C);
    auto D = greenStrainRate(0.0 * grad_u, grad_v);  // symmetric velocity gradient, allowing grad_v = zero

    auto B = dot(F, transpose(F));
    auto trB = tr(B);
    auto B_bar = B - (trB / 3.0) * I;
    auto J = det(F);

    // get moduli
    auto Gm_eff = Gm0(gw) * f1(theta);
    auto Ge_eff = Ge0(gw);

    // calculate forward and reverse reaction rate
    auto kf = Af * exp(-E_af / (R * theta));
    auto kr = Ar * exp(-E_ar / (R * theta));

    // get mass fraction supplies, forward and reverse
    auto dwff = (xi - wfp) * kf * dt / (1. + kf * dt);
    auto dwer = (1. - xi - wep) * kr * dt / (1. + kr * dt);
    // get net mass fraction supply
    auto dwe = -dwff + dwer;

    auto aux1 = 0.0, aux2 = 0.0, aux3 = 0.0;
    // if dwh>0, I need to get the new equivalent Fhsi
    if (dwe > 0 && wep == 0) {
      aux1 = 1.0;          // initialize Fhsi as the inverse of F at the current time
    } else if (dwe > 0) {  // calculate the current elastic deformation of the high-T material
      aux2 = 1.0;          // update the effective value of Fhsi
    } else {
      aux3 = 1.0;
    }

    auto Fesi = aux1 * inv(F) + aux2 * (wep / (wep + dwe)) * Fesip + aux3 * Fesip;
    auto Fe = dot(F, Fesi);
    auto Ce = dot(transpose(Fe), Fe);
    auto Ue = sqrt_symm(Ce);
    // state.Fesi = get_value(Fesi);

    // update mass fractions
    auto we = wep + dwe;  // 1 + wep + dwe -wep-dwe;

    // std::cout << theta << "," << kf << "," << kr << "," << dwe << "," << we << "," << wep << "\n";

    // calculate B_bar, J based on Fh
    auto Be = dot(Fe, transpose(Fe));
    auto trBe = tr(Be);
    auto Be_bar = Be - (trBe / 3.0) * I;

    // calculate kirchoff stress
    auto Tm = Gm_eff * pow(J, -2. / 3.) * B_bar + J * Km * (J - 1. - betam * (theta - Tr)) * I;  // + etal * D;
    auto Te = Ge_eff * pow(Je, -2. / 3.) * Be_bar + Je * Ke * (Je - 1.) * I;                     // + etah * D;

    auto TK = wm * Tm + (1. - wm) * we * Te + 2 * ((1. - we) * etam + we * etae) * D;

    // 1st Piola from Kirchhoff
    const auto Piola = dot(TK, inv(transpose(F)));

    // heat flux
    const auto q0 = -kappa * grad_theta;
    // std::cout << dwe+wep << "," << we << "," << wep << "\n";
    // state.w_e = get_value(we);
    // std::cout << state.w_e << "\n";
    // state.Cp = get_value(dot(transpose(F),F));

    // internal heat power
    auto green_strain_rate = greenStrainRate(grad_u, grad_v);
    // viscous stress
    auto Sv = 2 * ((1. - we) * etam + we * etae) * dot(Ci, dot(green_strain_rate, Ci));
    // derivative of elastic S with respect to T
    auto dtmdT = Gm0(gw) * df1(theta) * pow(J, -2. / 3) * B_bar - Km * J * betam * I;
    auto dSedT = dot(inv(F), dot(wm * dtmdT, transpose(inv(F))));
    const auto s0 = tr(dot(Sv + theta * dSedT, green_strain_rate));
    // const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);

    auto alpha_new = SymmetricStatePacking<d>::template pack<sd>(we, Ue);
    return smith::tuple{Piola, C_v, s0, q0, alpha_new};
  }

  // template <typename T1, typename T2, typename T3, typename T4, int dim>
  // auto operator()(double dt, State& state, const smith::tensor<T1, dim, dim>& grad_u, const smith::tensor<T2, dim,
  // dim>& grad_v, T3 theta,
  //                 const tensor<T4, dim>& grad_theta) const
  // {
  //   using std::pow, std::exp;

  //   // Tr is a double but I need auto to add to theta
  //   auto tempref = Tr; // 353.0;

  //   theta=theta+tempref;

  //   auto wep = state.w_e;     // previous entangled fraction
  //   auto wfp = 1.0-wep;       // previous free fraction
  //   auto Fesip = state.Fesi;  // previous inverse of mapping F^{es}

  //   // get equilibrium wl=xi
  //   auto xi = equilibrium_xi(theta);
  //   //std::cout << "wh: " << wh << "\n";

  //   // get kinematics
  //   constexpr auto I = Identity<dim>();

  //   auto F = grad_u + I;
  //   auto FeIni = dot(F,Fesip); // Fe for the extant entangled material, called Fh1 in my notes about the relaxation
  //   method auto Je = det(FeIni);

  //   auto C = dot(transpose(F), F);
  //   auto Ci = inv(C);
  //   auto D = 0.5*(grad_v+transpose(grad_v));//dot(inv(transpose(F)),CdFi)*0.5;

  //   auto B = dot(F, transpose(F));
  //   auto trB = tr(B);
  //   auto B_bar = B - (trB / 3.0) * I;
  //   auto J = det(F);

  //   // get moduli
  //   auto Gm_eff = Gm0(gw)*f1(theta);
  //   auto Ge_eff = Ge0(gw);

  //   // calculate forward and reverse reaction rate
  //   auto kf = Af * exp(-E_af / (R*theta));
  //   auto kr = Ar * exp(-E_ar / (R*theta));

  //   // get mass fraction supplies, forward and reverse
  //   auto dwff = (xi-wfp)*kf*dt/(1.+kf*dt);
  //   auto dwer = (1.-xi-wep)*kr*dt/(1.+kr*dt);
  //   // get net mass fraction supply
  //   auto dwe = -dwff + dwer;

  //   auto aux1 = 0.0, aux2 = 0.0, aux3 = 0.0;
  //   // if dwh>0, I need to get the new equivalent Fhsi
  //   if (dwe>0 && wep==0) {
  //     aux1 = 1.0; // initialize Fhsi as the inverse of F at the current time
  //   }
  //   else if (dwe>0) { // calculate the current elastic deformation of the high-T material
  //     aux2 = 1.0; // update the effective value of Fhsi
  //   }
  //   else {
  //     aux3 = 1.0;
  //   }

  //   auto Fesi = aux1 * inv(F) + aux2 * (wep/(wep+dwe))*Fesip + aux3 * Fesip;
  //   auto Fe = dot(F,Fesi);
  //   state.Fesi = get_value(Fesi);

  //   // update mass fractions
  //   auto we = wep+dwe;//1 + wep + dwe -wep-dwe;

  //   //std::cout << theta << "," << kf << "," << kr << "," << dwe << "," << we << "," << wep << "\n";

  // // calculate B_bar, J based on Fh
  //   auto Be = dot(Fe, transpose(Fe));
  //   auto trBe = tr(Be);
  //   auto Be_bar = Be - (trBe / 3.0) * I;

  //   // calculate kirchoff stress
  //   auto Tm = Gm_eff * pow(J, -2./3.) * B_bar + J * Km * (J - 1. - betam*(theta-Tr)) * I; // + etal * D;
  //   auto Te = Ge_eff * pow(Je, -2./3.) * Be_bar + Je * Ke * (Je - 1.) * I; // + etah * D;

  //   auto TK = wm * Tm + (1. - wm) * we * Te + 2*((1.-we)*etam+we*etae)*D;

  //   // 1st Piola from Kirchhoff
  //   const auto Piola = dot(TK, inv(transpose(F)));

  //   // heat flux
  //   const auto q0 = -kappa * grad_theta;
  //   std::cout << dwe+wep << "," << we << "," << wep << "\n";
  //   state.w_e = get_value(we);
  //   std::cout << state.w_e << "\n";
  //   state.Cp = get_value(dot(transpose(F),F));

  //   // internal heat power
  //   auto greenStrainRate =
  //       0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
  //   // viscous stress
  //   auto Sv = 2*((1.-we)*etam+we*etae)*dot(Ci,dot(greenStrainRate,Ci));
  //   // derivative of elastic S with respect to T
  //   auto dtmdT = Gm0(gw)*df1(theta)*pow(J,-2./3)*B_bar-Km*J*betam*I;
  //   auto dSedT = dot(inv(F),dot(wm*dtmdT,transpose(inv(F))));
  //   const auto s0 = tr(dot(Sv+theta*dSedT,greenStrainRate));
  //   //const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);

  //   return smith::tuple{Piola, C_v, s0, q0};
  // }
  static constexpr int numParameters() { return 1; }
};
};  // namespace extended_thermomechanics_materials
