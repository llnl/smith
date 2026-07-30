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

/// Viscoelastic PNC thermal stiffening material model
struct ViscoThermalStiffeningMaterial {
  double K;      ///< material bulk modulus
  double beta;   ///< material volumetric thermal expansion coefficient
  
  double Gm_beta;///< matrix nonequilibrium shear modulus scale factor, Gmneq=Gmeq*Gm_beta
  double etam;   ///< matrix viscosity, MPa-s

  double Gc_beta;///< chain nonequilibrium shear modulus scale factor, Gcneq=Gceq*Gc_beta
  double etac;   ///< chain viscosity, MPa-s
  double Jcm;    ///< relative volume change between matrix and chains (maybe not linked to Fcm yet)

  double C_v;    ///< material volumetric heat capacity (must account for matrix+chain+particle)
  double kappa;  ///< material thermal conductivity (must account for matrix+chain+particle)
  double rho0;   ///< material initial density

  double c;      ///< rate-dependent entanglement dissipation term
  double mscale; ///< scaling parameter since moduli are hard-coded as Pa

  // E_a and R can be SI units since they cancel out in the exponent
  double Af;     ///< forward (low-high) exponential prefactor, 1/s
  double E_af;   ///< forward (low-high) activation energy, J/mol
  double Ar;     ///< reverse exponential prefactor, 1/s
  double E_ar;   ///< reverse activation energy, J/mol
  double R;      ///< universal gas constant, J/mol/K
  double Tr;     ///< reference temperature, K

  double gw;     ///< particle weight fraction

  using State = smith::Empty;

  template <int d>
  struct FullStatePacking {
  static constexpr int sz = 3;
  static constexpr int tensor_size = sz * sz;
  static constexpr int num_tensors = 3;
  static constexpr int packed_size = 2 + num_tensors * tensor_size;

  template <int sd, typename ScalarW, typename ScalarT, typename T0, typename T1, typename T2>
  static auto pack(const ScalarW& wp,
                   const ScalarT& Tp,
                   const smith::tensor<T0, d, d>& A,
                   const smith::tensor<T1, d, d>& B,
                   const smith::tensor<T2, d, d>& C)
  {
    static_assert(sd == packed_size, "Packed state size mismatch.");
    using PackedValue = decltype(wp + Tp + A(0,0) + B(0,0) + C(0,0));

    smith::tensor<PackedValue, sd> out{};

    // store mass fraction w
    out[0] = wp;
    // store previous temperature
    out[1] = Tp;
    // iterate through Fcm tensor
    int k = 2;
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        out[k++] = A(i, j);
      }
    }
    // iterate through Fmv tensor
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        out[k++] = B(i, j);
      }
    }
    // iterate through Fcv tensor
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        out[k++] = C(i, j);
      }
    }

    return out;
  }

  template <typename T, int sd>
  static auto unpack(const smith::tensor<T, sd>& in)
  {
    static_assert(sd == packed_size, "Packed state size mismatch.");

    // unpack entangled mass fraction
    T wp = in[0];
    // unpack previous temperature
    T Tp = in[1];

    smith::tensor<T, d, d> A{};
    smith::tensor<T, d, d> B{};
    smith::tensor<T, d, d> C{};
   
    // unpack Fcm 
    int k = 2;
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        A(i, j) = in[k++];
      }
    }
    // unpack Fmv
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        B(i, j) = in[k++];
      }
    }
    // unpack Fcv
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        C(i, j) = in[k++];
      }
    }

    return smith::tuple{wp, Tp, A, B, C};
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

  template <typename scalar>
  SMITH_HOST_DEVICE auto fGmeq(scalar g) const
  {
    using std::pow;
    // percolation parameters
    auto Gr = 0.017;      //GPa, rigid modulus
    auto Gs0 = 1.7e-6;    //GPa, soft modulus
    auto Xc = 0.24;       //critical percolation volume fraction
    auto rhof = 2.65;     //g/cc, filler density
    auto rhom = 1.06;     //g/cc, matrix density
    auto n = 0.4;         //percolation exponent

    // Guth-Gold parameters
    auto a1 = 2.5;        //linear X_eff correction
    auto a2 = 14.1;       //quadratic X_eff correction
    auto r = 100.0;       //particle radius, nm
    auto d = 50.0;        //interphase thickness, nm

    auto X = g*rhom/(rhof+g*(rhom-rhof));            //convert weight fraction gw to volume fraction X
    auto X_eff = X*pow(r+d,3)/pow(r,3);              //effective volume fraction
    auto Gs = Gs0*(1.+a1*X_eff+a2*X_eff*X_eff);      //Guth-Gold correction
    auto psi = 0.;
    if (X>Xc) { // do I want to use X or X_eff here?
      psi = X*pow((X-Xc)/(1.-Xc),n);
    }
    auto Gnum = (1.-2.*psi+psi*X)*Gr*Gs+(1.-X)*psi*Gr*Gr;
    auto Gdenom = (1.-X)*Gr+(X-psi)*Gs;
    auto G = Gnum/Gdenom; // this is in GPa
    
    return (G*1.e9)/mscale;        // convert to Pa*mscale
  }

  template <typename scalar>
  SMITH_HOST_DEVICE auto fGmneq(scalar g) const
  {
    using std::pow;
    // percolation parameters
    auto Gr = 0.017;      //GPa, rigid modulus
    auto Gs0 = 1.7e-6;    //GPa, soft modulus
    auto Xc = 0.24;       //critical percolation volume fraction
    auto rhof = 2.65;     //g/cc, filler density
    auto rhom = 1.06;     //g/cc, matrix density
    auto n = 0.4;         //percolation exponent

    // Guth-Gold parameters
    auto a1 = 2.5;        //linear X_eff correction
    auto a2 = 14.1;       //quadratic X_eff correction
    auto r = 100.0;       //particle radius, nm
    auto d = 50.0;        //interphase thickness, nm

    auto X = g*rhom/(rhof+g*(rhom-rhof));            //convert weight fraction gw to volume fraction X
    auto X_eff = X*pow(r+d,3)/pow(r,3);              //effective volume fraction
    auto Gs = Gs0*(1.+a1*X_eff+a2*X_eff*X_eff);      //Guth-Gold correction
    auto psi = 0.;
    if (X>Xc) { // do I want to use X or X_eff here?
      psi = X*pow((X-Xc)/(1.-Xc),n);
    }
    auto Gnum = (1.-2.*psi+psi*X)*Gr*Gs+(1.-X)*psi*Gr*Gr;
    auto Gdenom = (1.-X)*Gr+(X-psi)*Gs;
    auto G = Gnum/Gdenom; // this is in GPa
    
    return (Gm_beta*G*1.e9)/mscale;        // convert to Pa*mscale
  }

  template <typename scalar>
  SMITH_HOST_DEVICE auto F1(scalar T) const
  {
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return exp(-N * (T - Tr));
  }
/*
  template <typename scalar>
  SMITH_HOST_DEVICE auto dF1(scalar T) const
  {
    using std::exp;
    // derivative of thermal softening function for low-T modulus
    auto N = 0.02;
    return -N * exp(-N * (T - Tr));
  }
*/
  template <typename scalar>
  SMITH_HOST_DEVICE auto fGceq(scalar g) const
  {
    using std::pow;
    // percolation parameters
    auto Gr = 0.12;      //GPa, rigid modulus
    auto Gs0 = 6.5e-7;    //GPa, soft modulus
    auto Xc = 0.05;       //critical percolation volume fraction
    auto rhof = 2.65;     //g/cc, filler density
    auto rhom = 1.06;     //g/cc, matrix density
    auto n = 1.2;         //percolation exponent

    // Guth-Gold parameters
    auto a1 = 2.5;        //linear X_eff correction
    auto a2 = 14.1;       //quadratic X_eff correction
    auto r = 100.0;       //particle radius, nm
    auto d = 50.0;        //interphase thickness, nm

    auto X = g*rhom/(rhof+g*(rhom-rhof));            //convert weight fraction gw to volume fraction X
    auto X_eff = X*pow(r+d,3)/pow(r,3);              //effective volume fraction
    auto Gs = Gs0*(1.+a1*X_eff+a2*X_eff*X_eff);      //Guth-Gold correction
    auto psi = 0.;
    if (X>Xc) { // do I want to use X or X_eff here?
      psi = X*pow((X-Xc)/(1.-Xc),n);
    }
    auto Gnum = (1.-2.*psi+psi*X)*Gr*Gs+(1.-X)*psi*Gr*Gr;
    auto Gdenom = (1.-X)*Gr+(X-psi)*Gs;
    auto G = Gnum/Gdenom; // this is in GPa
    
    return (G*1.e9)/mscale;        // convert to Pa*mscale
  }

  template <typename scalar>
  SMITH_HOST_DEVICE auto fGcneq(scalar g) const
  {
    using std::pow;
    // percolation parameters
    auto Gr = 0.12;      //GPa, rigid modulus
    auto Gs0 = 6.5e-7;    //GPa, soft modulus
    auto Xc = 0.05;       //critical percolation volume fraction
    auto rhof = 2.65;     //g/cc, filler density
    auto rhom = 1.06;     //g/cc, matrix density
    auto n = 1.2;         //percolation exponent

    // Guth-Gold parameters
    auto a1 = 2.5;        //linear X_eff correction
    auto a2 = 14.1;       //quadratic X_eff correction
    auto r = 100.0;       //particle radius, nm
    auto d = 50.0;        //interphase thickness, nm

    auto X = g*rhom/(rhof+g*(rhom-rhof));            //convert weight fraction gw to volume fraction X
    auto X_eff = X*pow(r+d,3)/pow(r,3);              //effective volume fraction
    auto Gs = Gs0*(1.+a1*X_eff+a2*X_eff*X_eff);      //Guth-Gold correction
    auto psi = 0.;
    if (X>Xc) { // do I want to use X or X_eff here?
      psi = X*pow((X-Xc)/(1.-Xc),n);
    }
    auto Gnum = (1.-2.*psi+psi*X)*Gr*Gs+(1.-X)*psi*Gr*Gr;
    auto Gdenom = (1.-X)*Gr+(X-psi)*Gs;
    auto G = Gnum/Gdenom; // this is in GPa
    
    return (Gc_beta*G*1.e9)/mscale;        // convert to Pa*mscale
  }

  template <typename scalar>
  SMITH_HOST_DEVICE auto fS(scalar g) const
  {
    using std::pow;
    //auto G4 = 1.
    //auto A4 = 1.e16;
    //auto AG4 = A4*G4;
    auto rhof = 2.65;     //g/cc, filler density
    auto rhom = 1.06;     //g/cc, matrix density
    auto X = g*rhom/(rhof+g*(rhom-rhof));            //convert weight fraction gw to volume fraction X
    if (X==0.0) {
      auto gwn = 0.01;
      X = gwn*rhom/(rhof+gwn*(rhom-rhof));
    }
    auto d = 55; //nm
    auto TwoRG = 24.534; //exact gw=0.4 value
    auto ID = d*(pow(2./(X*3.14159265358979323846),1./3)-1.0);
    auto fID = pow(ID/TwoRG,2.0);
    
    return 1./fID; //multiply this by Af0
  }
/*
  template <typename scalar>
  SMITH_HOST_DEVICE auto fAr(scalar g) const
  {
    using std::pow;
    //auto G4 = 1.
    //auto A4 = 0.5e-20;
    //auto AG4 = A4*G4;
    auto rhof = 2.65;     //g/cc, filler density
    auto rhom = 1.06;     //g/cc, matrix density
    auto X = g*rhom/(rhof+g*(rhom-rhof));            //convert weight fraction gw to volume fraction X
    if (X==0.0) {
      auto gwn = 0.01;
      X = gwn*rhom/(rhof+gwn*(rhom-rhof));
    }
    auto d = 55; //nm
    auto TwoRG = 24.534; //exact gw=0.4 value
    auto ID = d*(pow(2./(X*3.14159265358979323846),1./3)-1.0);
    auto fID = pow(ID/TwoRG,2.0);
    
    return 1./fID; //multiply this by Ar0
  }
*/
  template <typename T1, typename T2, typename T3, typename T4, typename T5, int d, int sd>
  auto operator()(double dt, State&, const smith::tensor<T1, d, d>& grad_u, const T2& grad_v, T3 theta,
                  const smith::tensor<T4, d>& grad_theta, const smith::tensor<T5, sd>& alpha_old) const
  {
    // Calculate Alpha new using the old variables to be used

    auto [wp, thetap, Fcmp, Fmvp, Fcvp] = FullStatePacking<d>::template unpack<T5, sd>(alpha_old);

    using std::pow, std::exp;

    // Tr is a double but I need auto to add to theta
    auto tempref = Tr;  // 353.0;
    theta = theta + tempref;

    // get kinematics
    static constexpr auto I = smith::Identity<d>();
    auto F = grad_u + I;

    // calculate forward and reverse reaction rates 
    // (haven't removed thetap as internal variable yet, +- here just so I don't get error)
    auto kf = Af * fS(gw) * exp(-E_af / (R * theta)) +thetap-thetap;
    auto kr = Ar * fS(gw) * exp(-E_ar / (R * theta));
    auto ksum = kf+kr;
    // exponential integration to get new entanglement fraction
    auto w = 0.0*kf;
    if (ksum > 0.0) {
      auto winf = kf/ksum;
      w = winf+(wp-winf)*exp(-ksum*dt);
    }
    else {
      w = wp;
    }
    
    // get mass fraction increment and rate for later
    auto dw = w - wp;
    auto what = dw/dt;

    // get moduli
    auto Gmeq = fGmeq(gw) * F1(theta);
    auto Gmneq = fGmneq(gw) * F1(theta);
    auto Gceq = fGceq(gw);
    auto Gcneq = fGcneq(gw);
    // do some kinematics
    auto J = det(F);
    auto B = dot(F,transpose(F));

    // equilibrium Cauchy stress terms
    auto Tvol = K*(J-1.-beta*(theta-tempref))*I;
    auto Tmeq = Gmeq*pow(J,-5./3)*dev(B); 

    // nonequilibrium matrix stuff
    // trial values
    auto Fmtr = dot(F,inv(Fmvp));
    auto Cmtr = dot(transpose(Fmtr),Fmtr);
    auto Hmtr = 0.5*log_symm(Cmtr);
    auto Mm = 2.0*Gmneq*dev(Hmtr);
    // update Dmv
    auto Dmv = Mm/(2.*(etam+dt*Gmneq));
    Mm = Mm - 2.*dt*Gmneq*Dmv; //or Mm = 2.*etam*Dvm
    // update Fmv
    auto Fmv = dot(exp_symm(dt*Dmv),Fmvp);
    // get Fme
    auto Fme = dot(F,inv(Fmv));
    // nonequilibrium Cauchy matrix stress
    auto Tmneq = dot(transpose(inv(Fme)),dot(Mm,transpose(Fme)))/J;

    // shift of chain deformation stuff
    // trial values
    auto Fctr = dot(F,inv(Fcmp));
    auto Cctr = dot(transpose(Fctr),Fctr);
    auto Hctr = 0.5*log_symm(Cctr);
    auto Mcm = 2.0*w*Jcm*Gceq*dev(Hctr);
    // get approximation to lambda_dot
    auto junk = double_dot(Mcm,Mcm);
    auto lamdot = 0.0 * junk;
    if (dw > 0) {
      auto Qtr = double_dot(Mcm,Mcm);
      if (Qtr != 0.0) {
        auto alph = 2.0*w*Jcm*Gceq*dt;
        auto Aa = 1.0+0.5*dw/w;
        auto Bb = 2.0*alph*c*what*what/Qtr;
        //todo: believe I need the positive root but then need check to avoid 1-2*Aa*Bb<0
        //auto MM = (0.5*Qtr/(Aa*Aa))*(1.0-Aa*Bb+sqrt(1.0-2.*Aa*Bb));
        auto MM = (0.5*Qtr/(Aa*Aa))*(1.0-Aa*Bb+sqrt(abs(1.0-2.*Aa*Bb)));
        lamdot = what/(4.*w*w*Jcm*Gceq) + c*what*what/MM;
      }
    }
    // update Dcm as needed
    auto Dcm = Mcm*lamdot/(1.+2*lamdot*w*Jcm*Gceq*dt);
    //Mcm = 0.0 * Dcm;
    if (dw > 0 && lamdot != 0.0) {
      // get the Mandel stress
      Mcm = Dcm/lamdot;
    }
    // update Fcm
    auto Fcm = Fcmp + 0.0 * Dcm; // necessary for typing reasons
    if (dw > 0) {
      Fcm = dot(exp_symm(dt*Dcm),Fcmp);
    }
    // get Fc
    auto Fc = dot(F,inv(Fcm));
    // equilibrium Cauchy chain stress
    auto Tceq = dot(transpose(inv(Fc)),dot(Mcm,transpose(Fc)))/J;

    // nonequilibrium chain stuff
    // trial values
    auto Fcetr = dot(Fc,inv(Fcvp));
    auto Ccetr = dot(transpose(Fcetr),Fcetr);
    auto Hcetr = 0.5*log_symm(Ccetr);
    auto Mce = 2.0*w*Jcm*Gcneq*dev(Hcetr);
    // update Dcv
    auto Dcv = Mce/(2.*(etac+w*Jcm*Gcneq*dt));
    Mce = Mce - 2.*w*Jcm*Gcneq*dt*Dcv;
    // update Fcv
    auto Fcv = dot(exp_symm(dt*Dcv),Fcvp);
    // get Fce
    auto Fce = dot(Fc,inv(Fcv));
    // nonequilibrium Cauchy chain stress
    auto Tcneq = dot(transpose(inv(Fce)),dot(Mce,transpose(Fce)))/J;

    // total Cauchy stress
    auto T = Tvol + Tmeq + Tmneq + Tceq + Tcneq;
    // Kirchoff stress
    auto TK = J*T;
    // 1st Piola from Kirchhoff
    const auto Piola = dot(TK, inv(transpose(F)));

    // heat flux
    const auto q0 = -kappa * grad_theta;

    // did not update this part with new theory yet
    // leaving generic expression here just so something gets done
    auto green_strain_rate = greenStrainRate(grad_u, grad_v);
    const auto s0 = -K * beta * theta * tr(green_strain_rate);
    // viscous stress
    //auto Sv = 2 * ((1. - we) * etam + we * etae) * dot(Ci, dot(green_strain_rate, Ci));
    // derivative of elastic S with respect to T
    //auto dtmdT = Gm0(gw) * df1(theta) * pow(J, -2. / 3) * B_bar - Km * J * betam * I;
    //auto dSedT = dot(inv(F), dot(wm * dtmdT, transpose(inv(F))));
    //tr(dot(Sv + theta * dSedT, green_strain_rate));
    // const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);

    auto alpha_new = FullStatePacking<d>::template pack<sd>(w, theta, Fcm, Fmv, Fcv);
    return smith::tuple{Piola, C_v, s0, q0, alpha_new};
  }
  static constexpr int numParameters() { return 1; }
};

};  // namespace extended_thermomechanics_materials
