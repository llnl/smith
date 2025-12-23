// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/physics/materials/green_saint_venant_thermoelastic.hpp"

/// Thermomechanics helper data types
namespace smith::thermomechanics {

/**
 * @brief Green-Saint Venant isotropic thermoelastic material model
 *
 */
struct ParameterizedThermoelasticMaterial {
  double density;    ///< density
  double E0;         ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha0;     ///< reference value of thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa0;     ///< thermal conductivity

  /// internal variables for the material model
  struct State {
    double strain_trace;  ///< trace of Green-Saint Venant strain tensor
  };

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 3; }

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   * @tparam T4 Type of the coefficient of thermal expansion scale factor
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in] DeltaE Parameterized Young's modulus offset
   * @param[in] DeltaKappa Parameterized thermal conductivity offset
   * @param[in] ScaleAlpha Parameterized thermal conductivity offset
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename DispGradType, typename TempType, typename TempGradType, typename YoungsType, typename ConductType,
            typename CoupleType, int dim>
  auto operator()(State& state, const tensor<DispGradType, dim, dim>& grad_u, TempType theta,
                  const tensor<TempGradType, dim>& grad_theta, YoungsType DeltaE, ConductType DeltaKappa,
                  CoupleType ScaleAlpha) const
  {
    auto E = E0 * get<0>(DeltaE);
    auto kappa = kappa0 + get<0>(DeltaKappa);
    auto alpha = alpha0 * get<0>(ScaleAlpha);

    auto K = E / (3.0 * (1.0 - 2.0 * nu));
    auto G = 0.5 * E / (1.0 + nu);
    static constexpr auto I = Identity<dim>();
    auto F = grad_u + I;
    const auto Eg = greenStrain(grad_u);
    const auto trEg = tr(Eg);

    // stress
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    const auto Piola = dot(F, S);

    // internal heat source
    const auto s0 = -3.0 * K * alpha * theta * (trEg - state.strain_trace);

    // heat flux
    const auto q0 = -kappa * grad_theta;

    state.strain_trace = get_value(trEg);

    return smith::tuple{Piola, C_v, s0, q0};
  }
};

/**
 * @brief Green-Saint Venant isotropic thermoelastic material model
 *
 */
struct ParameterizedThermalStiffeningMaterial {
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
  double kappa_;    ///< net thermal conductivity (must account for matrix+chain+particle)

  // E_a and R can be SI units since they cancel out in the exponent
  double Af;       ///< forward (low-high) exponential prefactor, 1/s
  double E_af;     ///< forward (low-high) activation energy, J/mol
  double Ar;       ///< reverse exponential prefactor, 1/s
  double E_ar;     ///< reverse activation energy, J/mol
  double R;        ///< universal gas constant, J/mol/K
  double Tr;       ///< reference temperature, K

  double gw;       ///< particle weight fraction
  double wm_;       ///< matrix mass fraction (set to 0.5, not real for now)

  /// internal variables for the material model
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
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 1; }

  /**
   * Evaluate constitutive variables for thermomechanics
   *
   * @param grad_u Displacement gradient
   * @param grad_v Velocity gradient
   * @param theta Temperature
   * @param grad_theta Temperature gradient
   * @param state State variables for this material
   * @return tuple: First Piola stress, volumetric heat capacity, heat generation rate, referential heat flux
   */
  template <typename DispGradType, typename VelocGradType, typename TempType, typename TempGradType, 
            typename volFracParamType, int dim>
  auto operator()(double dt, State& state, const tensor<DispGradType, dim, dim>& grad_u,
                  const tensor<VelocGradType, dim, dim>& grad_v, TempType theta,
                  const tensor<TempGradType, dim>& grad_theta, volFracParamType volFracParam) const
  {
  using std::pow, std::exp;
    auto tempref = Tr;
    theta = theta + tempref;

    // Constants (why hardcoded? Need context. Same for any user?)
    auto Tt = 443.0;
    auto k = 36.0;
    auto N = 0.02;

    auto wep = state.w_e;     // previous entangled fraction
    auto wfp = 1.0 - wep;     // previous free fraction
    auto Fesip = state.Fesi;  // previous mapping

    // Equilibrium entangled fraction
    auto xi = exp(-(pow(theta/Tt,k)));

    // Kinematics
    constexpr auto I = Identity<dim>();
    auto F = grad_u + I;
    auto FeIni = dot(F, Fesip);
    auto Je = det(FeIni);

    auto C = dot(transpose(F), F);
    auto Ci = inv(C);
    auto D = 0.5 * (grad_v + transpose(grad_v));

    auto B = dot(F, transpose(F));
    auto trB = tr(B);
    auto B_bar = B - (trB / 3.0) * I;
    auto J = det(F);

    // Moduli
    auto f1 = exp(-N * (theta - Tr));
    auto Gm_eff = gw * f1;
    auto Ge_eff = gw;

    // Reaction rates
    auto kf = Af * exp(-E_af / (R * theta));
    auto kr = Ar * exp(-E_ar / (R * theta));

    // Mass fraction updates
    auto dwff = (xi - wfp) * kf * dt / (1.0 + kf * dt);
    auto dwer = (1.0 - xi - wep) * kr * dt / (1.0 + kr * dt);
    auto dwe = -dwff + dwer;

    auto aux1 = 0.0, aux2 = 0.0, aux3 = 0.0;
    if (dwe > 0 && wep == 0) {
        aux1 = 1.0; // initialize Fhsi
    } else if (dwe > 0) {
        aux2 = 1.0; // update Fhsi
    } else {
        aux3 = 1.0;
    }

    auto Fesi = aux1 * inv(F) + aux2 * (wep / (wep + dwe)) * Fesip + aux3 * Fesip;
    auto Fe = dot(F, Fesi);
    state.Fesi = get_value(Fesi);

    // Update mass fractions
    auto wm = wm_ * get<0>(volFracParam);
    auto we = (wep + dwe) * get<0>(volFracParam);

    // Elastic left Cauchy-Green for entangled fraction
    auto Be = dot(Fe, transpose(Fe));
    auto trBe = tr(Be);
    auto Be_bar = Be - (trBe / 3.0) * I;

    // Kirchhoff stresses
    auto Tm = Gm_eff * pow(J, -2.0 / 3.0) * B_bar + J * Km * (J - 1.0 - betam * (theta - Tr)) * I;
    auto Te = Ge_eff * pow(Je, -2.0 / 3.0) * Be_bar + Je * Ke * (Je - 1.0) * I;

    auto TK = wm * Tm + (1.0 - wm) * we * Te + 2.0 * ((1.0 - we) * etam + we * etae) * D;

    // First Piola-Kirchhoff stress
    const auto Piola = dot(TK, inv(transpose(F)));

    // Heat flux
    auto kappa = kappa_ * get<0>(volFracParam);
    const auto q0 = -kappa * grad_theta;

    state.w_e = get_value(we);
    state.Cp = get_value(dot(transpose(F), F));

    // Internal heat power
    auto greenStrainRate = 0.5 * (grad_v + transpose(grad_v) +
        dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
        
    // Viscous stress
    auto Sv = 2.0 * ((1.0 - we) * etam + we * etae) * dot(Ci, dot(greenStrainRate, Ci));

    // Thermal contribution to stress
    auto df1 = -N*exp(-N * (theta - Tr));
    auto dtmdT = gw * df1 * pow(J, -2.0 / 3.0) * B_bar - Km * J * betam * I;
    auto dSedT = dot(inv(F), dot(wm * dtmdT, transpose(inv(F))));
    const auto s0 = tr(dot(Sv + theta * dSedT, greenStrainRate));

    return smith::tuple{Piola, C_v, s0, q0};
  }
};
}  // namespace smith::thermomechanics
