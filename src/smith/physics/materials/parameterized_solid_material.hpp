// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file parameterized_solid_material.hpp
 *
 * @brief The material and load types for the solid functional physics module
 */

#pragma once

#include "smith/numerics/functional/functional.hpp"
#include "smith/physics/materials/solid_material.hpp"

/// SolidMechanics helper data types
namespace smith::solid_mechanics {

/**
 * @brief Linear isotropic elasticity material model
 *
 */
struct ParameterizedLinearIsotropicSolid {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a linear isotropic material model
   *
   * @tparam dim Spatial dimension of the mesh
   * @tparam DispGradType Displacement gradient type
   * @tparam BulkType Bulk modulus type
   * @tparam ShearType Shear modulus type
   * @param du_dX Displacement gradient with respect to the reference configuration (displacement_grad)
   * @param DeltaK The bulk modulus offset
   * @param DeltaG The shear modulus offset
   * @return The calculated material response (stress) for the material
   */
  template <int dim, typename DispGradType, typename BulkType, typename ShearType>
  SMITH_HOST_DEVICE auto operator()(State& /*state*/, const smith::tensor<DispGradType, dim, dim>& du_dX,
                                    const BulkType& DeltaK, const ShearType& DeltaG) const
  {
    constexpr auto I = Identity<dim>();
    auto K = K0 + get<0>(DeltaK);
    auto G = G0 + get<0>(DeltaG);
    auto lambda = K - (2.0 / dim) * G;
    auto epsilon = 0.5 * (transpose(du_dX) + du_dX);
    return lambda * tr(epsilon) * I + 2.0 * G * epsilon;
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 2; }

  double density;  ///< mass density
  double K0;       ///< base bulk modulus
  double G0;       ///< base shear modulus
};

/**
 * @brief Neo-Hookean material model
 *
 */
struct ParameterizedNeoHookeanSolid {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * @tparam dim The spatial dimension of the mesh
   * @tparam DispGradType Displacement gradient type
   * @tparam BulkType Bulk modulus type
   * @tparam ShearType Shear modulus type
   * @param du_dX Displacement gradient with respect to the reference configuration (displacement_grad)
   * @param DeltaK The bulk modulus offset
   * @param DeltaG The shear modulus offset
   * @return The calculated material response (first Piola stress) for the material
   */
  template <int dim, typename DispGradType, typename BulkType, typename ShearType>
  SMITH_HOST_DEVICE auto operator()(State& /*state*/, const smith::tensor<DispGradType, dim, dim>& du_dX,
                                    const BulkType& DeltaK, const ShearType& DeltaG) const
  {
    using std::log1p;
    constexpr auto I = Identity<dim>();
    auto K = K0 + get<0>(DeltaK);
    auto G = G0 + get<0>(DeltaG);
    auto lambda = K - (2.0 / dim) * G;
    auto B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    auto logJ = log1p(detApIm1(du_dX));

    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return dot(TK, inv(transpose(F)));
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 2; }

  double density;  ///< mass density
  double K0;       ///< base bulk modulus
  double G0;       ///< base shear modulus
};

/**
 * @brief Infers type resulting from algebraic expressions of a group of variables
 *
 * Useful if one needs to create a variable that is dual-valued if any operands are dual.
 */
template <typename... T>
struct underlying_scalar {
  using type = decltype((T{} + ...));  ///< type of the sum of the parameters
};

/// @brief J2 material with Voce hardening, with hardening parameters exposed as differentiable parameters
struct ParameterizedJ2Nonlinear {
  static constexpr int dim = 3;         ///< dimension of space
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual for accepting return map solution

  double E;        ///< Young's modulus
  double nu;       ///< Poisson's ratio
  double density;  ///< mass density

  /// @brief variables required to characterize the hysteresis response
  struct State {
    tensor<double, dim, dim> plastic_strain;  ///< plastic strain
    double accumulated_plastic_strain;        ///< uniaxial equivalent plastic strain
  };

  /// @brief calculate the first Piola stress, given the displacement gradient and previous material state
  template <typename DisplacementGradient, typename YieldStrength, typename SaturationStrength, typename StrainConstant>
  auto operator()(State& state, const DisplacementGradient du_dX, const YieldStrength sigma_y,
                  const SaturationStrength sigma_sat, const StrainConstant strain_constant) const
  {
    // The output stress tensor should use dual numbers if any of the parameters are dual.
    // This slightly ugly trick to accomplishes that by picking up any dual number types
    // from the parameters into the dummy variable "one".
    // Another possiblity would be to cast the results to the correct type at the end, which
    // would avoid doing any unneccessary arithmetic with dual numbers.
    using T = typename underlying_scalar<YieldStrength, SaturationStrength, StrainConstant>::type;
    T one;
    one = 1.0;

    using std::sqrt;
    constexpr auto I = Identity<dim>();
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    const double rel_tol = tol * get_value(sigma_y);

    // (i) elastic predictor
    auto el_strain = sym(du_dX) - state.plastic_strain;
    auto p = K * tr(el_strain);
    auto s = 2.0 * G * dev(el_strain) * one;  // multiply by "one" to get type correct for parameter derivatives
    auto q = sqrt(1.5) * norm(s);

    // (ii) admissibility
    const double eqps_old = state.accumulated_plastic_strain;
    auto residual = [eqps_old, G, *this](auto delta_eqps, auto trial_mises, auto y0, auto ysat, auto e0) {
      auto Y = this->flow_strength(eqps_old + delta_eqps, y0, ysat, e0);
      return trial_mises - 3.0 * G * delta_eqps - Y;
    };
    if (residual(0.0, get_value(q), get_value(sigma_y), get_value(sigma_sat), get_value(strain_constant)) > rel_tol) {
      // (iii) return mapping

      // Note the tolerance for convergence is the same as the tolerance for entering the return map.
      // This ensures that if the constitutive update is called again with the updated internal
      // variables, the return map won't be repeated.
      ScalarSolverOptions opts{.xtol = 0, .rtol = rel_tol, .max_iter = 25};
      double lower_bound = 0.0;
      double upper_bound = (get_value(q) - flow_strength(eqps_old, get_value(sigma_y), get_value(sigma_sat),
                                                         get_value(strain_constant))) /
                           (3.0 * G);
      auto [delta_eqps, status] =
          solve_scalar_equation(residual, 0.0, lower_bound, upper_bound, opts, q, sigma_y, sigma_sat, strain_constant);

      auto Np = 1.5 * s / q;

      s = s - 2.0 * G * delta_eqps * Np;
      state.accumulated_plastic_strain += get_value(delta_eqps);
      state.plastic_strain += get_value(delta_eqps) * get_value(Np);
    }

    return s + p * I;
  }

  /** @brief Computes flow strength from Voce's hardening law */
  template <typename PlasticStrain, typename YieldStrength, typename SaturationStrength, typename StrainConstant>
  auto flow_strength(const PlasticStrain accumulated_plastic_strain, const YieldStrength sigma_y,
                     const SaturationStrength sigma_sat, const StrainConstant strain_constant) const
  {
    using std::exp;
    return sigma_sat - (sigma_sat - sigma_y) * exp(-accumulated_plastic_strain / strain_constant);
  };
};

}  // namespace smith::solid_mechanics
