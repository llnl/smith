// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_material.hpp
 *
 * @brief The material and load types for the solid functional physics module
 */

#pragma once

#include "smith/numerics/functional/functional.hpp"

/// SolidMechanics helper data types
namespace smith::solid_mechanics {

/**
 * @brief Linear isotropic elasticity material model
 *
 */
struct LinearIsotropic {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a linear isotropic material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX Displacement gradient with respect to the reference configuration
   * @return The stress
   */
  template <typename T, int dim>
  SMITH_HOST_DEVICE auto operator()(State& /* state */, const tensor<T, dim, dim>& du_dX) const
  {
    auto I = Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto epsilon = 0.5 * (transpose(du_dX) + du_dX);
    return lambda * tr(epsilon) * I + 2.0 * G * epsilon;
  }

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus
};

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T, int dim>
auto greenStrain(const tensor<T, dim, dim>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

/// @brief St. Venant Kirchhoff hyperelastic model
struct StVenantKirchhoff {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a St. Venant Kirchhoff material model
   *
   * @tparam T Type of the displacement gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   *
   * @return The first Piola stress
   */
  template <typename T, int dim>
  auto operator()(State&, const tensor<T, dim, dim>& grad_u) const
  {
    static constexpr auto I = Identity<dim>();
    auto F = grad_u + I;
    const auto E = greenStrain(grad_u);

    // stress
    const auto S = K * tr(E) * I + 2.0 * G * dev(E);
    return dot(F, S);
  }

  double density;  ///< density
  double K;        ///< Bulk modulus
  double G;        ///< Shear modulus
};

/**
 * @brief Neo-Hookean material model
 *
 */
struct NeoHookean {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The first Piola stress
   */
  template <typename T, int dim>
  SMITH_HOST_DEVICE auto operator()(State& /* state */, const tensor<T, dim, dim>& du_dX) const
  {
    using std::log1p;
    constexpr auto I = Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return dot(TK, inv(transpose(F)));
  }

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus
};

/// Neo-Hookean material version with additive split of deviatoric and volumetric behavior
struct NeoHookeanAdditiveSplit {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief Piola stress calculation
   *
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration
   * @return The first Piola stress
   */
  template <typename T, int dim>
  SMITH_HOST_DEVICE auto operator()(State& /* state */, const tensor<T, dim, dim>& du_dX) const
  {
    using std::pow;
    constexpr auto I = Identity<dim>();
    auto F = I + du_dX;
    auto J = det(F);
    auto Jm13 = pow(J, -1.0 / 3.0);
    auto F_bar = Jm13 * F;
    auto Pdev = G * Jm13 * (F_bar - 1.0 / 3.0 * inner(F_bar, F_bar) * inv(transpose(F_bar)));

    // any volumetric model could be substituted here
    using std::log1p;
    auto logJ = log1p(detApIm1(du_dX));
    auto Pvol = K * logJ * inv(transpose(F));

    return Pdev + Pvol;
  }

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus
};

/**
 * Overstress for linear viscosity
 */
template <typename T>
auto overstress(double viscosity, T accumulated_plastic_strain_rate)
{
  return viscosity * accumulated_plastic_strain_rate;
}

/**
 * @brief Linear isotropic hardening law
 */
struct LinearHardening {
  double sigma_y;  ///< yield strength
  double Hi;       ///< Isotropic hardening modulus
  double eta;      ///< viscosity for linear rate sensitivity

  /**
   * @brief Computes the flow stress
   *
   * @tparam T1 Number-like type
   * @tparam T2 Number-like type
   * @param accumulated_plastic_strain The uniaxial equivalent accumulated plastic strain
   * @param accumulated_plastic_strain_rate The rate of the uniaxial equivalent accumulated plastic strain
   * @return Flow stress value
   */
  template <typename T1, typename T2>
  auto operator()(T1 accumulated_plastic_strain, T2 accumulated_plastic_strain_rate) const
  {
    return sigma_y + Hi * accumulated_plastic_strain + overstress(eta, accumulated_plastic_strain_rate);
  };
};

/**
 * @brief Power-law isotropic hardening law
 */
struct PowerLawHardening {
  double sigma_y;  ///< yield strength
  double n;        ///< hardening index in reciprocal form
  double eps0;     ///< reference value of accumulated plastic strain
  double eta;      ///< viscosity for linear rate sensitivity

  /**
   * @brief Computes the flow stress
   *
   * @tparam T1 Number-like type
   * @tparam T2 Number-like type
   * @param accumulated_plastic_strain The uniaxial equivalent accumulated plastic strain
   * @param accumulated_plastic_strain_rate The rate of the uniaxial equivalent accumulated plastic strain
   * @return Flow stress value
   */
  template <typename T1, typename T2>
  auto operator()(T1 accumulated_plastic_strain, T2 accumulated_plastic_strain_rate) const
  {
    using std::pow;
    return sigma_y * pow(1.0 + accumulated_plastic_strain / eps0, 1.0 / n) +
           overstress(eta, accumulated_plastic_strain_rate);
  };
};

/**
 * @brief Voce's isotropic hardening law
 *
 * This form has an exponential saturation character.
 */
struct VoceHardening {
  double sigma_y;          ///< yield strength
  double sigma_sat;        ///< saturation value of flow strength
  double strain_constant;  ///< The constant dictating how fast the exponential decays
  double eta;              ///< viscosity for linear rate sensitivity

  /**
   * @brief Computes the flow stress
   *
   * @tparam T1 Number-like type
   * @tparam T2 Number-like type
   * @param accumulated_plastic_strain The uniaxial equivalent accumulated plastic strain
   * @param accumulated_plastic_strain_rate The rate of the uniaxial equivalent accumulated plastic strain
   * @return Flow stress value
   */
  template <typename T1, typename T2>
  auto operator()(T1 accumulated_plastic_strain, T2 accumulated_plastic_strain_rate) const
  {
    using std::exp;
    auto& epsilon_p = accumulated_plastic_strain;
    auto Y_eq = sigma_sat - (sigma_sat - sigma_y) * exp(-epsilon_p / strain_constant);
    auto& epsilon_p_dot = accumulated_plastic_strain_rate;
    auto Y_vis = overstress(eta, epsilon_p_dot);
    return Y_eq + Y_vis;
  };
};

/// @brief J2 material with nonlinear isotropic hardening and linear kinematic hardening
template <typename HardeningType>
struct J2SmallStrain {
  static constexpr int dim = 3;         ///< spatial dimension
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

  double E;                 ///< Young's modulus
  double nu;                ///< Poisson's ratio
  HardeningType hardening;  ///< Flow stress hardening model
  double Hk;                ///< Kinematic hardening modulus
  double density;           ///< Mass density

  /// @brief variables required to characterize the hysteresis response
  struct State {
    tensor<double, dim, dim> plastic_strain;  ///< plastic strain
    double accumulated_plastic_strain;        ///< uniaxial equivalent plastic strain
  };

  /** @brief calculate the first Piola stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State& state, double dt, const tensor<T, dim, dim>& du_dX) const
  {
    using std::sqrt;
    constexpr auto I = Identity<dim>();
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    // (i) elastic predictor
    auto el_strain = sym(du_dX) - state.plastic_strain;
    auto p = K * tr(el_strain);
    auto s = 2.0 * G * dev(el_strain);
    auto sigma_b = 2.0 / 3.0 * Hk * state.plastic_strain;
    auto eta = s - sigma_b;
    auto q = sqrt(1.5) * norm(eta);

    // (ii) admissibility
    const double eqps_old = state.accumulated_plastic_strain;
    auto residual = [eqps_old, G, dt, *this](auto delta_eqps, auto trial_q) {
      auto eqps_dot = dt > 0.0 ? delta_eqps / dt : 0.0 * delta_eqps;
      return trial_q - (3.0 * G + Hk) * delta_eqps - this->hardening(eqps_old + delta_eqps, eqps_dot);
    };
    if (residual(0.0, get_value(q)) > tol * hardening.sigma_y) {
      // (iii) return mapping

      // Note the tolerance for convergence is the same as the tolerance for entering the return map.
      // This ensures that if the constitutive update is called again with the updated internal
      // variables, the return map won't be repeated.
      ScalarSolverOptions opts{.xtol = 0, .rtol = tol * hardening.sigma_y, .max_iter = 25};
      double lower_bound = 0.0;
      double upper_bound = (get_value(q) - hardening(eqps_old, 0.0)) / (3.0 * G + Hk);
      auto [delta_eqps, status] = solve_scalar_equation(residual, 0.0, lower_bound, upper_bound, opts, q);

      auto Np = 1.5 * eta / q;

      s = s - 2.0 * G * delta_eqps * Np;
      state.accumulated_plastic_strain += get_value(delta_eqps);
      state.plastic_strain += get_value(delta_eqps) * get_value(Np);
    }

    return s + p * I;
  }
};

/// @brief Finite deformation version of J2 material with nonlinear isotropic hardening.
template <typename HardeningType>
struct J2 {
  static constexpr int dim = 3;         ///< spatial dimension
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

  double E;                 ///< Young's modulus
  double nu;                ///< Poisson's ratio
  HardeningType hardening;  ///< Flow stress hardening model
  double density;           ///< mass density

  /// @brief variables required to characterize the hysteresis response
  struct State {
    tensor<double, dim, dim> Fpinv = DenseIdentity<3>();  ///< inverse of plastic distortion tensor
    double accumulated_plastic_strain;                    ///< uniaxial equivalent plastic strain
  };

  /** @brief calculate the first Piola stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State& state, double dt, const tensor<T, dim, dim>& du_dX) const
  {
    using std::sqrt;
    constexpr auto I = Identity<dim>();
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    // (i) elastic predictor
    auto F = du_dX + I;
    auto Fe = dot(F, state.Fpinv);
    auto Ee = 0.5 * log_symm(dot(transpose(Fe), Fe));
    // From this point until the state variable update, the algorithm exactly coincides with the
    // small strain one.
    auto p = K * tr(Ee);
    auto s = 2.0 * G * dev(Ee);
    double q_val = sqrt(1.5) * norm(get_value(s));

    // (ii) admissibility
    const double eqps_old = state.accumulated_plastic_strain;
    auto residual = [eqps_old, G, dt, *this](auto delta_eqps, auto trial_mises) {
      auto eqps_dot = dt > 0.0 ? delta_eqps / dt : 0.0 * delta_eqps;
      return trial_mises - 3.0 * G * delta_eqps - this->hardening(eqps_old + delta_eqps, eqps_dot);
    };
    if (residual(0.0, q_val) > tol * hardening.sigma_y) {
      // (iii) return mapping

      // Note the tolerance for convergence is the same as the tolerance for entering the return map.
      // This ensures that if the constitutive update is called again with the updated internal
      // variables, the return map won't be repeated.
      ScalarSolverOptions opts{.xtol = 0, .rtol = tol * hardening.sigma_y, .max_iter = 25};
      double lower_bound = 0.0;
      double upper_bound = (q_val - hardening(eqps_old, 0.0)) / (3.0 * G);
      // safe to compute derivative of mises stress now that we're in yielding branch
      auto q = sqrt(1.5) * norm(s);
      auto [delta_eqps, status] = solve_scalar_equation(residual, 0.0, lower_bound, upper_bound, opts, q);

      if (!status.converged) {
        SLIC_WARNING("J2 solve failed to converge.");
      }

      auto Np = 1.5 * s / q;

      s = s - 2.0 * G * delta_eqps * Np;
      auto A = exp_symm(-delta_eqps * Np);

      auto Fpinv = dot(state.Fpinv, A);

      state.accumulated_plastic_strain += get_value(delta_eqps);
      state.Fpinv = get_value(Fpinv);
      // Mandel stress
      auto M = s + p * I;
      // convert to Piola
      Fe = dot(Fe, A);
      auto P = transpose(dot(dot(Fpinv, M), inv(Fe)));
      return P;
    } else {
      // I want to unify this branch with the yielding branch, but I'd need to declare Fpinv above,
      // and I don't know how to declare its type. BT 11/5/24
      // Mandel stress
      auto M = s + p * I;
      // convert to Piola
      auto P = transpose(dot(dot(state.Fpinv, M), inv(Fe)));
      return P;
    }
  }
};

/**
 * @brief Transform the Kirchhoff stress to the Piola stress
 *
 * @tparam T1 number-like type of the displacement gradient components
 * @tparam T1 number-like type of the Kirchhoff stress components
 * @tparam dim number of spatial dimensions
 *
 * @param displacement_gradient Displacement gradient
 * @param kirchhoff_stress Kirchhoff stress
 * @return Piola stress
 */
template <typename T1, typename T2, int dim>
auto KirchhoffToPiola(const tensor<T1, dim, dim>& kirchhoff_stress, const tensor<T2, dim, dim>& displacement_gradient)
{
  return transpose(dot(inv(displacement_gradient + Identity<dim>()), kirchhoff_stress));
}

/**
 * @brief Transform the Cauchy stress to the Piola stress
 *
 * @tparam T1 number-like type of the Cauchy stress components
 * @tparam T2 number-like type of the displacement gradient components
 * @tparam dim number of spatial dimensions
 *
 * @param displacement_gradient Displacement gradient
 * @param cauchy_stress Cauchy stress
 * @return Piola stress
 */
template <typename T1, typename T2, int dim>
auto CauchyToPiola(const tensor<T1, dim, dim>& cauchy_stress, const tensor<T2, dim, dim>& displacement_gradient)
{
  auto kirchhoff_stress = det(displacement_gradient + Identity<dim>()) * cauchy_stress;
  return KirchhoffToPiola(kirchhoff_stress, displacement_gradient);
}

/// Constant body force model
template <int dim>
struct ConstantBodyForce {
  /// The constant body force
  tensor<double, dim> force_;

  /**
   * @brief Evaluation function for the constant body force model
   *
   * @tparam T Position type
   * @tparam dim The dimension of the problem
   * @return The body force value
   */
  template <typename T>
  SMITH_HOST_DEVICE tensor<double, dim> operator()(const tensor<T, dim>& /* x */, const double /* t */) const
  {
    return force_;
  }
};

/// Constant traction boundary condition model
template <int dim>
struct ConstantTraction {
  /// The constant traction
  tensor<double, dim> traction_;

  /**
   * @brief Evaluation function for the constant traction model
   *
   * @tparam T Position type
   * @return The traction value
   */
  template <typename T>
  SMITH_HOST_DEVICE tensor<double, dim> operator()(const tensor<T, dim>& /* x */, const tensor<T, dim>& /* n */,
                                                   const double /* t */) const
  {
    return traction_;
  }
};

/**
 * @brief Neo-Hookean material model
 * This struct differs in style relative to the older materials as it needs to evaluate both stress
 * and density.  As a result, we want to clearly name these functions.
 * This is likely going to be a new design going forward, at the moment it works with the
 * SolidResidual class.
 *
 */
struct NeoHookeanWithFieldDensity {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a NeoHookean material model
   * @tparam T type of float or dual in tensor
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   * @return The first Piola stress
   */
  template <typename T, int dim, typename Density>
  SMITH_HOST_DEVICE auto pkStress(State& /* state */, const tensor<T, dim, dim>& du_dX, const Density&) const
  {
    using std::log1p;
    constexpr auto I = Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return dot(TK, inv(transpose(F)));
  }

  /// @brief interpolates density field
  template <typename Density>
  SMITH_HOST_DEVICE auto density(const Density& density) const
  {
    return get<VALUE>(density);
  }

  double K;  ///< bulk modulus
  double G;  ///< shear modulus
};

}  // namespace smith::solid_mechanics
