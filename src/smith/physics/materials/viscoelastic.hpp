// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file viscoelastic.hpp
 *
 * @brief A finite deformation viscoelastic material model
 */

#include "smith/infrastructure/accelerator.hpp"
#include "smith/numerics/functional/tensor.hpp"


#pragma once

namespace smith {

/**
 * @brief Finite deformation viscoelastic model
 */
struct Viscoelastic {
  static constexpr int dim = 3; ///< This model is implemented in 3D only. 
  template <typename T> using InternalState = tensor<T, dim*dim>;  ///< Internal state variable: inelastic distortion tensor (flattened)
  template <typename T> using Tensor = tensor<T, dim, dim>;

  /** 
   * @brief Stress due to the equilibrium branch
   */
  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto equilibrium_stress(const Tensor<T1>& H, T2 theta) const
  {
    // The spatial version of Hencky elasticity is used as it avoids the need
    // to compute the rotation tensor to pull back to the Piola stress
    auto BmI = H + transpose(H) + H*transpose(H);
    auto E = 0.5*logIp_symm(BmI);
    auto Em = E - alpha_inf*(theta - theta_sf)*Identity<dim>();
    auto M = 2.0*G_inf*dev(Em) + K_inf*tr(Em)*Identity<dim>();
    // pull back to Piola stress
    // Since the response is isotropic, the conjugate stress M coincides with
    // the Kirchhoff stress.
    auto F = H + Identity<dim>();
    return M*inv(transpose(F));
  }


  /**
   * @brief WLF shift factor for time-temperature superposition
   */
  template <typename T>
  SMITH_HOST_DEVICE auto shift_factor(T theta) const {
    auto dT = theta - theta_r;
    // The WLF equation makes sense only or dT > -C2, otherwise results are not physical
    SLIC_WARNING_IF(dT < -C2, "Temperature difference from reference is out of range for WLF shift");
    using std::pow;
    return pow(10.0, -C1*dT/(C2 + dT));
  }

  /** 
   * @brief Compute updated Piola stress and viscous deformation tensor
   */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto update(const InternalState<T1>& Q, double dt, const Tensor<T2>& du_dX, T3 theta) const
  {
    auto P_inf = equilibrium_stress(du_dX, theta);

    // Compute the stress in the spring-dashpot branch.
    // Possible generalizations:
    // - add more branches, giving more distinct relaxation time scales
    // - add spring-dashpot branch(es) for the volumetric response, and a glassy thermal expansion term
    // - make the viscosity-strain rate relationship nonlinear, such as a power law

    // Reshape the flattened inelastic distortion back into a tensor
    auto Fv = make_tensor<dim, dim>([&Q](int i, int j) { return Q[dim*i + j]; });
    auto F = du_dX + Identity<dim>();
    // Trial elastic values
    auto Fe = F*inv(Fv);
    auto Ce = transpose(Fe)*Fe;
    auto Ee = 0.5*log_symm(Ce);
    auto devM = 2.0*G_0*dev(Ee);
    auto M = devM; // + volumetric part, if a viscous volumetric response is added
    auto tau_bar = std::sqrt(0.5)*norm(devM);
    // Guard against division by zero.
    // Value of the denominator in the tau_bar = 0 case is unimportant,
    // since nothing evolves in that case.
    auto denom = (tau_bar > 0)? (tau_bar) : (1.0 + tau_bar);
    auto N = 0.5*devM/denom;
    
    auto a = shift_factor(theta);
    // Change in equivalent shear strain across dashpot.
    // If the viscosity relation is made nonlinear, this explicit relation will
    // be replaced with a nonlinear solve.
    auto dg = tau_bar/(a*eta_0/dt + G_0);
    // update elastic trial stress
    M -= 2*G_0*dg*N;
    // update inelastic distortion tensor
    auto Fv_new = exp_symm(dg*N)*Fv;
    // Change stress measure to Piola
    Fe = F*inv(Fv_new);
    auto P_0 = transpose(inv(Fv_new)*M*inv(Fe));

    // Flatten the inelastic distortion tensor for packing into the global array
    auto internal_state_new = make_tensor<dim*dim>(
      [&Fv_new](int ij) { 
        int i = ij / dim;
        int j = ij % dim;
        return Fv_new[i][j];
      });
    
    return make_tuple(P_inf + P_0, internal_state_new);
  }

  /**
   * @brief Return updated Piola stress
   */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto pkStress(const InternalState<T1>& Q, double dt, const Tensor<T2>& du_dX, T3 theta) const
  {
    auto [P, Q_new] = update(Q, dt, du_dX, theta);
    return P;
  }

  /**
   * @brief Return updated internal state variables
   */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto intenalState(const InternalState<T1>& Q, double dt, const Tensor<T2>& du_dX, T3 theta) const
  {
    auto [P, Q_new] = update(Q, dt, du_dX, theta);
    return Q_new;
  }

  /**
   * @brief interpolates density field
   */
  SMITH_HOST_DEVICE auto density() const
  {
    return rho_r;
  }

  /**
   * @brief Discrete potential (pseudo-enregy) which generates the stress relation
   *
   * This model has the special property that the stress derives from a scalar
   * potential. That is, there is a scalar energy density-like quantity W
   * such that P = ∂W / ∂F. This method computes the scalar
   * algorithmic potential W.
   *
   * The potential formulation allows us to use robust minimization-based
   * solvers (such as the trust region solver in Smith).
   *
   * For background on this subject, see:
   * Ortiz, M. and Stainier, L., 1999. The variational formulation of 
   * viscoplastic constitutive updates. Computer methods in applied mechanics
   * and engineering, 171(3-4), pp.419-444.
   */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto potential(const InternalState<T1>& Q, double dt, const Tensor<T2>& du_dX, T3 theta) const
  {
    auto BmI = du_dX + transpose(du_dX) + du_dX*transpose(du_dX);
    auto E = 0.5*logIp_symm(BmI);
    auto Em = E - alpha_inf*(theta - theta_sf)*Identity<dim>();
    auto devEm = dev(Em);
    auto trEm = tr(Em);
    auto psi_inf = G_inf*inner(devEm, devEm) + 0.5*K_inf*trEm*trEm;

    auto Fv = make_tensor<dim, dim>([&Q](int i, int j) { return Q[dim*i + j]; });
    auto F = du_dX + Identity<dim>();
    // Trial elastic values
    auto Fe = F*inv(Fv);
    auto Ce = transpose(Fe)*Fe;
    auto Ee = 0.5*log_symm(Ce);
    auto devM = 2.0*G_0*dev(Ee);
    auto tau_bar = std::sqrt(0.5)*norm(devM);
    auto denom = (tau_bar > 0)? (tau_bar) : (1.0 + tau_bar);
    auto N = 0.5*devM/denom;
    
    auto a = shift_factor(theta);
    auto dg = tau_bar/(a*eta_0/dt + G_0);
    Ee = Ee - dg*N;
    auto devEe = dev(Ee);
    auto psi_0 = G_0*inner(devEe, devEe);

    // discrete dual kinetic potential
    auto Pi = 0.5*eta_0/dt*dg*dg;

    return psi_inf + psi_0 + Pi;
  }

  double K_inf;  ///< equiibrium bulk modulus
  double G_inf;  ///< equilibrium shear modulus
  double alpha_inf; ///< equilibrium thermal expansion coefficient
  double theta_sf; /// < reference temperature for thermal expansion (that is, stress-free temperature)

  double G_0; ///< shear modulus in branch 0
  double eta_0; ///< viscosity in branch 0

  double theta_r; ///< reference temperature for temperature-dependent behaviors
  double C1; ///< first WLF factor (dimensionless)
  double C2; ///< second SLF factor (temperature units)

  double rho_r; ///< density in the reference configuration
};

} // namespace smith
