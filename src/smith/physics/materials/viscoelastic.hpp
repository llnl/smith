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

 #include "smith/numerics/functional/tensor.hpp"

#pragma once

namespace smith {

struct Viscoelastic {
  static constexpr int dim = 3;
  template <typename T> using InternalState = tensor<T, dim*dim>;  ///< Fv
  template <typename T> using Tensor = tensor<T, dim, dim>;

  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto equilibrium_stress(const Tensor<T1>& H, T2 theta) const
  {
    auto CmI = H + transpose(H) + transpose(H)*H;
    auto E = 0.5*logIp_symm(CmI);
    auto Ee = E - alpha_inf*(theta - theta_r)*Identity<dim>();
    auto M = 2.0*G_inf*dev(Ee) + K_inf*tr(Ee)*Identity<dim>();
    // pull back to Piola stress
    auto F = H + Identity<dim>();
    return M*inv(transpose(F));
  }

  template <typename T1>
  SMITH_HOST_DEVICE auto kinetic(T1 tau_bar) {
    return tau_bar / eta_0;
  }

  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto update(double dt, const InternalState<T1>& Q, const Tensor<T2>& du_dX, T3 theta) const
  {
    auto P_inf = equilibrium_stress(du_dX, theta);

    // Consider 
    auto Fv = make_tensor<dim, dim>([&Q](int i, int j) { return Q[dim*i + j]; });
    auto F = du_dX + Identity<dim>();
    auto Fe = F*inv(Fv);
    auto Ce = transpose(Fe)*Fe;
    auto Ee = 0.5*log_symm(Ce);
    auto devM = 2.0*G_0*dev(Ee);
    auto M = devM;
    auto tau_bar = std::sqrt(0.5)*norm(devM);
    auto denom = (tau_bar > 0)? (1.0/tau_bar) : 1.0;
    auto N = 0.5*devM/denom;
    
    auto dg = tau_bar/(eta_0/dt + G_0);
    M -= 2*G_0*dg*N;
    auto P_0 = M*inv(transpose(F));

    auto Fv_new = exp_symm(dg*N)*Fv;
    auto internal_state_new = make_tensor<dim*dim>(
      [&Fv_new](int ij) { 
        int i = ij / dim;
        int j = ij % dim;
        return Fv_new[i][j];
      });
    
    return make_tuple(P_inf + P_0, internal_state_new);
  }

  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto pkStress(double dt, const InternalState<T1>& Q, const Tensor<T2>& du_dX, T3 theta) const
  {
    auto [P, Q_new] = update(dt, Q, du_dX, theta);
    return P;
  }

  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto update_intenal_state(double dt, const InternalState<T1>& Q, const Tensor<T2>& du_dX, T3 theta) const
  {
    auto [P, Q_new] = update(dt, Q, du_dX, theta);
    return Q_new;
  }

  /// @brief interpolates density field
  SMITH_HOST_DEVICE auto density() const
  {
    return rho_r;
  }

  double K_inf;  ///< equiibrium bulk modulus
  double G_inf;  ///< equilibrium shear modulus
  double alpha_inf; ///< equilibrium thermal expansion coefficient

  double G_0; ///< shear modulus in branch 0
  double eta_0; ///< viscosity in branch 0

  double theta_r; /// < reference temperature for thermal expansion
  double rho_r; ///< density in the reference configuration
};

} // namespace smith