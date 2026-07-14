// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file viscoelastic.hpp
 *
 * @brief Finite-deformation viscoelastic material model.
 */

#pragma once

#include <cmath>

#include "smith/infrastructure/accelerator.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple_tensor_dual_functions.hpp"

namespace smith::solid_mechanics {

/**
 * @brief Single-branch finite-deformation viscoelastic model.
 */
struct Viscoelastic {
  static constexpr int dim = 3;

  template <typename T>
  using Tensor = tensor<T, dim, dim>;

  struct State {
    tensor<double, dim, dim> Fv{DenseIdentity<dim>()};
  };

  static State initialInternalState() { return {DenseIdentity<dim>()}; }

  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto equilibriumStress(const Tensor<T1>& H, T2 theta) const
  {
    const auto BmI = H + transpose(H) + H * transpose(H);
    const auto E = 0.5 * logIp_symm(BmI);
    const auto Em = E - alpha_inf * (theta - theta_sf) * Identity<dim>();
    const auto M = 2.0 * G_inf * dev(Em) + K_inf * tr(Em) * Identity<dim>();
    const auto F = H + Identity<dim>();
    return M * inv(transpose(F));
  }

  template <typename T>
  SMITH_HOST_DEVICE auto shiftFactor(T theta) const
  {
    using std::pow;
    const auto dT = theta - theta_r;
    return pow(10.0, -C1 * dT / (C2 + dT));
  }

  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto update(const State& internal_state, double dt, const Tensor<T1>& du_dX, T2 temperature) const
  {
    const auto theta = get<0>(temperature);
    const auto P_inf = equilibriumStress(du_dX, theta);

    const auto& Fv = internal_state.Fv;
    const auto F = du_dX + Identity<dim>();
    const auto Fe = F * inv(Fv);
    auto Ee = 0.5 * log_symm(transpose(Fe) * Fe) - 0.0 * theta * Identity<dim>();
    const auto devM = 2.0 * G_0 * dev(Ee);
    const double nrm = norm(get_value(devM));
    const auto tau_bar =
        nrm > 0.0 ? std::sqrt(0.5) * norm(devM) : std::sqrt(0.5) * norm(devM + 1.0e-8 * Identity<dim>());
    const auto N = 0.5 * devM / tau_bar;

    const auto a = shiftFactor(theta);
    const auto dg = tau_bar / (a * eta_0 / (dt + 1.0e-6) + G_0);
    const auto M = devM - (2.0 * G_0 * dg) * N;
    const auto Fv_new = exp_symm(dg * N) * Fv;
    const auto Fe_new = F * inv(Fv_new);
    const auto P_0 = transpose(inv(Fv_new) * M * inv(Fe_new));
    State internal_state_new{get_value(Fv_new)};
    return make_tuple(P_inf + P_0, internal_state_new);
  }

  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto pkStress(const State& Q, double dt, const Tensor<T1>& du_dX, T2 temperature) const
  {
    auto [P, Q_new] = update(Q, dt, du_dX, temperature);
    return P;
  }

  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto internalState(const State& Q, double dt, const Tensor<T1>& du_dX, T2 temperature) const
  {
    auto [P, Q_new] = update(Q, dt, du_dX, temperature);
    return Q_new;
  }

  SMITH_HOST_DEVICE auto density() const { return rho_r; }

  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto potential(const State& Q, double dt, const Tensor<T1>& du_dX, T2 temperature) const
  {
    const auto theta = get<0>(temperature);
    const auto BmI = du_dX + transpose(du_dX) + du_dX * transpose(du_dX);
    const auto E = 0.5 * logIp_symm(BmI);
    const auto Em = E - alpha_inf * (theta - theta_sf) * Identity<dim>();
    const auto devEm = dev(Em);
    const auto trEm = tr(Em);
    const auto psi_inf = G_inf * inner(devEm, devEm) + 0.5 * K_inf * trEm * trEm;

    const auto& Fv = Q.Fv;
    const auto F = du_dX + Identity<dim>();
    const auto Fe = F * inv(Fv);
    auto Ee = 0.5 * log_symm(transpose(Fe) * Fe) - 0.0 * theta * Identity<dim>();
    const auto devM = 2.0 * G_0 * dev(Ee);
    const auto tau_bar = std::sqrt(0.5) * norm(devM);
    const auto denom = (tau_bar > 0.0) ? tau_bar : (1.0 + tau_bar);
    const auto N = 0.5 * devM / denom;

    const auto a = shiftFactor(theta);
    const auto dg = tau_bar / (a * eta_0 / dt + G_0);
    Ee = Ee - dg * N;
    const auto devEe = dev(Ee);
    const auto psi_0 = G_0 * inner(devEe, devEe);
    const auto Pi = 0.5 * eta_0 / dt * dg * dg;

    return psi_inf + psi_0 + Pi;
  }

  double K_inf;
  double G_inf;
  double alpha_inf;
  double theta_sf;
  double G_0;
  double eta_0;
  double theta_r;
  double C1;
  double C2;
  double rho_r;
};

struct ViscoelasticOldInterface {
  static constexpr int dim = 3;
  using State = Viscoelastic::State;

  template <typename T>
  using Tensor = tensor<T, dim, dim>;

  ViscoelasticOldInterface(double density, double K_inf, double G_inf, double alpha_inf, double theta_sf, double G_0,
                           double eta_0, double theta_r, double C1, double C2)
      : material{K_inf, G_inf, alpha_inf, theta_sf, G_0, eta_0, theta_r, C1, C2, density}, density{density}
  {
  }

  template <typename T1, typename T2>
  SMITH_HOST_DEVICE auto operator()(State& Q, double dt, const Tensor<T1>& du_dX, T2 temperature) const
  {
    auto [P, Q_new] = material.update(Q, dt, du_dX, temperature);
    Q = Q_new;
    return P;
  }

  Viscoelastic material;
  double density;
};

}  // namespace smith::solid_mechanics
