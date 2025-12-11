// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/solid_mechanics.hpp"

#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"

namespace smith {

namespace solid_mechanics {

namespace detail {

void adjoint_integrate(double dt_n, double dt_np1, mfem::HypreParMatrix* m_mat, mfem::HypreParMatrix* k_mat,
                       mfem::HypreParVector& disp_adjoint_load_vector, mfem::HypreParVector& velo_adjoint_load_vector,
                       mfem::HypreParVector& accel_adjoint_load_vector, mfem::HypreParVector& adjoint_displacement_,
                       mfem::HypreParVector& implicit_sensitivity_displacement_start_of_step_,
                       mfem::HypreParVector& implicit_sensitivity_velocity_start_of_step_,
                       mfem::HypreParVector& adjoint_essential_, BoundaryConditionManager& bcs_,
                       mfem::Solver& lin_solver)
{
  // there are hard-coded here for now
  static constexpr double beta = 0.25;
  static constexpr double gamma = 0.5;
  // reminder, gathering info from the various layers of time integration
  // c0 = fac3 * dt * dt
  // c1 = fac4 * dt

  double fac1 = 0.5 - beta;
  double fac2 = 1.0 - gamma;
  double fac3 = beta;
  double fac4 = gamma;

  // J = M + c0 * K
  auto J_ = std::unique_ptr<mfem::HypreParMatrix>(mfem::Add(1.0, *m_mat, fac3 * dt_n * dt_n, *k_mat));
  auto J_T = std::unique_ptr<mfem::HypreParMatrix>(J_->Transpose());

  // recall that temperature_adjoint_load_vector and d_temperature_dt_adjoint_load_vector were already multiplied by
  // -1 above

  mfem::HypreParVector adjoint_rhs(accel_adjoint_load_vector);
  adjoint_rhs.Add(fac4 * dt_n, velo_adjoint_load_vector);
  adjoint_rhs.Add(fac3 * dt_n * dt_n, disp_adjoint_load_vector);

  adjoint_rhs.Add(-dt_np1 * fac2 - dt_n * fac4, implicit_sensitivity_velocity_start_of_step_);
  adjoint_rhs.Add(-dt_np1 * dt_np1 * fac1 - dt_n * dt_n * fac3 - dt_n * dt_np1 * fac4,
                  implicit_sensitivity_displacement_start_of_step_);

  for (const auto& bc : bcs_.essentials()) {
    bc.apply(*J_T, adjoint_rhs, adjoint_essential_);
  }

  lin_solver.SetOperator(*J_T);
  // really adjoint acceleration
  lin_solver.Mult(adjoint_rhs, adjoint_displacement_);

  // the current residual has no given dependence on velo, otherwise, there would be another term
  // implicit_sensitivity_velocity_start_of_step_.Add(-1.0, velo_adjoint_load_vector);
  implicit_sensitivity_velocity_start_of_step_.Add(dt_np1, implicit_sensitivity_displacement_start_of_step_);

  // the 1.0, 1.0 means += the implicit sensitivity
  k_mat->MultTranspose(adjoint_displacement_, implicit_sensitivity_displacement_start_of_step_, 1.0, 1.0);

  implicit_sensitivity_displacement_start_of_step_.Add(-1.0, disp_adjoint_load_vector);
}

}  // namespace detail
}  // namespace solid_mechanics
}  // namespace smith
