// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/odes.hpp"

#include <utility>
#include <vector>

namespace smith::mfem_ext {

SecondOrderODE::SecondOrderODE(int n, State&& state, const EquationSolver& solver, const BoundaryConditionManager& bcs)
    : mfem::SecondOrderTimeDependentOperator(n, 0.0), state_(std::move(state)), solver_(solver), bcs_(bcs), zero_(n)
{
  zero_ = 0.0;
  U_minus_.SetSize(n);
  U_.SetSize(n);
  U_plus_.SetSize(n);
  dU_dt_.SetSize(n);
  d2U_dt2_.SetSize(n);
}

void SecondOrderODE::SetTimestepper(const smith::TimestepMethod timestepper)
{
  timestepper_ = timestepper;

  switch (timestepper) {
    case smith::TimestepMethod::Newmark:
      second_order_ode_solver_ = std::make_unique<mfem::NewmarkSolver>();
      break;
    case smith::TimestepMethod::HHTAlpha:
      second_order_ode_solver_ = std::make_unique<mfem::HHTAlphaSolver>();
      break;
    case smith::TimestepMethod::WBZAlpha:
      second_order_ode_solver_ = std::make_unique<mfem::WBZAlphaSolver>();
      break;
    case smith::TimestepMethod::AverageAcceleration:
      // WARNING: apparently mfem's implementation of AverageAccelerationSolver
      // is NOT equivalent to Newmark (beta = 0.25, gamma = 0.5), so the
      // stability analysis of using DirichletEnforcementMethod::DirectControl
      // with Newmark methods does not apply.
      //
      // TODO: do a more thorough stability analysis for mfem::GeneralizedAlpha2Solver
      // to characterize which parameter combinations work with time-varying
      // dirichlet constraints
      SLIC_WARNING_ROOT(
          "Cannot guarantee stability for AverageAcceleration with time-dependent Dirichlet Boundary Conditions");
      second_order_ode_solver_ = std::make_unique<mfem::AverageAccelerationSolver>();
      break;
    case smith::TimestepMethod::LinearAcceleration:
      second_order_ode_solver_ = std::make_unique<mfem::LinearAccelerationSolver>();
      break;
    case smith::TimestepMethod::CentralDifference:
      second_order_ode_solver_ = std::make_unique<mfem::CentralDifferenceSolver>();
      break;
    case smith::TimestepMethod::FoxGoodwin:
      second_order_ode_solver_ = std::make_unique<mfem::FoxGoodwinSolver>();
      break;
    case smith::TimestepMethod::BackwardEuler:
      first_order_system_ode_solver_ = std::make_unique<mfem::BackwardEulerSolver>();
      break;
    default:
      SLIC_ERROR_ROOT("Timestep method was not a supported second-order ODE method");
  }

  if (second_order_ode_solver_) {
    second_order_ode_solver_->Init(*this);
  } else if (first_order_system_ode_solver_) {
    // we need to adjust the width of this operator
    width *= 2;
    first_order_system_ode_solver_->Init(*this);
  } else {
    // there is no ode_solver
    SLIC_ERROR("Neither second_order_ode_solver_ nor first_order_system_ode_solver_ specified");
  }
}

void SecondOrderODE::Step(mfem::Vector& x, mfem::Vector& dxdt, double& time, double& dt)
{
  if (second_order_ode_solver_) {
    // if we used a 2nd order method
    second_order_ode_solver_->Step(x, dxdt, time, dt);

    if (enforcement_method_ == DirichletEnforcementMethod::FullControl) {
      U_minus_ = 0.0;
      U_ = 0.0;
      U_plus_ = 0.0;
      for (const auto& bc : bcs_.essentials()) {
        bc.setDofs(U_minus_, t - epsilon);
        bc.setDofs(U_, t);
        bc.setDofs(U_plus_, t + epsilon);
      }

      auto constrained_dofs = bcs_.allEssentialTrueDofs();
      for (int i = 0; i < constrained_dofs.Size(); i++) {
        x[i] = U_[i];
        dxdt[i] = (U_plus_[i] - U_minus_[i]) / (2.0 * epsilon);
      }
    }

  } else if (first_order_system_ode_solver_) {
    // Would be better if displacement and velocity were from a block vector?
    mfem::Array<int> boffsets(3);
    boffsets[0] = 0;
    boffsets[1] = x.Size();
    boffsets[2] = x.Size() + dxdt.Size();
    mfem::BlockVector bx(boffsets);
    bx.GetBlock(0) = x;
    bx.GetBlock(1) = dxdt;

    first_order_system_ode_solver_->Step(bx, time, dt);

    // Copy back
    x = bx.GetBlock(0);
    dxdt = bx.GetBlock(1);
  } else {
    SLIC_ERROR_ROOT("Neither second_order_ode_solver_ nor first_order_system_ode_solver_ specified");
  }
}

void SecondOrderODE::ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt)
{
  /* A second order o.d.e can be recast as a first order system
    u_next = u_prev + dt * v_next
    v_next = v_prev + dt * a_next

    This means:
    u_next = u_prev + dt * (v_prev + dt * a_next);
    u_next = (u_prev + dt * v_prev) + dt*dt*a_next
  */

  // Split u in half and du_dt in half
  mfem::Array<int> boffsets(3);
  boffsets[0] = 0;
  boffsets[1] = u.Size() / 2;
  boffsets[2] = u.Size();

  const mfem::BlockVector bu(u.GetData(), boffsets);

  mfem::BlockVector bdu_dt(du_dt.GetData(), boffsets);

  mfem::Vector u_next(bu.GetBlock(0));
  u_next.Add(dt, bu.GetBlock(1));

  Solve(t, dt * dt, dt, u_next,
        bu.GetBlock(1),       // v_next
        bdu_dt.GetBlock(1));  // a_next

  bdu_dt.GetBlock(0) = bu.GetBlock(1);
  bdu_dt.GetBlock(0).Add(dt, bdu_dt.GetBlock(1));
}

void SecondOrderODE::Solve(const double time, const double c0, const double c1, const mfem::Vector& u,
                           const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) const
{
  // assign these values to variables with greater scope,
  // so that the residual operator can see them
  state_.time = time;
  state_.c0 = c0;
  state_.c1 = c1;
  state_.u = u;
  state_.du_dt = du_dt;

  // evaluate the constraint functions at a 3-point
  // stencil of times centered on the time of interest
  // in order to compute finite-difference approximations
  // to the time derivatives that appear in the residual
  U_minus_ = 0.0;
  U_ = 0.0;
  U_plus_ = 0.0;
  for (const auto& bc : bcs_.essentials()) {
    bc.setDofs(U_minus_, time - epsilon);
    bc.setDofs(U_, time);
    bc.setDofs(U_plus_, time + epsilon);
  }

  bool implicit = (c0 != 0.0 || c1 != 0.0);
  if (implicit) {
    if (enforcement_method_ == DirichletEnforcementMethod::DirectControl) {
      // TODO: MFEM PR#3064 explicitly deleted the const vector operator-.  This
      // is in active discusion and may be un-deleted. Original line commented.
      // d2U_dt2_ = (U_ - u) / c0;
      subtract(1.0 / c0, U_, u, d2U_dt2_);
      dU_dt_ = du_dt;
      U_ = u;
    }

    if (enforcement_method_ == DirichletEnforcementMethod::RateControl) {
      // d2U_dt2_ = ((U_plus_ - U_minus_) / (2.0 * epsilon) - du_dt) / c1;
      subtract(U_plus_, U_minus_, d2U_dt2_);
      d2U_dt2_ /= 2.0 * epsilon;
      d2U_dt2_ -= du_dt;
      d2U_dt2_ /= c1;

      dU_dt_ = du_dt;
      U_ = u;
    }

    if (enforcement_method_ == DirichletEnforcementMethod::FullControl) {
      // d2U_dt2_ = (U_minus_ - 2.0 * U_ + U_plus_) / (epsilon * epsilon);
      add(1.0, U_minus_, -2.0, U_, d2U_dt2_);
      d2U_dt2_ += U_plus_;
      d2U_dt2_ /= epsilon * epsilon;

      // d2U_dt2_ = ((U_plus_ - U_minus_) / (2.0 * epsilon) - du_dt) / c1;
      subtract(U_plus_, U_minus_, d2U_dt2_);
      d2U_dt2_ /= 2.0 * epsilon;
      d2U_dt2_ -= du_dt;
      d2U_dt2_ /= c1;

      // dU_dt_   = (U_plus_ - U_minus_) / (2.0 * epsilon) - c1 * d2U_dt2_;
      subtract(U_plus_, U_minus_, dU_dt_);
      dU_dt_ /= 2.0 * epsilon;
      dU_dt_.Add(-1.0 * c1, d2U_dt2_);

      // U_ = U_ - c0 * d2U_dt2_;
      U_.Add(-1.0 * c0, d2U_dt2_);
    }
  } else {
    // d2U_dt2_ = (U_minus_ - 2.0 * U_ + U_plus_) / (epsilon * epsilon);
    add(1.0, U_minus_, -2.0, U_, d2U_dt2_);
    d2U_dt2_ += U_plus_;
    d2U_dt2_ /= epsilon * epsilon;

    // dU_dt_   = (U_plus_ - U_minus_) / (2.0 * epsilon);
    subtract(U_plus_, U_minus_, dU_dt_);
    dU_dt_ /= 2.0 * epsilon;
  }

  auto constrained_dofs = bcs_.allEssentialTrueDofs();
  state_.u.SetSubVector(constrained_dofs, 0.0);
  U_.SetSubVectorComplement(constrained_dofs, 0.0);
  state_.u += U_;

  state_.du_dt.SetSubVector(constrained_dofs, 0.0);
  dU_dt_.SetSubVectorComplement(constrained_dofs, 0.0);
  state_.du_dt += dU_dt_;

  // use 0 as our starting guess for unconstrained dofs
  d2u_dt2 = state_.d2u_dt2;
  d2u_dt2.SetSubVector(constrained_dofs, 0.0);
  d2u_dt2 += d2U_dt2_;
  d2u_dt2.SetSubVectorComplement(constrained_dofs, 0.0);

  solver_.solve(d2u_dt2);
  SLIC_WARNING_ROOT_IF(!solver_.nonlinearSolver().GetConverged(), "Newton Solver did not converge.");

  state_.d2u_dt2 = d2u_dt2;
}

FirstOrderODE::FirstOrderODE(int n, FirstOrderODE::State&& state, const EquationSolver& solver,
                             const BoundaryConditionManager& bcs)
    : mfem::TimeDependentOperator(n, 0.0), state_(std::move(state)), solver_(solver), bcs_(bcs), zero_(n)
{
  zero_ = 0.0;
  U_minus_.SetSize(n);
  U_.SetSize(n);
  U_plus_.SetSize(n);
  dU_dt_.SetSize(n);
}

void FirstOrderODE::SetTimestepper(const smith::TimestepMethod timestepper)
{
  timestepper_ = timestepper;

  switch (timestepper) {
    case smith::TimestepMethod::BackwardEuler:
      ode_solver_ = std::make_unique<mfem::BackwardEulerSolver>();
      break;
    case smith::TimestepMethod::SDIRK33:
      ode_solver_ = std::make_unique<mfem::SDIRK33Solver>();
      break;
    case smith::TimestepMethod::ForwardEuler:
      ode_solver_ = std::make_unique<mfem::ForwardEulerSolver>();
      break;
    case smith::TimestepMethod::RK2:
      ode_solver_ = std::make_unique<mfem::RK2Solver>(0.5);
      break;
    case smith::TimestepMethod::RK3SSP:
      ode_solver_ = std::make_unique<mfem::RK3SSPSolver>();
      break;
    case smith::TimestepMethod::RK4:
      ode_solver_ = std::make_unique<mfem::RK4Solver>();
      break;
    case smith::TimestepMethod::GeneralizedAlpha:
      ode_solver_ = std::make_unique<mfem::GeneralizedAlphaSolver>(0.5);
      break;
    case smith::TimestepMethod::ImplicitMidpoint:
      ode_solver_ = std::make_unique<mfem::ImplicitMidpointSolver>();
      break;
    case smith::TimestepMethod::SDIRK23:
      ode_solver_ = std::make_unique<mfem::SDIRK23Solver>();
      break;
    case smith::TimestepMethod::SDIRK34:
      ode_solver_ = std::make_unique<mfem::SDIRK34Solver>();
      break;
    default:
      SLIC_ERROR_ROOT("Timestep method was not a supported first-order ODE method");
  }
  ode_solver_->Init(*this);
}

void FirstOrderODE::Solve(const double time, const double dt, const mfem::Vector& u, mfem::Vector& du_dt) const
{
  // assign these values to variables with greater scope,
  // so that the residual operator can see them
  state_.time = time;
  state_.dt = dt;
  state_.u = u;

  // evaluate the constraint functions at a 3-point
  // stencil of times centered on the time of interest
  // in order to compute finite-difference approximations
  // to the time derivatives that appear in the residual
  U_minus_ = 0.0;
  U_ = 0.0;
  U_plus_ = 0.0;
  for (const auto& bc : bcs_.essentials()) {
    bc.setDofs(U_minus_, time - epsilon);
    bc.setDofs(U_, time);
    bc.setDofs(U_plus_, time + epsilon);
  }

  bool implicit = (dt != 0.0);
  if (implicit) {
    if (enforcement_method_ == DirichletEnforcementMethod::DirectControl) {
      // TODO: MFEM PR#3064 explicitly deleted the const vector operator-.  This
      // is in active discusion and may be un-deleted. Original line commented.
      // dU_dt_ = (U_ - u) / dt;
      subtract(1.0 / dt, U_, u, dU_dt_);
      U_ = u;
    }

    if (enforcement_method_ == DirichletEnforcementMethod::RateControl) {
      // dU_dt_ = (U_plus_ - U_minus_) / (2.0 * epsilon);
      subtract(1.0 / (2.0 * epsilon), U_plus_, U_minus_, dU_dt_);
      U_ = u;
    }

    if (enforcement_method_ == DirichletEnforcementMethod::FullControl) {
      // dU_dt_ = (U_plus_ - U_minus_) / (2.0 * epsilon);
      subtract(1.0 / (2.0 * epsilon), U_plus_, U_minus_, dU_dt_);
      // U_     = U_ - dt * dU_dt_;
      U_.Add(-1.0 * dt, dU_dt_);
    }
  } else {
    // dU_dt_ = (U_plus_ - U_minus_) / (2.0 * epsilon);
    subtract(1.0 / (2.0 * epsilon), U_plus_, U_minus_, dU_dt_);
  }

  auto constrained_dofs = bcs_.allEssentialTrueDofs();
  state_.u.SetSubVector(constrained_dofs, 0.0);
  U_.SetSubVectorComplement(constrained_dofs, 0.0);
  state_.u += U_;

  du_dt = state_.du_dt;
  du_dt.SetSubVector(constrained_dofs, 0.0);
  dU_dt_.SetSubVectorComplement(constrained_dofs, 0.0);
  du_dt += dU_dt_;

  solver_.solve(du_dt);
  SLIC_WARNING_ROOT_IF(!solver_.nonlinearSolver().GetConverged(), "Newton Solver did not converge.");

  state_.du_dt = du_dt;
  state_.previous_dt = dt;
}

}  // namespace smith::mfem_ext
