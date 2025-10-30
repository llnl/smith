// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file odes.hpp
 *
 * @brief Wrapper classes for using MFEM's ODE solver objects
 */

#pragma once

#include <functional>
#include <memory>

#include "mfem.hpp"

#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"

namespace smith::mfem_ext {

/**
 * @brief SecondOrderODE is a class wrapping mfem::SecondOrderTimeDependentOperator
 *   so that the user can use std::function to define the implementations of
 *   mfem::SecondOrderTimeDependentOperator::Mult and
 *   mfem::SecondOrderTimeDependentOperator::ImplicitSolve
 *
 * The main benefit of this approach is that lambda capture lists allow
 * for a flexible inline representation of the overloaded functions,
 * without having to manually define a separate functor class.
 */
class SecondOrderODE : public mfem::SecondOrderTimeDependentOperator {
 public:
  /**
   * @brief a small number used to compute finite difference approximations
   * to time derivatives of boundary conditions.
   *
   * Note: this is intended to be temporary
   * Ideally, epsilon should be "small" relative to the characteristic
   * time of the ODE, but we can't ensure that at present (we don't have
   * a critical timestep estimate)
   */
  static constexpr double epsilon = 0.0001;

  /**
   * @brief A set of references to physics-module-owned variables
   * used by the residual operator
   */
  struct State {
    /**
     * @brief Time value at which the ODE solver wants to compute a residual
     */
    double& time;

    /**
     * @brief coefficient used to calculate updated displacement: u_{n + 1} := u + c0 * d2u_dt2
     */
    double& c0;

    /**
     * @brief coefficient used to calculate updated velocity: du_dt_{n+1} := du_dt + c1 * d2u_dt2
     */
    double& c1;

    /**
     * @brief Predicted true DOFs
     */
    mfem::Vector& u;

    /**
     * @brief  Predicted du_dt
     */
    mfem::Vector& du_dt;

    /**
     * @brief Previous value of d^2u_dt^2
     */
    mfem::Vector& d2u_dt2;
  };

  /**
   * @brief Constructor defining the size and specific system of ordinary differential equations to be solved
   *
   * @param[in] n The number of components in each vector of the ODE
   * @param[in] state The collection of references to input/output variables from the physics module
   * @param[in] solver The solver that operates on the residual
   * @param[in] bcs The set of Dirichlet conditions to enforce
   *
   * Implements mfem::SecondOrderTimeDependentOperator::Mult and mfem::SecondOrderTimeDependentOperator::ImplicitSolve
   *      (described in more detail here:
   * https://mfem.github.io/doxygen/html/classmfem_1_1SecondOrderTimeDependentOperator.html)
   *
   * where
   *
   * mfem::SecondOrderTimeDependentOperator::Mult corresponds to the case where fac0, fac1 are both zero
   * mfem::SecondOrderTimeDependentOperator::ImplicitSolve corresponds to the case where either of fac0, fac1 are
   * nonzero
   *
   */
  SecondOrderODE(int n, State&& state, const EquationSolver& solver, const BoundaryConditionManager& bcs);

  /**
   * @brief Solves the equation d2u_dt2 = f(u, du_dt, t)
   *
   * @param[in] u The true DOFs
   * @param[in] du_dt The first time derivative of u
   * @param[out] d2u_dt2 The second time derivative of u
   */
  void Mult(const mfem::Vector& u, const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) const override
  {
    Solve(t, 0.0, 0.0, u, du_dt, d2u_dt2);
  }

  /**
   * @brief Solves the equation d2u_dt2 = f(u + c0 * d2u_dt2, du_dt + c1 * d2u_dt2, t)
   *
   * @param[in] c0 coefficient on d2u_dt2 appearing the first argument of f
   * @param[in] c1 coefficient on d2u_dt2 appearing the second argument of f
   * @param[in] u The true DOFs
   * @param[in] du_dt The first time derivative of u
   * @param[out] d2u_dt2 The second time derivative of u
   */
  void ImplicitSolve(const double c0, const double c1, const mfem::Vector& u, const mfem::Vector& du_dt,
                     mfem::Vector& d2u_dt2) override
  {
    Solve(t, c0, c1, u, du_dt, d2u_dt2);
  }

  /**
     The FirstOrder recast that can be used by a first order ode solver
   */
  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) override;

  /**
   * @brief Configures the Dirichlet enforcement method to use
   * @param[in] method The selected method
   */
  void SetEnforcementMethod(const DirichletEnforcementMethod method) { enforcement_method_ = method; }

  /**
   * @brief Set the time integration method
   *
   * @param[in] timestepper The timestepping method for the solver
   */
  void SetTimestepper(const smith::TimestepMethod timestepper);

  /**
   * @brief Performs a time step
   *
   * @param[inout] x The predicted solution
   * @param[inout] dxdt The predicted rate
   * @param[inout] time The current time
   * @param[inout] dt The desired time step
   *
   * @see mfem::SecondOrderODESolver::Step
   */
  void Step(mfem::Vector& x, mfem::Vector& dxdt, double& time, double& dt);

  /**
   * @brief Get a reference to the current state
   */
  const State& GetState() { return state_; }

  /**
   * @brief Query the timestep method for the ode solver
   *
   * @return The timestep method used by the underlying ode solver
   */
  TimestepMethod GetTimestepper() { return timestepper_; }

 private:
  /**
   * @brief Internal implementation used for mfem::SOTDO::Mult and mfem::SOTDO::ImplicitSolve
   *        Solves the equation d2u_dt2 = f(u + c0 * d2u_dt2, du_dt + c1 * d2u_dt2, t)
   *
   * @param[in] time The current time
   * @param[in] c0 coefficient on d2u_dt2 appearing the first argument of f
   * @param[in] c1 coefficient on d2u_dt2 appearing the second argument of f
   * @param[in] u The true DOFs
   * @param[in] du_dt The first time derivative of u
   * @param[out] d2u_dt2 The second time derivative of u
   */
  void Solve(const double time, const double c0, const double c1, const mfem::Vector& u, const mfem::Vector& du_dt,
             mfem::Vector& d2u_dt2) const;

  /**
   * @brief Set of references to external variables used by residual operator
   */
  State state_;
  /**
   * @brief The method of enforcing time-varying dirichlet boundary conditions
   */
  DirichletEnforcementMethod enforcement_method_ = smith::DirichletEnforcementMethod::RateControl;
  /**
   * @brief Reference to the equationsolver used to solve for d2u_dt2
   */
  const EquationSolver& solver_;
  /**
   * @brief MFEM solver object for second-order ODEs
   */
  std::unique_ptr<mfem::SecondOrderODESolver> second_order_ode_solver_;

  /**
   * @brief MFEM solver object for second-order ODEs recast as first order
   */
  std::unique_ptr<mfem::ODESolver> first_order_system_ode_solver_;

  /**
   * @brief Reference to boundary conditions used to constrain the solution
   */
  const BoundaryConditionManager& bcs_;
  mfem::Vector zero_;

  /**
   * @brief Working vectors for ODE outputs prior to constraint enforcement
   */
  mutable mfem::Vector U_minus_;
  mutable mfem::Vector U_;
  mutable mfem::Vector U_plus_;
  mutable mfem::Vector dU_dt_;
  mutable mfem::Vector d2U_dt2_;

  smith::TimestepMethod timestepper_;
};

/**
 * @brief FirstOrderODE is a class wrapping mfem::TimeDependentOperator
 *   so that the user can use std::function to define the implementations of
 *   mfem::TimeDependentOperator::Mult and
 *   mfem::TimeDependentOperator::ImplicitSolve
 *
 * The main benefit of this approach is that lambda capture lists allow
 * for a flexible inline representation of the overloaded functions,
 * without having to manually define a separate functor class.
 */
class FirstOrderODE : public mfem::TimeDependentOperator {
 public:
  /**
   * @brief a small number used to compute finite difference approximations
   * to time derivatives of boundary conditions.
   *
   * Note: this is intended to be temporary
   * Ideally, epsilon should be "small" relative to the characteristic
   * time of the ODE, but we can't ensure that at present (we don't have
   * a critical timestep estimate)
   */
  static constexpr double epsilon = 0.000001;

  /**
   * @brief A set of references to physics-module-owned variables
   * used by the residual operator
   */
  struct State {
    /**
     * @brief Time value at which the ODE solver wants to compute a residual
     */
    double& time;

    /**
     * @brief Predicted true DOFs
     */
    mfem::Vector& u;

    /**
     * @brief Current time step
     */
    double& dt;

    /**
     * @brief Previous value of du_dt
     */
    mfem::Vector& du_dt;

    /**
     * @brief Previous value of dt
     */
    double& previous_dt;
  };

  /**
   * @brief Constructor defining the size and specific system of ordinary differential equations to be solved
   *
   *
   * @param[in] n The number of components in each vector of the ODE
   * @param[in] state The collection of references to input/output variables from the physics module
   * @param[in] solver The solver that operates on the residual
   * @param[in] bcs The set of Dirichlet conditions to enforce
   *
   * Implements mfem::TimeDependentOperator::Mult and mfem::TimeDependentOperator::ImplicitSolve
   *      (described in more detail here: https://mfem.github.io/doxygen/html/classmfem_1_1TimeDependentOperator.html)
   *
   * where
   *
   * mfem::TimeDependentOperator::Mult corresponds to the case where dt is zero
   * mfem::TimeDependentOperator::ImplicitSolve corresponds to the case where dt is nonzero
   *
   */
  FirstOrderODE(int n, FirstOrderODE::State&& state, const EquationSolver& solver, const BoundaryConditionManager& bcs);

  /**
   * @brief Solves the equation du_dt = f(u, t)
   *
   * @param[in] u The true DOFs
   * @param[in] du_dt The first time derivative of u
   */
  void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const { Solve(t, 0.0, u, du_dt); }

  /**
   * @brief Solves the equation du_dt = f(u + dt * du_dt, t)
   *
   * @param[in] dt The time step
   * @param[in] u The true DOFs
   * @param[in] du_dt The first time derivative of u
   */
  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) { Solve(t, dt, u, du_dt); }

  /**
   * @brief Configures the Dirichlet enforcement method to use
   * @param[in] method The selected method
   */
  void SetEnforcementMethod(const DirichletEnforcementMethod method) { enforcement_method_ = method; }

  /**
   * @brief Set the time integration method
   *
   * @param[in] timestepper The timestepping method for the solver
   */
  void SetTimestepper(const smith::TimestepMethod timestepper);

  /**
   * @brief Performs a time step
   *
   * @param[inout] x The predicted solution
   * @param[inout] time The current time
   * @param[inout] dt The desired time step
   *
   * @see mfem::ODESolver::Step
   */
  void Step(mfem::Vector& x, double& time, double& dt)
  {
    if (ode_solver_) {
      ode_solver_->Step(x, time, dt);
    } else {
      SLIC_ERROR("ode_solver_ unspecified");
    }
  }

  /**
   * @brief Query the timestep method for the ode solver
   *
   * @return The timestep method used by the underlying ode solver
   */
  TimestepMethod GetTimestepper() { return timestepper_; }

 private:
  /**
   * @brief Internal implementation used for mfem::TDO::Mult and mfem::TDO::ImplicitSolve\
   * @param[in] time The current time
   * @param[in] dt The time step
   * @param[in] u The true DOFs
   * @param[in] du_dt The first time derivative of u
   */
  virtual void Solve(const double time, const double dt, const mfem::Vector& u, mfem::Vector& du_dt) const;

  /**
   * @brief Set of references to external variables used by residual operator
   */
  FirstOrderODE::State state_;

  /**
   * @brief The method of enforcing time-varying dirichlet boundary conditions
   */
  DirichletEnforcementMethod enforcement_method_ = smith::DirichletEnforcementMethod::RateControl;
  /**
   * @brief Reference to the equationsolver used to solve for du_dt
   */
  const EquationSolver& solver_;
  /**
   * @brief MFEM solver object for first-order ODEs
   */
  std::unique_ptr<mfem::ODESolver> ode_solver_;
  /**
   * @brief Reference to boundary conditions used to constrain the solution
   */
  const BoundaryConditionManager& bcs_;
  mfem::Vector zero_;

  /**
   * @brief Working vectors for ODE outputs prior to constraint enforcement
   */
  mutable mfem::Vector U_minus_;
  mutable mfem::Vector U_;
  mutable mfem::Vector U_plus_;
  mutable mfem::Vector dU_dt_;

  TimestepMethod timestepper_;
};

}  // namespace smith::mfem_ext
