// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/lumped_mass_explicit_newmark_state_advancer.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/explicit_dynamic_solve.hpp"
#include "smith/differentiable_numerics/timestep_estimator.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"

namespace smith {

/// @brief uses the constrained dofs on the bc_manager to zero the corresponding dofs in FieldState s.
FieldState applyZeroBoundaryConditions(const FieldState& s, const BoundaryConditionManager* bc_manager)
{
  auto s_bc = s.clone({s});

  s_bc.set_eval([=](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    auto s_new = std::make_shared<FiniteElementState>(*inputs[0].get<FEFieldPtr>());
    s_new->SetSubVector(bc_manager->allEssentialTrueDofs(), 0.0);
    output.set<FEFieldPtr, FEDualPtr>(s_new);
  });

  s_bc.set_vjp([=](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    FiniteElementDual tmp(*output.get_dual<FEDualPtr, FEFieldPtr>());
    tmp.SetSubVector(bc_manager->allEssentialTrueDofs(), 0.0);
    inputs[0].get_dual<FEDualPtr, FEFieldPtr>()->Add(1.0, tmp);
  });

  return s_bc.finalize();
}

std::vector<FieldState> LumpedMassExplicitNewmarkStateAdvancer::advanceState(
    const TimeInfo& time_info, const FieldState& shape_disp, const std::vector<FieldState>& states_in,
    const std::vector<FieldState>& params) const
{
  SMITH_MARK_FUNCTION;
  SLIC_ERROR_IF(states_in.size() != 3, "ExplicitNewmark is a 2nd order time integrator requiring 3 states.");

  enum STATES
  {
    DISP,
    VELO,
    ACCEL
  };

  enum PARAMS
  {
    DENSITY
  };

  std::vector<FieldState> states = states_in;

  if (time_info.cycle() == 0 || !m_diag_inv) {
    //  Calculate a_pred, lumped mass version
    auto lumped_mass = computeLumpedMass(mass_residual_eval_.get(), shape_disp, states[DISP], params[DENSITY]);
    auto diag_inv = diagInverse(lumped_mass);  // should return inverse of diagonal matrix as a field state
    m_diag_inv = std::make_unique<FieldState>(diag_inv);
    auto zero_mass_res = evalResidual(residual_eval_.get(), shape_disp, states, params, time_info, ACCEL);
    auto a_initial = negativeComponentWiseMult(*m_diag_inv, zero_mass_res, bc_manager_.get());
    states[ACCEL] = a_initial;
  }

  double start_time = time_info.time();
  double end_time = time_info.time() + time_info.dt();

  DoubleState stable_dt = ts_estimator_->dt(shape_disp, states, params);
  DoubleState time =
      gretl::clone_state([=](double) { return start_time; }, [](double, double, double&, double) {}, stable_dt);

  while (time.get() < end_time) {
    if (time.get() + stable_dt.get() > end_time) {
      stable_dt = end_time - time;
    }

    time = time + stable_dt;

    // grabs initial states
    const FieldState& u = states[DISP];
    const FieldState& v = states[VELO];
    const FieldState& a = states[ACCEL];

    // first pass of setting u and v predictors
    FieldState v_half_step = v + 0.5 * (stable_dt * a);
    FieldState u_pred = u + stable_dt * v_half_step;

    // zeroing out u predictor dofs associated with zero BCs
    u_pred = applyZeroBoundaryConditions(u_pred, bc_manager_.get());
    // create a vector of type FieldState called state_pred and put the u and v predictors into it
    std::vector<FieldState> state_pred{u_pred, v_half_step, zeroCopy(a)};

    // should return the evaluation of the residual for the current state variables
    auto zero_mass_res = evalResidual(residual_eval_.get(), shape_disp, state_pred, params,
                                      TimeInfo(time.get(), time_info.dt(), time_info.cycle()), ACCEL);

    // m_diag_inv*zero_mass_res; // calculate the acceleration
    auto a_pred = negativeComponentWiseMult(*m_diag_inv, zero_mass_res, bc_manager_.get());

    // update the v predictor after a predictor solves
    FieldState v_pred = v_half_step + 0.5 * (stable_dt * a_pred);

    states = std::vector<FieldState>{u_pred, v_pred, a_pred};

    if (time.get() < end_time) {
      stable_dt = ts_estimator_->dt(shape_disp, states, params);
    }
  }

  // place all solved updated states into the output
  return states;
}

}  // namespace smith
