#include "serac/differentiable_numerics/lumped_mass_explicit_newmark_state_advancer.hpp"
#include "serac/physics/weak_form.hpp"
#include "serac/differentiable_numerics/field_state.hpp"
#include "serac/differentiable_numerics/explicit_dynamic_solve.hpp"
#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"

namespace serac {

/// @brief uses the constrained docs on the bc_manager to zero the corresponding dofs in FieldState s.
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

std::vector<FieldState> LumpedMassExplicitNewmarkStateAdvancer::advanceState(const FieldState& shape_disp,
                                                                             const std::vector<FieldState>& states,
                                                                             const std::vector<FieldState>& params,
                                                                             const TimeInfo& time_info) const
{
  SERAC_MARK_FUNCTION;
  SLIC_ERROR_IF(states.size() != 3, "ExplicitNewmark is a 2nd order time integrator requiring 3 states.");

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

  // grabs initial states
  const FieldState& u = states[DISP];
  const FieldState& v = states[VELO];
  const FieldState& a = states[ACCEL];

  // first pass of setting u and v predictors
  FieldState v_half_step = v + 0.5 * (time_info.dt() * a);  // axpby(1.0, v, 0.5 * dt, a);
  FieldState u_pred = u + time_info.dt() * v_half_step;     // auto u_pred = axpby(1.0, u, dt, v_half_step);

  // zeroing out u predictor dofs associated with zero BCs
  u_pred = applyZeroBoundaryConditions(u_pred, bc_manager_.get());
  // create a vector of type FieldState called state_pred and put the u and v predictors into it
  std::vector<FieldState> state_pred{u_pred, v_half_step, zero_copy(a)};

  // if (cycle == 0) {
  //  Calculate a_pred, lumped mass version
  //  Note that this could maybe done at a higher up level and then flowed down, which would be more efficient?
  auto lumped_mass = computeLumpedMass(mass_residual_eval_.get(), shape_disp, states[DISP], params[DENSITY]);
  auto diag_inv = diagInverse(lumped_mass);  // should return inverse of diagonal matrix as a field state
  // m_diag_inv = std::make_unique<FieldState>(diag_inv);
  //}

  // should return the evaluation of the residual for the current state variables
  auto zero_mass_res = evalResidual(residual_eval_.get(), shape_disp, state_pred, params, time_info, ACCEL);
  // m_diag_inv*zero_mass_res; // calculate the acceleration
  // auto a_pred = componentWiseMult(*m_diag_inv, zero_mass_res, bc_manager.get());
  auto a_pred = componentWiseMult(diag_inv, zero_mass_res, bc_manager_.get());
  // update the v predictor after a predictor solves
  FieldState v_pred = v_half_step + 0.5 * (time_info.dt() * a_pred);
  // place all solved updated states into the output
  return std::vector<FieldState>{u_pred, v_pred, a_pred};
}

}  // namespace serac
