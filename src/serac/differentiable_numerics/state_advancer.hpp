// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state_advancer.hpp
 *
 * @brief Specifies parametrized residuals and various linearized evaluations for arbitrary nonlinear systems of
 * equations
 */

#pragma once

#include <vector>
#include "serac/differentiable_numerics/field_state.hpp"

namespace serac {

class DifferentiableSolver;
class WeakForm;
class BoundaryConditionManager;

class StateAdvancer {
 public:
  virtual ~StateAdvancer() {}

  virtual std::tuple<std::vector<FieldState>, double> advanceState(const FieldState& shape_disp,
                                                                   const std::vector<FieldState>& states,
                                                                   const std::vector<FieldState>& params, double time,
                                                                   double dt, size_t cycle) const = 0;

  virtual std::vector<FieldState> initializeState(const FieldState& /*hape_disp*/,
                                                  const std::vector<FieldState>& states,
                                                  const std::vector<FieldState>& /*params*/, double /*time*/) const
  {
    return states;
  }
};

class LumpedMassExplicitNewmark : public StateAdvancer {
 public:
  LumpedMassExplicitNewmark(const std::shared_ptr<WeakForm>& r, const std::shared_ptr<WeakForm>& mr,
                            std::shared_ptr<BoundaryConditionManager> bc)
      : residual_eval(r), mass_residual_eval(mr), bc_manager(bc)
  {
  }

  std::tuple<std::vector<FieldState>, double> advanceState(const FieldState& shape_disp,
                                                           const std::vector<FieldState>& states,
                                                           const std::vector<FieldState>& params, double time,
                                                           double dt, size_t cycle) const override;

 private:
  const std::shared_ptr<WeakForm> residual_eval;
  const std::shared_ptr<WeakForm> mass_residual_eval;
  const std::shared_ptr<BoundaryConditionManager> bc_manager;
  mutable std::unique_ptr<FieldState> m_diag_inv;
};

}  // namespace serac
