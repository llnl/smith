// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file lumped_mass_explicit_newmark_state_advancer.hpp
 *
 * @brief Implementation of explicit Newmark
 */

#pragma once

#include <vector>
#include "gretl/double_state.hpp"
#include "smith/physics/common.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"

namespace smith {

class WeakForm;
class TimestepEstimator;
class BoundaryConditionManager;

/// Lumped mass explicit dynamics implementation for the StateAdvancer interface
class LumpedMassExplicitNewmarkStateAdvancer : public StateAdvancer {
 public:
  /// Constructor for lumped mass explicit Newmark implementation
  LumpedMassExplicitNewmarkStateAdvancer(const std::shared_ptr<WeakForm>& r, const std::shared_ptr<WeakForm>& mr,
                                         const std::shared_ptr<TimestepEstimator>& ts,
                                         std::shared_ptr<BoundaryConditionManager> bc)
      : residual_eval_(r), mass_residual_eval_(mr), ts_estimator_(ts), bc_manager_(bc)
  {
  }

  /// @overload
  std::vector<FieldState> advanceState(const TimeInfo& time_info, const FieldState& shape_disp,
                                       const std::vector<FieldState>& states,
                                       const std::vector<FieldState>& params) const override;

 private:
  const std::shared_ptr<WeakForm> residual_eval_;               ///< weak form to evaluate mechanical forces
  const std::shared_ptr<WeakForm> mass_residual_eval_;          ///< weak form to evaluate lumped masses
  const std::shared_ptr<TimestepEstimator> ts_estimator_;       ///< evaluates stable timesteps
  const std::shared_ptr<BoundaryConditionManager> bc_manager_;  ///< tracks information on which dofs are constrainted
  mutable std::unique_ptr<FieldState>
      m_diag_inv;  ///< save off FieldState for inverse lumped mass.  This can be computed up front and reused every
                   ///< timestep to avoid recomputing the mass each step.
};

}  // namespace smith
