// Copyright (c), Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state_advancer.hpp
 *
 * @brief Interface and implementations for advancing from one step to the next.  Typically these are time integrators.
 */

#pragma once

#include <vector>
#include "serac/differentiable_numerics/field_state.hpp"
#include "serac/gretl/double_state.hpp"
#include "serac/physics/common.hpp"

namespace serac {

class DifferentiableSolver;
class WeakForm;
class BoundaryConditionManager;

/// Base state advancer class, allows specification for quasi-static solve strategies, or time integration algorithms
class StateAdvancer {
 public:
  /// @brief destructor
  virtual ~StateAdvancer() {}

  /// @brief interface method to advance the states from a given cycle and time, to the next cycle (cycle+1) and time
  /// (time+dt). shape_disp and params are assumed to be fixed in this advance.  Time and time increment (dt) are
  /// gretl::State in order to record the duals on the reverse pass
  virtual std::tuple<std::vector<FieldState>, DoubleState> advanceState(const FieldState& shape_disp,
                                                                        const std::vector<FieldState>& states,
                                                                        const std::vector<FieldState>& params,
                                                                        DoubleState time, DoubleState dt,
                                                                        size_t cycle) const = 0;
};

/// Lumped mass explicit dynamics implementation for the StateAdvancer interface
class LumpedMassExplicitNewmark : public StateAdvancer {
 public:
  /// Constructor for lumped mass explicit Newmark implementation
  LumpedMassExplicitNewmark(const std::shared_ptr<WeakForm>& r, const std::shared_ptr<WeakForm>& mr,
                            std::shared_ptr<BoundaryConditionManager> bc)
      : residual_eval(r), mass_residual_eval(mr), bc_manager(bc)
  {
  }

  /// @overload
  std::tuple<std::vector<FieldState>, DoubleState> advanceState(const FieldState& shape_disp,
                                                                const std::vector<FieldState>& states,
                                                                const std::vector<FieldState>& params, DoubleState time,
                                                                DoubleState dt, size_t cycle) const override;

 private:
  const std::shared_ptr<WeakForm> residual_eval;               ///< weak form to evaluate mechanical forces
  const std::shared_ptr<WeakForm> mass_residual_eval;          ///< weak form to evaluate lumped masses
  const std::shared_ptr<BoundaryConditionManager> bc_manager;  ///< tracks information on which dofs are constrainted
  mutable std::unique_ptr<FieldState>
      m_diag_inv;  ///< save off FieldState for inverse lumped mass.  This can be computed up front and reused every
                   ///< timestep to avoid recomputing the mass each step.
};

inline TimeInfo create_time_info(DoubleState t, DoubleState dt, size_t cycle)
{
  return TimeInfo(t.get(), dt.get(), cycle);
}

}  // namespace serac
