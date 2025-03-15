// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "state_base.hpp"

namespace gretl {

void StateBase::evaluate_and_remove_disposable(double costFactor)
{
  stateData->evaluate();  // eventually could adjust cost factor by timing this?
  stateData->dataStore.clear_disposable_state(costFactor);
}

void StateBase::evaluate_vjp() { stateData->evaluate_vjp(); }

size_t StateBase::step_index() const { return stateData->stepIndex; }

void StateBase::clear() { stateData->clear_primal(); }

void StateBase::clear_dual() { stateData->clear_dual(); }

}  // namespace gretl