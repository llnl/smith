// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "state_base.hpp"
#include "upstream_state.hpp"

namespace gretl {

void StateBase::evaluate_forward()
{
  DownstreamState ds(dataStore_, step_);
  dataStore_->evals_[step_](dataStore_->upstreams_[step_], ds);
  dataStore_->erase_step_state_data(step_);
}

void StateBase::evaluate_vjp()
{
  const DownstreamState ds(dataStore_, step_);
  dataStore_->vjps_[step_](dataStore_->upstreams_[step_], ds);
}

}  // namespace gretl
