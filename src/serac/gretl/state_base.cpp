// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "state_base.hpp"
#include "upstream_state.hpp"

namespace gretl {

void StateBase::evaluate_and_remove_disposable_checkpoints()
{
  DownstreamState ds(dataStore_, step_);
  dataStore_->evals_[step_](dataStore_->upstreams_[step_], ds);
}

void StateBase::evaluate_vjp()
{
  std::cout << "trying eval j " << step_ << " " << dataStore_->upstreams_[step_].size() << std::endl;
  std::cout << "trying eval j " << dataStore_->vjps_.size() << std::endl;
  const DownstreamState ds(dataStore_, step_);
  for (auto s : dataStore_->upstreams_[step_].steps_) {
    std::cout << "up state = " << s << std::endl;
  }
  dataStore_->vjps_[step_](dataStore_->upstreams_[step_], ds);
  printf("did eval j\n");
}

}  // namespace gretl