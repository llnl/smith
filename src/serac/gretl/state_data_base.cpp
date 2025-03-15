// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "state_data_base.hpp"
#include "upstream_state.hpp"
#include "state_base.hpp"

namespace gretl {

StateDataBase::StateDataBase(DataStore& cpd, size_t step, const std::vector<StateBase>& ustreams)
    : stepIndex(step), dataStore(cpd)
{
  upstreams.reserve(ustreams.size());
  for (const auto& u : ustreams) {
    upstreams.emplace_back(u.stateData);
  }
}

bool StateDataBase::persistent() const { return upstreams.empty(); }

}  // namespace gretl