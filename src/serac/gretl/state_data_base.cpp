#include "state_data_base.hpp"
#include "upstream_state.hpp"
#include "state_base.hpp"

namespace gretl {

StateDataBase::StateDataBase(DataStore& cpd, size_t step, const std::vector<StateBase>& ustreams)
    : dataStore(cpd), stepIndex(step)
{
  upstreams.reserve(ustreams.size());
  for (const auto& u : ustreams) {
    upstreams.emplace_back(u.stateData);
  }
}

bool StateDataBase::persistent() const { return upstreams.empty(); }

}  // namespace gretl