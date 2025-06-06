#include "data_store.hpp"
#include "state.hpp"
#include <iostream>

namespace gretl {

DataStore::DataStore(size_t maxStates) : checkpoints{.maxNumStates = maxStates, .cps{}} {
  step = 0;
}

void DataStore::back_prop()
{
  for (size_t n = states.size(); n > 0; --n) {
    reverse_state();
  }
}

void DataStore::reset()
{
  for (size_t n = states.size(); n > 0; --n) {
    if (!states[n].data()->persistent()) {
      states[n].clear();
      states[n].clear_dual();
    }
  }
}

void DataStore::vjp(StateBase& state)
{
  state.evaluate_vjp();
}

StateBase DataStore::reverse_state()
{
  --step;
  vjp(states[step]);

  return states[step > 0 ? step - 1 : step];  // return step earlier, unless at 0
}

void DataStore::add_state(StateBase& newState)
{
  states.emplace_back(newState);
  ++step;
}

size_t DataStore::num_active_states() const
{
  return states.size();
}

size_t DataStore::num_dual_states() const
{
  return states.size();
}

}  // namespace gretl