#include "data_store.hpp"
#include "state.hpp"
#include <iostream>

namespace gretl {

DataStore::DataStore(size_t maxStates) : checkpoints{.maxNumStates = maxStates, .cps{}} { step = 0; }

void DataStore::back_prop()
{
  for (size_t n = states.size(); n > 0; --n) {
    reverse_state();
  }
}

void DataStore::reset()
{
  for (size_t n = states.size(); n > 0; --n) {
    if (!states[n - 1].data()->persistent()) {
      states[n - 1].clear();
    }
    states[n - 1].clear_dual();
  }
  step = 0;
}

void DataStore::vjp(StateBase& state) { state.evaluate_vjp(); }

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

void DataStore::fetch_state_data(size_t stepIndex)
{
  if (states[stepIndex].data()->primal_active()) {
    return;
  }

  fetch_state_data(stepIndex - 1);
  step = stepIndex;

  states[stepIndex].evaluate_and_remove_disposable_checkpoints();
}

size_t DataStore::num_active_states() const
{
  size_t numActive = 0;
  for (const auto& s : states) {
    if (s.data()->primal_active()) {
      ++numActive;
    }
  }
  return numActive;
}

size_t DataStore::num_dual_states() const
{
  size_t numActive = 0;
  for (const auto& s : states) {
    if (s.data()->dual_active()) {
      ++numActive;
    }
  }
  return numActive;
}

}  // namespace gretl