#include "data_store.hpp"
#include "state.hpp"
#include <iostream>

namespace gretl {

DataStore::DataStore(size_t /*maxStates*/) { step = 0; }

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


size_t DataStore::num_allocated_states() const
{
  std::set<const StateDataBase*> datas;
  for (const auto& s : states) {
    if (s.data()->primal_active()) {
      datas.emplace(s.data());
    }
    for (auto& g : *s.upstreamsForNextStep) {
      datas.emplace(g.get());
    }
  }
  return datas.size();
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

DynamicDataStore::DynamicDataStore(size_t maxStates)
    : DataStore(maxStates), checkpoints{.maxNumStates = maxStates, .cps{}}
{
}

void DynamicDataStore::add_state(StateBase& newState) 
{
  states.emplace_back(newState);
  auto allUpstreams = newState.data()->upstreams;
  for (auto& upstream : allUpstreams) {
    auto upState = upstream.get_state();
    size_t index = upState->step_index();
    for (size_t i=index+1; i < step; ++i) {
      auto previousState = states[i];
      previousState.upstreamsForNextStep->emplace(upState);
    }
  }
  step = states.size();
  std::cout << "step, states size = " << step << " " << states.size() << std::endl;
}

void DynamicDataStore::clear_disposable_state() 
{
  
}

void DynamicDataStore::fetch_state_data(size_t stepIndex)
{
  if (states[stepIndex].data()->primal_active()) {
    return;
  }

  fetch_state_data(stepIndex - 1);
  step = stepIndex;

  states[stepIndex].evaluate_and_remove_disposable_checkpoints();
}

}  // namespace gretl