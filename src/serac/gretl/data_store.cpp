#include <any>
#include "data_store.hpp"
#include "state.hpp"
#include <iostream>

namespace gretl {

DataStore::DataStore(size_t /*maxStates*/) { current_step_ = 0; }

void DataStore::back_prop()
{
  goingForward_ = false;
  printf("a\n");
  for (size_t n = states_.size(); n > 0; --n) {
    reverse_state();
  }
  goingForward_ = true;
}

void DataStore::reset()
{
  for (size_t n = states_.size(); n > 0; --n) {
    if (!upstreams_[n - 1].size()) {
      primals_[n-1] = nullptr;
    }
    duals_[n - 1] = nullptr;
  }
  current_step_ = 0;
}

void DataStore::vjp(StateBase& state) { state.evaluate_vjp(); }

StateBase DataStore::reverse_state()
{
  --current_step_;
  printf("b\n");
  vjp(states_[current_step_]);
  printf("c\n");
  return states_[current_step_ > 0 ? current_step_ - 1 : current_step_];  // return step earlier, unless at 0
}

void DataStore::add_state(StateBase newState, const std::vector<StateBase>& upstreams)
{
  ++current_step_;
  states_.emplace_back(newState);
  primals_.emplace_back(nullptr);
  duals_.emplace_back(nullptr);
  std::vector<Int> upstreamSteps;
  upstreamSteps.reserve(upstreams.size());
  for (auto& u : upstreams) {
    upstreamSteps.push_back(u.step_); 
  }
  upstreams_.emplace_back(*this, upstreamSteps);

  evals_.emplace_back([=](const UpstreamStates&, DownstreamState&) {
    std::cout << "you forgot to implement eval for step " << current_step_ << std::endl;
    exit(1);
  });

  vjps_.emplace_back([=](UpstreamStates&, const DownstreamState&) {
    std::cout << "you forgot to implement eval for step " << current_step_ << std::endl;
    exit(1);
  });

  assert(current_step_ == states_.size());
  assert(current_step_ == primals_.size());
  assert(current_step_ == duals_.size());
  assert(current_step_ == upstreams_.size());
  // states.emplace_back(newState);
  // duals.emplace_back(nullptr);
  //++step;
}

void DataStore::fetch_state_data(Int stepIndex)
{
  if (primals_[stepIndex]) {
    return;
  }

  fetch_state_data(stepIndex - 1);
  current_step_ = stepIndex;

  states_[stepIndex].evaluate_and_remove_disposable_checkpoints();
}

Int DataStore::num_allocated_states() const
{
  Int numActive = 0;
  for (const auto& s : states_) {
    if (primals_[s.step_]) {
      ++numActive;
    }
  }
  return numActive;
}

Int DataStore::num_active_states() const
{
  Int numActive = 0;
  for (const auto& s : states_) {
    if (primals_[s.step_]) {
      ++numActive;
    }
  }
  return numActive;
}

Int DataStore::num_dual_states() const
{
  Int numActive = 0;
  for (const auto& s : states_) {
    if (duals_[s.step_]) {
      ++numActive;
    }
  }
  return numActive;
}

DynamicDataStore::DynamicDataStore(size_t maxStates)
    : DataStore(maxStates), checkpoints{.maxNumStates = maxStates, .cps{}}
{
}

/*
void DynamicDataStore::add_state(StateBase& newState)
{
  states.emplace_back(newState);
  size_t stepToErase =
      checkpoints.add_checkpoint_and_get_index_to_remove(newState.step_index(), newState.data()->persistent());

  if (CheckpointManager::valid_checkpoint_index(stepToErase)) {
    states[stepToErase].stateData = nullptr;
  }

  std::cout << "use count at " << step_ << " = ";
  for (auto& s : states) {
    std::cout << s.stateData.use_count() << " ";
  }
  std::cout << std::endl;

  for (const auto& upstream : newState.data()->upstreams) {
    upstream.state->lastStepUsed = step_;
  }
  step_ = states.size();
}

void DynamicDataStore::clear_disposable_state() {}

void DynamicDataStore::fetch_state_data(size_t stepIndex)
{
  if (data[stepIndex]) {
    return;
  }
  fetch_state_data(stepIndex - 1);
  step_ = stepIndex;

  states[stepIndex].evaluate_and_remove_disposable_checkpoints();
}
*/

}  // namespace gretl