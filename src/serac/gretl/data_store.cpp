#include <any>
#include "data_store.hpp"
#include "state.hpp"
#include <iostream>

namespace gretl {

DataStore::DataStore(size_t /*maxStates*/) { current_step_ = 0; }

void DataStore::back_prop()
{
  goingForward_ = false;
  for (size_t n = states_.size(); n > 0; --n) {
    reverse_state();
  }
  goingForward_ = true;
}

void DataStore::reset()
{
  for (size_t n = states_.size(); n > 0; --n) {
    if (upstreams_[n - 1].size()) {
      states_[n - 1]->primal_ = nullptr;
    }
    duals_[n - 1] = nullptr;
  }
  current_step_ = 0;
}

void DataStore::vjp(StateBase& state) { state.evaluate_vjp(); }

void DataStore::reverse_state()
{
  --current_step_;
  if (upstreams_[current_step_].size()) {
    vjp(*states_[current_step_]);
    duals_[current_step_] = nullptr;
  }
}

void DataStore::add_state(std::unique_ptr<StateBase> newState, const std::vector<StateBase>& upstreams)
{
  ++current_step_;

  states_.emplace_back(std::move(newState));
  duals_.emplace_back(nullptr);

  std::vector<Int> upstreamSteps;
  upstreamSteps.reserve(upstreams.size());
  for (auto& u : upstreams) {
    Int upstreamStep = u.step_;
    upstreamSteps.push_back(upstreamStep);
    // repopulate the upstreams data, in case the checkpointer has decided to remove it.
    if (upstreams_[upstreamStep].size()) {
      if (!states_[u.step_]->primal_) {
        states_[u.step_]->primal_ = u.primal_;
      }
    }
  }
  upstreams_.emplace_back(*this, upstreamSteps);

  evals_.emplace_back([=](const UpstreamStates&, DownstreamState&) {
    std::cout << "you forgot to implement eval for step " << current_step_ << std::endl;
    gretl_assert(false);
  });

  vjps_.emplace_back([=](UpstreamStates&, const DownstreamState&) {
    std::cout << "you forgot to implement vjp for step " << current_step_ << std::endl;
    gretl_assert(false);
  });

  assert(current_step_ == states_.size());
  assert(current_step_ == duals_.size());
  assert(current_step_ == upstreams_.size());
  assert(current_step_ == vjps_.size());
}

void DataStore::add_state(std::unique_ptr<StateBase> newState, const std::vector<StateBase>& upstreams, const std::shared_ptr<std::any>& value)
{
  add_state(std::move(newState), upstreams);
  gretl_assert(current_step_ - 1 < states_.size());
  states_[current_step_ - 1]->primal_ = value;
}

std::shared_ptr<std::any>& DataStore::any_primal(Int step)
{
  return states_[step]->primal_;
}

void DataStore::fetch_state_data(Int stepIndex)
{
  if (states_[stepIndex]->primal_) {
    return;
  }
  
  current_step_ = stepIndex;

  states_[stepIndex]->evaluate_and_remove_disposable_checkpoints();
}

Int DataStore::num_active_states() const
{
  Int numActive = 0;
  for (const auto& s : states_) {
    if (states_[s->step_]->primal_) {
      ++numActive;
    }
  }
  return numActive;
}

Int DataStore::num_dual_states() const
{
  Int numActive = 0;
  for (const auto& s : states_) {
    if (duals_[s->step_]) {
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