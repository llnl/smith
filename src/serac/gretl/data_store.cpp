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
  // MRT, will need a virtual to clear out the activeCount
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
    states_[current_step_]->primal_ = nullptr;
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
    // check if the upstream has upstreams (is not persistent).
    if (upstreams_[upstreamStep].size()) {
      if (!states_[u.step_]->primal_) {
        gretl::print("at step ", current_step_-1, "resetting ", u.step_);
        states_[u.step_]->primal_ = u.primal_;
      } else {
        gretl_assert(states_[u.step_]->primal_ == u.primal_);
      }
    }
  }
  upstreams_.emplace_back(*this, upstreamSteps);

  evals_.emplace_back([=](const UpstreamStates&, DownstreamState&) {
    std::cout << "eval not implemented for step " << current_step_ << std::endl;
    gretl_assert(false);
  });

  vjps_.emplace_back([=](UpstreamStates&, const DownstreamState&) {
    std::cout << "vjp not implemented for step " << current_step_ << std::endl;
    gretl_assert(false);
  });

  assert(current_step_ == states_.size());
  assert(current_step_ == duals_.size());
  assert(current_step_ == upstreams_.size());
  assert(current_step_ == vjps_.size());
}

std::shared_ptr<std::any>& DataStore::any_primal(Int step) { return states_[step]->primal_; }

void DataStore::fetch_state_data(Int stepIndex)
{
  if (states_[stepIndex]->primal_) {
    return;
  }
  current_step_ = stepIndex;
  states_[stepIndex]->evaluate_forward();
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
    : DataStore(maxStates), checkpointManager{.maxNumStates = maxStates, .cps{}}
{
}

void printv(const std::vector<Int>& v) {
  
  size_t c=0;
  for (auto s : v) {
    std::cout << c << ":" << s << " ";
    ++c;
  }
  std::cout << std::endl;
}

void printv(const std::vector<StateBase>& v) {
  
  size_t c=0;
  for (auto s : v) {
    std::cout << c << ":" << s.step_ << " ";
    ++c;
  }
  std::cout << std::endl;
}

/// @overload
void DynamicDataStore::add_state(std::unique_ptr<StateBase> newState, const std::vector<StateBase>& upstreams) 
{
  Int step = newState->step_;
  
  lastStepUsed_.push_back(step);
  passthroughs_.push_back({});
  activeCount_.push_back(1);

  for (auto& u : upstreams) {
    if (upstreams_[u.step_].size()) { // is not persistent upstream 
      Int lastLastStepUsed = lastStepUsed_[u.step_];
      lastLastStepUsed = std::max(lastLastStepUsed, u.step_+1);
      Int stepPassingThrought = u.step_;
      for (Int stepBeingPassedThrough = lastLastStepUsed; stepBeingPassedThrough < step; ++stepBeingPassedThrough) {
        passthroughs_[stepBeingPassedThrough].push_back(stepPassingThrought);
        activeCount_[stepPassingThrought]++;
      }
      lastStepUsed_[stepPassingThrought] = step;
    }
  }

  DataStore::add_state(std::move(newState), upstreams);
}

/// @overload
void DynamicDataStore::fetch_state_data(Int stepIndex) 
{
  printf("not here yet\n");
  exit(1);
  if (states_[stepIndex]->primal_) {
    return;
  }

  size_t lastCheckpoint = checkpointManager.last_checkpoint_step();
  gretl_assert(states_[lastCheckpoint]->primal_);

  for (size_t i=lastCheckpoint; i < stepIndex; ++i) {
    states_[i+1]->evaluate_forward();
    size_t stepToErase = checkpointManager.add_checkpoint_and_get_index_to_remove(i+1);
    if (checkpointManager.valid_checkpoint_index(stepToErase)) {
      states_[stepToErase]->primal_ = nullptr;
    }
  }
}


/// @overload
void DynamicDataStore::remove_things(Int step) {
  if (upstreams_[step].size()) {
    size_t stepToErase = checkpointManager.add_checkpoint_and_get_index_to_remove(step);
    if (checkpointManager.valid_checkpoint_index(stepToErase)) {
      gretl_assert(activeCount_[stepToErase]);
      if (stepToErase==11) {
        gretl::print("at", step, "remove 11, active = ", activeCount_[11]);
      }
      activeCount_[stepToErase]--;
      if (activeCount_[stepToErase]==0) {
        if (stepToErase==11) {printf("null\n");}
        states_[stepToErase]->primal_ = nullptr;
      }
      for (Int stepPassingThrough : passthroughs_[stepToErase]) {
        gretl_assert(activeCount_[stepPassingThrough]);
        activeCount_[stepPassingThrough]--;
        if (activeCount_[stepPassingThrough]==0) {
          states_[stepPassingThrough]->primal_ = nullptr;
        }
      }
    }
  } else {
    bool persistentCheckpoint = true;
    checkpointManager.add_checkpoint_and_get_index_to_remove(step, persistentCheckpoint);
  }
}

/// @overload
void DynamicDataStore::print() const {
  for (Int i=0; i < current_step_; ++i) {
    std::cout << i << ", act: " << activeCount_[i] << ":" << (states_[i]->primal_!=nullptr) << ", ups: ";
    for (auto& v : upstreams_[i].steps_) {
      std::cout << v << " ";
    }
    std::cout << ", pass: ";
    for (auto& v : passthroughs_[i]) {
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }
  std::cout << checkpointManager << std::endl;
}

}  // namespace gretl