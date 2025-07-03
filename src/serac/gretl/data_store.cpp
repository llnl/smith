#include <any>
#include "data_store.hpp"
#include "state.hpp"
#include <iostream>

namespace gretl {

DataStore::DataStore(size_t /*maxStates*/) { current_step_ = 0; }

void DataStore::back_prop()
{
  isGoingForward = false;
  for (size_t n = states_.size(); n > 0; --n) {
    reverse_state();
  }
  isGoingForward = true;
}

template <typename Func>
void for_each_active_upstream(const DataStore* ds, size_t step, const Func& func) {
  auto dataStore = dynamic_cast<const DynamicDataStore*>(ds);
  if (!dataStore) { return; }
  for (Int upstream : dataStore->upstreams_[step].steps_) {
    if (!dataStore->is_persistent(upstream)) {
      func(upstream);
    }
  }
  for (Int upstreamStepPassingThrough : dataStore->passthroughs_[step]) {
    func(upstreamStepPassingThrough);
  }
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
  gretl::print("reverse from step", current_step_);
  if (upstreams_[current_step_].size()) {
    vjp(*states_[current_step_]);
    duals_[current_step_] = nullptr;
    states_[current_step_]->primal_ = nullptr;
  }
}

bool DataStore::is_persistent(Int step) const {
  return !upstreams_[step].size();
}

void DynamicDataStore::reverse_state()
{
  // must erase the final step in the cp manager before we get started
  if (current_step_ == states_.size()) {
    checkpointManager.erase_step(current_step_-1);
  }
  --current_step_;
  gretl::print("reverse from step", current_step_);
  if (upstreams_[current_step_].size()) {
    fetch_state_data(current_step_-1);
    vjp(*states_[current_step_]);
    duals_[current_step_] = nullptr;
    states_[current_step_]->primal_ = nullptr;
    activeCount_[current_step_] = 0;
    for (auto pass : passthroughs_[current_step_]) {
      activeCount_[pass]--;
      if (activeCount_[pass]==0) {
        states_[pass]->primal_ = nullptr;
      }
    }
    checkpointManager.erase_step(current_step_-1);
  }
}

void DataStore::add_state(std::unique_ptr<StateBase> newState, const std::vector<StateBase>& upstreams)
{
  /*
  states_.emplace_back(std::move(newState));
  duals_.emplace_back(nullptr);
  activeCount_.push_back(1);
  active_.push_back(true);

  std::vector<Int> upstreamSteps;
  upstreamSteps.reserve(upstreams.size());
  for (auto& u : upstreams) {
    Int upstreamStep = u.step_;
    upstreamSteps.push_back(upstreamStep);
    // repopulate the upstreams data, in case the checkpointer has decided to remove it.
    if (!is_persistent(upstreamStep)) {
      if (!states_[upstreamStep]->primal_) {
        gretl::print("at step ", current_step_, "resetting ", upstreamStep);
        activeCount_[upstreamStep]++;
        states_[upstreamStep]->primal_ = u.primal_;
      } else {
        gretl_assert(states_[upstreamStep]->primal_ == u.primal_);
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

  print();

  ++current_step_;
  assert(current_step_ == states_.size());
  assert(current_step_ == duals_.size());
  assert(current_step_ == upstreams_.size());
  assert(current_step_ == vjps_.size());
  */
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

  states_.emplace_back(std::move(newState));
  duals_.emplace_back(nullptr);
  activeCount_.push_back(1);
  active_.push_back(true);
  lastStepUsed_.push_back(step);
  passthroughs_.push_back({});

  bool persistent = upstreams.size()==0;
  if (persistent) {
    checkpointManager.add_checkpoint_and_get_index_to_remove(step, persistent);
  }

  bool resetting = false;

  std::vector<Int> upstreamSteps;
  upstreamSteps.reserve(upstreams.size());
  for (auto& u : upstreams) {
    Int upstreamStep = u.step_;
    upstreamSteps.push_back(upstreamStep);
    if (!is_persistent(upstreamStep)) {

      // we are know using this upstream again, add to count of uses
      activeCount_[upstreamStep]++;

      // check if step fully deleted, 
      if (!states_[upstreamStep]->primal_) {
        gretl_assert(activeCount_[upstreamStep]==1);
        gretl::print("at step ", current_step_, "resetting ", upstreamStep);
        resetting = true;
        //print();
        // gretl_assert(check_validity());
        states_[upstreamStep]->primal_ = u.primal_;
      } else {
        gretl_assert(states_[upstreamStep]->primal_ == u.primal_);
      }

      // knowing this upstream is used here, push the passthroughts forward from their last known use to the previous step
      Int lastLastStepUsed = std::max(lastStepUsed_[u.step_], u.step_+1);
      Int upstreamStepPassingThrough = u.step_;
      for (Int stepBeingPassedThrough = lastLastStepUsed; stepBeingPassedThrough < step; ++stepBeingPassedThrough) {
        passthroughs_[stepBeingPassedThrough].push_back(upstreamStepPassingThrough);
        if (active_[upstreamStepPassingThrough]) {
          activeCount_[upstreamStepPassingThrough]++;
        }
      }
      lastStepUsed_[upstreamStepPassingThrough] = step;
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
  
  //if (resetting) {
  print();
  gretl_assert(check_validity());
  //}

  ++current_step_;
  gretl_assert(current_step_ == states_.size());
  gretl_assert(current_step_ == duals_.size());
  gretl_assert(current_step_ == upstreams_.size());
  gretl_assert(current_step_ == passthroughs_.size());

  gretl_assert(current_step_ == active_.size());
  gretl_assert(current_step_ == activeCount_.size());
  gretl_assert(current_step_ == vjps_.size());
  gretl_assert(current_step_ == lastStepUsed_.size());
}

/// @overload
void DynamicDataStore::fetch_state_data(Int stepIndex) 
{
  gretl_assert(!isGoingForward);

  printf("fetching, deal with later\n");
  exit(1);

  Int lastCheckpoint = static_cast<Int>(checkpointManager.last_checkpoint_step());

  gretl_assert(lastCheckpoint <= stepIndex);

  gretl_assert(states_[lastCheckpoint]->primal_);
  for (auto& passThroughState : passthroughs_[lastCheckpoint]) {
    gretl_assert(activeCount_[passThroughState]);
  }

  for (Int i=lastCheckpoint; i < stepIndex; ++i) {
    gretl::print("at step", current_step_, ", trying to eval", i+1);
    print();
    for (auto& passThroughState : passthroughs_[i+1]) {
      gretl_assert(activeCount_[passThroughState]);
      activeCount_[passThroughState]++;
    }
    states_[i+1]->evaluate_forward();
  }
}


/// @overload
void DynamicDataStore::remove_things(Int step) {
  if (!is_persistent(step)) {
    size_t stepToErase = checkpointManager.add_checkpoint_and_get_index_to_remove(step);
    if (checkpointManager.valid_checkpoint_index(stepToErase)) {
      gretl::print("removing step", stepToErase, "at step", step);
      gretl_assert(activeCount_[stepToErase]);
      activeCount_[stepToErase]--;
      active_[stepToErase] = false;
      if (activeCount_[stepToErase]==0) {
        states_[stepToErase]->primal_ = nullptr;
      }
      for_each_active_upstream(this, stepToErase,[&](Int upstream) {
        gretl_assert(activeCount_[upstream]);
        activeCount_[upstream]--;
        if (activeCount_[upstream]==0) {
          states_[upstream]->primal_ = nullptr;
        }
      });
    }
  }
  gretl::print("in remove");
  print();
  if (!check_validity()) {
    gretl::print("issue in the remove");
    gretl_assert(check_validity());
  }
  
}


bool DynamicDataStore::check_validity() const {

  bool valid = true;
  // first check that our version of the saved states matches the cp manager
  // we are allowed to be saving an extra step here at the end
  for (size_t i=0; i < states_.size()-1; ++i) {
    if (active_[i]) {
      bool cp_has_i = false;
      for (auto& cp : checkpointManager.cps) {
        if (cp.step==i) {
          cp_has_i = true;
          break;
        }
      }
      if (!cp_has_i) {
        gretl::print("step", i, "not consistent with checkpoint manager");
        valid = false;
      }
    }
  }

  std::vector<int> my_active_count(states_.size());
  for (size_t i=0; i < states_.size(); ++i) {
    if (active_[i]) {
      my_active_count[i]++;
      for_each_active_upstream(this, i, [&](Int u) {
        my_active_count[u]++;
      });
    }
  }
  for (size_t i=0; i < states_.size(); ++i) {
    if (my_active_count[i] > 0 && !states_[i]->primal_) {
      gretl::print("step", i, "has an active count, but is deallocated");
      valid = false;
    }
    if (my_active_count[i] != activeCount_[i]) {
      gretl::print("step", i, "active count =", activeCount_[i], " my c=", my_active_count[i]);
      valid = false;
    }
  }

  return valid;
} 

/// @overload
void DynamicDataStore::print() const {
  for (Int i=0; i < states_.size(); ++i) {
    std::cout << i << ", act: " << active_[i] << ":" << activeCount_[i] << ":" << (states_[i]->primal_!=nullptr) << ", ups: ";
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