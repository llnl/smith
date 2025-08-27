// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <any>
#include "data_store.hpp"
#include "state.hpp"
#include <iostream>
#include <iomanip>


namespace gretl {

DataStore::DataStore(size_t maxStates) : checkpointManager_{.maxNumStates = maxStates, .cps{}} { current_step_ = 0; }

void DataStore::back_prop()
{
  stillConstructingGraph_ = false;
  current_step_ = static_cast<Int>(states_.size());
  for (size_t n = states_.size(); n > 0; --n) {
    reverse_state();
  }
}

template <typename Func>
void for_each_active_upstream(const DataStore* dataStore, size_t step, const Func& func)
{
  for (const auto& upstream : dataStore->upstreams_[step].states()) {
    if (!dataStore->is_persistent(upstream.step_)) {
      func(upstream.step_);
    }
  }
  for (Int upstreamStepPassingThrough : dataStore->passthroughs_[step]) {
    func(upstreamStepPassingThrough);
  }
}

void DataStore::clear_usage(Int step)
{
  duals_[step] = nullptr;
  states_[step]->primal() = nullptr;
  active_[step] = false;
  usageCount_[step] = 0;
  try_to_free(step);
  for_each_active_upstream(this, step, [&](Int u) {
    usageCount_[u]--;
    try_to_free(u);
  });
}

bool DataStore::state_in_use(Int step) const
{
  return (active_[step] || (usageCount_[step] > 0)) && states_[step]->primal();
}

void DataStore::reset()
{
  Int num_persistent = 0;
  for (size_t n = states_.size(); n > 0; --n) {
    Int stepToClear = static_cast<Int>(n - 1);
    if (!is_persistent(stepToClear)) {
      clear_usage(stepToClear);
    } else {
      num_persistent++;
      duals_[stepToClear] = nullptr;
    }
  }
  checkpointManager_.reset();
  current_step_ = num_persistent;
}

void DataStore::reset_graph()
{
  Int num_persistent = 0;
  while (is_persistent(num_persistent)) {
    duals_[num_persistent] = nullptr;
    ++num_persistent;
  }
  states_.resize(num_persistent);
  duals_.resize(num_persistent);
  upstreams_.resize(num_persistent);
  evals_.resize(num_persistent);
  vjps_.resize(num_persistent);
  active_.resize(num_persistent);
  usageCount_.resize(num_persistent);
  lastStepUsed_.resize(num_persistent);
  passthroughs_.resize(num_persistent);

  checkpointManager_.reset();
  current_step_ = num_persistent;
}

void DataStore::reset_for_backprop()
{
  current_step_ = static_cast<Int>(states_.size());
  fetch_state_data(current_step_ - 1);
  for (auto& dual : duals_) {
    dual = nullptr;
  }
}

void DataStore::vjp(StateBase& state) { state.evaluate_vjp(); }

bool DataStore::is_persistent(Int step) const { return !upstreams_[step].size(); }

void DataStore::reverse_state()
{
  // must erase the final step in the cp manager before we get started
  if (current_step_ == states_.size()) {
    checkpointManager_.erase_step(current_step_ - 1);
  }
  --current_step_;
  if (upstreams_[current_step_].size()) {
    fetch_state_data(current_step_ - 1);
    vjp(*states_[current_step_]);
    clear_usage(current_step_);
    checkpointManager_.erase_step(current_step_ - 1);
  }
}

std::shared_ptr<std::any>& DataStore::any_primal(Int step) { return states_[step]->primal(); }

void printv(const std::vector<Int>& v)
{
  size_t c = 0;
  for (auto s : v) {
    std::cout << c << ":" << s << " ";
    ++c;
  }
  std::cout << std::endl;
}

void printv(const std::vector<StateBase>& v)
{
  size_t c = 0;
  for (auto s : v) {
    std::cout << c << ":" << s.step() << " ";
    ++c;
  }
  std::cout << std::endl;
}

void DataStore::add_state(std::unique_ptr<StateBase> newState, const std::vector<StateBase>& upstreams)
{
  Int step = newState->step();

  states_.emplace_back(std::move(newState));
  duals_.emplace_back(nullptr);
  usageCount_.push_back(0);
  active_.push_back(true);
  lastStepUsed_.push_back(step);
  passthroughs_.push_back({});

  bool persistent = upstreams.size() == 0;
  if (persistent) {
    checkpointManager_.add_checkpoint_and_get_index_to_remove(step, persistent);
  }

  std::vector<Int> upstreamSteps;
  upstreamSteps.reserve(upstreams.size());
  for (auto& u : upstreams) {
    Int upstreamStep = u.step();
    upstreamSteps.push_back(upstreamStep);
  }
  upstreams_.emplace_back(*this, upstreamSteps);

  for (auto& u : upstreams) {
    Int upstreamStep = u.step();
    if (!is_persistent(upstreamStep)) {
      // we are now using this upstream (again), add to count of uses
      usageCount_[upstreamStep]++;

      // check if step fully deleted,
      if (!states_[upstreamStep]->primal()) {
        gretl_assert(usageCount_[upstreamStep] == 1);
        states_[upstreamStep]->primal() = u.primal();
      } else {
        gretl_assert(states_[upstreamStep]->primal() == u.primal());
      }

      // knowing this upstream is used here, push the passthroughs forward from their last known use to the previous
      // step
      Int lastLastStepUsed = std::max(lastStepUsed_[u.step()], u.step() + 1);
      Int upstreamStepPassingThrough = u.step();
      for (Int stepBeingPassedThrough = lastLastStepUsed; stepBeingPassedThrough < step; ++stepBeingPassedThrough) {
        passthroughs_[stepBeingPassedThrough].push_back(upstreamStepPassingThrough);
        if (active_[stepBeingPassedThrough]) {
          usageCount_[upstreamStepPassingThrough]++;
        }
      }
      lastStepUsed_[upstreamStepPassingThrough] = step;
    }
  }

  evals_.emplace_back([=](const UpstreamStates&, DownstreamState&) {
    std::cout << "eval not implemented for step " << current_step_ << std::endl;
    gretl_assert(false);
  });

  vjps_.emplace_back([=](UpstreamStates&, const DownstreamState&) {
    std::cout << "vjp not implemented for step " << current_step_ << std::endl;
    gretl_assert(false);
  });

  gretl_assert(check_validity());

  ++current_step_;
  gretl_assert(current_step_ == states_.size());
  gretl_assert(current_step_ == duals_.size());
  gretl_assert(current_step_ == upstreams_.size());
  gretl_assert(current_step_ == passthroughs_.size());
  gretl_assert(current_step_ == active_.size());
  gretl_assert(current_step_ == usageCount_.size());
  gretl_assert(current_step_ == vjps_.size());
  gretl_assert(current_step_ == lastStepUsed_.size());
}

void DataStore::fetch_state_data(Int stepIndex)
{
  gretl_assert_msg(!stillConstructingGraph_, "not allowed to fetch state before the graph is constructed");
  Int lastCheckpoint = static_cast<Int>(checkpointManager_.last_checkpoint_step());
  gretl_assert_msg(lastCheckpoint <= stepIndex, "last checkpoint cannot be ahead of the currently requested step");
  gretl_assert_msg(state_in_use(lastCheckpoint),
                   "cannot confirm that last checkpointed state is actually currently in memory");
  for_each_active_upstream(this, lastCheckpoint, [&](Int upstream) { gretl_assert(state_in_use(upstream)); });
  for (Int i = lastCheckpoint; i < stepIndex; ++i) {
    Int iEval = i + 1;
    for_each_active_upstream(this, iEval, [&](Int u) {
      gretl_assert_msg(state_in_use(u), "upstream is not in use");
      usageCount_[u]++;
    });
    gretl_assert(!active_[iEval]);
    active_[iEval] = true;

    if (states_[iEval]->primal()) {
      for_each_active_upstream(this, iEval, [&](Int upstream) { gretl_assert(state_in_use(upstream)); });
      erase_step_state_data(iEval);
    } else {
      states_[iEval]->evaluate_forward();
    }

    gretl_assert(check_validity());
  }
}

void DataStore::try_to_free(Int step)
{
  if (states_[step] && states_[step]->data_) {
    if (usageCount_[step] == 0 && !active_[step] && states_[step]->data_.use_count() <= 1) {
      states_[step]->primal() = nullptr;
    }
  }
}

void DataStore::erase_step_state_data(Int step)
{
  if (!is_persistent(step)) {
    size_t stepToErase = checkpointManager_.add_checkpoint_and_get_index_to_remove(step);
    if (checkpointManager_.valid_checkpoint_index(stepToErase)) {
      active_[stepToErase] = false;
      try_to_free(static_cast<Int>(stepToErase));
      for_each_active_upstream(this, stepToErase, [&](Int upstream) {
        gretl_assert(usageCount_[upstream]);
        usageCount_[upstream]--;
        try_to_free(upstream);
      });
    }
  }
  if (!check_validity()) {
    gretl_assert(check_validity());
  }
}

bool DataStore::check_validity() const
{
  bool valid = true;
  // first check that our version of the saved states matches the cp manager
  // we are allowed to be saving an extra step here at the end
  for (size_t i = 0; i < current_step_; ++i) {
    if (active_[i]) {
      bool cp_has_i = false;
      for (auto& cp : checkpointManager_.cps) {
        if (cp.step == i) {
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

  std::vector<Int> my_active_count(states_.size(), 0);
  for (size_t i = 0; i < states_.size(); ++i) {
    if (active_[i]) {
      for_each_active_upstream(this, i, [&](Int u) { my_active_count[u]++; });
    }
  }
  for (size_t i = 0; i < states_.size(); ++i) {
    if (my_active_count[i] > 0 && !states_[i]->primal()) {
      gretl::print("step", i, "has an active count, but is deallocated");
      valid = false;
    }
    if (my_active_count[i] != usageCount_[i]) {
      gretl::print("step", i, "usage count =", usageCount_[i], " graph usage count=", my_active_count[i]);
      valid = false;
    }
    if (usageCount_[i] == 0 && !active_[i] && states_[i]->primal()) {
      // print_graph();
      if (states_[i]->data_.use_count() == 1) {
        gretl::print("step", i, "has a no usage count, but is still allocated");
        valid = false;
        print_graph();
      }
    }
  }

  return valid;
}

void DataStore::print_graph() const
{
  for (Int i = 0; i < states_.size(); ++i) {
    std::cout << i << ", act: " << std::setw(3) << active_[i] << ":" << std::setw(3) << usageCount_[i] << ":"
              << std::setw(3) << states_[i]->data_.use_count() << ":" << std::setw(3)
              << (states_[i]->primal() != nullptr) << ",    ups: ";
    for (auto& v : upstreams_[i].states()) {
      std::cout << v.step_ << " ";
    }
    std::cout << ", pass: ";
    for (auto& v : passthroughs_[i]) {
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }
  // std::cout << checkpointManager_ << std::endl;
}

}  // namespace gretl
