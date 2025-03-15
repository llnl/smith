#include "data_store.hpp"
#include "state.hpp"
#include <iostream>

namespace gretl {

DataStore::DataStore(size_t maxStates) : checkpoints{.maxNumStates = maxStates} {}

void DataStore::back_prop()
{
  for (size_t n = states.size(); n > 0; --n) {
    reverse_state(n - 1);
  }
}

void DataStore::reset()
{
  for (size_t n = states.size(); n > 0; --n) {
    if (checkpoints.erase_step(n)) {
      clear_measure(states[n]);
      clear_measure_dual(states[n]);
    }
  }
}

void DataStore::clear_measure(Measure& measure)
{
  measure.first.clear();
  for (auto&& ghostStateToClear : measure.second) {
    ghostStateToClear->clear_primal();
  }
}

void DataStore::clear_measure_dual(Measure& measure)
{
  measure.first.clear_dual();
  for (auto&& ghostStateToClear : measure.second) {
    ghostStateToClear->clear_dual();
  }
}

void DataStore::vjp_measure(Measure& measure)
{
  measure.first.evaluate_vjp();
  for (auto&& ghost : measure.second) {
    ghost->evaluate_vjp();
  }
}

bool DataStore::measure_primal_active(const Measure& measure) const
{
  // MRT: consider checking all have same active status
  return measure.first.stateData->primal_active();
}

bool DataStore::measure_dual_active(const Measure& measure) const
{
  // MRT: consider checking all have same active status
  return measure.first.stateData->dual_active();
}

// This function should really only be publically available for testing
// back_prop is the desired interface
StateBase DataStore::reverse_state(size_t step)
{
  vjp_measure(states[step]);

  if (checkpoints.erase_step(step)) {
    clear_measure(states[step]);
    clear_measure_dual(states[step]);
  }

  return states[step > 0 ? step - 1 : step].first;  // return step earlier, unless at 0
}

std::pair<bool, size_t> ghost_already_exists(const std::vector<DataStore::GhostState>& ghosts,
                                             const std::shared_ptr<StateDataBase>& parentState)
{
  for (size_t g = 0; g < ghosts.size(); ++g) {
    if (ghosts[g]->parent_step_index() == parentState->step_index()) return std::make_pair(true, g);
  }
  return std::make_pair(false, 0);  // first state is always persistent and never a ghost parent
};

std::shared_ptr<StateDataBase> DataStore::add_ghost_state(const std::shared_ptr<StateDataBase>& parent,
                                                          const std::shared_ptr<StateDataBase>& last, size_t stepToAdd)
{
  auto [ghostExists, ghostIndex] = ghost_already_exists(states[stepToAdd].second, parent);

  if (ghostExists) {
    const auto& existingGhost = states[stepToAdd].second[ghostIndex];
    if (!existingGhost->primal_active()) {
      existingGhost->evaluate();
    }
    return existingGhost;
  }

  std::shared_ptr<StateDataBase> nextUpstreamState = parent->create_ghost(parent, last, stepToAdd);
  states[stepToAdd].second.push_back(nextUpstreamState);
  return nextUpstreamState;
}

void DataStore::add_state(StateBase& newState)
{
  if (newState.stateData->persistent()) {
    if (!states.empty()) {
      assert(states.back().first.stateData->persistent());  // persistents must be added first
    }
  }

  check_consistency();
  states.emplace_back(std::make_pair(newState, std::vector<GhostState>{}));
  check_consistency_except_last(1);

  // loop over upstreams, see if they are not in last step
  // if they are not, go back and add a copy state all the way back from the upstreams
  // initial state
  // future: think about how to tell the most recent time that state was
  // added to not duplicate going back states

  auto& upstreams = newState.stateData->upstreams;

  for (auto& ustate : upstreams) {
    auto& uPtr = ustate.state;
    size_t nextStepIndex = newState.step_index();
    size_t parentStepIndex = uPtr->step_index();
    assert(nextStepIndex > uPtr->step_index());

    if (!uPtr->persistent() && (nextStepIndex != parentStepIndex + 1)) {
      // the upstream is not from the last step (nor persistent)
      // must create ghost states to ensure checkpointed states are always fully reproducable

      bool upstreamIsCp = uPtr->primal_active();
      if (!upstreamIsCp) {
        uPtr->evaluate();
      }

      auto nextUpstreamGhost = add_ghost_state(uPtr, uPtr, parentStepIndex + 1);
      assert(nextUpstreamGhost->primal_active());
      if (!upstreamIsCp) {
        uPtr->clear_primal();
      }

      for (size_t s = parentStepIndex + 2; s < nextStepIndex; ++s) {
        // create a clone of this ghost state at the next step... up until the just before current step
        // then at the end, set upstream's data to be that ghost
        assert(nextUpstreamGhost->primal_active());
        auto lastUpstreamGhost = nextUpstreamGhost;
        nextUpstreamGhost = add_ghost_state(uPtr, lastUpstreamGhost, s);
        assert(nextUpstreamGhost->primal_active());

        assert(checkpoints.contains_step(s - 1) == measure_primal_active(states[s - 1]));
        if (!checkpoints.contains_step(s - 1)) {
          lastUpstreamGhost->clear_primal();
        }
      }

      ustate.state = nextUpstreamGhost;
    }

    assert(uPtr->persistent() || newState.step_index() == uPtr->step_index() + 1);
  }

  check_consistency_except_last(1);

  size_t stateStepAdded = states.size() - 1;
  bool persistent = upstreams.empty();
  nextStepToClear = checkpoints.add_checkpoint_and_get_index_to_remove(stateStepAdded, persistent);
  if (checkpoints.valid_checkpoint_index(nextStepToClear)) {
    assert(nextStepToClear < stateStepAdded);  // MRT this should fail with 1 less
  }
}

void DataStore::fetch_state_data(size_t stepIndex)
{
  // this can be called when an upstream is from a previous step
  // and it happens to not currently be a checkpoint
  // so ensure we dont lose the last known step on the forward pass
  // but, what if user asks for some step in the middle of the reverse pass
  // should we save off the end state step as well?
  Checkpoint secondToLastCp;
  bool savingSecondToLast = false;
  if (checkpoints.last_checkpoint_step() > stepIndex) {
    if (measure_primal_active(states[states.size() - 2])) {
      savingSecondToLast = true;
      secondToLastCp = checkpoints.remove_checkpoint(states.size() - 2);
    }
  }

  // we are trying to get an earlier state.  throw away states until we get back to it
  while (checkpoints.last_checkpoint_step() > stepIndex) {
    size_t lastStep = checkpoints.last_checkpoint_step();
    if (checkpoints.erase_step(lastStep)) {
      clear_measure(states[lastStep]);
    }
  }

  assert(checkpoints.last_checkpoint_step() <= stepIndex);
  assert(stepIndex < states.size());

  while (checkpoints.last_checkpoint_step() < stepIndex) {
    size_t lastCp = checkpoints.last_checkpoint_step();

    nextStepToClear = checkpoints.add_checkpoint_and_get_index_to_remove(lastCp + 1);
    if (checkpoints.valid_checkpoint_index(nextStepToClear)) {
      assert(nextStepToClear <= lastCp);
    }

    auto& measure = states[lastCp + 1];
    for (auto&& subState : measure.second) {
      subState->evaluate();
    }
    measure.first.evaluate_and_remove_disposable(
        1.0);  // this call deletes disposable state, MRT, think about cost factor estimate here?

    if (savingSecondToLast) {
      check_consistency_except_last(2);
    } else {
      check_consistency();
    }
  }

  if (savingSecondToLast) {
    checkpoints.insert_checkpoint(secondToLastCp);
  }
}

size_t DataStore::num_active_states() const
{
  size_t numStates = 0;
  for (auto&& measure : states) {
    numStates += measure_primal_active(measure);
  }
  return numStates;
}

size_t DataStore::num_dual_states() const
{
  size_t numStates = 0;
  for (const auto& measure : states) {
    numStates += measure_dual_active(measure);
  }
  return numStates;
}

void DataStore::clear_disposable_state(double costFactor)
{
  if (checkpoints.valid_checkpoint_index(nextStepToClear)) {
    clear_measure(states[nextStepToClear]);
  }
  nextStepToClear = checkpoints.invalidCheckpointIndex;
}

void DataStore::print_state_info() const
{
  for (const auto& s : states) {
    std::cout << " " << s.first.step_index() << "; active =" << s.first.stateData->primal_active() << std::endl;
    for (const auto& t : s.second) {
      std::cout << "  " << t->step_index() << ", " << t->parent_step_index() << "; active =" << t->primal_active()
                << std::endl;
    }
  }
}

void DataStore::check_consistency() const { check_consistency_except_last(0); }

void DataStore::check_consistency_except_last(size_t fromBack) const
{
  for (size_t i = 0; i < states.size() - fromBack; ++i) {
    const auto& s = states[i];
    const auto& parent = s.first;
    bool cpHasState = checkpoints.contains_step(parent.step_index());
    // EXPECT_EQ(cpHasState, parent.stateData->primal_active()) << "parent step = " << parent.step_index() << "\n" <<
    // checkpoints << std::endl;
    assert(cpHasState == parent.stateData->primal_active());
    for (auto& ghost : s.second) {
      // EXPECT_EQ(cpHasState, ghost->primal_active()) << "parent step = " << parent.step_index() << ", ghost = " <<
      // ghost->parent_step_index() << "\n" << checkpoints << std::endl;
      if (cpHasState != ghost->primal_active()) {
        print_state_info();
      }
      assert(cpHasState == ghost->primal_active());
    }
  }
}

}  // namespace gretl