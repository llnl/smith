// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file checkpoint.hpp
 */

#pragma once

#include <set>
#include <map>
#include <ostream>
#include <iostream>
#include <cassert>
#include <limits>

/// @brief gretl_assert that prints line and file info before throwing in release and halting in debug
#define gretl_assert(x)                                                                                          \
  if (!(x))                                                                                                      \
    throw std::runtime_error{"Error on line " + std::to_string(__LINE__) + " in file " + std::string(__FILE__)}; \
  assert(x);

/// @brief gretl_assert_msg that prints message, line and file info before throwing in release and halting in debug
#define gretl_assert_msg(x, msg_name_)                                                                           \
  if (!(x))                                                                                                      \
    throw std::runtime_error{"Error on line " + std::to_string(__LINE__) + " in file " + std::string(__FILE__) + \
                             std::string(", ") + std::string(msg_name_)};                                        \
  assert(x);

namespace gretl {

/// @brief checkpoint struct which tracks level and step per "Minimal Repetition Dynamic Checkpointing Algorithm for
/// Unsteady Adjoint Calculation", Wang, et al. , 2009.
struct Checkpoint {
  size_t level;  ///< level
  size_t step;   ///< step
  static constexpr size_t infinity()
  {
    return std::numeric_limits<size_t>::max();
  }  ///< The largest possible step and level value
};

/// @brief comparison operator between two checkpoints to determine which is most disposable per the dynamic
/// checkpointing algorithm
inline bool operator<(const Checkpoint& a, const Checkpoint& b)
{
  if (a.level == Checkpoint::infinity() && b.level == Checkpoint::infinity()) {
    return a.step > b.step;
  }
  if (a.level == Checkpoint::infinity()) return false;
  if (b.level == Checkpoint::infinity()) return true;
  return a.step > b.step;
}

/// @brief output stream for a single checkpoint
inline std::ostream& operator<<(std::ostream& stream, const Checkpoint& p);

/// @brief CheckpointManager class which encapsulates the logic of when and which steps should be dynamically saved a
/// fetched
struct CheckpointManager {
  static constexpr size_t invalidCheckpointIndex =
      std::numeric_limits<size_t>::max();  ///< magic number of invalid checkpoint

  /// @brief utilty for checking if an index is valid.  There is a magic number, invalidCheckpointIndex, which
  /// represents an invalid checkpoint
  static bool valid_checkpoint_index(size_t i) { return i != invalidCheckpointIndex; }

  /// @brief returns const_iterator to currently most dispensable checkpoint step
  std::set<gretl::Checkpoint>::const_iterator most_dispensable() const
  {
    size_t maxHigherTimeLevel = 0;
    for (auto rIter = cps.begin(); rIter != cps.end(); ++rIter) {
      if (rIter->level < maxHigherTimeLevel) {
        return rIter;
      }
      maxHigherTimeLevel = std::max(rIter->level, maxHigherTimeLevel);
    }
    return cps.end();
  }

  /// @brief this does multiple things
  /// 1. it adds checkpoints into the database, and updates internal data structures
  /// 2. it determines if a checkpoint needs to be removed
  /// 3. if a checkpoint needs to be removed, it returns the index for that checkpoint
  /// 4. otherwise, it returns zero
  size_t add_checkpoint_and_get_index_to_remove(size_t step, bool persistent = false)
  {
    size_t levelupAmount = 1;  //= relativeCost >= 2.0 ? 3 : 1;

    Checkpoint nextStep{.level = levelupAmount - 1, .step = step};

    size_t nextEraseStep = invalidCheckpointIndex;

    // don't include persistent data in data quota.  MRT, this might change
    if (persistent) {
      maxNumStates++;
      nextStep.level = Checkpoint::infinity();
      gretl_assert(cps.size() < maxNumStates);
    }

    if (cps.size() < maxNumStates) {
      cps.insert(nextStep);
    } else {
      auto iterToMostDispensable = most_dispensable();
      if (iterToMostDispensable != cps.end()) {
        nextEraseStep = iterToMostDispensable->step;
        cps.erase(iterToMostDispensable);
        cps.insert(nextStep);
      } else {
        nextEraseStep = cps.begin()->step;
        nextStep.level = cps.begin()->level + levelupAmount;

        cps.erase(cps.begin());
        cps.insert(nextStep);
      }
    }

    return nextEraseStep;
  }

  /// @brief return largest currently checkpointed step
  size_t last_checkpoint_step() const { return cps.begin()->step; }

  /// @brief erase
  bool erase_step(size_t stepIndex)
  {
    for (std::set<Checkpoint>::iterator it = cps.begin(); it != cps.end(); ++it) {
      if (it->step == stepIndex) {
        if (it->level != Checkpoint::infinity()) {
          cps.erase(it);
          return true;
        }
      }
    }
    return false;
  }

  /// @brief check if this step is currently checkpointed. This could potentially use performance optimization down the
  /// way.
  bool contains_step(size_t stepIndex) const
  {
    for (auto& c : cps) {
      if (c.step == stepIndex) {
        return true;
      }
    }
    return false;
  }

  /// @brief erase all non persistent checkpoints
  void reset()
  {
    for (auto cp_it = cps.begin(); cp_it != cps.end(); ++cp_it) {
      if (cp_it->level == Checkpoint::infinity()) {
        cps.erase(cps.begin(), cp_it);
        break;
      }
    }
  }

  size_t maxNumStates = 20;  ///< The max number of non-persistent, not-in-scope states stored by the CheckpointManager
  std::set<Checkpoint> cps;  ///< Vector of checkpoints
};

/// @brief interface to run forward with a linear graph, checkpoint, then automatically backpropagate the sensitivities
/// given the reverse_callback vjp.
/// @tparam T type of each state's data
/// @param numSteps number of forward iterations
/// @param storageSize maximum states to save in memory at a time
/// @param x initial condition
/// @param update_func function which evaluates the forward response
/// @param reverse_callback vjp function (action of Jacobian-transposed) to back propagate sensitivities
/// @return
template <typename T>
T advance_and_reverse_steps(size_t numSteps, size_t storageSize, T x, std::function<T(size_t n, const T&)> update_func,
                            std::function<void(size_t n, const T&)> reverse_callback)
{
  gretl::CheckpointManager cps{.maxNumStates = storageSize, .cps{}};
  std::map<size_t, T> savedCps;
  savedCps[0] = x;

  cps.add_checkpoint_and_get_index_to_remove(0, true);
  for (size_t i = 0; i < numSteps; ++i) {
    x = update_func(i, savedCps[i]);
    size_t eraseStep = cps.add_checkpoint_and_get_index_to_remove(i + 1, false);
    if (cps.valid_checkpoint_index(eraseStep)) {
      savedCps.erase(eraseStep);
    }

    savedCps[i + 1] = x;
  }

  double xf = x;

  for (size_t i = numSteps; i + 1 > 0; --i) {
    while (cps.last_checkpoint_step() < i) {
      size_t lastCp = cps.last_checkpoint_step();
      x = update_func(lastCp, savedCps[lastCp]);
      size_t eraseStep = cps.add_checkpoint_and_get_index_to_remove(lastCp + 1, false);
      if (cps.valid_checkpoint_index(eraseStep)) {
        savedCps.erase(eraseStep);
      }
      savedCps[lastCp + 1] = x;
    }
    reverse_callback(i, savedCps[i]);

    cps.erase_step(i);
    savedCps.erase(i);
  }

  return xf;
}

/// @brief ostream operator for writing out checkpoint information
inline std::ostream& operator<<(std::ostream& stream, const Checkpoint& p)
{
  return stream << "   lvl=" << p.level << ", step=" << p.step;
}

/// @brief ostream operator for writing out information about the entire checkpoint manager to see the set of currently
/// checkpointed states
inline std::ostream& operator<<(std::ostream& stream, const CheckpointManager& set)
{
  stream << "CHECKPOINTS: capacity = " << set.maxNumStates << std::endl;
  for (const auto& s : set.cps) {
    stream << s << "\n";
  }
  return stream;
}

}  // namespace gretl
