
#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <any>
#include "checkpoint.hpp"

#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace gretl {

/**
 * @brief Return string of given parameter's type
 * @tparam T the type to get a string name for
 * @param[in] var the variable to get the type of
 * @return string representation of the type
 */
template <typename T>
std::string typeString(T& var)
{
  // Remove reference, but keep the const/volatile qualifiers.
  const char* name = typeid(var).name();
#ifdef __GNUG__
  int status = -4;  // Arbitrary value to eliminate the compiler warning
  char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string result((status == 0) ? demangled : name);
  std::free(demangled);
  if constexpr (std::is_const_v<T>) {
    result = "const " + result;
  }
  return result;
#else
  // Return name if compiler doesn't support GNU's extensions (most do)
  return name;
#endif
}

// Base case for the recursive variadic macro
inline void print() {
  std::cout << std::endl;
}

// Recursive case for the variadic macro
template <typename T, typename... Args>
void print(T value, Args... args) {
    // Find the current variable name in the comma-separated string
    std::cout << value << " ";
    print(args...); // Recurse for remaining arguments
}

// Base case for the recursive variadic macro
inline void printt() {
  std::cout << std::endl;
}

// Recursive case for the variadic macro
template <typename T, typename... Args>
void printt(T value, Args... args) {
    // Find the current variable name in the comma-separated string
    std::cout << value << " (" << typeString(value) << ") ";
    printt(args...); // Recurse for remaining arguments
}

// Base case for the recursive variadic macro
inline void printtype() {
  std::cout << std::endl;
}

// Recursive case for the variadic macro
template <typename T, typename... Args>
void printtype(T value, Args... args) {
    // Find the current variable name in the comma-separated string
    std::cout << typeString(value) << " ";
    printtype(args...); // Recurse for remaining arguments
}

using Int = unsigned int;

struct StateBase;

template <typename T, typename D = T>
struct State;

struct UpstreamStates;

struct DownstreamState;

template <typename T, typename D>
using InitializeZeroDual = std::function<D(const T&)>;

template <typename T>
struct defaultInitializeZeroDual {
  T operator()(const T&) { return T(0.0); }
};

class DataStore {
 public:
  DataStore(size_t maxStates);
  virtual ~DataStore() {}

  /// @brief create a new state in the graph, store it, return it
  template <typename T, typename D = T>
  State<T, D> create_state(const T& t, InitializeZeroDual<T, D> initial_zero_dual = defaultInitializeZeroDual<D>())
  {
    State<T, D> newState(*this, states_.size(), initial_zero_dual);
    add_state(newState, {});
    gretl_assert(current_step_-1 < primals_.size());
    primals_[current_step_-1] = std::make_shared<std::any>(t);
    return newState;
  }

  /// @brief  unwind one step of the graph
  StateBase reverse_state();

  /// @brief unwind the entire graph
  void back_prop();

  /// @brief clear all but persistent state, keeping the graph
  void reset();

  Int num_allocated_states() const;
  Int num_active_states() const;
  Int num_dual_states() const;

  friend struct StateBase;

  template <typename T, typename D>
  friend struct State;

  friend struct UpstreamState;
  friend struct DownstreamState;

 protected:
  // create a new state in the graph, store it, return it
  template <typename T, typename D, typename InitDualFromValue>
  State<T, D> create_empty_state(InitDualFromValue initial_zero_dual, const std::vector<StateBase>& upstreams)
  {
    gretl_assert(!upstreams.empty());
    State<T, D> newState(*this, states_.size(), initial_zero_dual);
    add_state(newState, upstreams);
    return newState;
  }

  /// @brief vjp
  void vjp(StateBase& state);

  /// @brief function for safely adding new states to graph and checkpoint
  virtual void add_state(StateBase newState, const std::vector<StateBase>& upstreams);

  /// @brief method for clearing states in the past that are no longer needed
  virtual void clear_disposable_state() {}

  /// @brief method for fetching states at a particular moment in time
  virtual void fetch_state_data(Int stepIndex);

  using EvalT = std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)>;
  using VjpT = std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)>;

  std::vector<StateBase> states_;

  std::vector<UpstreamStates> upstreams_;
  std::vector<EvalT> evals_;
  std::vector<VjpT> vjps_;

  mutable std::vector<std::shared_ptr<std::any>> primals_;
  mutable std::vector<std::unique_ptr<std::any>> duals_;

  template <typename T>
  const T& get_primal(Int step) const
  {
    gretl_assert(primals_[step]);
    auto tptr = std::any_cast<T>(primals_[step].get());
    gretl_assert(tptr);
    return *tptr;
  }

  template <typename T>
  void set_primal(Int step, const T& t)
  {
    gretl_assert(!primals_[step]);
    primals_[step] = std::make_shared<std::any>(t);
  }

  template <typename D, typename T=D>
  D& get_dual(Int step) const
  {
    print("trying to get upstream dual");
    if (!duals_[step]) {
      print("nope");
      const T& thisPrimal = get_primal<T>(step);
      printtype("primal = ", thisPrimal);
      auto thisState = dynamic_cast<const State<T,D>*>(&states_[step]);
      gretl_assert(thisState);
      duals_[step] = std::make_unique<std::any>(thisState->initialize_zero_dual_(thisPrimal));
    }
    auto dualData = std::any_cast<D>(duals_[step].get());
    std::cout << "at step " << step << " / " << duals_.size() << " getting type = " << typeString(dualData) << std::endl; // << " " << *dualData << std::endl;
    gretl_assert(dualData);
    return *dualData;
  }

  template <typename D>
  void set_dual(Int step, const D& d) const {
    if (!duals_[step]) {
      duals_[step] = std::move(std::make_unique<std::any>(d));
    }
    auto dualData = std::any_cast<D>(duals_[step].get());
    std::cout << "at step " << step << " / " << duals_.size() << " setting dual type = " << typeString(dualData) << std::endl;
    gretl_assert(dualData);
    *dualData = d;
  }

  /// step counter
  Int current_step_;

  // is going forward
  bool goingForward_ = true;
};

class DynamicDataStore : public DataStore {
 public:
  DynamicDataStore(size_t maxStates);
  virtual ~DynamicDataStore() {}

 protected:
  friend struct StateBase;
  friend struct UpstreamState;

  /// @overload
  // virtual void add_state(StateBase& newState) override;

  /// @overload
  // void clear_disposable_state() override;

  /// @overload
  // void fetch_state_data(size_t stepIndex) override;

  /// container which track the states in the graph with allocated data
  CheckpointManager checkpoints;
};

}  // namespace gretl