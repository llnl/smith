#pragma once

#include <vector>
#include "state_data.hpp"

namespace gretl {

struct StateBase {
  StateBase() {}
  StateBase(const StateBase&) = default;
  StateBase& operator=(const StateBase&) = default;
  virtual ~StateBase() = default;

  template <typename T>
  const T& get() const
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T>>(stateData);
    assert(typedStateData);
    return typedStateData->get_primal();
  }

  template <typename T>
  const T& get_dual() const
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T>>(stateData);
    assert(typedStateData);
    return typedStateData->get_dual();
  }

  template <typename T>
  void set(const T& t)
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T>>(stateData);
    assert(typedStateData);
    typedStateData->set(t);
  }

  template <typename T>
  void set(T&& t)
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T>>(stateData);
    assert(typedStateData);
    typedStateData->set(std::move(t));
  }

  template <typename T>
  void set_dual(const T& t)
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T>>(stateData);
    assert(typedStateData);
    typedStateData->set_dual(t);
  }

  template <typename T>
  void set_dual(T&& t)
  {
    auto typedStateData = std::dynamic_pointer_cast<StateData<T>>(stateData);
    assert(typedStateData);
    typedStateData->set_dual(std::move(t));
  }

  template <typename T, typename ZeroCopyT>
  State<T> create_state(const std::vector<StateBase>& upstreams, const ZeroCopyT zero_clone)
  {
    return stateData->dataStore.create_empty_state(zero_clone, upstreams);
  }

  template <typename S>
  State<S> create_state(const std::vector<StateBase>& upstreams) const
  {
    auto zero_clone = [](const S&) -> S { return S(); };
    return stateData->dataStore.create_empty_state<S>(zero_clone, upstreams);
  }

  friend class DataStore;
  friend class StateDataBase;

 protected:
  size_t step_index() const;
  void clear();
  void clear_dual();

  void evaluate_and_remove_disposable(double costFactor);
  void evaluate_vjp();

  std::shared_ptr<StateDataBase> stateData;
};

}  // namespace gretl