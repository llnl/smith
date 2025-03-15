#pragma once

#include "data_store.hpp"

namespace gretl {

class UpstreamState;
class DownstreamState;
using UpstreamStates = std::vector<UpstreamState>;

template <typename T>
struct StateData;

struct StateDataBase {
  virtual ~StateDataBase() {}

  StateDataBase(const StateDataBase&) = delete;
  StateDataBase& operator=(const StateDataBase&) = delete;

  template <typename T>
  T& get() const
  {
    auto typedStateData = dynamic_cast<const StateData<T>*>(this);
    assert(typedStateData);
    return typedStateData->get_primal();
  }

  template <typename T>
  T& get_dual() const
  {
    auto typedStateData = dynamic_cast<const StateData<T>*>(this);
    assert(typedStateData);
    return typedStateData->get_dual();
  }

  template <typename T>
  void set_primal(const T& t) const
  {
    auto typedStateData = dynamic_cast<const StateData<T>*>(this);
    assert(typedStateData);
    typedStateData->set_primal(t);
  }

  template <typename T>
  void set_primal_move(T&& t) const
  {
    auto typedStateData = dynamic_cast<const StateData<T>*>(this);
    assert(typedStateData);
    typedStateData->set_primal_move(std::move(t));
  }

  template <typename T>
  void set_dual(const T& t) const
  {
    auto typedStateData = dynamic_cast<const StateData<T>*>(this);
    assert(typedStateData);
    typedStateData->set_dual(t);
  }

  template <typename T>
  void set_dual(T&& t) const
  {
    auto typedStateData = dynamic_cast<const StateData<T>*>(this);
    assert(typedStateData);
    typedStateData->set_dual(std::move(t));
  }

  virtual void evaluate() = 0;
  virtual void evaluate_vjp() = 0;

  virtual void clear_primal() = 0;
  virtual void clear_dual() = 0;

  virtual bool primal_active() const = 0;
  virtual bool dual_active() const = 0;

  bool persistent() const;

  size_t step_index() const { return stepIndex; }
  virtual size_t parent_step_index() const { return 0; }

  virtual std::shared_ptr<StateDataBase> create_ghost(const std::shared_ptr<StateDataBase>& parentGhost,
                                                      const std::shared_ptr<StateDataBase>& lastGhost,
                                                      size_t step) const = 0;

  size_t stepIndex;
  DataStore& dataStore;
  UpstreamStates upstreams;

  float computeCost;
  float memorySize;

  friend class DataStore;

 protected:
  StateDataBase(DataStore& cpd, size_t step, const std::vector<StateBase>& ustreams);
};

}  // namespace gretl