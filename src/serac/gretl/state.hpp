#pragma once

#include <functional>
#include "state_base.hpp"
#include "upstream_state.hpp"

namespace gretl {

template <typename T>
struct State : public StateBase {
  using type = T;

  void set(const T& t) { get_data().set_primal(t); }

  void set(T&& t) { get_data().set_primal(std::move(t)); }

  void set_dual(const T& t) { get_data().set_dual(t); }

  void set_dual(T&& t) { get_data().set_dual(std::move(t)); }

  const T& get() const { return get_data().get_primal(); }

  const T& get_dual() const { return get_data().get_dual(); }

  State<T> clone(const std::vector<StateBase>& upstreams) const { return get_data().clone(upstreams); }

  void set_eval(const std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)>& e)
  {
    get_data().eval = e;
  }

  void set_vjp(const std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)>& v)
  {
    get_data().vjp = v;
  }

  State<T> finalize(double costFactor = 1.0)
  {
    this->evaluate_and_remove_disposable(costFactor);
    return *this;
  }

  friend class DataStore;

 protected:
  template <typename ZeroCloneFromT>
  State(DataStore& store, const T& t, size_t step, ZeroCloneFromT zero_clone, const std::vector<StateBase>& ustreams)
  {
    stateData = std::make_shared<StateData<T>>(store, t, step, zero_clone, ustreams);
  }

  template <typename ZeroCloneFromT>
  State(DataStore& store, size_t step, ZeroCloneFromT zero_clone, const std::vector<StateBase>& ustreams)
  {
    stateData = std::make_shared<StateData<T>>(store, step, zero_clone, ustreams);
  }

  StateData<T>& get_data() const
  {
    auto typedData = std::dynamic_pointer_cast<StateData<T>>(stateData);
    assert(typedData);
    return *typedData;
  }
};

inline State<double> set_as_objective(gretl::State<double> o)
{
  o.set_dual(1.0);
  return o;
}

}  // namespace gretl