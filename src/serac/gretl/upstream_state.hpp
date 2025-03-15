#pragma once

#include "state_data_base.hpp"

namespace gretl {

struct UpstreamState {
  UpstreamState(const std::shared_ptr<StateDataBase>& s) : state(s) {}

  bool valid() const { return state->primal_active(); }
  bool dual_valid() const { return state->dual_active(); }

  template <typename T>
  const T& get() const
  {
    return state->template get<T>();
  }
  template <typename T>
  T& get_dual()
  {
    return state->template get_dual<T>();
  }
  template <typename T>
  void set_dual(const T& t)
  {
    return state->set_dual(t);
  }
  template <typename T>
  void set_dual(T&& t)
  {
    return state->set_dual(std::move(t));
  }

  friend class DataStore;

 private:
  std::shared_ptr<StateDataBase> state;
};

struct DownstreamState {
  DownstreamState(StateDataBase& s) : state(s) {}

  bool dual_valid() const { return state.dual_active(); }

  template <typename T>
  void set(const T& t)
  {
    return state.set_primal(t);
  }
  template <typename T>
  void set(T&& t)
  {
    return state.set_primal_move(std::move(t));
  }
  template <typename T>
  const T& get() const
  {
    return state.template get<T>();
  }
  template <typename T>
  const T& get_dual() const
  {
    return state.template get_dual<T>();
  }

  friend class DataStore;

 private:
  StateDataBase& state;
};

}  // namespace gretl