#pragma once

#include <functional>
#include <memory>
#include "state_data_base.hpp"
#include "upstream_state.hpp"

namespace gretl {

template <typename T, typename D>
struct StateData : public StateDataBase {
  virtual ~StateData() {}

  void clear_primal() override { p = nullptr; }

  void clear_dual() override { d = nullptr; }

  void evaluate() override
  {
    DownstreamState ds(*this);
    gretl_assert(!primal_active());
    eval(this->upstreams, ds);
  }

  void evaluate_vjp() override { vjp(this->upstreams, DownstreamState(*this)); }

  State<T, D> clone(const std::vector<StateBase>& upstreams_) const
  {
    return dataStore.create_empty_state<T, D>(initialize_zero_dual, upstreams_);
  }

  std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)> eval = [](const UpstreamStates&,
                                                                                              DownstreamState&) {};

  std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)> vjp = [](UpstreamStates&,
                                                                                             const DownstreamState&) {};

  virtual T& get_primal() const
  {
    if (!p) {
      dataStore.fetch_state_data(stepIndex);
    }
    gretl_assert(p);
    return *p;
  }

  std::shared_ptr<T>& primal_ptr() const { return p; }

  virtual D& get_dual() const
  {
    if (!d) {
      d = std::make_unique<D>(initialize_zero_dual(get_primal()));
    }
    gretl_assert(d);
    return *d;
  }

  virtual void set_primal(const T& t) const { p = std::make_shared<T>(t); }

  virtual void set_primal_move(T&& t) const { p = std::make_shared<T>(std::move(t)); }

  virtual void set_dual(const D& t) const { d = std::make_unique<D>(t); }

  virtual void set_dual(D&& t) const { d = std::make_unique<D>(std::move(t)); }

  bool primal_active() const override { return p != nullptr; }

  virtual bool dual_active() const override { return d != nullptr; }

  StateData(DataStore& cpd, size_t step, const InitializeZeroDual<T, D>& zc, const std::vector<StateBase>& ustreams)
      : StateDataBase(cpd, step, ustreams), p(nullptr), d(nullptr), initialize_zero_dual(zc)
  {
  }

  StateData(DataStore& cpd, const T& t, size_t step, const InitializeZeroDual<T, D>& zc,
            const std::vector<StateBase>& ustreams)
      : StateData(cpd, step, zc, ustreams)
  {
    p = std::make_shared<T>(t);
  }

 protected:
  mutable std::shared_ptr<T> p;
  mutable std::unique_ptr<D> d;
  InitializeZeroDual<T, D> initialize_zero_dual;
};

}  // namespace gretl