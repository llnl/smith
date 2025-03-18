#pragma once

#include <functional>
#include <memory>
#include "state_data_base.hpp"
#include "upstream_state.hpp"

namespace gretl {

template <typename T, typename D>
struct GhostStateData;

template <typename T, typename D>
struct StateData : public StateDataBase {
  virtual ~StateData() {}

  void clear_primal() override
  {
    if (debugPrint) printf("clearing main primal at step %zu\n", stepIndex);
    p = nullptr;
  }

  void clear_dual() override { d = nullptr; }

  void evaluate() override
  {
    DownstreamState ds(*this);
    assert(!primal_active());
    eval(this->upstreams, ds);
  }

  void evaluate_vjp() override
  {
    // MRT, why not this? assert(dual_active());
    // the vjp for the downstream may not exist
    // but only when that value is never used
    // perhaps try to identify such cases and
    // remove from graph...
    vjp(this->upstreams, DownstreamState(*this));
  }

  State<T,D> clone(const std::vector<StateBase>& upstreams) const
  {
    return dataStore.create_empty_state<T,D>(zero_clone, upstreams);
  }

  std::function<void(const UpstreamStates& upstreams, DownstreamState& downstream)> eval =
      [](const UpstreamStates&, DownstreamState&) {};

  std::function<void(UpstreamStates& upstreams, const DownstreamState& downstream)> vjp =
      [](UpstreamStates&, const DownstreamState&) {};

  virtual T& get_primal() const
  {
    if (!p) {
      dataStore.fetch_state_data(stepIndex);
    }
    assert(p);
    return *p;
  }

  std::shared_ptr<T>& primal_ptr() const { return p; }

  virtual D& get_dual() const
  {
    if (!d) {
      d = std::make_unique<D>(zero_clone(get_primal()));
    }
    assert(d);
    return *d;
  }

  virtual void set_primal(const T& t) const { p = std::make_shared<T>(t); }

  virtual void set_primal_move(T&& t) const { p = std::make_shared<T>(std::move(t)); }

  virtual void set_dual(const D& t) const { d = std::make_unique<D>(t); }

  virtual void set_dual(D&& t) const { d = std::make_unique<D>(std::move(t)); }

  bool primal_active() const override { return p != nullptr; }

  virtual bool dual_active() const override { return d != nullptr; }

  StateData(DataStore& cpd, size_t step, const ZeroClone<T,D>& zc, const std::vector<StateBase>& ustreams)
      : StateDataBase(cpd, step, ustreams), p(nullptr), d(nullptr), zero_clone(zc)
  {
  }

  StateData(DataStore& cpd, const T& t, size_t step, const ZeroClone<T,D>& zc, const std::vector<StateBase>& ustreams)
      : StateData(cpd, step, zc, ustreams)
  {
    p = std::make_shared<T>(t);
  }

  std::shared_ptr<StateDataBase> create_ghost(const std::shared_ptr<StateDataBase>& parentGhost,
                                              const std::shared_ptr<StateDataBase>& lastGhost,
                                              size_t step) const override
  {
    // this is the parent ghost
    assert(parentGhost);
    assert(lastGhost);
    assert(lastGhost->stepIndex >= parentGhost->stepIndex);
    assert(step > parentGhost->stepIndex);
    assert(step == lastGhost->stepIndex + 1);
    auto ghostState = std::make_shared<GhostStateData<T,D>>(dataStore, step, zero_clone, parentGhost, lastGhost);
    return ghostState;
  }

 protected:
  ZeroClone<T,D> zero_clone;
  mutable std::shared_ptr<T> p;

 private:
  mutable std::unique_ptr<D> d;
};

template <typename T,typename D>
struct GhostEvalFunctor {
  void operator()(const UpstreamStates& upstreams, DownstreamState& downstream)
  {
    assert(lastGhost->primal_ptr());
    assert(!nextGhost.primal_ptr());
    nextGhost.primal_ptr() = lastGhost->primal_ptr();
  }

  std::shared_ptr<StateData<T,D>> lastGhost;
  GhostStateData<T,D>& nextGhost;
};

template <typename T, typename D>
struct GhostStateData : public StateData<T,D> {
  template <typename ZeroCloneFromT>
  GhostStateData(DataStore& store, size_t step, ZeroCloneFromT zero_clone, const std::shared_ptr<StateDataBase>& pGhost,
                 const std::shared_ptr<StateDataBase>& lGhost)
      : StateData<T,D>(store, step, zero_clone, {}), parentGhost(std::dynamic_pointer_cast<StateData<T,D>>(pGhost))
  {
    std::shared_ptr<StateData<T,D>> lastGhost = std::dynamic_pointer_cast<StateData<T,D>>(lGhost);
    GhostEvalFunctor<T,D> ghostEval{.lastGhost = lastGhost, .nextGhost = *this};

    this->eval = ghostEval;
    this->vjp = [](UpstreamStates& upstreams, const DownstreamState& downstream) {};

    assert(!this->primal_active());
    DownstreamState ds(*this);
    this->eval(this->upstreams, ds);
  }

  T& get_primal() const override { return StateData<T,D>::get_primal(); }

  T& get_dual() const override { return parentGhost->get_dual(); }

  void set_primal(const T& t) const override
  {
    assert(false);  // should never be setting upstream primals, ghosts can only ever be upstreams
  }

  void set_primal_move(T&& t) const override
  {
    assert(false);  // should never be setting upstream primals, ghosts can only ever be upstreams
  }

  void set_dual(const T& t) const override
  {
    assert(!parentGhost->dual_active());
    parentGhost->set_dual(t);
  }

  void set_dual(T&& t) const override
  {
    assert(!parentGhost->dual_active());
    parentGhost->set_dual(std::move(t));
  }

  bool dual_active() const override { return parentGhost->dual_active(); }

  void clear_primal() override
  {
    if (debugPrint)
      printf("clearing ghost primal at step = %zu, parent = %zu\n", this->stepIndex, parentGhost->stepIndex);
    StateData<T,D>::clear_primal();
  }

  void clear_dual() override
  {
    // ghost are not responsible for dual memory
  }

  size_t parent_step_index() const override { return parentGhost->step_index(); }

 protected:
  std::shared_ptr<StateData<T,D>> parentGhost;
};

}  // namespace gretl