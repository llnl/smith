#pragma once

#include "serac/gretl/data_store.hpp"
#include "serac/gretl/state.hpp"
#include "serac/gretl/create_state.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

using FEFieldPtr = std::shared_ptr<FiniteElementState>;      ///< typedef
using FEDualPtr = std::shared_ptr<FiniteElementDual>;        ///< typedef
using FieldState = gretl::State<FEFieldPtr, FEDualPtr>;      ///< typedef
using ResultantState = gretl::State<FEDualPtr, FEFieldPtr>;  ///< typedef

/// @brief functor which takes a std::shared_ptr<FiniteElementState>, and returns a zero-valued
/// std::shared_ptr<FiniteElementDual> with the same space
struct zero_dual_from_state {
  /// @brief functor operator
  auto operator()(const serac::FEFieldPtr& f) const
  {
    return std::make_shared<serac::FiniteElementDual>(f->space(), f->name() + "_dual");
  };
};

/// @brief functor which takes a std::shared_ptr<FiniteElementDual>, and returns a zero-valued
/// std::shared_ptr<FiniteElementState> with the same space
struct zero_state_from_dual {
  /// @brief functor operator
  auto operator()(const serac::FEDualPtr& f) const
  {
    return std::make_shared<serac::FiniteElementState>(f->space(), f->name() + "_undual");
  };
};

/// @brief initialize on the gretl::DataStore a FieldState with values from s
inline FieldState create_field_state(gretl::DataStore& dataStore, const serac::FEFieldPtr& s)
{
  return dataStore.create_state<serac::FEFieldPtr, serac::FEDualPtr>(s, zero_dual_from_state());
}

/// @brief initialize on the gretl::DataStore a FieldState from a FiniteElementState of given space, name and mesh.
template <typename function_space>
FieldState create_field_state(gretl::DataStore& dataStore, function_space space, const std::string& name,
                              const std::string& mesh_tag)
{
  auto f = std::make_shared<FiniteElementState>(StateManager::newState(space, name, mesh_tag));
  return create_field_state(dataStore, f);
}

/// @brief initialize on the gretl::DataStore a ResultantState with values from s
inline ResultantState create_field_resultant(gretl::DataStore& dataStore, const serac::FEDualPtr& s)
{
  return dataStore.create_state<serac::FEDualPtr, serac::FEFieldPtr>(s, zero_state_from_dual());
}

/// @brief initialize on the gretl::DataStore a ResultantState from a FiniteElementDual of given space, name and mesh.
template <typename function_space>
ResultantState create_field_resultant(gretl::DataStore& dataStore, function_space space, const std::string& name,
                                      const std::string& mesh_tag)
{
  auto f = std::make_shared<FiniteElementDual>(StateManager::newDual(space, name, mesh_tag));
  return create_field_resultant(dataStore, f);
}

/// @brief gretl-function to compute the inner product (vector l2-norm) of state and state
inline FieldState square(const FieldState& state)
{
  using T = FieldState::type;
  using D = FieldState::dual_type;

  auto newState = state.clone(std::vector<gretl::StateBase>{state});

  newState.set_eval([](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    auto s = inputs[0].get<T>();
    auto next = std::make_shared<FiniteElementState>(s->space(), s->name() + "_squared");
    *next = *s;
    *next *= *s;
    output.set<T, D>(next);
  });

  newState.set_vjp([](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    const FiniteElementDual& output_ = *output.get_dual<D, T>();
    const FiniteElementState& input = *inputs[0].get<T>();
    FiniteElementDual& input_ = *inputs[0].get_dual<D, T>();
    int sz = input_.Size();
    for (int i = 0; i < sz; ++i) {
      input_[i] += 2.0 * input[i] * output_[i];
    }
  });

  return newState.finalize();
}

/// @brief gretl-function to compute the inner product (vector l2-norm) of a and b
inline gretl::State<double> inner_product(const FieldState& a, const FieldState& b)
{
  using T = FieldState::type;
  using D = FieldState::dual_type;

  gretl::State<double> c = a.create_state<double>({a, b});

  c.set_eval([](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    auto A = upstreams[0].get<T>();
    auto B = upstreams[1].get<T>();
    double prod = serac::innerProduct(*A, *B);
    downstream.set(prod);
  });

  c.set_vjp([](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const double& Cbar = downstream.get_dual<double, double>();
    auto& a_ = upstreams[0];
    auto& b_ = upstreams[1];

    const FiniteElementState& A = *a_.get<T>();
    const FiniteElementState& B = *b_.get<T>();

    FiniteElementDual& Abar = *a_.get_dual<D, T>();
    Abar.Add(Cbar, B);

    FiniteElementDual& Bbar = *b_.get_dual<D, T>();
    Bbar.Add(Cbar, A);
  });

  return c.finalize();
}

/// @brief gretl-function to compute a*x + b*y
inline FieldState axpby(double a, const FieldState& x, double b, const FieldState& y)
{
  auto z = x.clone({x, y});

  z.set_eval([a, b](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const FEFieldPtr& X = upstreams[0].get<FEFieldPtr>();
    const FEFieldPtr& Y = upstreams[1].get<FEFieldPtr>();
    FEFieldPtr Z = std::make_shared<FiniteElementState>(X->space(), "axpby");
    add(a, *X, b, *Y, *Z);
    downstream.set<FEFieldPtr, FEDualPtr>(Z);
  });

  z.set_vjp([a, b](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const FEDualPtr& Z_dual = downstream.get_dual<FEDualPtr, FEFieldPtr>();
    FEDualPtr& X_dual = upstreams[0].get_dual<FEDualPtr, FEFieldPtr>();
    FEDualPtr& Y_dual = upstreams[1].get_dual<FEDualPtr, FEFieldPtr>();
    add(*X_dual, a, *Z_dual, *X_dual);
    add(*Y_dual, b, *Z_dual, *Y_dual);
  });

  return z.finalize();
}

/// @brief gretl-function to make a deep-copy of a FieldState and initialize it to 0.
inline FieldState zero_copy(const FieldState& x)
{
  auto z = x.clone({x});

  z.set_eval([](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const FEFieldPtr& X = upstreams[0].get<FEFieldPtr>();
    FEFieldPtr Z = std::make_shared<FiniteElementState>(X->space(), "zero");
    *Z *= 0.0;
    downstream.set<FEFieldPtr, FEDualPtr>(Z);
  });

  z.set_vjp([](gretl::UpstreamStates&, const gretl::DownstreamState&) {});

  return z.finalize();
}

/// @brief gretl-function to compute the weighted average a * weight + b * (1-weight)
inline FieldState weighted_average(const FieldState& a, const FieldState& b, double weight)
{
  return axpby(weight, a, 1.0 - weight, b);
}

/// @brief gretl-function to add two FieldState
inline FieldState operator+(const FieldState& a, const FieldState& b) { return axpby(1.0, a, 1.0, b); }

/// @brief gretl-function to subtract two FieldState
inline FieldState operator-(const FieldState& a, const FieldState& b) { return axpby(1.0, a, -1.0, b); }

/// @brief gretl-function to add three FieldState
inline FieldState add(FieldState a, FieldState b, FieldState c)
{
  auto g = clone_state(
      [](const FEFieldPtr& A, const FEFieldPtr& B, const FEFieldPtr& C) -> FEFieldPtr {
        FEFieldPtr D = std::make_shared<FiniteElementState>(A->space(), "d");
        *D = *A;
        *D += *B;
        *D += *C;
        return D;
      },
      [](const FEFieldPtr& /*A*/, const FEFieldPtr& /*B*/, const FEFieldPtr& /*C*/, const FEFieldPtr& /*D*/,
         FEDualPtr& A_dual_v, FEDualPtr& B_dual_v, FEDualPtr& C_dual_v, const FEDualPtr& D_dual_v) -> void {
        *A_dual_v += *D_dual_v;
        *B_dual_v += *D_dual_v;
        *C_dual_v += *D_dual_v;
      },
      a, b, c);

  return g;
}

}  // namespace serac
