// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file field_state.hpp
 */

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

/// @brief axpby using State<double> and FieldState
FieldState axpby(const gretl::State<double>& a, const FieldState& x, const gretl::State<double>& b,
                 const FieldState& y);

/// @brief compute the differentiable weighted sum of fields, weighted by both double weights, and also
/// gret::State<double> differentiable weights.  The differentiable_scale_factors are applied to the differentiable
/// weights to enable negation and scalar muliplication of weights.
FieldState weighted_sum(const std::vector<double>& weights, const std::vector<FieldState>& weighted_fields,
                        const std::vector<gretl::State<double>>& differentiable_weights = {},
                        const std::vector<FieldState>& differentiably_weighted_fields = {},
                        const std::vector<double>& differentiable_scale_factors = {});

/// @brief temporary object to register the multiplication of a gretl::State<d  ouble> with a FieldState.  Casts back
struct FieldStateWeightedSum {
  /// @brief construct from double weights, and fields
  FieldStateWeightedSum(const std::vector<double>& w, const std::vector<FieldState>& f) : weights(w), weighted_fields(f)
  {
  }

  /// @brief construct from State<double> weights, and fields
  FieldStateWeightedSum(const std::vector<gretl::State<double>>& w, const std::vector<FieldState>& f)
      : differentiable_weights(w), differentiably_weighted_fields(f), differentiable_scale_factors(w.size(), 1.0)
  {
  }

  /// @brief default copy
  FieldStateWeightedSum(const FieldStateWeightedSum& old) = default;

  /// @brief default assignment
  FieldStateWeightedSum& operator=(const FieldStateWeightedSum& old) = default;

  /// @brief add another weighted sum in place
  FieldStateWeightedSum& operator+=(const FieldStateWeightedSum& b);

  /// @brief subtract another weighted sum in place
  FieldStateWeightedSum& operator-=(const FieldStateWeightedSum& b);

  /// @brief mulitply by a fixed scalar
  FieldStateWeightedSum& operator*=(double weight);

  /// @brief negate
  FieldStateWeightedSum operator-() const;

  std::vector<double> weights;                               ///< non-differentiable weights
  std::vector<FieldState> weighted_fields;                   ///< fields to weight by non-differentiable weights
  std::vector<gretl::State<double>> differentiable_weights;  ///< differentiable weights
  std::vector<FieldState> differentiably_weighted_fields;    ///< fields to weight by differentiable weights
  std::vector<double> differentiable_scale_factors;          ///< flag differentiable weights to be negated

  /// @brief conversion operator to a FieldState
  operator FieldState() const
  {
    return weighted_sum(weights, weighted_fields, differentiable_weights, differentiably_weighted_fields,
                        differentiable_scale_factors);
  }
};

/// @brief multiply scalar by a FieldState to get a temporary FieldStateWeightedSum which can cast back to a FieldState
FieldStateWeightedSum operator*(double a, const FieldState& b);

/// @brief multiply scalar by a FieldState to get a temporary FieldStateWeightedSum which can cast back to a FieldState
FieldStateWeightedSum operator*(const FieldState& b, double a);

/// @brief multiply scalar by a FieldStateWeightedSum to get a temporary FieldStateWeightedSum which can cast back to a
/// FieldState
FieldStateWeightedSum operator*(double a, const FieldStateWeightedSum& b);

/// @brief multiply scalar by a FieldStateWeightedSum to get a temporary FieldStateWeightedSum which can cast back to a
/// FieldState
FieldStateWeightedSum operator*(const FieldStateWeightedSum& b, double a);

/// @brief multiply scalar by a FieldState to get a temporary FieldStateWeightedSum which can cast back to a FieldState
FieldStateWeightedSum operator*(const gretl::State<double>& a, const FieldState& b);

/// @brief multiply scalar by a FieldState to get a temporary FieldStateWeightedSum which can cast back to a FieldState
FieldStateWeightedSum operator*(const FieldState& b, const gretl::State<double>& a);

/// @brief add two FieldState
FieldStateWeightedSum operator+(const FieldState& x, const FieldState& y);

/// @brief subtract two FieldState
FieldStateWeightedSum operator-(const FieldState& x, const FieldState& y);

/// @brief add two FieldStateWeightedSum
FieldStateWeightedSum operator+(const FieldStateWeightedSum& ax, const FieldStateWeightedSum& by);

/// @brief subtract two FieldStateWeightedSum
FieldStateWeightedSum operator-(const FieldStateWeightedSum& ax, const FieldStateWeightedSum& by);

/// @brief add FieldStateWeightedSum and FieldState
FieldStateWeightedSum operator+(const FieldStateWeightedSum& ax, const FieldState& y);

/// @brief add FieldStateWeightedSum and FieldState
FieldStateWeightedSum operator+(const FieldState& y, const FieldStateWeightedSum& ax);

/// @brief subtract FieldState from FieldStateWeightedSum
FieldStateWeightedSum operator-(const FieldStateWeightedSum& ax, const FieldState& by);

/// @brief subtract FieldStateWeightedSum from FieldState
FieldStateWeightedSum operator-(const FieldState& ax, const FieldStateWeightedSum& by);

// TO DO
// Add multplication of WeightedSum by a differentiable State<double> for improve efficiency
// Consider adding divide operators, maybe component-wise things as well

// Utilty functions for quickly getting spaces from FieldStates

/// @brief Get the space from the primal field of a field states
inline const mfem::ParFiniteElementSpace& space(FieldState field) { return field.get()->space(); }

/// @brief Get the spaces from the primal fields of a vector of field states
inline std::vector<const mfem::ParFiniteElementSpace*> spaces(std::vector<FieldState> states)
{
  std::vector<const mfem::ParFiniteElementSpace*> spaces;
  for (auto s : states) {
    spaces.push_back(&s.get()->space());
  }
  return spaces;
};

}  // namespace serac
