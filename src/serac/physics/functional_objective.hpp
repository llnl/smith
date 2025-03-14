// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional_objective.hpp
 *
 * @brief Implements the objective interface for some average motion constraints
 */

#pragma once

#include "serac/physics/objective.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/common.hpp"

namespace mfem {
class Vector;
class HypreParMatrix;
}  // namespace mfem

namespace serac {

template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class FunctionalObjective;

/**
 * @brief Center of mass constraint
 *
 * @tparam order The order of the discretization of the displacement and velocity fields
 * @tparam dim The spatial dimension of the mesh
 */
template <int order, int dim, typename... parameter_space, int... parameter_indices>
class FunctionalObjective<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>
    : public Objective {
 public:
  static constexpr auto NUM_STATE_VARS = 0;  ///< _
  static constexpr auto NUM_PRE_PARAMS = 1;  ///< shape_displacement
  static constexpr auto NUM_FIELD_OFFSET =
      NUM_STATE_VARS + NUM_PRE_PARAMS;  ///< sum of num states and num preset params

  /// @brief construct a center of mass objective
  FunctionalObjective(const mfem::ParFiniteElementSpace& mfem_shape_disp_space,
                      std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces = {},
                      std::vector<std::string> parameter_names = {})
  {
    std::array<const mfem::ParFiniteElementSpace*, NUM_STATE_VARS + sizeof...(parameter_space)> mfem_spaces;

    SLIC_ERROR_ROOT_IF(
        sizeof...(parameter_space) != parameter_names.size(),
        axom::fmt::format("{} parameter spaces given in the template argument but {} parameter names were supplied.",
                          sizeof...(parameter_space), parameter_names.size()));

    if constexpr (sizeof...(parameter_space) > 0) {
      for_constexpr<sizeof...(parameter_space)>(
          [&](auto i) { mfem_spaces[i + NUM_STATE_VARS] = parameter_fe_spaces[i]; });
    }

    objective_ = std::make_unique<ShapeAwareFunctional<shape_space, double(parameter_space...)>>(&mfem_shape_disp_space,
                                                                                                 mfem_spaces);
  }

  /// @brief using
  using FieldPtr = FiniteElementState*;

  /**
   * @brief register a custom domain integral calculation as part of the residual
   *
   * @tparam active_parameters a list of indices, describing which parameters to pass to the q-function
   * @param domain which elements should evaluate the provided qfunction
   * @param qfunction a callable that returns a tuple of body-force and stress
   */
  template <int... active_parameters, typename FuncOfTimeSpaceAndParams>
  void addDomainIntegral(DependsOn<active_parameters...>, serac::Domain& domain,
                         const FuncOfTimeSpaceAndParams& qfunction)
  {
    objective_->AddDomainIntegral(serac::Dimension<dim>{}, serac::DependsOn<active_parameters...>{}, qfunction, domain);
  }

  /// @overload
  virtual double evaluate(double time, const std::vector<FieldPtr>& fields) const
  {
    return evaluateObjective(std::make_integer_sequence<int, sizeof...(parameter_indices) + NUM_FIELD_OFFSET>{}, time,
                             fields);
  }

  /// @overload
  virtual mfem::Vector gradient(double time, const std::vector<FieldPtr>& fields, int direction) const
  {
    auto grads = gradientEvaluators(std::make_integer_sequence<int, sizeof...(parameter_indices) + NUM_FIELD_OFFSET>{},
                                    time, fields);
    auto g = serac::get<DERIVATIVE>(grads[static_cast<size_t>(direction)](time, fields));
    return *assemble(g);
  }

 private:
  /// @brief Utility to evaluate residual using all fields in vector
  template <int... i>
  auto evaluateObjective(std::integer_sequence<int, i...>, double time, const std::vector<FieldPtr>& fs) const
  {
    return (*objective_)(time, *fs[i]...);
  };

  /// @brief Utility to get array of jacobian functions, one for each input field in fs
  template <int... i>
  auto gradientEvaluators(std::integer_sequence<int, i...>, double time, const std::vector<FieldPtr>& fs) const
  {
    using JacFuncType = std::function<decltype((*objective_)(DifferentiateWRT<1>{}, time, *fs[i]...))(
        double, const std::vector<FieldPtr>&)>;
    return std::array<JacFuncType, sizeof...(i)>{[=](double _time, const std::vector<FieldPtr>& _fs) {
      return (*objective_)(DifferentiateWRT<i>{}, _time, *_fs[i]...);
    }...};
  };

  using shape_space = H1<order, dim>;

  std::unique_ptr<ShapeAwareFunctional<shape_space, double(parameter_space...)>> objective_;
};

}  // namespace serac