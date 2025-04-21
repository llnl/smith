// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional_objective.hpp
 *
 * @brief Implements the scalar objective interface using shape aware functional's scalar output capability
 */

#pragma once

#include "serac/physics/scalar_objective.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

namespace serac {

template <int spatial_dim, typename ShapeDispSpace, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class FunctionalObjective;

/**
 * @brief Construct a new FunctionalObjective object
 *
 * @param physics_name A name for the physics module instance
 * @param mesh The serac mesh
 * @param shape_disp_space Shape displacement space
 * @param input_mfem_spaces Vector of finite element spaces which are arguments to the residual
 */
template <int spatial_dim, typename ShapeDispSpace, typename... InputSpaces, int... parameter_indices>
class FunctionalObjective<spatial_dim, ShapeDispSpace, Parameters<InputSpaces...>,
                          std::integer_sequence<int, parameter_indices...>> : public ScalarObjective {
 public:
  using SpacesT = std::vector<const mfem::ParFiniteElementSpace*>;  ///< typedef

  /// @brief construct a center of mass objective
  FunctionalObjective(const std::string& physics_name, std::shared_ptr<Mesh> mesh,
                      const mfem::ParFiniteElementSpace& shape_disp_space, const SpacesT& input_mfem_spaces)
      : ScalarObjective(physics_name), mesh_(mesh)
  {
    std::array<const mfem::ParFiniteElementSpace*, sizeof...(InputSpaces)> mfem_spaces;

    SLIC_ERROR_ROOT_IF(
        sizeof...(InputSpaces) != input_mfem_spaces.size(),
        axom::fmt::format("{} parameter spaces given in the template argument but {} parameter names were supplied.",
                          sizeof...(InputSpaces), input_mfem_spaces.size()));

    if constexpr (sizeof...(InputSpaces) > 0) {
      for_constexpr<sizeof...(InputSpaces)>([&](auto i) { mfem_spaces[i] = input_mfem_spaces[i]; });
    }

    objective_ =
        std::make_unique<ShapeAwareFunctional<ShapeDispSpace, double(InputSpaces...)>>(&shape_disp_space, mfem_spaces);
  }

  /// @brief using
  using FieldPtr = FiniteElementState*;

  /**
   * @brief register a custom domain integral calculation as part of the residual
   *
   * @tparam active_parameters a list of indices, describing which parameters to pass to the q-function
   * @param body_name string specifing the domain to integrate over
   * @param qfunction a callable that returns a tuple of body-force and stress
   */
  template <int... active_parameters, typename FuncOfTimeSpaceAndParams>
  void addBodyIntegral(DependsOn<active_parameters...>, std::string body_name,
                       const FuncOfTimeSpaceAndParams& qfunction)
  {
    objective_->AddDomainIntegral(serac::Dimension<spatial_dim>{}, serac::DependsOn<active_parameters...>{}, qfunction,
                                  mesh_->domain(body_name));
  }

  /// @overload
  virtual double evaluate(double time, double dt, const std::vector<FieldPtr>& fields) const
  {
    dt_ = dt;
    return evaluateObjective(std::make_integer_sequence<int, sizeof...(parameter_indices) + 1>{}, time, fields);
  }

  /// @overload
  virtual mfem::Vector gradient(double time, double dt, const std::vector<FieldPtr>& fields, int direction) const
  {
    dt_ = dt;
    auto grads = gradientEvaluators(std::make_integer_sequence<int, sizeof...(parameter_indices) + 1>{}, time, fields);
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

  /// @brief timestep, this needs to be held here and modified for rate dependent applications
  mutable double dt_ = std::numeric_limits<double>::max();

  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;

  /// @brief scalar output shape aware functional
  std::unique_ptr<ShapeAwareFunctional<ShapeDispSpace, double(InputSpaces...)>> objective_;
};

}  // namespace serac