// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional_objective.hpp
 *
 * @brief Implements the scalar objective interface using shape aware functional's scalar output capability
 */

#pragma once

#include "smith/physics/scalar_objective.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/numerics/functional/shape_aware_functional.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_dual.hpp"

namespace smith {

template <int spatial_dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class FunctionalObjective;

/**
 * @brief FunctionalObjective object, implements to the ScalarFunctional interface using smith::ShapeAwareFunctional
 */
template <int spatial_dim, typename... InputSpaces, int... parameter_indices>
class FunctionalObjective<spatial_dim, Parameters<InputSpaces...>, std::integer_sequence<int, parameter_indices...>>
    : public ScalarObjective {
 public:
  using SpacesT = std::vector<const mfem::ParFiniteElementSpace*>;  ///< typedef

  using ShapeDispSpace = H1<1, spatial_dim>;  ///< typedef

  /** @brief construct a FunctionalObjective
   * @param physics_name name for the physics module instance
   * @param mesh Smith mesh
   * @param input_mfem_spaces vector of finite element spaces which are arguments to the residual
   */
  FunctionalObjective(const std::string& physics_name, std::shared_ptr<Mesh> mesh, const SpacesT& input_mfem_spaces)
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

    const auto& shape_disp_space = mesh_->shapeDisplacementSpace();

    objective_ =
        std::make_unique<ShapeAwareFunctional<ShapeDispSpace, double(InputSpaces...)>>(&shape_disp_space, mfem_spaces);
  }

  /**
   * @brief register a custom domain integral calculation as part of the residual
   *
   * @tparam active_parameters a list of indices, describing which parameters to pass to the q-function
   * @param body_name string specifying the domain to integrate over
   * @param qfunction a callable that returns a tuple of body-force and stress
   */
  template <int... active_parameters, typename FuncOfTimeSpaceAndParams>
  void addBodyIntegral(DependsOn<active_parameters...>, std::string body_name,
                       const FuncOfTimeSpaceAndParams& qfunction)
  {
    objective_->AddDomainIntegral(smith::Dimension<spatial_dim>{}, smith::DependsOn<active_parameters...>{}, qfunction,
                                  mesh_->domain(body_name));
  }

  /// @overload
  virtual double evaluate(TimeInfo time_info, ConstFieldPtr shape_disp,
                          const std::vector<ConstFieldPtr>& fields) const override
  {
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    return evaluateObjective(std::make_integer_sequence<int, sizeof...(parameter_indices)>{}, time_info.time(),
                             shape_disp, fields);
  }

  /// @overload
  virtual mfem::Vector gradient(TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                                size_t field_ordinal) const override
  {
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    auto grads = gradientEvaluators(std::make_integer_sequence<int, sizeof...(parameter_indices)>{}, time_info.time(),
                                    shape_disp, fields);
    auto g = smith::get<DERIVATIVE>(grads[field_ordinal](time_info.time(), shape_disp, fields));
    return *assemble(g);
  }

  /// @overload
  virtual mfem::Vector mesh_coordinate_gradient(TimeInfo time_info, ConstFieldPtr shape_disp,
                                                const std::vector<ConstFieldPtr>& fields) const override
  {
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    auto g = smith::get<DERIVATIVE>(
        (*objective_)(DifferentiateWRT<0>{}, time_info.time(), *shape_disp, *fields[parameter_indices]...));
    return *assemble(g);
  }

 private:
  /// @brief Utility to evaluate residual using all fields in vector
  template <int... i>
  auto evaluateObjective(std::integer_sequence<int, i...>, double time, ConstFieldPtr shape_disp,
                         const std::vector<ConstFieldPtr>& fs) const
  {
    return (*objective_)(time, *shape_disp, *fs[i]...);
  };

  /// @brief Utility to get array of jacobian functions, one for each input field in fs
  template <int... i>
  auto gradientEvaluators(std::integer_sequence<int, i...>, double time, ConstFieldPtr shape_disp,
                          const std::vector<ConstFieldPtr>& fs) const
  {
    using JacFuncType = std::function<decltype((*objective_)(DifferentiateWRT<1>{}, time, *shape_disp, *fs[i]...))(
        double, ConstFieldPtr, const std::vector<ConstFieldPtr>&)>;
    return std::array<JacFuncType, sizeof...(i)>{
        [=](double _time, ConstFieldPtr _shape_disp, const std::vector<ConstFieldPtr>& _fs) {
          return (*objective_)(DifferentiateWRT<i + 1>{}, _time, *_shape_disp, *_fs[i]...);
        }...};
  };

  /// @brief timestep, this needs to be held here and modified for rate dependent applications.
  mutable double dt_ = std::numeric_limits<double>::max();

  /// @brief cycle or step or iteration.  This counter is useful for certain time integrators.
  mutable size_t cycle_ = 0;

  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;

  /// @brief scalar output shape aware functional
  std::unique_ptr<ShapeAwareFunctional<ShapeDispSpace, double(InputSpaces...)>> objective_;
};

}  // namespace smith
