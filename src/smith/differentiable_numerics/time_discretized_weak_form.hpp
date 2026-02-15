// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file time_discretized_weak_form.hpp
 *
 * @brief Wraps FunctionalWeakForm to provide TimeInfo (time, dt, cycle) to integrands instead of just time.
 *
 * This class provides a thin wrapper around FunctionalWeakForm that automatically converts the time
 * parameter into a TimeInfo struct containing time, timestep size (dt), and cycle number. This allows
 * physics systems to access timestep information needed for time integration.
 *
 * Key features:
 * - All integrands receive TimeInfo instead of just time
 * - Systems are responsible for manually applying time integration rules
 * - Supports body integrals, boundary integrals, boundary fluxes, and interior boundary integrals
 * - Default behavior passes ALL input fields to integrands (can be overridden with DependsOn)
 */

#pragma once

#include "smith/physics/functional_weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"

namespace smith {

template <int spatial_dim, typename OutputSpace, typename inputs = Parameters<>>
class TimeDiscretizedWeakForm;

/**
 * @brief A time-discretized weak form that provides TimeInfo to integrands.
 *
 * This wraps FunctionalWeakForm to pass TimeInfo (containing time, dt, and cycle) to all
 * integrand functions instead of just the time value. This allows physics models to access
 * timestep information for time integration.
 *
 * @tparam spatial_dim The spatial dimension for the problem.
 * @tparam OutputSpace The output residual for the weak form (test-space).
 * @tparam InputSpaces All the input FiniteElementState fields (trial-spaces).
 */
template <int spatial_dim, typename OutputSpace, typename... InputSpaces>
class TimeDiscretizedWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>>
    : public FunctionalWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>> {
 public:
  using Base = FunctionalWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>>;  ///< Base class alias

  /**
   * @brief Construct a time-discretized weak form.
   * @param physics_name Unique name for this physics module.
   * @param mesh The computational mesh.
   * @param output_mfem_space The test function space (output/residual space).
   * @param input_mfem_spaces Vector of trial function spaces (input field spaces).
   */
  TimeDiscretizedWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh,
                          const mfem::ParFiniteElementSpace& output_mfem_space,
                          const typename Base::SpacesT& input_mfem_spaces)
      : Base(physics_name, mesh, output_mfem_space, input_mfem_spaces)
  {
  }

  /**
   * @brief Add a body integral with TimeInfo.
   *
   * The integrand receives TimeInfo (containing time, dt, cycle) instead of just time.
   * The system is responsible for manually applying time integration rules inside the integrand
   * to compute current state values from the raw field history.
   *
   * @tparam active_parameters Indices of fields this integral depends on.
   * @tparam BodyIntegralType The integrand function type.
   * @param depends_on Dependency specification for which input fields to pass.
   * @param body_name The name of the domain.
   * @param integrand Function with signature (TimeInfo, X, inputs...) -> residual.
   */
  template <int... active_parameters, typename BodyIntegralType>
  void addBodyIntegral(DependsOn<active_parameters...> depends_on, std::string body_name, BodyIntegralType integrand)
  {
    const double* dt = &this->dt_;
    const size_t* cycle = &this->cycle_;
    Base::addBodyIntegral(depends_on, body_name, [dt, cycle, integrand](double t, auto X, auto... inputs) {
      TimeInfo time_info(t, *dt, *cycle);
      return integrand(time_info, X, inputs...);
    });
  }

  /**
   * @brief Add a body integral with TimeInfo (defaults to all input fields).
   * @tparam BodyIntegralType The integrand function type.
   * @param body_name The name of the domain.
   * @param integrand Function with signature (TimeInfo, X, inputs...) -> residual.
   */
  template <typename BodyIntegralType>
  void addBodyIntegral(std::string body_name, BodyIntegralType integrand)
  {
    constexpr int num_inputs = sizeof...(InputSpaces);
    addBodyIntegralWithAllParams(body_name, integrand, std::make_integer_sequence<int, num_inputs>{});
  }

  /**
   * @brief Add a body source (body load) with TimeInfo.
   * @tparam active_parameters Indices of fields this source depends on.
   * @tparam BodyLoadType The load function type.
   * @param depends_on Dependency specification.
   * @param body_name The name of the domain.
   * @param load_function Function with signature (TimeInfo, X, inputs...) -> load vector.
   */
  template <int... active_parameters, typename BodyLoadType>
  void addBodySource(DependsOn<active_parameters...> depends_on, std::string body_name, BodyLoadType load_function)
  {
    addBodyIntegral(depends_on, body_name, [load_function](auto t_info, auto X, auto... inputs) {
      return smith::tuple{-load_function(t_info, get<VALUE>(X), get<VALUE>(inputs)...), smith::zero{}};
    });
  }

  /// @overload - defaults to all parameters
  template <typename BodyLoadType>
  void addBodySource(std::string body_name, BodyLoadType load_function)
  {
    constexpr int num_inputs = sizeof...(InputSpaces);
    addBodySourceWithAllParams(body_name, load_function, std::make_integer_sequence<int, num_inputs>{});
  }

  /**
   * @brief Add a boundary integral with TimeInfo.
   * @tparam active_parameters Indices of fields this integral depends on.
   * @tparam BoundaryIntegralType The boundary integrand function type.
   * @param depends_on Dependency specification.
   * @param boundary_name The name of the boundary.
   * @param integrand Function with signature (TimeInfo, X, inputs...) -> residual.
   */
  template <int... active_parameters, typename BoundaryIntegralType>
  void addBoundaryIntegral(DependsOn<active_parameters...> depends_on, std::string boundary_name,
                           BoundaryIntegralType integrand)
  {
    const double* dt = &this->dt_;
    const size_t* cycle = &this->cycle_;
    Base::addBoundaryIntegral(depends_on, boundary_name, [dt, cycle, integrand](double t, auto X, auto... inputs) {
      TimeInfo time_info(t, *dt, *cycle);
      return integrand(time_info, X, inputs...);
    });
  }

  /// @overload - defaults to all parameters
  template <typename BoundaryIntegralType>
  void addBoundaryIntegral(std::string boundary_name, BoundaryIntegralType integrand)
  {
    constexpr int num_inputs = sizeof...(InputSpaces);
    addBoundaryIntegralWithAllParams(boundary_name, integrand, std::make_integer_sequence<int, num_inputs>{});
  }

  /**
   * @brief Add a boundary flux with TimeInfo.
   * @tparam active_parameters Indices of fields this integral depends on.
   * @tparam BoundaryFluxType The flux function type.
   * @param depends_on Dependency specification.
   * @param boundary_name The name of the boundary.
   * @param flux_function Function with signature (TimeInfo, X, n, inputs...) -> flux.
   */
  template <int... active_parameters, typename BoundaryFluxType>
  void addBoundaryFlux(DependsOn<active_parameters...> depends_on, std::string boundary_name,
                       BoundaryFluxType flux_function)
  {
    const double* dt = &this->dt_;
    const size_t* cycle = &this->cycle_;
    Base::addBoundaryFlux(depends_on, boundary_name,
                          [dt, cycle, flux_function](double t, auto X, auto n, auto... inputs) {
                            TimeInfo time_info(t, *dt, *cycle);
                            return flux_function(time_info, X, n, inputs...);
                          });
  }

  /// @overload - defaults to all parameters
  template <typename BoundaryFluxType>
  void addBoundaryFlux(std::string boundary_name, BoundaryFluxType flux_function)
  {
    constexpr int num_inputs = sizeof...(InputSpaces);
    addBoundaryFluxWithAllParams(boundary_name, flux_function, std::make_integer_sequence<int, num_inputs>{});
  }

  /**
   * @brief Add an interior boundary integral with TimeInfo.
   * @tparam active_parameters Indices of fields this integral depends on.
   * @tparam InteriorIntegralType The integrand function type.
   * @param depends_on Dependency specification.
   * @param interior_name The name of the interior boundary.
   * @param integrand Function with signature (TimeInfo, X, inputs...) -> residual.
   */
  template <int... active_parameters, typename InteriorIntegralType>
  void addInteriorBoundaryIntegral(DependsOn<active_parameters...> depends_on, std::string interior_name,
                                   InteriorIntegralType integrand)
  {
    const double* dt = &this->dt_;
    const size_t* cycle = &this->cycle_;
    Base::addInteriorBoundaryIntegral(depends_on, interior_name,
                                      [dt, cycle, integrand](double t, auto X, auto... inputs) {
                                        TimeInfo time_info(t, *dt, *cycle);
                                        return integrand(time_info, X, inputs...);
                                      });
  }

  /// @overload - defaults to all parameters
  template <typename InteriorIntegralType>
  void addInteriorBoundaryIntegral(std::string interior_name, InteriorIntegralType integrand)
  {
    constexpr int num_inputs = sizeof...(InputSpaces);
    addInteriorBoundaryIntegralWithAllParams(interior_name, integrand, std::make_integer_sequence<int, num_inputs>{});
  }

 private:
  template <typename BodyIntegralType, int... all_params>
  void addBodyIntegralWithAllParams(std::string body_name, BodyIntegralType integrand,
                                    std::integer_sequence<int, all_params...>)
  {
    addBodyIntegral(DependsOn<all_params...>{}, body_name, integrand);
  }

  template <typename BodyLoadType, int... all_params>
  void addBodySourceWithAllParams(std::string body_name, BodyLoadType load_function,
                                  std::integer_sequence<int, all_params...>)
  {
    addBodySource(DependsOn<all_params...>{}, body_name, load_function);
  }

  template <typename BoundaryIntegralType, int... all_params>
  void addBoundaryIntegralWithAllParams(std::string boundary_name, BoundaryIntegralType integrand,
                                        std::integer_sequence<int, all_params...>)
  {
    addBoundaryIntegral(DependsOn<all_params...>{}, boundary_name, integrand);
  }

  template <typename InteriorIntegralType, int... all_params>
  void addInteriorBoundaryIntegralWithAllParams(std::string interior_name, InteriorIntegralType integrand,
                                                std::integer_sequence<int, all_params...>)
  {
    addInteriorBoundaryIntegral(DependsOn<all_params...>{}, interior_name, integrand);
  }

  template <typename BoundaryFluxType, int... all_params>
  void addBoundaryFluxWithAllParams(std::string boundary_name, BoundaryFluxType flux_function,
                                    std::integer_sequence<int, all_params...>)
  {
    addBoundaryFlux(DependsOn<all_params...>{}, boundary_name, flux_function);
  }
};

}  // namespace smith
