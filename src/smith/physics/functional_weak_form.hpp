// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional_weak_form.hpp
 *
 * @brief Implements the WeakForm interface using smith::ShapeAwareFunctional.
 * Allows for generic specification of body and boundary integrals
 */

#pragma once

#include "smith/physics/weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/numerics/functional/shape_aware_functional.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_dual.hpp"

namespace smith {

template <int spatial_dim, typename OutputSpace, typename inputs = Parameters<>,
          typename input_indices = std::make_integer_sequence<int, inputs::n>>
class FunctionalWeakForm;

/**
 * @brief A nonlinear WeakForm class implemented using smith::Functional
 *
 * This uses Functional to compute fairly general residuals and tangent
 * stiffness matrices based on body and boundary weak form integrals.
 *
 */
template <int spatial_dim, typename OutputSpace, typename... InputSpaces, int... input_indices>
class FunctionalWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>,
                         std::integer_sequence<int, input_indices...>> : public WeakForm {
 public:
  using SpacesT = std::vector<const mfem::ParFiniteElementSpace*>;  ///< typedef

  using ShapeDispSpace = H1<SHAPE_ORDER, spatial_dim>;  ///< typedef

  /**
   * @brief Construct a new FunctionalWeakForm object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The Smith mesh
   * @param output_mfem_space Test space
   * @param input_mfem_spaces Vector of finite element spaces which are arguments to the residual
   */
  FunctionalWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh,
                     const mfem::ParFiniteElementSpace& output_mfem_space, const SpacesT& input_mfem_spaces)
      : WeakForm(physics_name), mesh_(mesh)
  {
    std::array<const mfem::ParFiniteElementSpace*, sizeof...(InputSpaces)> trial_spaces;
    std::array<const mfem::ParFiniteElementSpace*, sizeof...(InputSpaces) + 1> vector_residual_trial_spaces{
        &output_mfem_space};

    SLIC_ERROR_ROOT_IF(
        sizeof...(InputSpaces) != input_mfem_spaces.size(),
        std::format("{} parameter spaces given in the template argument but {} input mfem spaces were supplied.",
                    sizeof...(InputSpaces), input_mfem_spaces.size()));

    // Validate output space
    validateSpace<OutputSpace>(output_mfem_space, "output");

    // Validate input spaces
    if constexpr (sizeof...(InputSpaces) > 0) {
      // Validate each input space using a helper
      validateInputSpaces<0>(input_mfem_spaces);

      for_constexpr<sizeof...(InputSpaces)>([&](auto i) { trial_spaces[i] = input_mfem_spaces[i]; });
      for_constexpr<sizeof...(InputSpaces)>(
          [&](auto i) { vector_residual_trial_spaces[i + 1] = input_mfem_spaces[i]; });
    }

    const auto& shape_disp_space = mesh->shapeDisplacementSpace();

    weak_form_ = std::make_unique<ShapeAwareFunctional<ShapeDispSpace, OutputSpace(InputSpaces...)>>(
        &shape_disp_space, &output_mfem_space, trial_spaces);

    v_dot_weak_form_residual_ =
        std::make_unique<ShapeAwareFunctional<ShapeDispSpace, double(OutputSpace, InputSpaces...)>>(
            &shape_disp_space, vector_residual_trial_spaces);
  }

  /**
   * @brief Add a body integral contribution to the residual
   *
   * // DependsOn<active_parameters...> can be indices into fields which the body integral may depend on
   * @tparam BodyIntegralType The type of the body integral
   * @param body_name The name of the registered domain over which the body integrals are evaluated.
   * @param integrand A function describing the body force applied.  Our convention for the sign of the residual
   * vector is that it is expected to be a 'negative force', so the mass terms show up with a positive sign in the
   * residual.  This also ensures that the Jacobian of the residual is positive definite for most physics.  A body
   * integrand involving 'right hand side' contributions like a body load, should be supplied by the user with a
   * negative sign.
   * @pre integrand must be a object that can be called with the following arguments:
   *    1. `double t` the time
   *    2. `tuple{tensor<T,dim>, isoparametric derivative} X` the spatial coordinates for the quadrature point and the
   * coordinate's isoparametric derivative.
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and spatial derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
  template <int... active_parameters, typename BodyIntegralType>
  void addBodyIntegral(DependsOn<active_parameters...>, std::string body_name, BodyIntegralType integrand)
  {
    const double* dt = &dt_;
    const size_t* cycle = &cycle_;
    weak_form_->AddDomainIntegral(
        Dimension<spatial_dim>{}, DependsOn<active_parameters...>{},
        [dt, cycle, integrand](double time, auto X, auto... inputs) {
          return invokeTimeAwareIntegrand(integrand, time, *dt, *cycle, X, inputs...);
        },
        mesh_->domain(body_name));

    v_dot_weak_form_residual_->AddDomainIntegral(
        Dimension<spatial_dim>{}, DependsOn<0, 1 + active_parameters...>{},
        [dt, cycle, integrand](double time, auto X, auto V, auto... inputs) {
          auto orig_tuple = invokeTimeAwareIntegrand(integrand, time, *dt, *cycle, X, inputs...);
          return smith::inner(get<VALUE>(V), get<VALUE>(orig_tuple)) +
                 smith::inner(get<DERIVATIVE>(V), get<DERIVATIVE>(orig_tuple));
        },
        mesh_->domain(body_name));
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyIntegral(std::string body_name, BodyForceType body_integral)
  {
    addBodyIntegralWithAllParams(body_name, body_integral, std::make_integer_sequence<int, sizeof...(InputSpaces)>{});
  }

  /**
   * @brief Add a body source (body load) to the weak form
   *
   * @tparam active_parameters Type for indices into fields which the body integral may depend on
   * @tparam BodyLoadType The type of the body load function
   * @param body_name The name of the registered domain over which the body loads are applied.
   * @param depends_on Indices into fields which the body integral may depend on
   * @param load_function A function describing the body force applied.
   * @pre load_function must be a object that can be called with the following arguments:
   *    1. `double t` the time
   *    2. `tensor<T,dim> X` the spatial coordinates for the quadrature point.
   *    3. `value`, a variadic list of field values, one tuple for each of the trial spaces specified in the
   * `DependsOn<...>` argument.
   *    The expected return is the value of the source at X.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
  template <int... active_parameters, typename BodyLoadType>
  void addBodySource(DependsOn<active_parameters...> depends_on, std::string body_name, BodyLoadType load_function)
  {
    addBodyIntegral(depends_on, body_name, [load_function](const TimeInfo& t_info, auto X, auto... inputs) {
      return smith::tuple{-invokeTimeAwareSource(load_function, t_info, get<VALUE>(X), get<VALUE>(inputs)...),
                          smith::zero{}};
    });
  }

  /// @overload
  template <int... active_parameters, typename BodyLoadType>
  void addBodySource(std::string body_name, BodyLoadType load_function)
  {
    return addBodySourceWithAllParams(body_name, load_function,
                                      std::make_integer_sequence<int, sizeof...(InputSpaces)>{});
  }

  /**
   * @brief Add a boundary integral term to the weak form
   *
   * * // DependsOn<active_parameters...> can be indices into fields which the body integral may depend on
   * @tparam BoundaryIntegrandType The type of the boundary integral function.
   * @param boundary_name The name of the registered domain over which the boundary integral is applied.
   * @param integrand A function describing the boundary integral term to include in the weak form.
   * Our convention for the sign of the residual
   * vector is that it is expected to be a 'negative force', so the mass terms show up with a positive sign in the
   * residual.  This also ensures that the Jacobian of the residual is positive definite for most physics.  A body
   * integrand involving 'right hand side' contributions like a body load, should be supplied by the user with a
   * negative sign.  Otherwise, the addBoundaryFlux method can be used.
   * @pre integrand must be a object that can be called with the following arguments:
   *    1. `double t` the time
   *    2. `tuple{tensor<T,dim>, surface isoparametric derivative} X` the spatial coordinates for the quadrature point
   *    3. `tuple{value, surface isoparametric derivative}`, a variadic list of tuples (each with a values and
   * derivative), one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
  template <int... active_parameters, typename BoundaryIntegrandType>
  void addBoundaryIntegral(DependsOn<active_parameters...>, std::string boundary_name, BoundaryIntegrandType integrand)
  {
    const double* dt = &dt_;
    const size_t* cycle = &cycle_;
    weak_form_->AddBoundaryIntegral(
        Dimension<spatial_dim - 1>{}, DependsOn<active_parameters...>{},
        [dt, cycle, integrand](double time, auto X, auto... params) {
          return invokeTimeAwareIntegrand(integrand, time, *dt, *cycle, X, params...);
        },
        mesh_->domain(boundary_name));

    v_dot_weak_form_residual_->AddBoundaryIntegral(
        Dimension<spatial_dim - 1>{}, DependsOn<0, 1 + active_parameters...>{},
        [dt, cycle, integrand](double time, auto X, auto V, auto... params) {
          auto orig_surface_flux = invokeTimeAwareIntegrand(integrand, time, *dt, *cycle, X, params...);
          return smith::inner(get<VALUE>(V), orig_surface_flux);
        },
        mesh_->domain(boundary_name));
  }

  /// @overload
  template <typename BoundaryIntegrandType>
  void addBoundaryIntegral(std::string boundary_name, const BoundaryIntegrandType& integrand)
  {
    addBoundaryIntegralWithAllParams(boundary_name, integrand,
                                     std::make_integer_sequence<int, sizeof...(InputSpaces)>{});
  }

  /**
   * @brief Add a interior boundary integral term to the weak form
   *
   * * // DependsOn<active_parameters...> can be indices into fields which the body integral may depend on
   * @tparam InteriorIntegrandType The type of the interior boundary integral function.
   * @param interior_name The name of the registered domain over which the interior boundary integral is applied.
   * @param integrand A function describing the interior boundary integral term to include in the weak form.
   * @pre integrand must be a object that can be called with the following arguments:
   *    1. `double t` the time
   *    2. `tuple{tensor<T,dim>, surface isoparametric derivative} X` the spatial coordinates for the quadrature point
   *    3. `tuple{value, surface isoparametric derivative}`, a variadic list of tuples (each with a values and
   * derivative), one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
  template <int... active_parameters, typename InteriorIntegrandType>
  void addInteriorBoundaryIntegral(DependsOn<active_parameters...>, std::string interior_name,
                                   InteriorIntegrandType integrand)
  {
    const double* dt = &dt_;
    const size_t* cycle = &cycle_;
    weak_form_->AddInteriorFaceIntegral(
        Dimension<spatial_dim - 1>{}, DependsOn<active_parameters...>{},
        [dt, cycle, integrand](double time, auto X, auto... params) {
          return invokeTimeAwareIntegrand(integrand, time, *dt, *cycle, X, params...);
        },
        mesh_->domain(interior_name));

    v_dot_weak_form_residual_->AddInteriorFaceIntegral(
        Dimension<spatial_dim - 1>{}, DependsOn<0, 1 + active_parameters...>{},
        [dt, cycle, integrand](double time, auto X, auto V, auto... params) {
          auto [V1, V2] = V;
          auto orig_surface_flux = invokeTimeAwareIntegrand(integrand, time, *dt, *cycle, X, params...);
          auto [flux_pos, flux_neg] = orig_surface_flux;
          return smith::inner(V1, flux_pos) + smith::inner(V2, flux_neg);
        },
        mesh_->domain(interior_name));
  }

  /// @overload
  template <typename InteriorIntegrandType>
  void addInteriorBoundaryIntegral(std::string interior_name, const InteriorIntegrandType& integrand)
  {
    addInteriorBoundaryIntegralWithAllParams(interior_name, integrand,
                                             std::make_integer_sequence<int, sizeof...(InputSpaces)>{});
  }

  /**
   * @brief Add a boundary flux term to the weak form
   *
   * @tparam active_parameters Type for indices into fields which the body integral may depend on
   * @tparam BoundaryFluxType The type of the traction load
   * @param depends_on Indices into fields which the body integral may depend on
   * @param boundary_name The name of the registered domain over which the boundary integral is applied.
   * @param flux_function A function describing the outward normal flux applied.
   * @pre flux_function must be a object that can be called with the following arguments:
   *    1. `double t` the time
   *    1. `tensor<T,dim> X` the spatial coordinates for the quadrature point
   *    3. `tensor<T,dim> n` the outward-facing unit normal for the quadrature point
   *    4. `value`, a variadic list of tuples of field values at quadrature points,
   *            one for each of the trial spaces specified in the `DependsOn<...>` argument.
   *   The expected return is the value of the boundary flux oriented in the sense of the outward normal.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
  template <int... active_parameters, typename BoundaryFluxType>
  void addBoundaryFlux(DependsOn<active_parameters...> depends_on, std::string boundary_name,
                       BoundaryFluxType flux_function)
  {
    addBoundaryIntegral(depends_on, boundary_name, [flux_function](const TimeInfo& t_info, auto X, auto... inputs) {
      auto n = cross(get<DERIVATIVE>(X));
      return -invokeTimeAwareFlux(flux_function, t_info, get<VALUE>(X), normalize(n), get<VALUE>(inputs)...);
    });
  }

  /// @overload
  template <typename BoundaryFluxType>
  void addBoundaryFlux(std::string boundary_name, const BoundaryFluxType& integrand)
  {
    addBoundaryFluxWithAllParams(boundary_name, integrand, std::make_integer_sequence<int, sizeof...(InputSpaces)>{});
  }

  /// @overload
  mfem::Vector residual(TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                        [[maybe_unused]] const std::vector<ConstQuadratureFieldPtr>& quad_fields = {}) const override
  {
    validateFields(fields, "residual");
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();
    auto ret = (*weak_form_)(time_info.time(), *shape_disp, *fields[input_indices]...);
    return ret;
  }

  /// @overload
  std::unique_ptr<mfem::HypreParMatrix> jacobian(
      TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
      const std::vector<double>& jacobian_weights,
      [[maybe_unused]] const std::vector<ConstQuadratureFieldPtr>& quad_fields = {}) const override
  {
    validateFields(fields, "jacobian");
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    std::unique_ptr<mfem::HypreParMatrix> J;

    auto addToJ = [&J](double factor, std::unique_ptr<mfem::HypreParMatrix> jac_contrib) {
      if (J) {
        SLIC_ERROR_IF(J->N() != jac_contrib->N(),
                      "Multiple nonzero jacobian weights are being used on inconsistently sized input arguments.");
        SLIC_ERROR_IF(J->M() != jac_contrib->M(),
                      "Multiple nonzero jacobian weights are being used on inconsistently sized input arguments.");
        J->Add(factor, *jac_contrib);
      } else {
        J.reset(jac_contrib.release());
        if (factor != 1.0) (*J) *= factor;
      }
    };

    auto jacs = jacobianFunctions(std::make_integer_sequence<int, sizeof...(input_indices)>{}, time_info.time(),
                                  shape_disp, fields);

    for (size_t input_col = 0; input_col < jacobian_weights.size(); ++input_col) {
      if (jacobian_weights[input_col] != 0.0) {
        auto K = smith::get<DERIVATIVE>(jacs[input_col](time_info.time(), shape_disp, fields));
        addToJ(jacobian_weights[input_col], assemble(K));
      }
    }

    return J;
  }

  /// @overload
  void jvp(TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
           [[maybe_unused]] const std::vector<ConstQuadratureFieldPtr>& quad_fields,
           [[maybe_unused]] ConstFieldPtr v_shape_disp, const std::vector<ConstFieldPtr>& v_fields,
           [[maybe_unused]] const std::vector<ConstQuadratureFieldPtr>& v_quad_fields,
           DualFieldPtr jvp_reaction) const override
  {
    validateFields(fields, "jvp");
    SLIC_ERROR_IF(v_fields.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");

    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    auto jacs = jacobianFunctions(std::make_integer_sequence<int, sizeof...(input_indices)>{}, time_info.time(),
                                  shape_disp, fields);

    *jvp_reaction = 0.0;

    for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
      if (v_fields[input_col] != nullptr) {
        auto K = smith::get<DERIVATIVE>(jacs[input_col](time_info.time(), shape_disp, fields));
        K.AddMult(*v_fields[input_col], *jvp_reaction);
      }
    }
  }

  /// @overload
  void vjp(TimeInfo time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
           [[maybe_unused]] const std::vector<ConstQuadratureFieldPtr>& quad_fields, ConstFieldPtr v_field,
           DualFieldPtr vjp_shape_disp_sensitivity, const std::vector<DualFieldPtr>& vjp_sensitivities,
           [[maybe_unused]] const std::vector<QuadratureFieldPtr>& vjp_quad_field_sensitivities) const override
  {
    validateFields(fields, "vjp");
    SLIC_ERROR_IF(vjp_sensitivities.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");

    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    auto vecJacs = vectorJacobianFunctions(std::make_integer_sequence<int, sizeof...(input_indices)>{},
                                           time_info.time(), shape_disp, v_field, fields);
    {
      auto shape_vjp = smith::get<DERIVATIVE>((*v_dot_weak_form_residual_)(
          DifferentiateWRT<0>{}, time_info.time(), *shape_disp, *v_field, *fields[input_indices]...));
      auto shape_vjp_vector = assemble(shape_vjp);
      *vjp_shape_disp_sensitivity += *shape_vjp_vector;
    }

    for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
      if (vjp_sensitivities[input_col] != nullptr) {
        auto vec_jac = smith::get<DERIVATIVE>(vecJacs[input_col](time_info.time(), shape_disp, v_field, fields));
        auto vec_jac_mfem_vector = assemble(vec_jac);
        *vjp_sensitivities[input_col] += *vec_jac_mfem_vector;
      }
    }
  }

  /// @brief Accessor to get a reference to the underlying ShapeAwareFunctional in case more direct access is needed.
  /// @return Reference to ShapeAwareFunctional instance.
  ShapeAwareFunctional<ShapeDispSpace, OutputSpace(InputSpaces...)>& getShapeAwareResidual() { return *weak_form_; }

  /// @brief Accessor to get a reference to the underlying ShapeAwareFunctional vector-residual in case more direct
  /// access is needed.
  /// @return Reference to ShapeAwareFunctional instance.
  ShapeAwareFunctional<ShapeDispSpace, double(OutputSpace, InputSpaces...)>& getShapeAwareVectorTimesResidual()
  {
    return *v_dot_weak_form_residual_;
  }

 protected:
  template <typename BodyIntegralType, int... all_params>
  void addBodyIntegralWithAllParams(std::string body_name, BodyIntegralType integrand,
                                    std::integer_sequence<int, all_params...>)
  {
    this->template addBodyIntegral<all_params...>(DependsOn<all_params...>{}, body_name, integrand);
  }

  template <typename BodyLoadType, int... all_params>
  void addBodySourceWithAllParams(std::string body_name, BodyLoadType load_function,
                                  std::integer_sequence<int, all_params...>)
  {
    this->template addBodySource<all_params...>(DependsOn<all_params...>{}, body_name, load_function);
  }

  template <typename BoundaryIntegrandType, int... all_params>
  void addBoundaryIntegralWithAllParams(std::string boundary_name, BoundaryIntegrandType integrand,
                                        std::integer_sequence<int, all_params...>)
  {
    this->template addBoundaryIntegral<all_params...>(DependsOn<all_params...>{}, boundary_name, integrand);
  }

  template <typename BoundaryFluxType, int... all_params>
  void addBoundaryFluxWithAllParams(std::string boundary_name, BoundaryFluxType flux_function,
                                    std::integer_sequence<int, all_params...>)
  {
    this->template addBoundaryFlux<all_params...>(DependsOn<all_params...>{}, boundary_name, flux_function);
  }

  template <typename InteriorIntegrandType, int... all_params>
  void addInteriorBoundaryIntegralWithAllParams(std::string interior_name, InteriorIntegrandType integrand,
                                                std::integer_sequence<int, all_params...>)
  {
    this->template addInteriorBoundaryIntegral<all_params...>(DependsOn<all_params...>{}, interior_name, integrand);
  }

  template <typename IntegrandType, typename XType, typename... InputTypes>
  static auto invokeTimeAwareIntegrand(const IntegrandType& integrand, double time, double dt, size_t cycle, XType&& X,
                                       InputTypes&&... inputs)
  {
    if constexpr (requires {
                    integrand(TimeInfo(time, dt, cycle), std::forward<XType>(X), std::forward<InputTypes>(inputs)...);
                  }) {
      return integrand(TimeInfo(time, dt, cycle), std::forward<XType>(X), std::forward<InputTypes>(inputs)...);
    } else {
      return integrand(time, std::forward<XType>(X), std::forward<InputTypes>(inputs)...);
    }
  }

  template <typename LoadFunctionType, typename XType, typename... InputTypes>
  static auto invokeTimeAwareSource(const LoadFunctionType& load_function, const TimeInfo& t_info, XType&& X,
                                    InputTypes&&... inputs)
  {
    if constexpr (requires { load_function(t_info, std::forward<XType>(X), std::forward<InputTypes>(inputs)...); }) {
      return load_function(t_info, std::forward<XType>(X), std::forward<InputTypes>(inputs)...);
    } else {
      return load_function(t_info.time(), std::forward<XType>(X), std::forward<InputTypes>(inputs)...);
    }
  }

  template <typename FluxFunctionType, typename XType, typename NType, typename... InputTypes>
  static auto invokeTimeAwareFlux(const FluxFunctionType& flux_function, const TimeInfo& t_info, XType&& X, NType&& n,
                                  InputTypes&&... inputs)
  {
    if constexpr (requires {
                    flux_function(t_info, std::forward<XType>(X), std::forward<NType>(n),
                                  std::forward<InputTypes>(inputs)...);
                  }) {
      return flux_function(t_info, std::forward<XType>(X), std::forward<NType>(n), std::forward<InputTypes>(inputs)...);
    } else {
      return flux_function(t_info.time(), std::forward<XType>(X), std::forward<NType>(n),
                           std::forward<InputTypes>(inputs)...);
    }
  }

  /// @brief Helper to validate input spaces recursively (for constructor)
  template <size_t I>
  void validateInputSpaces(const SpacesT& input_mfem_spaces) const
  {
    if constexpr (I < sizeof...(InputSpaces)) {
      using Space = typename std::tuple_element<I, std::tuple<InputSpaces...>>::type;
      validateSpace<Space>(*input_mfem_spaces[I], axom::fmt::format("input[{}]", I));
      validateInputSpaces<I + 1>(input_mfem_spaces);
    }
  }

  /// @brief Helper to validate fields passed to residual/jacobian/vjp/jvp
  template <size_t I>
  void validateFieldsRecursive(const std::vector<ConstFieldPtr>& fields, const std::string& method_name) const
  {
    if constexpr (I < sizeof...(InputSpaces)) {
      using Space = typename std::tuple_element<I, std::tuple<InputSpaces...>>::type;
      validateSpace<Space>(fields[I]->space(),
                           axom::fmt::format("{}(): field[{}] ('{}')", method_name, I, fields[I]->name()));
      validateFieldsRecursive<I + 1>(fields, method_name);
    }
  }

  /// @brief Validate that fields vector matches the templated input spaces
  void validateFields(const std::vector<ConstFieldPtr>& fields, const std::string& method_name) const
  {
    SLIC_ERROR_ROOT_IF(fields.size() != sizeof...(InputSpaces),
                       axom::fmt::format("{}(): fields.size()={} but weak form expects {} InputSpaces", method_name,
                                         fields.size(), sizeof...(InputSpaces)));

    if constexpr (sizeof...(InputSpaces) > 0) {
      validateFieldsRecursive<0>(fields, method_name);
    }
  }

  /// @brief Validate that an mfem space matches the template space parameters
  template <typename Space>
  static void validateSpace(const mfem::ParFiniteElementSpace& mfem_space, const std::string& space_name)
  {
    const auto* fec = mfem_space.FEColl();

    // Note: We check using the name string rather than dynamic_cast because MFEM
    // has multiple FECollection types that represent H1 and L2 spaces
    if constexpr (Space::family == Family::H1) {
      std::string fec_name = fec->Name();
      bool is_h1 =
          (fec_name.find("H1") != std::string::npos || fec_name.find("ND_") != std::string::npos ||  // Nedelec elements
           fec_name.find("Linear") != std::string::npos);
      SLIC_ERROR_ROOT_IF(!is_h1, axom::fmt::format("Space '{}': Template specifies H1 family but mfem space uses '{}'",
                                                   space_name, fec_name));
    } else if constexpr (Space::family == Family::L2) {
      std::string fec_name = fec->Name();
      bool is_l2 = (fec_name.find("L2") != std::string::npos || fec_name.find("DG") != std::string::npos ||
                    fec_name.find("Const") != std::string::npos);
      SLIC_ERROR_ROOT_IF(
          !is_l2, axom::fmt::format("Space '{}': Template specifies L2/DG family but mfem space uses '{}'", space_name,
                                    fec_name));
    }

    // Validate order
    SLIC_ERROR_ROOT_IF(fec->GetOrder() != Space::order,
                       axom::fmt::format("Space '{}': Template specifies order {} but mfem space has order {}",
                                         space_name, Space::order, fec->GetOrder()));

    // Validate vector dimension
    SLIC_ERROR_ROOT_IF(
        mfem_space.GetVDim() != Space::components,
        axom::fmt::format("Space '{}': Template specifies {} components but mfem space has {} components (VDim)",
                          space_name, Space::components, mfem_space.GetVDim()));
  }

  /// @brief Utility to get array of jacobian functions, one for each input field in fs
  template <int... i>
  auto jacobianFunctions(std::integer_sequence<int, i...>, double time, ConstFieldPtr shape_disp,
                         const std::vector<ConstFieldPtr>& fs) const
  {
    using JacFuncType = std::function<decltype((*weak_form_)(DifferentiateWRT<1>{}, time, *shape_disp, *fs[i]...))(
        double, ConstFieldPtr, const std::vector<ConstFieldPtr>&)>;
    return std::array<JacFuncType, sizeof...(i)>{
        [this](double _time, ConstFieldPtr _shape_disp, const std::vector<ConstFieldPtr>& _fs) {
          return (*weak_form_)(DifferentiateWRT<i + 1>{}, _time, *_shape_disp, *_fs[i]...);
        }...};
  }

  /// @brief Utility to get array of jvp functions, one for each input field in fs
  template <int... i>
  auto vectorJacobianFunctions(std::integer_sequence<int, i...>, double time, ConstFieldPtr shape_disp, ConstFieldPtr v,
                               const std::vector<ConstFieldPtr>& fs) const
  {
    using GradFuncType =
        std::function<decltype((*v_dot_weak_form_residual_)(DifferentiateWRT<1>{}, time, *shape_disp, *v, *fs[i]...))(
            double, ConstFieldPtr, ConstFieldPtr, const std::vector<ConstFieldPtr>&)>;
    return std::array<GradFuncType, sizeof...(i)>{
        [this](double _time, ConstFieldPtr _shape_disp, ConstFieldPtr _v, const std::vector<ConstFieldPtr>& _fs) {
          return (*v_dot_weak_form_residual_)(DifferentiateWRT<i + 2>{}, _time, *_shape_disp, *_v, *_fs[i]...);
        }...};
  }

  /// @brief timestep, this needs to be held here and modified for rate dependent applications
  mutable double dt_ = std::numeric_limits<double>::max();

  /// @brief cycle or step or iteration.  This counter is useful for certain time integrators.
  mutable size_t cycle_ = 0;

  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;

  /// @brief functional residual evaluator, shape aware
  std::unique_ptr<ShapeAwareFunctional<ShapeDispSpace, OutputSpace(InputSpaces...)>> weak_form_;

  /// @brief functional residual times and arbitrary vector v (same space as residual) evaluator, shape aware
  std::unique_ptr<ShapeAwareFunctional<ShapeDispSpace, double(OutputSpace, InputSpaces...)>> v_dot_weak_form_residual_;
};

/// @brief Helper function to construct vector of spaces from an existing vector of FiniteElementState.
/// @param states vector of FiniteElementState
inline std::vector<const mfem::ParFiniteElementSpace*> getSpaces(const std::vector<smith::FiniteElementState>& states)
{
  std::vector<const mfem::ParFiniteElementSpace*> spaces;
  for (auto& f : states) {
    spaces.push_back(&f.space());
  }
  return spaces;
}

}  // namespace smith
