// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_residual.hpp
 *
 * @brief Implement the residual interface for solid mechanics physics
 */

#pragma once

#include "serac/physics/residual.hpp"
#include "serac/physics/common.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class SolidResidual;

template <int order, int dim, typename... parameter_space, int... parameter_indices>
class SolidResidual<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>
    : public Residual {
 public:
  static constexpr auto NUM_STATE_VARS = 3;  // displacement, velocity, acceleration
  static constexpr auto NUM_PRE_PARAMS = 1;  // shape_displacement
  static constexpr auto NUM_FIELD_OFFSET = NUM_STATE_VARS + NUM_PRE_PARAMS;

  /// @brief a container holding quadrature point data of the specified type
  /// @tparam T the type of data to store at each quadrature point
  template <typename T>
  using qdata_type = std::shared_ptr<QuadratureData<T>>;

  SolidResidual(std::string physics_name, std::string mesh_tag, const mfem::ParFiniteElementSpace& shape_disp_space,
                const mfem::ParFiniteElementSpace& test_space,
                std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces = {},
                std::vector<std::string> parameter_names = {})
      : Residual(physics_name), mesh_tag_(mesh_tag), mesh_(StateManager::mesh(mesh_tag_))
  {
    std::array<const mfem::ParFiniteElementSpace*, NUM_STATE_VARS + sizeof...(parameter_space)> trial_spaces;
    trial_spaces[0] = &test_space;
    trial_spaces[1] = &test_space;
    trial_spaces[2] = &test_space;

    SLIC_ERROR_ROOT_IF(
        sizeof...(parameter_space) != parameter_names.size(),
        axom::fmt::format("{} parameter spaces given in the template argument but {} parameter names were supplied.",
                          sizeof...(parameter_space), parameter_names.size()));

    if constexpr (sizeof...(parameter_space) > 0) {
      for_constexpr<sizeof...(parameter_space)>(
          [&](auto i) { trial_spaces[i + NUM_STATE_VARS] = parameter_fe_spaces[i]; });
    }

    residual_ = std::make_unique<ShapeAwareFunctional<shape_trial, test(trial, trial, trial, parameter_space...)>>(
        &shape_disp_space, &test_space, trial_spaces);
  }

  /**
   * @brief register a custom boundary integral calculation as part of the residual
   *
   * @tparam active_parameters a list of indices, describing which parameters to pass to the q-function
   * @param qfunction a callable that returns the traction on a boundary surface
   * @param optional_domain The domain over which the boundary integral is evaluated. If nothing is supplied the entire
   * boundary is used.
   * ~~~ {.cpp}
   *
   *  solid_mechanics.addCustomBoundaryIntegral(DependsOn<>{}, [](double t, auto position, auto displacement, auto
   * acceleration, auto shape){ auto [X, dX_dxi] = position;
   *
   *     auto [u, du_dxi] = displacement;
   *     auto f           = u * 3.0 (X[0] < 0.01);
   *     return f;  // define a displacement-proportional traction at a given support
   *  });
   *
   * ~~~
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename callable>
  void addCustomBoundaryIntegral(DependsOn<active_parameters...>, callable qfunction,
                                 const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireBoundary(mesh_);

    residual_->AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0, 1, 2, active_parameters + NUM_FIELD_OFFSET...>{},
                                   qfunction, domain);
  }

  /**
   * @brief Set the material stress response and mass properties for the physics module
   *
   * @tparam MaterialType The solid material type
   * @tparam StateType the type that contains the internal variables for MaterialType
   * @param material A material that provides a function to evaluate stress
   * @pre material must be a object that can be called with the following arguments:
   *    1. `MaterialType::State & state` an mutable reference to the internal variables for this quadrature point
   *    2. `tensor<T,dim,dim> du_dx` the displacement gradient at this quadrature point
   *    3. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @param qdata the buffer of material internal variables at each quadrature point
   *
   * @pre MaterialType must have a public member variable `density`
   * @pre MaterialType must define operator() that returns the Cauchy stress
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename MaterialType, typename StateType = Empty>
  void setMaterial(DependsOn<active_parameters...>, const MaterialType& material, Domain& domain,
                   qdata_type<StateType> qdata = EmptyQData)
  {
    static_assert(std::is_same_v<StateType, Empty> || std::is_same_v<StateType, typename MaterialType::State>,
                  "invalid quadrature data provided in setMaterial()");
    MaterialStressFunctor<MaterialType> material_functor(material);
    residual_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 2, active_parameters + NUM_STATE_VARS...>{},
                                 std::move(material_functor), domain, qdata);
  }

  /// @overload
  template <typename MaterialType, typename StateType = Empty>
  void setMaterial(const MaterialType& material, std::shared_ptr<QuadratureData<StateType>> qdata = EmptyQData)
  {
    setMaterial(DependsOn<>{}, material, qdata);
  }

  /**
   * @brief Set the material stress response and mass properties for the physics module
   *
   * @tparam MaterialType The solid material type
   * @tparam StateType the type that contains the internal variables for MaterialType
   * @param material A material that provides a function to evaluate stress
   * @pre material must be a object that can be called with the following arguments:
   *    1. `MaterialType::State & state` an mutable reference to the internal variables for this quadrature point
   *    2. `tensor<T,dim,dim> du_dx` the displacement gradient at this quadrature point
   *    3. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @param qdata the buffer of material internal variables at each quadrature point
   *
   * @pre MaterialType must have a public member variable `density`
   * @pre MaterialType must define operator() that returns the Cauchy stress
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename MaterialType, typename StateType = Empty>
  void setRateMaterial(DependsOn<active_parameters...>, const MaterialType& material, Domain& domain,
                       qdata_type<StateType> qdata = EmptyQData)
  {
    static_assert(std::is_same_v<StateType, Empty> || std::is_same_v<StateType, typename MaterialType::State>,
                  "invalid quadrature data provided in setMaterial()");
    RateMaterialStressFunctor<MaterialType> material_functor(material);
    residual_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{},
                                 std::move(material_functor), domain, qdata);
  }

  /// @overload
  template <typename MaterialType, typename StateType = Empty>
  void setRateMaterial(const MaterialType& material, std::shared_ptr<QuadratureData<StateType>> qdata = EmptyQData)
  {
    setRateMaterial(DependsOn<>{}, material, qdata);
  }

  /**
   * @brief Functor representing a body force integrand.  A functor is necessary instead
   * of an extended, generic lambda for compatibility with NVCC.
   */
  template <typename BodyForceType>
  struct BodyForceIntegrand {
    /// @brief Body force model
    BodyForceType body_force_;
    /// @brief Constructor for the functor
    BodyForceIntegrand(BodyForceType body_force) : body_force_(body_force) {}

    /**
     * @brief Body force call
     *
     * @tparam T temperature
     * @tparam Position Spatial position type
     * @tparam Displacement displacement
     * @tparam Velocity displacement
     * @tparam Acceleration acceleration
     * @tparam Params variadic parameters for call
     * @param[in] t temperature
     * @param[in] position position
     * @param[in] params parameter pack
     * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a linear
     * isotropic material
     */
    template <typename T, typename Position, typename Displacement, typename Velocity, typename Acceleration,
              typename... Params>
    auto SERAC_HOST_DEVICE operator()(T t, Position position, Displacement, Velocity, Acceleration,
                                      Params... params) const
    {
      return serac::tuple{-1.0 * body_force_(get<VALUE>(position), t, params...), zero{}};
    }
  };

  /**
   * @brief Set the body forcefunction
   *
   * @tparam BodyForceType The type of the body force load
   * @param body_force A function describing the body force applied
   * @param optional_domain The domain over which the body force is applied. If nothing is supplied the entire domain is
   * used.
   * @pre body_force must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `double t` the time (note: time will be handled differently in the future)
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename BodyForceType>
  void addBodyForce(DependsOn<active_parameters...>, BodyForceType body_force,
                    const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireDomain(mesh_);
    residual_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{},
                                 BodyForceIntegrand<BodyForceType>(body_force), domain);
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    addBodyForce(DependsOn<>{}, body_force, optional_domain);
  }

  /**
   * @brief Set the traction boundary condition
   *
   * @tparam TractionType The type of the traction load
   * @param traction_function A function describing the traction applied to a boundary
   * @param optional_domain The domain over which the traction is applied. If nothing is supplied the entire boundary is
   * used.
   * @pre TractionType must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `tensor<T,dim> n` the outward-facing unit normal for the quadrature point
   *    3. `double t` the time (note: time will be handled differently in the future)
   *    4. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @note This traction is applied in the reference (undeformed) configuration.
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename TractionType>
  void setTraction(DependsOn<active_parameters...>, TractionType traction_function,
                   const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireBoundary(mesh_);

    residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, active_parameters + NUM_FIELD_OFFSET...>{},
        [traction_function](double t, auto X, auto /* displacement */, auto... params) {
          auto n = cross(get<DERIVATIVE>(X));
          return -1.0 * traction_function(get<VALUE>(X), normalize(n), t, params...);
        },
        domain);
  }

  /// @overload
  template <typename TractionType>
  void setTraction(TractionType traction_function, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    setTraction(DependsOn<>{}, traction_function, optional_domain);
  }

  /**
   * @brief Set the pressure boundary condition
   *
   * @tparam PressureType The type of the pressure load
   * @param pressure_function A function describing the pressure applied to a boundary
   * @param optional_domain The domain over which the pressure is applied. If nothing is supplied the entire boundary is
   * used.
   * @pre PressureType must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the reference configuration spatial coordinates for the quadrature point
   *    2. `double t` the time (note: time will be handled differently in the future)
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @note This pressure is applied in the deformed (current) configuration if GeometricNonlinearities are on.
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename PressureType>
  void setPressure(DependsOn<active_parameters...>, PressureType pressure_function,
                   const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireBoundary(mesh_);

    residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, active_parameters + NUM_FIELD_OFFSET...>{},
        [pressure_function](double t, auto X, auto displacement, auto... params) {
          // Calculate the position and normal in the shape perturbed deformed configuration
          auto x = X + 0.0 * displacement;

          x = x + displacement;

          auto n = cross(get<DERIVATIVE>(x));

          // serac::Functional's boundary integrals multiply the q-function output by
          // norm(cross(dX_dxi)) at that quadrature point, but if we impose a shape displacement
          // then that weight needs to be corrected. The new weight should be
          // norm(cross(dX_dxi + du_dxi + dp_dxi)) where u is displacement and p is shape displacement. This implies:
          //
          //   pressure * normalize(normal_new) * w_new
          // = pressure * normalize(normal_new) * (w_new / w_old) * w_old
          // = pressure * normalize(normal_new) * (norm(normal_new) / norm(normal_old)) * w_old
          // = pressure * (normal_new / norm(normal_new)) * (norm(normal_new) / norm(normal_old)) * w_old
          // = pressure * (normal_new / norm(normal_old)) * w_old

          // We always query the pressure function in the undeformed configuration
          return pressure_function(get<VALUE>(X), t, params...) * (n / norm(cross(get<DERIVATIVE>(X))));
        },
        domain);
  }

  /// @overload
  template <typename PressureType>
  void setPressure(PressureType pressure_function, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    setPressure(DependsOn<>{}, pressure_function, optional_domain);
  }

  // computes residual outputs
  mfem::Vector residual(double time, const std::vector<FieldPtr>& fields, int block_row = 0) const override
  {
    SLIC_ERROR_IF(block_row != 0, "Invalid block row and column requested in fieldJacobian for SolidResidual");
    // SLIC_ERROR_IF(fields.size()!=4 + sizeof(parameter_indices), "Invalid number of input fields");
    auto& disp = *fields[0];
    auto& velo = *fields[1];
    auto& acceleration = *fields[2];
    auto& shape_disp = *fields[3];

    return (*residual_)(time, shape_disp, disp, velo, acceleration, *fields[parameter_indices + NUM_FIELD_OFFSET]...);
  }

  std::unique_ptr<mfem::HypreParMatrix> jacobian(double time, const std::vector<FieldPtr>& fields,
                                                 const std::vector<double>& argument_tangents,
                                                 int block_row = 0) const override
  {
    SLIC_ERROR_IF(block_row != 0, "Invalid block row and column requested in fieldJacobian for SolidResidual");
    auto& disp = *fields[0];
    auto& velo = *fields[1];
    auto& acceleration = *fields[2];
    auto& shape_disp = *fields[3];

    std::unique_ptr<mfem::HypreParMatrix> J;

    auto addToJ = [&J](double factor, std::unique_ptr<mfem::HypreParMatrix> jac_contrib) {
      if (J) {
        J->Add(factor, *jac_contrib);
      } else {
        J.reset(jac_contrib.release());
        if (factor != 1.0) (*J) *= factor;
      }
    };

    for (size_t col = 0; col < argument_tangents.size(); ++col) {
      if (argument_tangents[col] != 0.0) {
        if (col == 0) {
          auto K = serac::get<DERIVATIVE>((*residual_)(time, shape_disp, differentiate_wrt(disp), velo, acceleration,
                                                       *fields[parameter_indices + NUM_FIELD_OFFSET]...));
          addToJ(argument_tangents[col], assemble(K));
        } else if (col == 1) {
          // MRT: assume not velocity dependence for now
          // auto C = serac::get<DERIVATIVE>((*residual_)(time, shape_disp,
          //                                              disp, differentiate_wrt(velo), acceleration,
          //                                              *fields[parameter_indices+NUM_FIELD_OFFSET].get()...));
          // addToJ(argument_tangents[col], assemble(C));
        } else if (col == 2) {
          auto M = serac::get<DERIVATIVE>((*residual_)(time, shape_disp, disp, velo, differentiate_wrt(acceleration),
                                                       *fields[parameter_indices + NUM_FIELD_OFFSET]...));
          addToJ(argument_tangents[col], assemble(M));
        } else if (col == 3) {
          auto shape_K =
              serac::get<DERIVATIVE>((*residual_)(time, differentiate_wrt(shape_disp), disp, velo, acceleration,
                                                  *fields[parameter_indices + NUM_FIELD_OFFSET]...));
          addToJ(argument_tangents[col], assemble(shape_K));
        }
      }
    }

    return J;
  }

  // computes for each residual output: dr/du * fieldsV + dr/dp * parametersV
  void jvp([[maybe_unused]] double time, [[maybe_unused]] const std::vector<FieldPtr>& fields,
           [[maybe_unused]] const std::vector<FieldPtr>& fieldsV,
           [[maybe_unused]] std::vector<DualFieldPtr>& jacobianVectorProducts) const override
  {
    return;
  }

  // computes for each input field  (dr/du).T * vResidual
  // computes for each input parameter (dr/dp).T * vResidual
  // can early out if the vectors being requested are sized to 0?
  void vjp([[maybe_unused]] double time, [[maybe_unused]] const std::vector<FieldPtr>& fields,
           [[maybe_unused]] const std::vector<DualFieldPtr>& vResiduals,
           [[maybe_unused]] std::vector<DualFieldPtr>& fieldSensitivities) const override
  {
    SLIC_ERROR_IF(fieldSensitivities.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_IF(vResiduals.size() != 1, "Solid mechanics nonlinear system only support 1 output residual");

    auto& disp = *fields[0];
    auto& velo = *fields[1];
    auto& acceleration = *fields[2];
    auto& shape_disp = *fields[3];

    size_t num_fields = fields.size();

    /// @brief to get parameter sensitivities
    std::array<std::function<decltype((*residual_)(DifferentiateWRT<1>{}, time, shape_disp, disp, velo, acceleration,
                                                   *fields[parameter_indices + NUM_FIELD_OFFSET]...))()>,
               sizeof...(parameter_indices)>
        d_residual_d_ = {[&]() {
          return (*residual_)(DifferentiateWRT<parameter_indices + NUM_FIELD_OFFSET>{}, time, shape_disp, disp, velo,
                              acceleration, *fields[parameter_indices + NUM_FIELD_OFFSET]...);
        }...};

    for (size_t col = 0; col < num_fields; ++col) {
      if (col == 0) {
        auto K = serac::get<DERIVATIVE>((*residual_)(time, shape_disp, differentiate_wrt(disp), velo, acceleration,
                                                     *fields[parameter_indices + NUM_FIELD_OFFSET]...));
        std::unique_ptr<mfem::HypreParMatrix> J = assemble(K);
        J->MultTranspose(*vResiduals[0], *fieldSensitivities[col]);
      } else if (col == 1) {
        auto C = serac::get<DERIVATIVE>((*residual_)(time, shape_disp, disp, differentiate_wrt(velo), acceleration,
                                                     *fields[parameter_indices + NUM_FIELD_OFFSET]...));
        std::unique_ptr<mfem::HypreParMatrix> J = assemble(C);
        J->MultTranspose(*vResiduals[0], *fieldSensitivities[col]);
      } else if (col == 2) {
        auto M = serac::get<DERIVATIVE>((*residual_)(time, shape_disp, disp, velo, differentiate_wrt(acceleration),
                                                     *fields[parameter_indices + NUM_FIELD_OFFSET]...));
        std::unique_ptr<mfem::HypreParMatrix> J = assemble(M);
        J->MultTranspose(*vResiduals[0], *fieldSensitivities[col]);
      } else if (col == 3) {
        auto shape_K =
            serac::get<DERIVATIVE>((*residual_)(time, differentiate_wrt(shape_disp), disp, velo, acceleration,
                                                *fields[parameter_indices + NUM_FIELD_OFFSET]...));
        std::unique_ptr<mfem::HypreParMatrix> J = assemble(shape_K);
        J->MultTranspose(*vResiduals[0], *fieldSensitivities[col]);
      } else {
        auto param_K = serac::get<DERIVATIVE>(d_residual_d_[col - 4]());
        std::unique_ptr<mfem::HypreParMatrix> J = assemble(param_K);
        J->MultTranspose(*vResiduals[0], *fieldSensitivities[col]);
      }
    }

    return;
  }

 private:
  /**
   * @brief Functor representing a material stress.  A functor is used here instead of an
   * extended, generic lambda for compatibility with NVCC.
   */
  template <typename Material>
  struct MaterialStressFunctor {
    /// Constructor for the functor
    MaterialStressFunctor(Material material) : material_(material) {}

    /// Material model
    Material material_;

    /**
     * @brief Material stress response call
     *
     * @tparam X Spatial position type
     * @tparam State state
     * @tparam Displacement displacement
     * @tparam Acceleration acceleration
     * @tparam Params variadic parameters for call
     * @param[in] state state
     * @param[in] displacement displacement
     * @param[in] acceleration acceleration
     * @param[in] params parameter pack
     * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a linear
     * isotropic material
     */
    template <typename X, typename State, typename Displacement, typename Acceleration, typename... Params>
    auto SERAC_HOST_DEVICE operator()(double, X, State& state, Displacement displacement, Acceleration acceleration,
                                      Params... params) const
    {
      auto du_dX = get<DERIVATIVE>(displacement);
      auto d2u_dt2 = get<VALUE>(acceleration);

      auto stress = material_(state, du_dX, params...);

      return serac::tuple{material_.density_func(params...) * d2u_dt2, stress};
    }
  };

  /**
   * @brief Functor representing a material stress.  A functor is used here instead of an
   * extended, generic lambda for compatibility with NVCC.
   */
  template <typename Material>
  struct RateMaterialStressFunctor {
    /// Constructor for the functor
    RateMaterialStressFunctor(Material material) : material_(material) {}

    /// Material model
    Material material_;

    /**
     * @brief Material stress response call
     *
     * @tparam X Spatial position type
     * @tparam State state
     * @tparam Displacement displacement
     * @tparam Velocity velocity
     * @tparam Acceleration acceleration
     * @tparam Params variadic parameters for call
     * @param[in] state state
     * @param[in] displacement displacement
     * @param[in] acceleration acceleration
     * @param[in] params parameter pack
     * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a linear
     * isotropic material
     */
    template <typename X, typename State, typename Displacement, typename Velocity, typename Acceleration,
              typename... Params>
    auto SERAC_HOST_DEVICE operator()(double dt, X, State& state, Displacement displacement, Velocity velocity,
                                      Acceleration acceleration, Params... params) const
    {
      auto du_dX = get<DERIVATIVE>(displacement);
      auto dv_dX = get<DERIVATIVE>(velocity);
      auto d2u_dt2 = get<VALUE>(acceleration);

      auto stress = material_(dt, state, du_dX, dv_dX, params...);

      return serac::tuple{material_.density_func(params...) * d2u_dt2, stress};
    }
  };

  using trial = H1<order, dim>;
  using test = H1<order, dim>;
  using shape_trial = H1<order, dim>;

  /// @brief string tag for the mesh
  std::string mesh_tag_;

  /// @brief primary mesh
  mfem::ParMesh& mesh_;

  /// @brief functional
  std::unique_ptr<ShapeAwareFunctional<shape_trial, test(trial, trial, trial, parameter_space...)>> residual_;
};

template <int order, int dim, typename... parameter_space>
auto create_solid_residual(const std::string& physics_name, const serac::Mesh& mesh,
                           const std::vector<std::string>& parameter_names,
                           const std::vector<const serac::FiniteElementState*>& params)
{
  FiniteElementState field(mesh.mfemParMesh(), serac::H1<order, dim>{}, "temporary");

  std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces;
  if constexpr (sizeof...(parameter_space) > 0) {
    for_constexpr<sizeof...(parameter_space)>([&](auto) { parameter_fe_spaces.push_back(&params.back()->space()); });
  }

  using ResidualT = SolidResidual<order, dim, Parameters<parameter_space...>>;

  return std::make_shared<ResidualT>(physics_name, mesh.tag(), field.space(), field.space(), parameter_fe_spaces,
                                     parameter_names);
}

}  // namespace serac