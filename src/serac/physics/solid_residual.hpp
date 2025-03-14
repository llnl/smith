// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_residual.hpp
 *
 * @brief Implements the residual interface for solid mechanics physics.
 * Derives from functional_residual.
 */

#pragma once

#include "serac/physics/functional_residual.hpp"

namespace serac {

template <int order, int dim, typename InputSpaces = Parameters<>>
class SolidResidual;

/**
 * @brief Nonlinear residual class for solid mechanics
 *
 * This uses Functional to compute the solid mechanics residuals and tangent
 * stiffness matrices.
 *
 * @tparam order The order of the discretization of the displacement and velocity fields
 * @tparam dim The spatial dimension of the mesh
 */
template <int order, int dim, typename... InputSpaces>
class SolidResidual<order, dim, Parameters<InputSpaces...>>
    : public FunctionalResidual<dim, H1<order, dim>, H1<order, dim>,
                                Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>, InputSpaces...>> {
 public:
  /// @brief typedef for underlying functional type with templates
  using BaseResidualT = FunctionalResidual<dim, H1<order, dim>, H1<order, dim>,
                                           Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>, InputSpaces...>>;

  /// @brief a container holding quadrature point data of the specified type
  /// @tparam T the type of data to store at each quadrature point
  template <typename T>
  using qdata_type = std::shared_ptr<QuadratureData<T>>;

  /// @brief disp, velo, accel
  static constexpr int NUM_STATE_VARS = 3;

  /**
   * @brief Construct a new SolidResidual object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The serac Mesh
   * @param shape_disp_space Shape displacement space
   * @param test_space Test space
   * @param parameter_fe_spaces Vector of parameters spaces
   */
  SolidResidual(std::string physics_name, std::shared_ptr<Mesh> mesh,
                const mfem::ParFiniteElementSpace& shape_disp_space, const mfem::ParFiniteElementSpace& test_space,
                std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces = {})
      : BaseResidualT(physics_name, mesh, shape_disp_space, test_space,
                      constructAllSpaces(test_space, parameter_fe_spaces))
  {
  }

  /**
   * @brief Set the material stress response and mass properties for the physics module
   *
   * @tparam MaterialType The solid material type
   * @tparam StateType the type that contains the internal variables for MaterialType
   * @param body_name string name for a registered body Domain on the mesh
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
   * @pre MaterialType must have a public method `density` which can take FiniteElementState parameter inputs
   * @pre MaterialType must have a public method 'pkStress' which returns the first Piola-Kirchhoff stress
   *
   */
  template <int... active_parameters, typename MaterialType, typename StateType = Empty>
  void setMaterial(DependsOn<active_parameters...>, std::string body_name, const MaterialType& material,
                   qdata_type<StateType> qdata = EmptyQData)
  {
    static_assert(std::is_same_v<StateType, Empty> || std::is_same_v<StateType, typename MaterialType::State>,
                  "invalid quadrature data provided in setMaterial()");
    MaterialStressFunctor<MaterialType> material_functor(material);
    BaseResidualT::residual_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 2, active_parameters + NUM_STATE_VARS...>{}, std::move(material_functor),
        BaseResidualT::mesh_->domain(body_name), qdata);
  }

  /// @overload
  template <typename MaterialType, typename StateType = Empty>
  void setMaterial(std::string body_name, const MaterialType& material,
                   std::shared_ptr<QuadratureData<StateType>> qdata = EmptyQData)
  {
    setMaterial(DependsOn<>{}, body_name, material, qdata);
  }

  /**
   * @brief Set the material stress response and mass properties for the physics module
   *
   * @tparam MaterialType The solid material type
   * @tparam StateType the type that contains the internal variables for MaterialType
   * @param body_name string name for a registered domain on the mesh
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
   * @pre MaterialType must have a public method `density` which can take FiniteElementState parameter inputs
   * @pre MaterialType must have a public method 'pkStress' which returns the first Piola-Kirchhoff stress
   *
   */
  template <int... active_parameters, typename MaterialType, typename StateType = Empty>
  void setRateMaterial(DependsOn<active_parameters...>, std::string body_name, const MaterialType& material,
                       qdata_type<StateType> qdata = EmptyQData)
  {
    static_assert(std::is_same_v<StateType, Empty> || std::is_same_v<StateType, typename MaterialType::State>,
                  "invalid quadrature data provided in setMaterial()");
    RateMaterialStressFunctor<MaterialType> material_functor(material, &this->dt_);
    BaseResidualT::residual_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{}, std::move(material_functor),
        BaseResidualT::mesh_->domain(body_name), qdata);
  }

  /// @overload
  template <typename MaterialType, typename StateType = Empty>
  void setRateMaterial(std::string body_name, const MaterialType& material,
                       std::shared_ptr<QuadratureData<StateType>> qdata = EmptyQData)
  {
    setRateMaterial(DependsOn<>{}, body_name, material, qdata);
  }

  /**
   * @brief Set the pressure boundary condition
   *
   * @tparam active_parameters the indices for active non-state params (i.e., indexing starts just after disp, velo, or
   * accel)
   * @tparam PressureType The type of the pressure load
   * @param boundary_name string, name of boundary domain
   * @param pressure_function A function describing the pressure applied to a boundary
   * used.
   * @pre PressureType must be a object that can be called with the following arguments:
   *    1. `double t` the time (note: time will be handled differently in the future)
   *    2. `tensor<T,dim> x` the reference configuration spatial coordinates for the quadrature point
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @note This pressure is applied in the deformed (current) configuration if GeometricNonlinearities are on.
   */
  template <int... active_parameters, typename PressureType>
  void addPressure(DependsOn<active_parameters...>, std::string boundary_name, PressureType pressure_function)
  {
    BaseResidualT::residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, active_parameters + NUM_STATE_VARS...>{},
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
          return pressure_function(t, get<VALUE>(X), params...) * (n / norm(cross(get<DERIVATIVE>(X))));
        },
        BaseResidualT::mesh_->domain(boundary_name));
  }

  /// @overload
  template <typename PressureType>
  void addPressure(std::string boundary_name, PressureType pressure_function)
  {
    addPressure(DependsOn<>{}, boundary_name, pressure_function);
  }

 protected:
  /// @brief For use in the constructor, combined the correct number of state spaces (disp,velo,accel) with the vector
  /// of parameters
  /// @param state_space H1<order,dim> displacement space
  /// @param spaces parameter spaces
  /// @return
  std::vector<const mfem::ParFiniteElementSpace*> constructAllSpaces(
      const mfem::ParFiniteElementSpace& state_space, const std::vector<const mfem::ParFiniteElementSpace*>& spaces)
  {
    std::vector<const mfem::ParFiniteElementSpace*> all_spaces{&state_space, &state_space, &state_space};
    for (auto& s : spaces) {
      all_spaces.push_back(s);
    }
    return all_spaces;
  }

<<<<<<< HEAD
=======
  /// @overload
  void jvp([[maybe_unused]] double time, [[maybe_unused]] const std::vector<FieldPtr>& fields,
           [[maybe_unused]] const std::vector<FieldPtr>& vFields,
           [[maybe_unused]] std::vector<DualFieldPtr>& jvpReactions) const override
  {
    SLIC_ERROR_IF(vFields.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_IF(jvpReactions.size() != 1, "Solid mechanics nonlinear system only supports 1 output residual");

    auto fields_in_residual_order = reorder_fields(fields);
    auto jacs = jacobianFunctions(std::make_integer_sequence<int, sizeof...(parameter_indices) + NUM_FIELD_OFFSET>{},
                                  time, fields_in_residual_order);

    *jvpReactions[0] = 0.0;
    serac::FiniteElementDual tmp = *jvpReactions[0];

    for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
      if (vFields[input_col] != nullptr) {
        auto K = serac::get<DERIVATIVE>(jacs[residual_index(input_col)](time, fields_in_residual_order));
        tmp = 0.0;
        K.Mult(*vFields[input_col], tmp);
        *jvpReactions[0] += tmp;
      }
    }
    return;
  }

  /// @overload
  void vjp([[maybe_unused]] double time, [[maybe_unused]] const std::vector<FieldPtr>& fields,
           [[maybe_unused]] const std::vector<DualFieldPtr>& vReactions,
           [[maybe_unused]] std::vector<FieldPtr>& vjpFields) const override
  {
    SLIC_ERROR_IF(vjpFields.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_IF(vReactions.size() != 1, "Solid mechanics nonlinear system only supports 1 output residual");

    auto fields_in_residual_order = reorder_fields(fields);
    auto jacs = jacobianFunctions(std::make_integer_sequence<int, sizeof...(parameter_indices) + NUM_FIELD_OFFSET>{},
                                  time, fields_in_residual_order);

    for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
      auto K = serac::get<DERIVATIVE>(jacs[residual_index(input_col)](time, fields_in_residual_order));
      std::unique_ptr<mfem::HypreParMatrix> J = assemble(K);
      J->MultTranspose(1.0, *vReactions[0], 1.0, *vjpFields[input_col]);
    }

    return;
  }

 private:
  /// @brief Utility to evaluate residual using all fields in vector
  template <int... i>
  auto evaluateResidual(std::integer_sequence<int, i...>, double time, const std::vector<FieldPtr>& fs) const
  {
    return (*residual_)(time, *fs[i]...);
  };

  /// @brief Utility to get array of jacobian functions, one for each input field in fs
  template <int... i>
  auto jacobianFunctions(std::integer_sequence<int, i...>, double time, const std::vector<FieldPtr>& fs) const
  {
    using JacFuncType = std::function<decltype((*residual_)(DifferentiateWRT<1>{}, time, *fs[i]...))(
        double, const std::vector<FieldPtr>&)>;
    return std::array<JacFuncType, sizeof...(i)>{[=](double _time, const std::vector<FieldPtr>& _fs) {
      return (*residual_)(DifferentiateWRT<i>{}, _time, *_fs[i]...);
    }...};
  };

>>>>>>> d44aa2f57 (Rename some things regarding the objective.)
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

      auto stress = material_.pkStress(state, du_dX, params...);

      return serac::tuple{material_.density(params...) * d2u_dt2, stress};
    }
  };

  /**
   * @brief Functor representing a material stress.  A functor is used here instead of an
   * extended, generic lambda for compatibility with NVCC.
   */
  template <typename Material>
  struct RateMaterialStressFunctor {
    /// Constructor for the functor
    RateMaterialStressFunctor(Material material, const double* dt) : material_(material), dt_(dt) {}

    /// Material model
    Material material_;

    /// Time step
    const double* dt_;

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
     * @param[in] velocity velocity
     * @param[in] acceleration acceleration
     * @param[in] params parameter pack
     * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a linear
     * isotropic material
     */
    template <typename X, typename State, typename Displacement, typename Velocity, typename Acceleration,
              typename... Params>
    auto SERAC_HOST_DEVICE operator()(double /*t*/, X, State& state, Displacement displacement, Velocity velocity,
                                      Acceleration acceleration, Params... params) const
    {
      auto du_dX = get<DERIVATIVE>(displacement);
      auto dv_dX = get<DERIVATIVE>(velocity);
      auto d2u_dt2 = get<VALUE>(acceleration);

      auto stress = material_.pkStress(*dt_, state, du_dX, dv_dX, params...);

      return serac::tuple{material_.density(params...) * d2u_dt2, stress};
    }
  };
};

}  // namespace serac
