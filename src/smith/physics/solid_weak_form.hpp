// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_weak_form.hpp
 *
 * @brief Implements the WeakForm interface for solid mechanics physics.
 * Derives from FunctionalWeakForm.
 */

#pragma once

#include "smith/physics/functional_weak_form.hpp"

namespace smith {

template <int order, int dim, typename InputSpaces = Parameters<>>
class SolidWeakForm;

/**
 * @brief The weak form for solid mechanics
 *
 * This uses smith::unctional to compute the solid mechanics residuals and tangent
 * stiffness matrices.
 *
 * @tparam order The order of the discretization of the displacement and velocity fields
 * @tparam dim The spatial dimension of the mesh
 */
template <int order, int dim, typename... InputSpaces>
class SolidWeakForm<order, dim, Parameters<InputSpaces...>>
    : public FunctionalWeakForm<dim, H1<order, dim>,
                                Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>, InputSpaces...>> {
 public:
  /// @brief typedef for underlying functional type with templates
  using BaseWeakFormT = FunctionalWeakForm<dim, H1<order, dim>,
                                           Parameters<H1<order, dim>, H1<order, dim>, H1<order, dim>, InputSpaces...>>;

  /// @brief a container holding quadrature point data of the specified type
  /// @tparam T the type of data to store at each quadrature point
  template <typename T>
  using qdata_type = std::shared_ptr<QuadratureData<T>>;

  /// @brief disp, velo, accel
  static constexpr int NUM_STATE_VARS = 3;

  /// @brief enumeration of the required states
  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION,
    NUM_STATES
  };

  /**
   * @brief Construct a new SolidWeakForm object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The Smith Mesh
   * @param test_space Test space
   * @param parameter_fe_spaces Vector of parameters spaces
   */
  SolidWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh, const mfem::ParFiniteElementSpace& test_space,
                std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces = {})
      : BaseWeakFormT(physics_name, mesh, test_space, constructAllSpaces(test_space, parameter_fe_spaces))
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
    BaseWeakFormT::weak_form_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 2, active_parameters + NUM_STATE_VARS...>{}, std::move(material_functor),
        BaseWeakFormT::mesh_->domain(body_name), qdata);

    BaseWeakFormT::v_dot_weak_form_residual_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 1, 3, active_parameters + 1 + NUM_STATE_VARS...>{},
        [material_functor](double t, auto X, auto state, auto V, auto... params) {
          auto flux = material_functor(t, X, state, params...);
          return smith::inner(get<VALUE>(V), get<VALUE>(flux)) +
                 smith::inner(get<DERIVATIVE>(V), get<DERIVATIVE>(flux));
        },
        BaseWeakFormT::mesh_->domain(body_name), qdata);
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
    BaseWeakFormT::weak_form_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{}, std::move(material_functor),
        BaseWeakFormT::mesh_->domain(body_name), qdata);

    BaseWeakFormT::v_dot_weak_form_residual_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 1, 2, 3, active_parameters + 1 + NUM_STATE_VARS...>{},
        [material_functor, qdata](double t, auto X, auto state, auto V, auto... params) {
          auto flux = material_functor(t, X, state, params...);
          return smith::inner(get<VALUE>(V), get<VALUE>(flux)) +
                 smith::inner(get<DERIVATIVE>(V), get<DERIVATIVE>(flux));
        },
        BaseWeakFormT::mesh_->domain(body_name), qdata);
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
    BaseWeakFormT::weak_form_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, active_parameters + NUM_STATE_VARS...>{},
        [pressure_function](double t, auto X, auto displacement, auto... params) {
          // Calculate the position and normal in the shape perturbed deformed configuration
          auto x = X + displacement;
          auto n = cross(get<DERIVATIVE>(x));
          // smith::Functional's boundary integrals multiply the q-function output by
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
        BaseWeakFormT::mesh_->domain(boundary_name));

    BaseWeakFormT::v_dot_weak_form_residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, 1, active_parameters + 1 + NUM_STATE_VARS...>{},
        [pressure_function](double t, auto X, auto V, auto displacement, auto... params) {
          auto x = X + displacement;
          auto n = cross(get<DERIVATIVE>(x));
          auto pressure = pressure_function(t, get<VALUE>(X), params...) * (n / norm(cross(get<DERIVATIVE>(X))));
          return inner(get<VALUE>(V), pressure);
        },
        BaseWeakFormT::mesh_->domain(boundary_name));
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
    auto SMITH_HOST_DEVICE operator()(double, X, State& state, Displacement displacement, Acceleration acceleration,
                                      Params... params) const
    {
      auto du_dX = get<DERIVATIVE>(displacement);
      auto d2u_dt2 = get<VALUE>(acceleration);
      auto stress = material_.pkStress(state, du_dX, params...);
      return smith::tuple{material_.density(params...) * d2u_dt2, stress};
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
    auto SMITH_HOST_DEVICE operator()(double /*t*/, X, State& state, Displacement displacement, Velocity velocity,
                                      Acceleration acceleration, Params... params) const
    {
      auto du_dX = get<DERIVATIVE>(displacement);
      auto dv_dX = get<DERIVATIVE>(velocity);
      auto d2u_dt2 = get<VALUE>(acceleration);
      auto stress = material_.pkStress(*dt_, state, du_dX, dv_dX, params...);
      return smith::tuple{material_.density(params...) * d2u_dt2, stress};
    }
  };
};

/**
 * @brief Utility function for creating a shared_ptr<SolidWeakForm<>>
 */
template <int order, int dim, typename... ParameterSpaces>
auto create_solid_weak_form(const std::string& physics_name, std::shared_ptr<smith::Mesh> mesh,
                            const std::vector<const smith::FiniteElementState*>& states,  // u, v, a, e
                            const std::vector<const smith::FiniteElementState*>& params)
{
  /// Local enum to better document the expected indexing order to states
  enum FieldNumbering
  {
    DISP,
    VELO,
    ACCEL,
  };

  std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces;

  if constexpr (sizeof...(ParameterSpaces) > 0) {
    for_constexpr<sizeof...(ParameterSpaces)>([&](auto i) { parameter_fe_spaces.push_back(&params[i]->space()); });
  }

  using WeakFormT = SolidWeakForm<order, dim, Parameters<ParameterSpaces...>>;

  return std::make_shared<WeakFormT>(physics_name, mesh, states[DISP]->space(), parameter_fe_spaces);
}

}  // namespace smith
