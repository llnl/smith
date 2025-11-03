// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file heat_transfer_weak_form.hpp
 *
 * @brief Implements the WeakForm interface for heat transfer physics.
 * Derives from FunctionalWeakForm.
 */

#pragma once

#include "smith/physics/functional_weak_form.hpp"

namespace smith {

template <int order, int dim, typename InputSpaces = Parameters<>>
class HeatTransferWeakForm;

/**
 * @brief The weak form for heat transfer
 *
 * This uses smith::functional to compute the heat transfer residuals and tangent
 * stiffness matrices.
 *
 * @tparam order The order of the discretization of the temperature and temperature rate
 * @tparam dim The spatial dimension of the mesh
 */
template <int order, int dim, typename... InputSpaces>
class HeatTransferWeakForm<order, dim, Parameters<InputSpaces...>>
    : public FunctionalWeakForm<dim, H1<order>, Parameters<H1<order>, H1<order>, InputSpaces...>> {
 public:
  /// @brief typedef for underlying functional type with templates
  using BaseWeakFormT = FunctionalWeakForm<dim, H1<order>, Parameters<H1<order>, H1<order>, InputSpaces...>>;

  // /// @brief a container holding quadrature point data of the specified type
  // /// @tparam T the type of data to store at each quadrature point
  // template <typename T>
  // using qdata_type = std::shared_ptr<QuadratureData<T>>;

  /// @brief temperature, temperature rate
  static constexpr int NUM_STATE_VARS = 2;

  /// @brief enumeration of the required heat transfer states
  enum STATE
  {
    TEMPERATURE,
    TEMPERATURE_RATE,
    NUM_STATES
  };

  /**
   * @brief Construct a new HeatTransferWeakForm object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The Smith Mesh
   * @param test_space Test space
   * @param parameter_fe_spaces Vector of parameters spaces
   */
  HeatTransferWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh,
                       const mfem::ParFiniteElementSpace& test_space,
                       std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces = {})
      : BaseWeakFormT(physics_name, mesh, test_space, constructAllSpaces(test_space, parameter_fe_spaces))
  {
  }

  /**
   * @brief Set the thermal material model for the physics module
   *
   * @tparam MaterialType The thermal material type
   * @param body_name string name for a registered body Domain on the mesh
   * @param material A material that provides a function to evaluate heat capacity and thermal flux
   * @pre material must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial position of the material evaluation call
   *    2. `T temperature` the current temperature at the quadrature point
   *    3. `tensor<T,dim>` the spatial gradient of the temperature at the quadrature point
   *    4. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @pre MaterialType must return a smith::tuple of volumetric heat capacity and thermal flux when operator() is called
   * with the arguments listed above.
   *
   */
  template <int... active_parameters, typename MaterialType>
  void setMaterial(DependsOn<active_parameters...>, std::string body_name, const MaterialType& material)
  {
    ThermalMaterialFunctor<MaterialType> material_functor(material);
    BaseWeakFormT::weak_form_->AddDomainIntegral(Dimension<dim>{},
                                                 DependsOn<0, 1, active_parameters + NUM_STATE_VARS...>{},
                                                 std::move(material_functor), BaseWeakFormT::mesh_->domain(body_name));
    BaseWeakFormT::v_dot_weak_form_residual_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 1, 2, active_parameters + 1 + NUM_STATE_VARS...>{},
        [material_functor](double t, auto X, auto V, auto... params) {
          auto flux = material_functor(t, X, params...);
          return smith::inner(get<VALUE>(V), get<VALUE>(flux)) +
                 smith::inner(get<DERIVATIVE>(V), get<DERIVATIVE>(flux));
        },
        BaseWeakFormT::mesh_->domain(body_name));
  }

  /// @overload
  template <typename MaterialType>
  void setMaterial(std::string body_name, const MaterialType& material)
  {
    setMaterial(DependsOn<>{}, body_name, material);
  }

 protected:
  /// @brief For use in the constructor, combined the correct number of state spaces (disp,velo,accel) with the vector
  /// of parameters
  /// @param state_space H1<order> temperature space
  /// @param spaces parameter spaces
  /// @return
  std::vector<const mfem::ParFiniteElementSpace*> constructAllSpaces(
      const mfem::ParFiniteElementSpace& state_space, const std::vector<const mfem::ParFiniteElementSpace*>& spaces)
  {
    std::vector<const mfem::ParFiniteElementSpace*> all_spaces{&state_space, &state_space};
    for (auto& s : spaces) {
      all_spaces.push_back(s);
    }
    return all_spaces;
  }

  /**
   * @brief Functor representing a thermal material's heat capacity and flux.
   */
  template <typename MaterialType>
  struct ThermalMaterialFunctor {
    /**
     * @brief Construct a ThermalMaterialIntegrand functor with material model of type `MaterialType`.
     * @param[in] material A functor representing the material model.  Should be a functor, or a class/struct with
     * public operator() method.  Must NOT be a generic lambda, or Smith will not compile due to static asserts below.
     */
    ThermalMaterialFunctor(MaterialType material) : material_(material) {}

    /// Material model
    MaterialType material_;

    /**
     * @brief Thermal material response call
     *
     * @tparam X Spatial position type
     * @tparam Temperature temperature
     * @tparam dT_dt temperature rate
     * @tparam Params variadic parameters for call
     * @param[in] x spatial position
     * @param[in] temperature temperature
     * @param[in] dtemp_dt temperature rate
     * @param[in] params parameter pack
     * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a linear
     * isotropic material
     */
    template <typename X, typename Temperature, typename dT_dt, typename... Params>
    auto operator()(double /*time*/, X x, Temperature temperature, dT_dt dtemp_dt, Params... params) const
    {
      // Get the value and the gradient from the input tuple
      auto [u, du_dX] = temperature;
      auto du_dt = get<VALUE>(dtemp_dt);
      auto [heat_capacity, heat_flux] = material_(x, u, du_dX, params...);
      return smith::tuple{heat_capacity * du_dt, -heat_flux};
    }
  };
};

}  // namespace smith
