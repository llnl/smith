// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermomechanics_monolithic.hpp
 *
 * @brief An object containing an monolithic thermal structural solver
 * with operator-split options
 */

#pragma once

#include "mfem.hpp"

#include "smith/physics/base_physics.hpp"
#include "smith/physics/thermomechanics_input.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/numerics/functional/shape_aware_functional.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"

namespace smith {

namespace thermomechanics {

/// @brief the default direct solver option for solving the linear stiffness equations
#ifdef MFEM_USE_STRUMPACK
const smith::LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::Strumpack, .print_level = 0};
#else
const smith::LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::SuperLU, .print_level = 0};
#endif

/**
 * @brief Reasonable defaults for most thermomechanics nonlinear solver options
 */
const smith::NonlinearSolverOptions default_nonlinear_options = {.nonlin_solver = NonlinearSolver::Newton,
                                                                 .relative_tol = 1.0e-4,
                                                                 .absolute_tol = 1.0e-8,
                                                                 .max_iterations = 500,
                                                                 .print_level = 1};

}  // namespace thermomechanics

template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class ThermomechanicsMonolithic;

/**
 * @brief The monolithic thermal-structural solver with operator-split options
 *
 * Uses Functional to compute action of operators
 */
template <int order, int dim, typename... parameter_space, int... parameter_indices>
class ThermomechanicsMonolithic<order, dim, Parameters<parameter_space...>,
                                std::integer_sequence<int, parameter_indices...>> : public BasePhysics {
 public:
  //! @cond Doxygen_Suppress
  static constexpr int VALUE = 0, DERIVATIVE = 1;
  static constexpr int SHAPE = 0;
  static constexpr auto I = Identity<dim>();
  //! @endcond

  /// @brief The total number of non-parameter state variables (displacement, temperature) passed to the FEM
  /// integrators
  static constexpr auto NUM_STATE_VARS = 2;

  /// @brief a container holding quadrature point data of the specified type
  /// @tparam T the type of data to store at each quadrature point
  template <typename T>
  using qdata_type = std::shared_ptr<QuadratureData<T>>;

  /**
   * @brief Construct a new Thermomechanics object
   *
   * @param[in] nonlinear_opts The options for solving the nonlinear thermomechanics residual equations
   * @param[in] lin_opts The options for solving the linearized Jacobian thermomechanics equations
   * @param[in] physics_name A name for the physics module instance
   * @param[in] smith_mesh Smith mesh for the physics
   * @param[in] parameter_names A vector of the names of the requested parameter fields
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   * @param[in] checkpoint_to_disk A flag to save the transient states on disk instead of memory for the transient
   * adjoint solves
   */
  ThermomechanicsMonolithic(const NonlinearSolverOptions nonlinear_opts, const LinearSolverOptions lin_opts,
                            const std::string& physics_name, std::shared_ptr<smith::Mesh> smith_mesh,
                            std::vector<std::string> parameter_names = {}, int cycle = 0, double time = 0.0,
                            bool checkpoint_to_disk = false)
      : ThermomechanicsMonolithic(std::make_unique<EquationSolver>(nonlinear_opts, lin_opts, smith_mesh->getComm()),
                                  physics_name, smith_mesh, parameter_names, cycle, time, checkpoint_to_disk)
  {
  }

  /**
   * @brief Construct a new Thermal-SolidMechanics object
   *
   * @param[in] solver The nonlinear equation solver for the heat conduction equations
   * @param[in] physics_name A name for the physics module instance
   * @param[in] smith_mesh Smith mesh for the physics
   * @param[in] parameter_names A vector of the names of the requested parameter fields
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   * @param[in] checkpoint_to_disk A flag to save the transient states on disk instead of memory for the transient
   * adjoint solves
   */
  ThermomechanicsMonolithic(std::unique_ptr<EquationSolver> solver, const std::string& physics_name,
                            std::shared_ptr<smith::Mesh> smith_mesh, std::vector<std::string> parameter_names = {},
                            int cycle = 0, double time = 0.0, bool checkpoint_to_disk = false)
      : BasePhysics(physics_name, smith_mesh, cycle, time, checkpoint_to_disk),
        temperature_(StateManager::newState(H1<order>{}, detail::addPrefix(physics_name, "temperature"), mesh_->tag())),
        displacement_(
            StateManager::newState(H1<order, dim>{}, detail::addPrefix(physics_name, "displacement"), mesh_->tag())),
        temperature_adjoint_(
            StateManager::newState(H1<order>{}, detail::addPrefix(physics_name, "temperature_adjoint"), mesh_->tag())),
        displacement_adjoint_(StateManager::newState(
            H1<order, dim>{}, detail::addPrefix(physics_name, "dispacement_adjoint"), mesh_->tag())),
        temperature_adjoint_load_(StateManager::newDual(
            H1<order>{}, detail::addPrefix(physics_name, "temperature_adjoint_load"), mesh_->tag())),
        displacement_adjoint_load_(StateManager::newDual(
            H1<order, dim>{}, detail::addPrefix(physics_name, "displacement_adjoint_load"), mesh_->tag())),
        bcs_displacement_(mfemParMesh()),
        block_residual_with_bcs_(temperature_.space().TrueVSize() + displacement_.space().TrueVSize()),
        nonlin_solver_(std::move(solver))
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(mfemParMesh().Dimension() != dim,
                       axom::fmt::format("Compile time dimension, {0}, and runtime mesh dimension, {1}, mismatch", dim,
                                         mfemParMesh().Dimension()));
    SLIC_ERROR_ROOT_IF(!nonlin_solver_,
                       "EquationSolver argument is nullptr in ThermoMechanics constructor. It is possible that it was "
                       "previously moved.");

    is_quasistatic_ = true;

    states_.push_back(&temperature_);
    states_.push_back(&displacement_);

    adjoints_.push_back(&temperature_adjoint_);
    adjoints_.push_back(&displacement_adjoint_);

    const mfem::ParFiniteElementSpace* test_space_1 = &temperature_.space();
    const mfem::ParFiniteElementSpace* test_space_2 = &displacement_.space();
    const mfem::ParFiniteElementSpace* shape_space = &mesh_->shapeDisplacementSpace();

    std::array<const mfem::ParFiniteElementSpace*, NUM_STATE_VARS + sizeof...(parameter_space)> trial_spaces;
    trial_spaces[0] = &temperature_.space();
    trial_spaces[1] = &displacement_.space();

    SLIC_ERROR_ROOT_IF(
        sizeof...(parameter_space) != parameter_names.size(),
        axom::fmt::format("{} parameter spaces given in the template argument but {} parameter names were supplied.",
                          sizeof...(parameter_space), parameter_names.size()));

    if constexpr (sizeof...(parameter_space) > 0) {
      tuple<parameter_space...> types{};
      for_constexpr<sizeof...(parameter_space)>([&](auto i) {
        parameters_.emplace_back(mfemParMesh(), get<i>(types), detail::addPrefix(name_, parameter_names[i]));
        trial_spaces[i + NUM_STATE_VARS] = &(parameters_[i].state->space());
      });
    }

    residual_T_ = std::make_unique<
        ShapeAwareFunctional<shape_trial, scalar_test(scalar_trial, vector_trial, parameter_space...)>>(
        shape_space, test_space_1, trial_spaces);
    residual_u_ = std::make_unique<
        ShapeAwareFunctional<shape_trial, vector_test(scalar_trial, vector_trial, parameter_space...)>>(
        shape_space, test_space_2, trial_spaces);

    block_thermomech_offsets_.SetSize(NUM_STATE_VARS + 1);
    block_thermomech_offsets_[0] = 0;
    block_thermomech_offsets_[1] = temperature_.space().TrueVSize();
    block_thermomech_offsets_[2] = displacement_.space().TrueVSize();
    block_thermomech_offsets_.PartialSum();

    block_thermomech_ = std::make_unique<mfem::BlockVector>(block_thermomech_offsets_);

    block_thermomech_->GetBlock(0) = temperature_;
    block_thermomech_->GetBlock(1) = displacement_;

    block_thermomech_adjoint_ = std::make_unique<mfem::BlockVector>(block_thermomech_offsets_);
    block_thermomech_adjoint_->GetBlock(0) = temperature_adjoint_;
    block_thermomech_adjoint_->GetBlock(1) = displacement_adjoint_;

    nonlin_solver_->setOperator(block_residual_with_bcs_);

    block_nonlinear_oper_ = std::make_unique<mfem::BlockOperator>(block_thermomech_offsets_);
    block_nonlinear_oper_transpose_ = std::make_unique<mfem::BlockOperator>(block_thermomech_offsets_);

    initializeThermoMechanicsStates();
  }

  /// @brief Destroy the ThermoMechanics Functional object
  virtual ~ThermomechanicsMonolithic() {}

  /**
   * @brief Non virtual method to reset temperature and displacement states to zero.  This does not reset design
   * parameters or shape.
   *
   */
  void initializeThermoMechanicsStates()
  {
    dt_ = 0.0;
    time_end_step_ = 0.0;

    temperature_ = 0.0;
    displacement_ = 0.0;

    temperature_adjoint_ = 0.0;
    displacement_adjoint_ = 0.0;

    temperature_adjoint_load_ = 0.0;
    displacement_adjoint_load_ = 0.0;
  }

  /**
   * @brief Virtual implementation of required state reset method
   *
   */
  void resetStates(int cycle = 0, double time = 0.0) override
  {
    BasePhysics::initializeBasePhysicsStates(cycle, time);
    initializeThermoMechanicsStates();

    if (!checkpoint_to_disk_) {
      checkpoint_states_.clear();
      auto state_names = stateNames();
      for (const auto& state_name : state_names) {
        checkpoint_states_[state_name].push_back(state(state_name));
      }
    }
  }

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The boundary attributes on which to enforce a temperature
   * @param[in] temp The prescribed boundary temperature function
   *
   * @note This should be called prior to completeSetup()
   */
  void setTemperatureBCs(const std::set<int>& temp_bdr, std::function<double(const mfem::Vector& x, double t)> temp)
  {
    // Project the coefficient onto the grid function
    temp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(temp);

    bcs_.addEssential(temp_bdr, temp_bdr_coef_, temperature_.space());
  }

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced) matching setDisplacementBCs signature
   *
   * @param[in] applied_temperature Function specifying the applied temperature.
   * @param[in] domain Domain over which to apply the boundary condition.
   *
   * @note This should be called prior to completeSetup()
   */
  template <typename AppliedTemperatureFunction>
  void setTemperatureBCs(AppliedTemperatureFunction applied_temperature, Domain& domain)
  {
    auto mfem_coefficient_function = [applied_temperature](const mfem::Vector X_mfem, double t) {
      auto X = make_tensor<dim>([&X_mfem](int i) { return X_mfem[i]; });
      return applied_temperature(X, t);
    };

    temp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(mfem_coefficient_function);

    auto dof_list = domain.dof_list(&temperature_.space());

    bcs_.addEssential(dof_list, temp_bdr_coef_, temperature_.space(), 0);
  }

  /**
   * @brief Set the thermal flux boundary condition
   *
   * @tparam FluxType The type of the thermal flux object
   * @param flux_function A function describing the flux applied to a boundary
   * @param domain The domain over which the flux is applied.
   *
   * @pre FluxType must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `tensor<T,dim> n` the outward-facing unit normal for the quadrature point
   *    3. `double t` the time (note: time will be handled differently in the future)
   *    4. `T temperature` the current temperature at the quadrature point
   *    4. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename FluxType>
  void setFluxBCs(DependsOn<active_parameters...>, FluxType flux_function, Domain& domain)
  {
    residual_T_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, active_parameters + NUM_STATE_VARS...>{},
        [flux_function](double t, auto X, auto u, auto... params) {
          auto temp = get<VALUE>(u);
          auto n = cross(get<DERIVATIVE>(X));

          return flux_function(X, normalize(n), t, temp, params...);
        },
        domain);
  }

  /// @overload
  template <typename FluxType>
  void setFluxBCs(FluxType flux_function, Domain& domain)
  {
    setFluxBCs(DependsOn<>{}, flux_function, domain);
  }

  /**
   * @brief Set essential displacement boundary conditions on selected components
   *
   * @param[in] applied_displacement Function specifying the applied displacement vector.
   * @param[in] domain Domain over which to apply the boundary condition.
   * @param[in] components (optional) Indicator of vector components to be constrained.
   *            If argument is omitted, the default is to constrain all components.
   *
   * @note This method must be called prior to completeSetup()
   *
   * The signature of the applied_displacement callable is:
   * tensor<double, dim> applied_displacement(tensor<double, dim> X, double t)
   * Parameters:
   *   X - coordinates of node
   *   t - time
   * Returns:
   *   u, vector of applied displacements
   *
   * Usage examples:
   *
   * To constrain the Y component:
   * setDisplacementBCs(applied_displacement, domain, Component::Y);
   *
   * To constrain the X and Z components:
   * setDisplacementBCs(applied_displacement, domain, Component::X + Component::Z);
   *
   * To constrain all components:
   * setDisplacementBCs((applied_displacement, domain);
   */
  template <typename AppliedDisplacementFunction>
  void setDisplacementBCs(AppliedDisplacementFunction applied_displacement, Domain& domain,
                          Components components = Component::ALL)
  {
    for (int i = 0; i < dim; ++i) {
      if (components[size_t(i)]) {
        auto mfem_coefficient_function = [applied_displacement, i](const mfem::Vector& X_mfem, double t) {
          auto X = make_tensor<dim>([&X_mfem](int k) { return X_mfem[k]; });
          return applied_displacement(X, t)[i];
        };

        component_disp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(mfem_coefficient_function);

        auto dof_list = domain.dof_list(&displacement_.space());

        // scalar ldofs -> vector ldofs
        displacement_.space().DofsToVDofs(i, dof_list);

        bcs_displacement_.addEssential(dof_list, component_disp_bdr_coef_, displacement_.space(), i);
      }
    }
  }

  /**
   * @brief Shortcut to set selected components of displacements to zero for all time
   *
   * @param[in] domain Domain to apply the homogeneous boundary condition to
   * @param[in] components (optional) Indicator of vector components to be constrained.
   *            If argument is omitted, the default is to constrain all components.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setFixedBCs(Domain& domain, Components components = Component::ALL)
  {
    auto zero_vector_function = [](tensor<double, dim>, double) { return tensor<double, dim>{}; };
    setDisplacementBCs(zero_vector_function, domain, components);
  }

  /**
   * @brief Set the traction boundary condition
   *
   * @tparam TractionType The type of the traction load
   * @param traction_function A function describing the traction applied to a boundary
   * @param domain The domain over which the traction is applied.
   *
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
  void setTraction(DependsOn<active_parameters...>, TractionType traction_function, Domain& domain)
  {
    residual_u_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<NUM_STATE_VARS + active_parameters...>{},
        [traction_function](double t, auto X, auto... params) {
          auto n = cross(get<DERIVATIVE>(X));

          return -1.0 * traction_function(get<VALUE>(X), normalize(n), t, params...);
        },
        domain);
  }

  /// @overload
  template <typename TractionType>
  void setTraction(TractionType traction_function, Domain& domain)
  {
    setTraction(DependsOn<>{}, traction_function, domain);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed temperature
   *
   * @param temp The function describing the temperature field
   *
   * @note This will override any existing solution values in the temperature field
   */
  void setTemperature(std::function<double(const mfem::Vector& x, double t)> temp)
  {
    // Project the coefficient onto the grid function
    mfem::FunctionCoefficient temp_coef(temp);

    temp_coef.SetTime(time_);
    temperature_.project(temp_coef);
  }

  /// @overload
  void setTemperature(const FiniteElementState temp) { temperature_ = temp; }

  /**
   * @brief Set the underlying finite element state to a prescribed displacement
   *
   * @param disp The function describing the displacement field
   */
  void setDisplacement(std::function<void(const mfem::Vector& x, mfem::Vector& disp)> disp)
  {
    // Project the coefficient onto the grid function
    mfem::VectorFunctionCoefficient disp_coef(dim, disp);
    displacement_.project(disp_coef);
  }

  /// @overload
  void setDisplacement(const FiniteElementState& disp) { displacement_ = disp; }

  /**
   * @brief Functor representing the integrand of a thermal material.  Material type must be
   * a functor as well.
   */
  template <typename MaterialType>
  struct ThermalMaterialInterface {
    /**
     * @brief Construct a ThermalMaterialIntegrand functor with material model of type `MaterialType`.
     * @param[in] material A functor representing the material model.  Should be a functor, or a class/struct with
     * public operator() method.  Must NOT be a generic lambda, or Smith will not compile due to static asserts below.
     */
    ThermalMaterialInterface(MaterialType material) : material_(material) {}

    /**
     * @brief Evaluate integrand
     */
    // template <typename X, typename T, typename dT_dt, typename... Params>
    template <typename X, typename T, typename U, typename... Params>
    auto operator()(double /*time*/, X /* x */, T temperature, U displacement, Params... params) const
    {
      typename MaterialType::State state{};

      // Get the value and the gradient from the input tuple
      auto [theta, dtheta_dX] = temperature;
      auto du_dX = get<DERIVATIVE>(displacement);
      // auto du_dt = get<VALUE>(dtemp_dt);

      auto [stress, heat_accumulation, internal_heat_source, heat_flux] =
          material_(state, du_dX, theta, dtheta_dX, params...);

      // return smith::tuple{heat_capacity * du_dt, -1.0 * heat_flux};
      return smith::tuple{-1.0 * internal_heat_source, -1.0 * heat_flux};
    }

   private:
    MaterialType material_;
  };

  /**
   * @brief Functor representing a material stress.  A functor is used here instead of an
   * extended, generic lambda for compatibility with NVCC.
   */
  template <typename MaterialType>
  struct SolidMaterialInterface {
    /// @brief Constructor for the functor
    SolidMaterialInterface(MaterialType material) : material_(material) {}

    /**
     * @brief Material stress response call
     */
    template <typename X, typename T, typename U, typename... Params>
    auto operator()(double /* time */, X /* x */, T temperature, U displacement, Params... params) const
    {
      typename MaterialType::State state{};

      auto [theta, dtheta_dX] = temperature;
      auto du_dX = get<DERIVATIVE>(displacement);

      auto [stress, heat_accumulation, internal_heat_source, heat_flux] =
          material_(state, du_dX, theta, dtheta_dX, params...);

      return smith::tuple{smith::zero{}, stress};
    }

   private:
    MaterialType material_;
  };

  /**
   * @brief Set the thermomechanical material model for the physics solver
   *
   * @tparam MaterialType The thermomechanical material type
   * @param material A material containing heat capacity, thermal flux, stress, and internal heat source evaluation
   * information
   * @param domain which elements in the mesh are described by the specified material
   *
   * @pre material must be a object that can be called with the following arguments:
   *    1. `MaterialType::State & state` an mutable reference to the internal variables for this quadrature point
   *    2. `tensor<T,dim,dim> du_dx` the displacement gradient at this quadrature point
   *    3. `T temperature` the current temperature at the quadrature point
   *    4. `tensor<T,dim> dT_dx` the spatial gradient of the temperature at the quadrature point
   *    5. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @pre MaterialType must return a smith::tuple of volumetric heat capacity, thermal flux, stress,
   * and internal heat source when operator() is called with the arguments listed above.
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename MaterialType>
  void setMaterial(DependsOn<active_parameters...>, const MaterialType& material, Domain& domain)
  {
    residual_T_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, NUM_STATE_VARS + active_parameters...>{},
                                   ThermalMaterialInterface<MaterialType>(material), domain);
    residual_u_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, NUM_STATE_VARS + active_parameters...>{},
                                   SolidMaterialInterface<MaterialType>(material), domain);
  }

  /// @overload
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, Domain& domain)
  {
    setMaterial(DependsOn<>{}, material, domain);
  }

  /**
   * @brief Set the thermal source function
   *
   * @tparam SourceType The type of the source function
   * @param source_function A source function for a prescribed thermal load
   * @param domain The domain over which the source is applied.
   *
   * @pre source_function must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `double t` the time (note: time will be handled differently in the future)
   *    3. `T temperature` the current temperature at the quadrature point
   *    4. `tensor<T,dim>` the spatial gradient of the temperature at the quadrature point
   *    5. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename SourceType>
  void setSource(DependsOn<active_parameters...>, SourceType source_function, Domain& domain)
  {
    residual_T_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, active_parameters + NUM_STATE_VARS...>{},
        [source_function](double t, auto x, auto temperature, auto... params) {
          // Get the value and the gradient from the input tuple
          auto [T, dT_dX] = temperature;

          auto source = source_function(x, t, T, dT_dX, params...);

          // Return the source and the flux as a tuple
          return smith::tuple{-1.0 * source, smith::zero{}};
        },
        domain);
  }

  /// @overload
  template <typename SourceType>
  void setSource(SourceType source_function, Domain& domain)
  {
    setSource(DependsOn<>{}, source_function, domain);
  }

  /**
   * @brief Set the body force function
   *
   * @tparam BodyForceType The type of the body force load
   * @param body_force A function describing the body force applied
   * @param domain The domain over which the body force is applied.
   *
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
  void addBodyForce(DependsOn<active_parameters...>, BodyForceType body_force, Domain& domain)
  {
    residual_u_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<active_parameters + NUM_STATE_VARS...>{},
        [body_force](double t, auto x, auto... params) {
          auto bf = body_force(get<VALUE>(x), t, params...);
          return smith::tuple{-1.0 * bf, smith::zero{}};
        },
        domain);
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force, Domain& domain)
  {
    addBodyForce(DependsOn<>{}, body_force, domain);
  }

  /// @overload
  void completeSetup() override
  {
    // Block operator representing the nonlinear system of equations
    block_residual_with_bcs_ = mfem_ext::StdFunctionOperator(
        temperature_.space().TrueVSize() + displacement_.space().TrueVSize(),

        // A lambda representing the residual R(T, u, params...)
        // The input is the current state (T, u) in block vector form and the output is the block residual vector (r1,
        // r2)
        [this](const mfem::Vector& u, mfem::Vector& r) {
          mfem::BlockVector block_u(const_cast<mfem::Vector&>(u), block_thermomech_offsets_);
          mfem::BlockVector block_r(r, block_thermomech_offsets_);

          auto temperature = block_u.GetBlock(0);
          auto displacement = block_u.GetBlock(1);

          auto r_1 = (*residual_T_)(time_, shapeDisplacement(), temperature, displacement,
                                    *parameters_[parameter_indices].state...);
          r_1.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);

          auto r_2 = (*residual_u_)(time_, shapeDisplacement(), temperature, displacement,
                                    *parameters_[parameter_indices].state...);
          r_2.SetSubVector(bcs_displacement_.allEssentialTrueDofs(), 0.0);

          block_r.GetBlock(0) = r_1;
          block_r.GetBlock(1) = r_2;
        },

        [this](const mfem::Vector& u) -> mfem::Operator& {
          mfem::BlockVector block_u(const_cast<mfem::Vector&>(u), block_thermomech_offsets_);

          auto temperature = block_u.GetBlock(0);
          auto displacement = block_u.GetBlock(1);

          // Get the components of the block Jacobian via auto differentiation
          auto [r1, dr1_dT] = (*residual_T_)(time_, shapeDisplacement(), differentiate_wrt(temperature), displacement,
                                             *parameters_[parameter_indices].state...);
          auto [_1, dr1_du] = (*residual_T_)(time_, shapeDisplacement(), temperature, differentiate_wrt(displacement),
                                             *parameters_[parameter_indices].state...);

          auto [r2, dr2_dT] = (*residual_u_)(time_, shapeDisplacement(), differentiate_wrt(temperature), displacement,
                                             *parameters_[parameter_indices].state...);
          auto [_2, dr2_du] = (*residual_u_)(time_, shapeDisplacement(), temperature, differentiate_wrt(displacement),
                                             *parameters_[parameter_indices].state...);

          // Assemble the matrix-free Jacobian operators into hypre matrices
          J_11_ = assemble(dr1_dT);
          J_12_ = assemble(dr1_du);
          J_21_ = assemble(dr2_dT);
          J_22_ = assemble(dr2_du);

          // Eliminate the essential DoFs from the matrix
          auto ess_tdofs_T = bcs_.allEssentialTrueDofs();

          mfem::HypreParMatrix* JTempTemp = J_11_->EliminateRowsCols(ess_tdofs_T);
          mfem::HypreParMatrix* JDispTemp = J_21_->EliminateCols(ess_tdofs_T);
          J_12_->EliminateRows(ess_tdofs_T);

          delete JTempTemp;
          delete JDispTemp;

          auto ess_tdofs_u = bcs_displacement_.allEssentialTrueDofs();

          mfem::HypreParMatrix* JDispDisp = J_22_->EliminateRowsCols(ess_tdofs_u);
          mfem::HypreParMatrix* JTempDisp = J_12_->EliminateCols(ess_tdofs_u);
          J_21_->EliminateRows(ess_tdofs_u);

          delete JDispDisp;
          delete JTempDisp;

          // Fill the block operator with the individual Jacobian blocks
          block_nonlinear_oper_->SetBlock(0, 0, J_11_.get());
          block_nonlinear_oper_->SetBlock(0, 1, J_12_.get());
          block_nonlinear_oper_->SetBlock(1, 0, J_21_.get());
          block_nonlinear_oper_->SetBlock(1, 1, J_22_.get());

          return *block_nonlinear_oper_;
        });

    if (checkpoint_to_disk_) {
      outputStateToDisk();
    } else {
      checkpoint_states_.clear();
      auto state_names = stateNames();
      for (const auto& state_name : state_names) {
        checkpoint_states_[state_name].push_back(state(state_name));
      }
    }
  }

  /// @overload
  void advanceTimestep(double dt) override
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(!residual_T_ || !residual_u_,
                       "completeSetup() must be called prior to advanceTimestep(dt) in ThermoMechanics.");

    // If this is the first call, initialize the previous parameter values as the initial values
    if (cycle_ == 0) {
      for (auto& parameter : parameters_) {
        *parameter.previous_state = *parameter.state;
      }
    }

    time_ += dt;

    for (auto& bc : bcs_.essentials()) {
      bc.setDofs(temperature_, time_);
    }

    for (auto& bc : bcs_displacement_.essentials()) {
      bc.setDofs(displacement_, time_);
    }

    // Update the block vector representation with the current temperature and displacement
    block_thermomech_->GetBlock(0) = temperature_;
    block_thermomech_->GetBlock(1) = displacement_;

    // Perform a nonlinear solve using Newton's method
    nonlin_solver_->solve(*block_thermomech_);

    // Fill the independent temperature and displacement vectors from the block vector
    static_cast<mfem::Vector&>(temperature_) = block_thermomech_->GetBlock(0);
    static_cast<mfem::Vector&>(displacement_) = block_thermomech_->GetBlock(1);

    cycle_ += 1;

    if (checkpoint_to_disk_) {
      outputStateToDisk();
    } else {
      auto state_names = stateNames();
      for (const auto& state_name : state_names) {
        checkpoint_states_[state_name].push_back(state(state_name));
      }
    }

    if (cycle_ > max_cycle_) {
      timesteps_.push_back(dt);
      max_cycle_ = cycle_;
      max_time_ = time_;
    }
  }

  /**
   * @brief Set the loads for the adjoint reverse timestep solve
   *
   * @param loads The loads (e.g. right hand sides) for the adjoint problem
   *
   * @pre The adjoint load map is expected to contain an entry named "temperature" and "displacement"
   *
   * These loads are typically defined as derivatives of a downstream quantity of intrest with respect
   * to a primal solution field (in this case, temperature and displacement). For this physics module,
   * the unordered map is expected to have two entries with the keys "temperature" and "displacement".
   *
   */
  void setAdjointLoad(std::unordered_map<std::string, const smith::FiniteElementDual&> loads) override
  {
    SLIC_ERROR_ROOT_IF(loads.size() != 2,
                       "Adjoint load container is not the expected size of 2 in the thermomechanics module");
    auto temperature_adjoint_load = loads.find("temperature");

    SLIC_ERROR_ROOT_IF(temperature_adjoint_load == loads.end(), "Adjoint load for \"temperature\" not found.");

    temperature_adjoint_load_ = temperature_adjoint_load->second;
    // Add sign correction
    temperature_adjoint_load_ *= -1.0;

    auto displacement_adjoint_load = loads.find("displacement");

    SLIC_ERROR_ROOT_IF(displacement_adjoint_load == loads.end(), "Adjoint load for \"displacement\" not found.");

    displacement_adjoint_load_ = displacement_adjoint_load->second;
    // Add sign correction
    displacement_adjoint_load_ *= -1.0;
  }

  /// @overload
  void reverseAdjointTimestep() override
  {
    SMITH_MARK_FUNCTION;
    auto& linear_solver = nonlin_solver_->linearSolver();

    cycle_--;  // cycle is now at n \in [0,N-1]

    double dt = getCheckpointedTimestep(cycle_ + 1);

    auto end_step_solution = getCheckpointedStates(cycle_ + 1);
    temperature_ = end_step_solution.at("temperature");
    displacement_ = end_step_solution.at("displacement");

    auto [r1, dr1_dT] = (*residual_T_)(time_, shapeDisplacement(), differentiate_wrt(temperature_), displacement_,
                                       *parameters_[parameter_indices].state...);
    auto [_1, dr1_du] = (*residual_T_)(time_, shapeDisplacement(), temperature_, differentiate_wrt(displacement_),
                                       *parameters_[parameter_indices].state...);

    auto [r2, dr2_dT] = (*residual_u_)(time_, shapeDisplacement(), differentiate_wrt(temperature_), displacement_,
                                       *parameters_[parameter_indices].state...);
    auto [_2, dr2_du] = (*residual_u_)(time_, shapeDisplacement(), temperature_, differentiate_wrt(displacement_),
                                       *parameters_[parameter_indices].state...);

    J_11_ = assemble(dr1_dT);
    J_12_ = assemble(dr1_du);
    J_21_ = assemble(dr2_dT);
    J_22_ = assemble(dr2_du);

    auto ess_tdofs_T = bcs_.allEssentialTrueDofs();
    temperature_adjoint_load_.SetSubVector(ess_tdofs_T, 0.0);

    mfem::HypreParMatrix* JTempTemp = J_11_->EliminateRowsCols(ess_tdofs_T);
    mfem::HypreParMatrix* JDispTemp = J_21_->EliminateCols(ess_tdofs_T);
    J_12_->EliminateRows(ess_tdofs_T);

    delete JTempTemp;
    delete JDispTemp;

    auto ess_tdofs_u = bcs_displacement_.allEssentialTrueDofs();
    displacement_adjoint_load_.SetSubVector(ess_tdofs_u, 0.0);

    mfem::HypreParMatrix* JDispDisp = J_22_->EliminateRowsCols(ess_tdofs_u);
    mfem::HypreParMatrix* JTempDisp = J_12_->EliminateCols(ess_tdofs_u);
    J_21_->EliminateRows(ess_tdofs_u);

    delete JDispDisp;
    delete JTempDisp;

    // Adjoint problem uses the tranpose of the Jacobian operator
    J_11_transpose_ = std::unique_ptr<mfem::HypreParMatrix>(J_11_->Transpose());
    J_12_transpose_ = std::unique_ptr<mfem::HypreParMatrix>(J_12_->Transpose());
    J_21_transpose_ = std::unique_ptr<mfem::HypreParMatrix>(J_21_->Transpose());
    J_22_transpose_ = std::unique_ptr<mfem::HypreParMatrix>(J_22_->Transpose());

    block_nonlinear_oper_transpose_->SetBlock(0, 0, J_11_transpose_.get());
    block_nonlinear_oper_transpose_->SetBlock(0, 1, J_21_transpose_.get());
    block_nonlinear_oper_transpose_->SetBlock(1, 0, J_12_transpose_.get());
    block_nonlinear_oper_transpose_->SetBlock(1, 1, J_22_transpose_.get());

    linear_solver.SetOperator(*block_nonlinear_oper_transpose_);

    mfem::BlockVector block_thermomech_adjoint_load(block_thermomech_offsets_);
    block_thermomech_adjoint_load.GetBlock(0) = temperature_adjoint_load_;
    block_thermomech_adjoint_load.GetBlock(1) = displacement_adjoint_load_;

    linear_solver.Mult(block_thermomech_adjoint_load, *block_thermomech_adjoint_);

    // Fill the adjoint vectors
    static_cast<mfem::Vector&>(temperature_adjoint_) = block_thermomech_adjoint_->GetBlock(0);
    static_cast<mfem::Vector&>(displacement_adjoint_) = block_thermomech_adjoint_->GetBlock(1);

    // Reset the equation solver
    nonlin_solver_->setOperator(block_residual_with_bcs_);

    time_end_step_ = time_;
    time_ -= dt;
  }

  /// @overload
  FiniteElementDual computeTimestepSensitivity(size_t parameter_field) override
  {
    SLIC_ASSERT_MSG(parameter_field < sizeof...(parameter_indices),
                    axom::fmt::format("Invalid parameter index '{}' requested for sensitivity."));

    auto dr1_dparam = smith::get<DERIVATIVE>(d_residual_T_d_[parameter_field](time_end_step_));
    auto dr2_dparam = smith::get<DERIVATIVE>(d_residual_u_d_[parameter_field](time_end_step_));

    auto dr1_dparam_mat = assemble(dr1_dparam);
    auto dr2_dparam_mat = assemble(dr2_dparam);

    dr1_dparam_mat->MultTranspose(temperature_adjoint_, *parameters_[parameter_field].sensitivity);
    dr2_dparam_mat->AddMultTranspose(displacement_adjoint_, *parameters_[parameter_field].sensitivity);

    return *parameters_[parameter_field].sensitivity;
  }

  /// @overload
  const FiniteElementDual& computeTimestepShapeSensitivity() override
  {
    auto dr1_dshape =
        smith::get<DERIVATIVE>((*residual_T_)(time_end_step_, differentiate_wrt(shapeDisplacement()), temperature_,
                                              displacement_, *parameters_[parameter_indices].state...));
    auto dr2_dshape =
        smith::get<DERIVATIVE>((*residual_u_)(time_end_step_, differentiate_wrt(shapeDisplacement()), temperature_,
                                              displacement_, *parameters_[parameter_indices].state...));

    auto dr1_dshape_mat = assemble(dr1_dshape);
    auto dr2_dshape_mat = assemble(dr2_dshape);

    dr1_dshape_mat->MultTranspose(temperature_adjoint_, shape_displacement_dual_);
    dr2_dshape_mat->AddMultTranspose(displacement_adjoint_, shape_displacement_dual_);

    return shapeDisplacementSensitivity();
  }

  /// @overload
  const FiniteElementState& state(const std::string& state_name) const override
  {
    if (state_name == "temperature") {
      return temperature_;
    } else if (state_name == "displacement") {
      return displacement_;
    } else {
      SLIC_ERROR_ROOT(axom::fmt::format("State '{}' requested from thermomechanics solver '{}', but it doesn't exist",
                                        state_name, name_));
    }

    return temperature_;
  }

  void setState(const std::string& state_name, const FiniteElementState& state) override
  {
    if (state_name == "temperature") {
      temperature_ = state;
      return;
    } else if (state_name == "displacement") {
      displacement_ = state;
      return;
    }

    SLIC_ERROR_ROOT(axom::fmt::format(
        "setState for state name '{}' requested from thermomechanics module '{}', but it doesn't exist", state_name,
        name_));
  }

  std::vector<std::string> stateNames() const override
  {
    return std::vector<std::string>{{"temperature"}, {"displacement"}};
  }

  const FiniteElementState& adjoint(const std::string& state_name) const override
  {
    if (state_name == "temperature") {
      return temperature_adjoint_;
    } else if (state_name == "displacement") {
      return displacement_adjoint_;
    } else {
      SLIC_ERROR_ROOT(axom::fmt::format("adjoint '{}' requested from thermomechanics solver '{}', but it doesn't exist",
                                        state_name, name_));
    }

    return temperature_adjoint_;
  }

  /**
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const smith::FiniteElementState& temperature() const { return temperature_; };

  /**
   * @brief Get the displacement state
   *
   * @return A reference to the current displacement finite element state
   */
  const smith::FiniteElementState& displacement() const { return displacement_; };

 private:
  /// The compile-time finite element trial space for displacement (H1 of dimension dim and order p)
  using vector_trial = H1<order, dim>;

  /// The compile-time finite element trial space for temperature (H1 of order p)
  using scalar_trial = H1<order>;

  /// The compile-time finite element test space for displacement (H1 of dimension dim and order p)
  using vector_test = H1<order, dim>;

  /// The compile-time finite element test space for temperature (H1 of order p)
  using scalar_test = H1<order>;

  /// The compile-time finite element trial space for shape displacement (H1 of order 1, nodal displacements)
  /// The choice of polynomial order for the shape sensitivity is determined in the StateManager
  using shape_trial = smith::H1<smith::SHAPE_ORDER, dim>;

  /// The temperature finite element state
  FiniteElementState temperature_;

  /// The displacement finite element state
  FiniteElementState displacement_;

  /// The temperature adjoint finite element state
  FiniteElementState temperature_adjoint_;

  /// The displacement adjoint finite element state
  FiniteElementState displacement_adjoint_;

  /// The RHS for the temperature part of the adjoint problem, typically a downstream d(QoI)/dT
  smith::FiniteElementDual temperature_adjoint_load_;

  /// The RHS for the displacement part of the adjoint problem, typically a downstream d(QoI)/du
  smith::FiniteElementDual displacement_adjoint_load_;

  /// Additional boundary condition manager for displacement
  smith::BoundaryConditionManager bcs_displacement_;

  /// smith::Functional that is used to calculate the displacement residual and its derivatives
  std::unique_ptr<smith::ShapeAwareFunctional<shape_trial, scalar_test(scalar_trial, vector_trial, parameter_space...)>>
      residual_T_;

  /// smith::Functional that is used to calculate the temperature residual and its derivatives
  std::unique_ptr<smith::ShapeAwareFunctional<shape_trial, vector_test(scalar_trial, vector_trial, parameter_space...)>>
      residual_u_;

  /**
   * @brief The residual operator representing the PDE-based block residual operator
   * and its linearized block Jacobian.
   */
  mfem_ext::StdFunctionOperator block_residual_with_bcs_;

  /// The nonlinear solver for the block system of nonlinear residual equations
  std::unique_ptr<smith::EquationSolver> nonlin_solver_;

  /// A vector of offsets describing the block structure of the combined (displacement, temperature) block vector
  mfem::Array<int> block_thermomech_offsets_;

  /// A block vector combining displacement and temperature into a single vector
  std::unique_ptr<mfem::BlockVector> block_thermomech_;

  /// A block vector combining adjoint displacement and adjoint temperature into a single vector
  std::unique_ptr<mfem::BlockVector> block_thermomech_adjoint_;

  /// The operator representing the assembled block Jacobian
  std::unique_ptr<mfem::BlockOperator> block_nonlinear_oper_;

  /// The operator representing the assembled block Jacobian transpose
  std::unique_ptr<mfem::BlockOperator> block_nonlinear_oper_transpose_;

  /// (1,1) block of the assembled Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_11_;

  /// (1,2) block of the assembled Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_12_;

  /// (2,1) block of the assembled Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_21_;

  /// (2,2) block of the assembled Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_22_;

  /// Transpose of the (1,1) block of the assembled Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_11_transpose_;

  /// Transpose of the (1,2) block of the assembled Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_12_transpose_;

  /// Transpose of the (2,1) block of the assembled Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_21_transpose_;

  /// Transpose of the (2,2) block of the assembled Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_22_transpose_;

  /// @brief End of step time used in reverse mode so that the time can be decremented on reverse steps
  /// @note This time is important to save to evaluate various parameter sensitivities after each reverse step
  double time_end_step_;

  /// Coefficient containing the essential boundary values
  std::shared_ptr<mfem::Coefficient> temp_bdr_coef_;

  /// @brief Coefficient containing the displacement essential boundary values
  std::shared_ptr<mfem::Coefficient> component_disp_bdr_coef_;

  /// @brief Array functions computing the derivative of the residual with respect to each given parameter
  /// @note This is needed so the user can ask for a specific sensitivity at runtime as opposed to it being a
  /// template parameter.
  std::array<std::function<decltype((*residual_T_)(DifferentiateWRT<1>{}, 0.0, shape_displacement_, temperature_,
                                                   displacement_, *parameters_[parameter_indices].state...))(double)>,
             sizeof...(parameter_indices)>
      d_residual_T_d_ = {[&](double _t) {
        return (*residual_T_)(DifferentiateWRT<NUM_STATE_VARS + 1 + parameter_indices>{}, _t, shape_displacement_,
                              temperature_, displacement_, *parameters_[parameter_indices].state...);
      }...};

  std::array<std::function<decltype((*residual_u_)(DifferentiateWRT<1>{}, 0.0, shape_displacement_, temperature_,
                                                   displacement_, *parameters_[parameter_indices].state...))(double)>,
             sizeof...(parameter_indices)>
      d_residual_u_d_ = {[&](double _t) {
        return (*residual_u_)(DifferentiateWRT<NUM_STATE_VARS + 1 + parameter_indices>{}, _t, shape_displacement_,
                              temperature_, displacement_, *parameters_[parameter_indices].state...);
      }...};
};

}  // namespace smith
