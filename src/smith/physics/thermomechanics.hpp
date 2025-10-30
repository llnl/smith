// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermomechanics.hpp
 *
 * @brief An object containing an operator-split thermal structural solver
 */

#pragma once

#include "mfem.hpp"

#include "smith/physics/base_physics.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/thermomechanics_input.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/heat_transfer.hpp"
#include "smith/physics/materials/thermal_material.hpp"
#include "smith/physics/materials/solid_material.hpp"

namespace smith {

/**
 * @brief The operator-split thermal-structural solver
 *
 * Uses Functional to compute action of operators
 */
template <int order, int dim, typename... parameter_space>
class Thermomechanics : public BasePhysics {
 public:
  /**
   * @brief Construct a new coupled Thermal-SolidMechanics object
   *
   * @param thermal_nonlin_opts The options for solving the nonlinear heat conduction residual equations
   * @param thermal_lin_opts The options for solving the linearized Jacobian heat transfer equations
   * @param thermal_timestepping The timestepping options for the heat transfer operator
   * @param solid_nonlin_opts The options for solving the nonlinear solid mechanics residual equations
   * @param solid_lin_opts The options for solving the linearized Jacobian solid mechanics equations
   * @param solid_timestepping The timestepping options for the solid solver
   * @param physics_name A name for the physics module instance
   * @param smith_mesh Smith mesh for physics
   * @param cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param time The simulation time to initialize the physics module to
   */
  Thermomechanics(const NonlinearSolverOptions thermal_nonlin_opts, const LinearSolverOptions thermal_lin_opts,
                  TimesteppingOptions thermal_timestepping, const NonlinearSolverOptions solid_nonlin_opts,
                  const LinearSolverOptions solid_lin_opts, TimesteppingOptions solid_timestepping,
                  const std::string& physics_name, std::shared_ptr<smith::Mesh> smith_mesh, int cycle = 0,
                  double time = 0.0)
      : Thermomechanics(std::make_unique<EquationSolver>(thermal_nonlin_opts, thermal_lin_opts, smith_mesh->getComm()),
                        thermal_timestepping,
                        std::make_unique<EquationSolver>(solid_nonlin_opts, solid_lin_opts, smith_mesh->getComm()),
                        solid_timestepping, physics_name, smith_mesh, cycle, time)
  {
  }

  /**
   * @brief Construct a new coupled Thermal-SolidMechanics object
   *
   * @param thermal_solver The nonlinear equation solver for the heat conduction equations
   * @param thermal_timestepping The timestepping options for the thermal solver
   * @param solid_solver The nonlinear equation solver for the solid mechanics equations
   * @param solid_timestepping The timestepping options for the solid solver
   * @param physics_name A name for the physics module instance
   * @param smith_mesh Smith mesh for physics
   * @param cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param time The simulation time to initialize the physics module to
   */
  Thermomechanics(std::unique_ptr<EquationSolver> thermal_solver, TimesteppingOptions thermal_timestepping,
                  std::unique_ptr<EquationSolver> solid_solver, TimesteppingOptions solid_timestepping,
                  const std::string& physics_name, std::shared_ptr<smith::Mesh> smith_mesh, int cycle = 0,
                  double time = 0.0)
      : BasePhysics(physics_name, smith_mesh),
        thermal_(std::move(thermal_solver), thermal_timestepping, physics_name + "thermal", smith_mesh,
                 {"displacement"}, cycle, time),
        solid_(std::move(solid_solver), solid_timestepping, physics_name + "mechanical", smith_mesh, {"temperature"},
               cycle, time)
  {
    SLIC_ERROR_ROOT_IF(mfemParMesh().Dimension() != dim,
                       axom::fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

    states_.push_back(&thermal_.temperature());
    states_.push_back(&solid_.velocity());
    states_.push_back(&solid_.displacement());
  }

  /**
   * @brief Construct a new Thermal-SolidMechanics Functional object from input file options
   *
   * @param[in] thermal_options The thermal physics module input file option struct
   * @param[in] solid_options The solid physics module input file option struct
   * @param[in] physics_name A name for the physics module instance
   * @param[in] mesh_tag The tag for the mesh in the StateManager to construct the physics module on
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   */
  Thermomechanics(const HeatTransferInputOptions& thermal_options, const SolidMechanicsInputOptions& solid_options,
                  const std::string& physics_name, std::string mesh_tag, int cycle = 0, double time = 0.0)
      : Thermomechanics(thermal_options.nonlin_solver_options, thermal_options.lin_solver_options,
                        thermal_options.timestepping_options, solid_options.nonlin_solver_options,
                        solid_options.lin_solver_options, solid_options.timestepping_options, physics_name, mesh_tag,
                        cycle, time)
  {
  }

  /**
   * @brief Construct a new Thermal-SolidMechanics Functional object from input file options
   *
   * @param[in] options The thermal solid physics module input file option struct
   * @param[in] physics_name A name for the physics module instance
   * @param[in] mesh_tag The tag for the mesh in the StateManager to construct the physics module on
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   */
  Thermomechanics(const ThermomechanicsInputOptions& options, const std::string& physics_name, std::string mesh_tag,
                  int cycle = 0, double time = 0.0)
      : Thermomechanics(options.thermal_options, options.solid_options, physics_name, mesh_tag, cycle, time)
  {
    if (options.coef_thermal_expansion) {
      std::unique_ptr<mfem::Coefficient> cte(options.coef_thermal_expansion->constructScalar());
      std::unique_ptr<mfem::Coefficient> ref_temp(options.reference_temperature->constructScalar());

      // setThermalExpansion(std::move(cte), std::move(ref_temp));
    }
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * @note This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    thermal_.completeSetup();
    solid_.completeSetup();
  }

  /**
   * @brief Method to reset physics states to zero.  This does not reset design parameters or shape.
   *
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   */
  void resetStates(int cycle = 0, double time = 0.0) override
  {
    BasePhysics::initializeBasePhysicsStates(cycle, time);
    thermal_.resetStates(cycle, time);
    solid_.resetStates(cycle, time);
  }

  /**
   * @brief Accessor for getting named finite element state primal fields from the physics modules
   *
   * @param state_name The name of the Finite Element State to retrieve
   * @return The named Finite Element State
   */
  const FiniteElementState& state(const std::string& state_name) const override
  {
    if (state_name == "displacement") {
      return solid_.displacement();
    } else if (state_name == "velocity") {
      return solid_.velocity();
    } else if (state_name == "temperature") {
      return thermal_.temperature();
    }

    SLIC_ERROR_ROOT(axom::fmt::format("State '{}' requested from solid mechanics module '{}', but it doesn't exist",
                                      state_name, name_));
    return solid_.displacement();
  }

  /**
   * @brief Set the primal solution field (displacement, velocity, temperature) for the underlying thermomechanics
   * solver
   *
   * @param state_name The name of the field to initialize ("displacement", or "velocity")
   * @param state The finite element state vector containing the values for either the displacement or velocity fields
   *
   * It is expected that @a state has the same underlying finite element space and mesh as the selected primal solution
   * field.
   */
  void setState(const std::string& state_name, const FiniteElementState& state) override
  {
    if (state_name == "displacement") {
      const_cast<FiniteElementState&>(solid_.displacement()) = state;
      return;
    } else if (state_name == "velocity") {
      const_cast<FiniteElementState&>(solid_.velocity()) = state;
      return;
    } else if (state_name == "temperature") {
      const_cast<FiniteElementState&>(thermal_.temperature()) = state;
      return;
    }

    SLIC_ERROR_ROOT(axom::fmt::format(
        "setState for state named '{}' requested from thermomechanics module '{}', but it doesn't exist", state_name,
        name_));
  }

  /**
   * @brief Get a vector of the finite element state solution variable names
   *
   * @return The solution variable names
   */
  virtual std::vector<std::string> stateNames() const override
  {
    return std::vector<std::string>{"displacement", "velocity", "temperature"};
  }

  /**
   * @brief Accessor for getting named finite element adjoint fields from the physics modules
   *
   * @param state_name The name of the Finite Element State adjoint field to retrieve
   * @return The named Finite Element State adjoint
   */
  const FiniteElementState& adjoint(const std::string& state_name) const override
  {
    if (state_name == "displacement") {
      return solid_.adjoint("displacement");
    } else if (state_name == "temperature") {
      return thermal_.adjoint("temperature");
    }

    SLIC_ERROR_ROOT(axom::fmt::format("Adjoint '{}' requested from solid mechanics module '{}', but it doesn't exist",
                                      state_name, name_));
    return solid_.displacement();
  }

  /**
   * @brief Advance the timestep
   *
   * @param dt The increment of simulation time to advance the underlying thermomechanical problem
   */
  void advanceTimestep(double dt) override
  {
    thermal_.setParameter(0, solid_.displacement());
    thermal_.advanceTimestep(dt);

    solid_.setParameter(0, thermal_.temperature());
    solid_.advanceTimestep(dt);

    cycle_ += 1;
    time_ += dt;
  }

  /**
   * @brief Create a shared ptr to a quadrature data buffer for the given material type
   *
   * @tparam T the type to be created at each quadrature point
   * @param initial_state the value to be broadcast to each quadrature point
   * @return std::shared_ptr< QuadratureData<T> >
   */
  template <typename T>
  std::shared_ptr<QuadratureData<T>> createQuadratureDataBuffer(T initial_state)
  {
    return solid_.createQuadratureDataBuffer(initial_state);
  }

  /**
   * @brief This is an adaptor class that makes a thermomechanical material usable by
   * the thermal module, by discarding the solid-mechanics-specific information
   *
   * @tparam ThermalMechanicalMaterial the material model being wrapped
   */
  template <typename ThermalMechanicalMaterial>
  struct ThermalMaterialInterface {
    using State = typename ThermalMechanicalMaterial::State;  ///< internal variables for the wrapped material model

    const ThermalMechanicalMaterial mat;  ///< the wrapped material model

    /// constructor
    ThermalMaterialInterface(const ThermalMechanicalMaterial& m) : mat(m)
    {
      // empty
    }

    /**
     * @brief glue code to evaluate a thermomechanical material and extract the thermal outputs
     *
     * @tparam T1 the type of the spatial coordinate values
     * @tparam T2 the type of the temperature value
     * @tparam T3 the type of the temperature gradient values
     * @tparam T4 the type of the displacement gradient values
     * @tparam param_types the types of user-specified parameters
     * @param temperature the temperature at this quadrature point
     * @param temperature_gradient the gradient w.r.t. physical coordinates of the temperature
     * @param displacement the value and gradient w.r.t. physical coordinates of the displacement
     * @param parameters values and derivatives of any additional user-specified parameters
     */
    template <typename T1, typename T2, typename T3, typename T4, typename... param_types>
    SMITH_HOST_DEVICE auto operator()(const T1& /* x */, const T2& temperature, const T3& temperature_gradient,
                                      const T4& displacement, param_types... parameters) const
    {
      // BT: this will not update the state correctly. I just want to get the code compiling before plumbing the
      // state variables.
      State state{};

      auto [u, du_dX] = displacement;
      auto [T, heat_capacity, s0, q0] = mat(state, du_dX, temperature, temperature_gradient, parameters...);

      return smith::tuple{heat_capacity, q0};
    }
  };

  /**
   * @brief This is an adaptor class that makes a thermomechanical material usable by
   * the solid mechanics module, by discarding the thermal-specific information
   *
   * @tparam ThermalMechanicalMaterial the material model being wrapped
   */
  template <typename ThermalMechanicalMaterial>
  struct MechanicalMaterialInterface {
    using State = typename ThermalMechanicalMaterial::State;  ///< internal variables for the wrapped material model

    const ThermalMechanicalMaterial mat;  ///< the wrapped material model

    const double density;  ///< mass density

    /// constructor
    MechanicalMaterialInterface(ThermalMechanicalMaterial m) : mat(m), density(m.density)
    {
      // empty
    }

    /**
     * @brief glue code to evaluate a thermomechanical material and extract the stress
     *
     * @tparam T1 the type of the displacement gradient values
     * @tparam T2 the type of the temperature value
     * @tparam param_types the types of user-specified parameters
     * @param state any internal variables needed to evaluate this material
     * @param displacement_gradient the gradient w.r.t. physical coordinates of the displacement
     * @param temperature the temperature at this quadrature point
     * @param parameters values and derivatives of any additional user-specified parameters
     */
    template <typename T1, typename T2, typename... param_types>
    SMITH_HOST_DEVICE auto operator()(State& state, const T1& displacement_gradient, const T2& temperature,
                                      param_types... parameters) const
    {
      auto [theta, dtheta_dX] = temperature;
      auto [T, heat_capacity, s0, q0] = mat(state, displacement_gradient, theta, dtheta_dX, parameters...);
      return T;
    }
  };

  /**
   * @brief Set the thermomechanical material response
   *
   * @tparam MaterialType The thermomechanical material type
   * @tparam StateType The type that contains the internal variables for MaterialType
   * @param material A material that provides a function to evaluate stress, heat flux, density, and heat capacity
   * @param domain which elements in the mesh are described by the specified material
   * @param qdata the buffer of material internal variables at each quadrature point
   *
   * @pre material must be a object that can be called with the following arguments:
   *    1. `MaterialType::State & state` an mutable reference to the internal variables for this quadrature point
   *    2. `tensor<T,dim,dim> du_dx` the displacement gradient at this quadrature point
   *    3. `T temperature` the current temperature at the quadrature point
   *    4. `tensor<T,dim>` the spatial gradient of the temperature at the quadrature point
   *    5. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes
   * `tensor<dual<...>, 3>`)
   *
   * @pre MaterialType must return a smith::tuple of Cauchy stress, volumetric heat capacity, internal heat source,
   * and thermal flux when operator() is called with the arguments listed above.
   */
  template <int... active_parameters, typename MaterialType, typename StateType>
  void setMaterial(DependsOn<active_parameters...>, const MaterialType& material, Domain& domain,
                   std::shared_ptr<QuadratureData<StateType>> qdata)
  {
    // note: these parameter indices are offset by 1 since, internally, this module uses the first parameter
    // to communicate the temperature and displacement field information to the other physics module
    //
    thermal_.setMaterial(DependsOn<0, active_parameters + 1 ...>{}, ThermalMaterialInterface<MaterialType>{material},
                         domain);
    solid_.setMaterial(DependsOn<0, active_parameters + 1 ...>{}, MechanicalMaterialInterface<MaterialType>{material},
                       domain, qdata);
  }

  /// @overload
  template <typename MaterialType, typename StateType = Empty>
  void setMaterial(const MaterialType& material, Domain& domain,
                   std::shared_ptr<QuadratureData<StateType>> qdata = EmptyQData)
  {
    setMaterial(DependsOn<>{}, material, domain, qdata);
  }

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temperature_attributes The boundary attributes on which to enforce a temperature
   * @param[in] prescribed_value The prescribed boundary temperature function
   */
  void setTemperatureBCs(const std::set<int>& temperature_attributes,
                         std::function<double(const mfem::Vector& x, double t)> prescribed_value)
  {
    thermal_.setTemperatureBCs(temperature_attributes, prescribed_value);
  }

  /**
   * @brief Set essential displacement boundary conditions on selected components
   *
   * @param[in] applied_displacement Function specifying the applied displacement vector.
   * @param[in] domain Domain over which to apply the boundary condition.
   * @param[in] components (optional) (optional) Indicator of vector components to be constrained.
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
   */
  template <typename AppliedDisplacementFunction>
  void setDisplacementBCs(AppliedDisplacementFunction applied_displacement, Domain& domain,
                          Components components = Component::ALL)
  {
    solid_.setDisplacementBCs(applied_displacement, domain, components);
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
  void setFixedBCs(Domain& domain, Components components = Component::ALL) { solid_.setFixedBCs(domain, components); }

  /**
   * @brief Set the thermal flux boundary condition
   *
   * @tparam FluxType The type of the thermal flux object
   * @param flux_function A function describing the flux applied to a boundary
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
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes
   * `tensor<dual<...>, 3>`)
   *
   * @note: until mfem::GetFaceGeometricFactors implements their JACOBIANS option,
   * (or we implement a replacement kernel ourselves) we are not able to compute
   * shape sensitivities for boundary integrals.
   */
  template <typename FluxType>
  void setHeatFluxBCs(FluxType flux_function)
  {
    thermal_.setFluxBCs(flux_function);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed displacement
   *
   * @param displacement The function describing the displacement field
   */
  template <typename Callable>
  void setDisplacement(Callable displacement)
  {
    solid_.setDisplacement(displacement);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed temperature
   *
   * @param temperature The function describing the temperature field
   */
  void setTemperature(std::function<double(const mfem::Vector& x, double t)> temperature)
  {
    thermal_.setTemperature(temperature);
  }

  /**
   * @brief Set the body forcefunction
   *
   * @tparam BodyForceType The type of the body force load
   * @pre body_force_function must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `double t` the time (note: time will be handled differently in the future)
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes
   * `tensor<dual<...>, 3>`)
   *
   */
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force_function)
  {
    solid_.addBodyForce(body_force_function);
  }

  /**
   * @brief Set the thermal source function
   *
   * @tparam HeatSourceType The type of the source function
   * @param source_function A source function for a prescribed thermal load
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
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes
   * `tensor<dual<...>, 3>`)
   */
  template <typename HeatSourceType>
  void addHeatSource(HeatSourceType source_function)
  {
    thermal_.setSource(source_function);
  }

  /**
   * @brief Get the displacement state
   *
   * @return A reference to the current displacement finite element state
   */
  const smith::FiniteElementState& displacement() const { return solid_.displacement(); };

  /**
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const smith::FiniteElementState& temperature() const { return thermal_.temperature(); };

 protected:
  using displacement_field = H1<order, dim>;  ///< the function space for the displacement field
  using temperature_field = H1<order>;        ///< the function space for the temperature field

  /// Submodule to compute the heat transfer physics
  HeatTransfer<order, dim, Parameters<displacement_field, parameter_space...>> thermal_;

  /// Submodule to compute the mechanics
  SolidMechanics<order, dim, Parameters<temperature_field, parameter_space...>> solid_;
};

}  // namespace smith
