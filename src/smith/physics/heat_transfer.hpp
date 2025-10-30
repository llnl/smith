// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file heat_transfer.hpp
 *
 * @brief An object containing the solver for a heat transfer PDE
 */

#pragma once

#include "mfem.hpp"

#include "smith/physics/common.hpp"
#include "smith/physics/heat_transfer_input.hpp"
#include "smith/physics/base_physics.hpp"
#include "smith/numerics/odes.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/numerics/functional/shape_aware_functional.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/thermal_material.hpp"

namespace smith {

namespace heat_transfer {

/**
 * @brief Reasonable defaults for most thermal linear solver options
 */
const LinearSolverOptions default_linear_options = {.linear_solver = LinearSolver::GMRES,
                                                    .preconditioner = Preconditioner::HypreL1Jacobi,
                                                    .relative_tol = 1.0e-6,
                                                    .absolute_tol = 1.0e-12,
                                                    .max_iterations = 200};

/// the default direct solver option for solving the linear stiffness equations
#ifdef MFEM_USE_STRUMPACK
const LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::Strumpack, .print_level = 0};
#else
const LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::SuperLU, .print_level = 0};
#endif

/**
 * @brief Reasonable defaults for most thermal nonlinear solver options
 */
const NonlinearSolverOptions default_nonlinear_options = {.nonlin_solver = NonlinearSolver::Newton,
                                                          .relative_tol = 1.0e-4,
                                                          .absolute_tol = 1.0e-8,
                                                          .max_iterations = 500,
                                                          .print_level = 1};

/**
 * @brief Reasonable defaults for dynamic heat transfer simulations
 */
const TimesteppingOptions default_timestepping_options = {TimestepMethod::BackwardEuler,
                                                          DirichletEnforcementMethod::RateControl};

/**
 * @brief Reasonable defaults for static heat transfer simulations
 */
const TimesteppingOptions default_static_options = {TimestepMethod::QuasiStatic};

}  // namespace heat_transfer

/**
 * @brief An object containing the solver for a heat transfer PDE
 *
 * This is a generic linear thermal diffusion operator of the form
 *
 * \f[
 * \mathbf{M} \frac{\partial \mathbf{u}}{\partial t} = -\kappa \mathbf{K} \mathbf{u} + \mathbf{f}
 * \f]
 *
 *  where \f$\mathbf{M}\f$ is a mass matrix, \f$\mathbf{K}\f$ is a stiffness matrix, \f$\mathbf{u}\f$ is the
 *  temperature degree of freedom vector, and \f$\mathbf{f}\f$ is a thermal load vector.
 */
template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class HeatTransfer;

/// @overload
template <int order, int dim, typename... parameter_space, int... parameter_indices>
class HeatTransfer<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>
    : public BasePhysics {
 public:
  //! @cond Doxygen_Suppress
  static constexpr int VALUE = 0, DERIVATIVE = 1;
  static constexpr int SHAPE = 0;
  static constexpr auto I = Identity<dim>();
  //! @endcond

  /// @brief The total number of non-parameter state variables (temperature, dtemp_dt) passed to the FEM
  /// integrators
  static constexpr auto NUM_STATE_VARS = 2;

  /**
   * @brief Construct a new heat transfer object
   *
   * @param[in] nonlinear_opts The nonlinear solver options for solving the nonlinear residual equations
   * @param[in] lin_opts The linear solver options for solving the linearized Jacobian equations
   * @param[in] timestepping_opts The timestepping options for the heat transfer ordinary differential equations
   * @param[in] physics_name A name for the physics module instance
   * @param[in] smith_mesh The Smith mesh
   * @param[in] parameter_names A vector of the names of the requested parameter fields
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   * @param[in] checkpoint_to_disk A flag to save the transient states on disk instead of memory for the transient
   * adjoint solves
   *
   * @note On parallel file systems (e.g. lustre), significant slowdowns and occasional errors were observed when
   *       writing and reading the needed trainsient states to disk for adjoint solves
   */
  HeatTransfer(const NonlinearSolverOptions nonlinear_opts, const LinearSolverOptions lin_opts,
               const smith::TimesteppingOptions timestepping_opts, const std::string& physics_name,
               std::shared_ptr<smith::Mesh> smith_mesh, std::vector<std::string> parameter_names = {}, int cycle = 0,
               double time = 0.0, bool checkpoint_to_disk = false)
      : HeatTransfer(std::make_unique<EquationSolver>(nonlinear_opts, lin_opts, smith_mesh->getComm()),
                     timestepping_opts, physics_name, smith_mesh, parameter_names, cycle, time, checkpoint_to_disk)
  {
  }

  /**
   * @brief Construct a new heat transfer object
   *
   * @param[in] solver The nonlinear equation solver for the heat transfer equations
   * @param[in] timestepping_opts The timestepping options for the heat transfer ordinary differential equations
   * @param[in] physics_name A name for the physics module instance
   * @param[in] smith_mesh Smith mesh used for physics
   * @param[in] parameter_names A vector of the names of the requested parameter fields
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   * @param[in] checkpoint_to_disk A flag to save the transient states on disk instead of memory for the transient
   * adjoint solves
   *
   * @note On parallel file systems (e.g. lustre), significant slowdowns and occasional errors were observed when
   *       writing and reading the needed trainsient states to disk for adjoint solves
   */
  HeatTransfer(std::unique_ptr<smith::EquationSolver> solver, const smith::TimesteppingOptions timestepping_opts,
               const std::string& physics_name, std::shared_ptr<smith::Mesh> smith_mesh,
               std::vector<std::string> parameter_names = {}, int cycle = 0, double time = 0.0,
               bool checkpoint_to_disk = false)
      : BasePhysics(physics_name, smith_mesh, cycle, time, checkpoint_to_disk),
        temperature_(StateManager::newState(H1<order>{}, detail::addPrefix(physics_name, "temperature"), mesh_->tag())),
        temperature_rate_(
            StateManager::newState(H1<order>{}, detail::addPrefix(physics_name, "temperature_rate"), mesh_->tag())),
        adjoint_temperature_(
            StateManager::newState(H1<order>{}, detail::addPrefix(physics_name, "adjoint_temperature"), mesh_->tag())),
        implicit_sensitivity_temperature_start_of_step_(adjoint_temperature_.space(),
                                                        detail::addPrefix(physics_name, "total_deriv_wrt_temperature")),
        temperature_adjoint_load_(temperature_.space(), detail::addPrefix(physics_name, "temperature_adjoint_load")),
        temperature_rate_adjoint_load_(temperature_.space(),
                                       detail::addPrefix(physics_name, "temperature_rate_adjoint_load")),
        residual_with_bcs_(temperature_.space().TrueVSize()),
        nonlin_solver_(std::move(solver)),
        ode_(temperature_.space().TrueVSize(),
             {.time = time_, .u = u_, .dt = dt_, .du_dt = temperature_rate_, .previous_dt = previous_dt_},
             *nonlin_solver_, bcs_)
  {
    SLIC_ERROR_ROOT_IF(
        mfemParMesh().Dimension() != dim,
        axom::fmt::format("Compile time class dimension template parameter and runtime mesh dimension do not match"));

    SLIC_ERROR_ROOT_IF(
        !nonlin_solver_,
        "EquationSolver argument is nullptr in HeatTransfer constructor. It is possible that it was previously moved.");

    // Check for dynamic mode
    if (timestepping_opts.timestepper != TimestepMethod::QuasiStatic) {
      ode_.SetTimestepper(timestepping_opts.timestepper);
      ode_.SetEnforcementMethod(timestepping_opts.enforcement_method);
      is_quasistatic_ = false;
    } else {
      is_quasistatic_ = true;
    }

    states_.push_back(&temperature_);
    if (!is_quasistatic_) {
      states_.push_back(&temperature_rate_);
    }

    adjoints_.push_back(&adjoint_temperature_);

    // Create a pack of the primal field and parameter finite element spaces
    const mfem::ParFiniteElementSpace* test_space = &temperature_.space();
    const mfem::ParFiniteElementSpace* shape_space = &mesh_->shapeDisplacementSpace();

    std::array<const mfem::ParFiniteElementSpace*, sizeof...(parameter_space) + NUM_STATE_VARS> trial_spaces;
    trial_spaces[0] = &temperature_.space();
    trial_spaces[1] = &temperature_.space();

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

    residual_ =
        std::make_unique<ShapeAwareFunctional<shape_trial, test(scalar_trial, scalar_trial, parameter_space...)>>(
            shape_space, test_space, trial_spaces);

    nonlin_solver_->setOperator(residual_with_bcs_);

    int true_size = temperature_.space().TrueVSize();
    u_.SetSize(true_size);
    u_predicted_.SetSize(true_size);

    initializeThermalStates();
  }

  /**
   * @brief Construct a new Nonlinear HeatTransfer Solver object
   *
   * @param[in] input_options The solver information parsed from the input file
   * @param[in] physics_name A name for the physics module instance
   * @param smith_mesh Smith mesh used for physics
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   */
  HeatTransfer(const HeatTransferInputOptions& input_options, const std::string& physics_name,
               std::shared_ptr<smith::Mesh> smith_mesh, int cycle = 0, double time = 0.0)
      : HeatTransfer(input_options.nonlin_solver_options, input_options.lin_solver_options,
                     input_options.timestepping_options, physics_name, smith_mesh, {}, cycle, time)
  {
    for (const auto& mat : input_options.materials) {
      if (std::holds_alternative<smith::heat_transfer::LinearIsotropicConductor>(mat)) {
        setMaterial(std::get<smith::heat_transfer::LinearIsotropicConductor>(mat));
      } else if (std::holds_alternative<smith::heat_transfer::LinearConductor<dim>>(mat)) {
        setMaterial(std::get<smith::heat_transfer::LinearConductor<dim>>(mat));
      }
    }

    if (input_options.initial_temperature) {
      auto temp = input_options.initial_temperature->constructScalar();
      temperature_.project(*temp);
    }

    if (input_options.source_coef) {
      // TODO: Not implemented yet in input files
      // NOTE: cannot use std::functions that use mfem::vector
      SLIC_ERROR("'source' is not implemented yet in input files.");
    }

    // Process the BCs in sorted order for correct behavior with repeated attributes
    std::map<std::string, input::BoundaryConditionInputOptions> sorted_bcs(input_options.boundary_conditions.begin(),
                                                                           input_options.boundary_conditions.end());
    for (const auto& [bc_name, bc] : sorted_bcs) {
      // FIXME: Better naming for boundary conditions?
      if (bc_name.find("temperature") != std::string::npos) {
        std::shared_ptr<mfem::Coefficient> temp_coef(bc.coef_opts.constructScalar());
        bcs_.addEssential(bc.attrs, temp_coef, temperature_.space(), *bc.coef_opts.component);
      } else if (bc_name.find("flux") != std::string::npos) {
        // TODO: Not implemented yet in input files
        // NOTE: cannot use std::functions that use mfem::vector
        SLIC_ERROR("'flux' is not implemented yet in input files.");
      } else {
        SLIC_WARNING_ROOT("Ignoring boundary condition with unknown name: " << physics_name);
      }
    }
  }

  /**
   * @brief Non virtual method to reset thermal states to zero.  This does not reset design parameters or shape.
   */
  void initializeThermalStates()
  {
    dt_ = 0.0;
    previous_dt_ = -1.0;
    time_end_step_ = 0.0;

    u_ = 0.0;
    temperature_ = 0.0;
    temperature_rate_ = 0.0;
    adjoint_temperature_ = 0.0;
    implicit_sensitivity_temperature_start_of_step_ = 0.0;
    temperature_adjoint_load_ = 0.0;
    temperature_rate_adjoint_load_ = 0.0;
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
    initializeThermalStates();

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
   * @brief Advance the heat conduction physics module in time
   *
   * Advance the underlying ODE with the requested time integration scheme using the previously set timestep.
   *
   * @param dt The increment of simulation time to advance the underlying heat transfer problem
   */
  void advanceTimestep(double dt) override
  {
    if (is_quasistatic_) {
      time_ += dt;

      // Project the essential boundary coefficients
      for (auto& bc : bcs_.essentials()) {
        bc.setDofs(temperature_, time_);
      }
      nonlin_solver_->solve(temperature_);
    } else {
      // Step the time integrator
      // Note that the ODE solver handles the essential boundary condition application itself

      // The current ode interface tracks 2 times, one internally which we have a handle to via time_,
      // and one here via the step interface.
      // We are ignoring this one, and just using the internal version for now.
      // This may need to be revisited when more complex time integrators are required,
      // but at the moment, the double times creates a lot of confusion, so
      // we short circuit the extra time here by passing a dummy time and ignoring it.
      double time_tmp = time_;
      ode_.Step(temperature_, time_tmp, dt);
    }

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
   * @brief Functor representing the integrand of a thermal material.  Material type must be
   * a functor as well.
   */
  template <typename MaterialType>
  struct ThermalMaterialIntegrand {
    /**
     * @brief Construct a ThermalMaterialIntegrand functor with material model of type `MaterialType`.
     * @param[in] material A functor representing the material model.  Should be a functor, or a class/struct with
     * public operator() method.  Must NOT be a generic lambda, or Smith will not compile due to static asserts below.
     */
    ThermalMaterialIntegrand(MaterialType material) : material_(material) {}

    /**
     * @brief Evaluate integrand
     */
    template <typename X, typename T, typename dT_dt, typename... Params>
    auto operator()(double /*time*/, X x, T temperature, dT_dt dtemp_dt, Params... params) const
    {
      // Get the value and the gradient from the input tuple
      auto [u, du_dX] = temperature;
      auto du_dt = get<VALUE>(dtemp_dt);

      auto [heat_capacity, heat_flux] = material_(x, u, du_dX, params...);

      return smith::tuple{heat_capacity * du_dt, -1.0 * heat_flux};
    }

   private:
    MaterialType material_;
  };

  /**
   * @brief Set the thermal material model for the physics solver
   *
   * @tparam MaterialType The thermal material type
   * @param material A material containing heat capacity and thermal flux evaluation information
   * @param domain which elements in the mesh are described by the specified material
   *
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
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename MaterialType>
  void setMaterial(DependsOn<active_parameters...>, const MaterialType& material, Domain& domain)
  {
    residual_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, NUM_STATE_VARS + active_parameters...>{},
                                 ThermalMaterialIntegrand<MaterialType>(material), domain);
  }

  /// @overload
  template <typename MaterialType>
  void setMaterial(const MaterialType& material, Domain& domain)
  {
    setMaterial(DependsOn<>{}, material, domain);
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
    residual_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 1, active_parameters + NUM_STATE_VARS...>{},
        [source_function](double t, auto x, auto temperature, auto /* dtemp_dt */, auto... params) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dX] = temperature;

          auto source = source_function(x, t, u, du_dX, params...);

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
   * @brief Set the thermal flux boundary condition
   *
   * @tparam FluxType The type of the thermal flux object
   * @param flux_function A function describing the flux applied to a boundary
   * @param domain The domain over which the flux is applied
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
    residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, 1, active_parameters + NUM_STATE_VARS...>{},
        [flux_function](double t, auto X, auto u, auto /* dtemp_dt */, auto... params) {
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
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const smith::FiniteElementState& temperature() const { return temperature_; };

  /**
   * @brief Get the temperature rate of change state
   *
   * @return A reference to the current temperature rate of change finite element state
   */
  const smith::FiniteElementState& temperatureRate() const { return temperature_rate_; };

  /**
   * @brief Accessor for getting named finite element state primal solution from the physics modules
   *
   * @param state_name The name of the Finite Element State primal solution to retrieve
   * @return The named primal Finite Element State
   */
  const FiniteElementState& state(const std::string& state_name) const override
  {
    if (state_name == "temperature") {
      return temperature_;
    } else if (state_name == "temperature_rate") {
      return temperature_rate_;
    }

    SLIC_ERROR_ROOT(axom::fmt::format("State '{}' requested from solid mechanics module '{}', but it doesn't exist",
                                      state_name, name_));
    return temperature_;
  }

  /**
   * @brief Set the primal solution field (temperature) for the underlying heat transfer solver
   *
   * @param state_name The name of the field to initialize (must be "temperature")
   * @param state The finite element state vector containing the values for either the temperature field
   *
   * It is expected that @a state has the same underlying finite element space and mesh as the selected primal solution
   * field.
   */
  void setState(const std::string& state_name, const FiniteElementState& state) override
  {
    if (state_name == "temperature") {
      temperature_ = state;
      if (!checkpoint_to_disk_) {
        checkpoint_states_["temperature"][static_cast<size_t>(cycle_)] = temperature_;
      }
      return;
    }

    SLIC_ERROR_ROOT(axom::fmt::format(
        "setState for state named '{}' requested from heat transfer module '{}', but it doesn't exist", state_name,
        name_));
  }

  /**
   * @brief Get a vector of the finite element state primal solution names
   *
   * @return The primal solution names
   */
  virtual std::vector<std::string> stateNames() const override
  {
    if (is_quasistatic_) {
      return std::vector<std::string>{"temperature"};
    } else {
      return std::vector<std::string>{"temperature", "temperature_rate"};
    }
  }

  /**
   * @brief register a custom boundary integral calculation as part of the residual
   *
   * @tparam active_parameters a list of indices, describing which parameters to pass to the q-function
   * @param qfunction a callable that returns the normal heat flux on a boundary surface
   * @param domain The domain over which the integral is computed
   *
   * ~~~ {.cpp}
   *
   *  heat_transfer.addCustomBoundaryIntegral(
   *    DependsOn<>{},
   *    [](double t, auto position, auto temperature, auto temperature_rate) {
   *      auto [T, dT_dxi] = temperature;
   *      auto q           = 5.0*(T-25.0);
   *      return q;  // define a temperature-proportional heat-flux
   *    },
   *    domain
   *  );
   *
   * ~~~
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename callable>
  void addCustomBoundaryIntegral(DependsOn<active_parameters...>, callable qfunction, Domain& domain)
  {
    residual_->AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0, 1, active_parameters + NUM_STATE_VARS...>{},
                                   qfunction, domain);
  }

  /**
   * @brief Accessor for getting named finite element state adjoint solution from the physics modules
   *
   * @param state_name The name of the Finite Element State adjoint solution to retrieve
   * @return The named adjoint Finite Element State
   */
  const FiniteElementState& adjoint(const std::string& state_name) const override
  {
    if (state_name == "temperature") {
      return adjoint_temperature_;
    }

    SLIC_ERROR_ROOT(axom::fmt::format("Adjoint '{}' requested from solid mechanics module '{}', but it doesn't exist",
                                      state_name, name_));
    return adjoint_temperature_;
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * This must be called before advanceTimestep().
   */
  void completeSetup() override
  {
    if (is_quasistatic_) {
      residual_with_bcs_ = mfem_ext::StdFunctionOperator(
          temperature_.space().TrueVSize(),

          [this](const mfem::Vector& u, mfem::Vector& r) {
            const mfem::Vector res = (*residual_)(time_, shapeDisplacement(), u, temperature_rate_,
                                                  *parameters_[parameter_indices].state...);

            // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
            // tracking strategy
            // See https://github.com/mfem/mfem/issues/3531
            r = res;
            r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
          },

          [this](const mfem::Vector& u) -> mfem::Operator& {
            auto [r, drdu] = (*residual_)(time_, shapeDisplacement(), differentiate_wrt(u), temperature_rate_,
                                          *parameters_[parameter_indices].state...);
            J_ = assemble(drdu);
            J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            return *J_;
          });
    } else {
      residual_with_bcs_ = mfem_ext::StdFunctionOperator(
          temperature_.space().TrueVSize(),

          [this](const mfem::Vector& du_dt, mfem::Vector& r) {
            add(1.0, u_, dt_, du_dt, u_predicted_);
            const mfem::Vector res =
                (*residual_)(time_, shapeDisplacement(), u_predicted_, du_dt, *parameters_[parameter_indices].state...);

            // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
            // tracking strategy
            // See https://github.com/mfem/mfem/issues/3531
            r = res;
            r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
          },

          [this](const mfem::Vector& du_dt) -> mfem::Operator& {
            add(1.0, u_, dt_, du_dt, u_predicted_);

            // K := dR/du
            auto K = smith::get<DERIVATIVE>((*residual_)(time_, shapeDisplacement(), differentiate_wrt(u_predicted_),
                                                         du_dt, *parameters_[parameter_indices].state...));
            std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

            // M := dR/du_dot
            auto M =
                smith::get<DERIVATIVE>((*residual_)(time_, shapeDisplacement(), u_predicted_, differentiate_wrt(du_dt),
                                                    *parameters_[parameter_indices].state...));
            std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

            // J := M + dt K
            J_.reset(mfem::Add(1.0, *m_mat, dt_, *k_mat));
            J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

            return *J_;
          });
    }

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

  /**
   * @brief Set the loads for the adjoint reverse timestep solve
   *
   * @param loads The loads (e.g. right hand sides) for the adjoint problem
   *
   * @pre The adjoint load map is expected to contain an entry named "temperature"
   * @pre The adjoint load map may contain an entry named "temperature_rate"
   *
   * These loads are typically defined as derivatives of a downstream quantity of intrest with respect
   * to a primal solution field (in this case, temperature). For this physics module, the unordered
   * map is expected to have two entries with the keys "temperature" and "temperature_rate". Note that the
   * "temperature_rate" load is only used by transient (i.e. non-quasi-static) adjoint calculations.
   *
   */
  virtual void setAdjointLoad(std::unordered_map<std::string, const smith::FiniteElementDual&> loads) override
  {
    SLIC_ERROR_ROOT_IF(loads.size() == 0,
                       "Adjoint load container size must be greater than 0 in the heat transfer module.");

    auto temp_adjoint_load = loads.find("temperature");
    auto temp_rate_adjoint_load = loads.find("temperature_rate");  // does not need to be specified

    SLIC_ERROR_ROOT_IF(temp_adjoint_load == loads.end(), "Adjoint load for \"temperature\" not found.");

    temperature_adjoint_load_ = temp_adjoint_load->second;
    // Add the sign correction to move the term to the RHS
    temperature_adjoint_load_ *= -1.0;

    if (temp_rate_adjoint_load != loads.end()) {
      temperature_rate_adjoint_load_ = temp_rate_adjoint_load->second;
      temperature_rate_adjoint_load_ *= -1.0;
    }
  }

  /**
   * @brief Solve the adjoint problem
   * @pre It is expected that the forward analysis is complete and the current temperature state is valid
   * @pre It is expected that the adjoint load has already been set in HeatTransfer::setAdjointLoad
   * @note If the essential boundary dual is not specified, homogeneous essential boundary conditions are applied to
   * the adjoint system
   * @note We provide a quick derivation for the discrete adjoint equations and notation used here for backward
   * Euler. There are two equations satisfied at each step of the forward solve using backward Euler \n 1). \f$r^n(u^n,
   * v^n; d) = 0,\f$ \n where \f$u^n\f$ is the end-step primal value, \f$d\f$ are design parameters, and \f$v^n\f$ is
   * the central difference velocity, satisfying \n 2). \f$\Delta t \, v^n = u^n-u^{n-1}.\f$ \n We are interesting in
   * the implicit sensitivity of a qoi (quantity of interest), \f$\sum_{n=1}^N \pi^n(u^n, v^n; d)\f$, while maintaining
   * the above constraints. We construct a Lagrangian that adds 'zero' to this qoi using the free multipliers
   * \f$\lambda^n\f$ (which we call the adjoint temperature) and
   * \f$\mu^n\f$ (which can eventually be intepreted as the implicit sensitivity of the qoi with respect to the
   * start-of-step primal value \f$u^{n-1}\f$) with \f[ \mathcal{L} :=  \sum_{n=1}^N \pi(u^n, v^n; d) + \lambda^n \cdot
   * r^n(u^n, v^n; d) + \mu^n \cdot (\Delta t \, v^n - u^n + u^{n-1}).\f] We are interesting in the total derivative
   * \f[\frac{d\mathcal{L}}{dp} = \sum_{n=1}^N \pi^n_{,u} u^n_{,d} + \pi^n_{,v} v^n_{,d} + \pi^n_{,d} + \lambda^n \cdot
   * \left( r^n_{,u} u^n_{,d} + r^n_{,v} v^n_{,d} + r^n_{,d} \right) + \mu^n \cdot (\Delta t \, v^n_{,d} - u^n_{,d} +
   * u^{n-1}_{,d}). \f] We are free to choose \f$\lambda^n\f$ and \f$\mu^n\f$ in any convenient way, and the way we
   * choose is by grouping terms involving \f$u^n_{,d}\f$ and \f$v^n_{,d}\f$, and setting the multipliers such that
   * those terms cancel out in the final expression of the qoi sensitivity. In particular, by choosing \f[ \lambda^n = -
   * \left[ r_{,u}^n + \frac{r_{,v}^n}{\Delta t} \right]^{-T} \left( \pi^n_{,u} + \frac{\pi^n_{,v}}{\Delta t} +
   * \mu^{n+1} \right) \f] and \f[ \mu^n = -\frac{1}{\Delta t}\left( \pi^n_{,v} + \lambda^n \cdot r^n_{,v} \right), \f]
   * we find
   * \f[ \frac{d\mathcal{L}}{dp} = \sum_{n=1}^N \left( \pi^n_{,d} + \lambda^n \cdot r^n_{,d} \right) + \mu^1 \cdot
   * u^{0}_{,d}, \f] where the multiplier/adjoint equations are solved backward in time starting at \f$n=N\f$,
   * \f$\mu^{N+1} = 0\f$, and \f$u^{0}_{,d}\f$ is the sensitivity of the initial primal variable with respect to the
   * parameters.\n We call the quantities \f$\pi^n_{,u}\f$ and \f$\pi^n_{,v}\f$ the temperature and temperature-rate
   * adjoint loads, respectively.
   */
  void reverseAdjointTimestep() override
  {
    auto& lin_solver = nonlin_solver_->linearSolver();

    mfem::HypreParVector adjoint_essential(temperature_adjoint_load_);
    adjoint_essential = 0.0;

    SLIC_ERROR_ROOT_IF(cycle_ <= min_cycle_,
                       "Maximum number of adjoint timesteps exceeded! The number of adjoint timesteps must equal the "
                       "number of forward timesteps");

    cycle_--;  // cycle is now at n \in [0,N-1]

    double dt = getCheckpointedTimestep(cycle_);
    auto end_step_solution = getCheckpointedStates(cycle_ + 1);

    temperature_ = end_step_solution.at("temperature");

    if (is_quasistatic_) {
      // We store the previous timestep's temperature as the current temperature for use in the lambdas computing the
      // sensitivities.

      auto [_, drdu] = (*residual_)(time_, shapeDisplacement(), differentiate_wrt(temperature_), temperature_rate_,
                                    *parameters_[parameter_indices].state...);
      auto jacobian = assemble(drdu);
      auto J_T = std::unique_ptr<mfem::HypreParMatrix>(jacobian->Transpose());

      for (const auto& bc : bcs_.essentials()) {
        bc.apply(*J_T, temperature_adjoint_load_, adjoint_essential);
      }

      lin_solver.SetOperator(*J_T);
      lin_solver.Mult(temperature_adjoint_load_, adjoint_temperature_);
    } else {
      SLIC_ERROR_ROOT_IF(ode_.GetTimestepper() != TimestepMethod::BackwardEuler,
                         "Only backward Euler implemented for transient adjoint heat conduction.");

      temperature_rate_ = end_step_solution.at("temperature_rate");

      // K := dR/du
      auto K = smith::get<DERIVATIVE>((*residual_)(time_, shapeDisplacement(), differentiate_wrt(temperature_),
                                                   temperature_rate_, *parameters_[parameter_indices].state...));
      std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

      // M := dR/du_dot
      auto M = smith::get<DERIVATIVE>((*residual_)(time_, shapeDisplacement(), temperature_,
                                                   differentiate_wrt(temperature_rate_),
                                                   *parameters_[parameter_indices].state...));
      std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

      J_.reset(mfem::Add(1.0, *m_mat, dt, *k_mat));
      auto J_T = std::unique_ptr<mfem::HypreParMatrix>(J_->Transpose());

      // recall that temperature_adjoint_load_vector and d_temperature_dt_adjoint_load_vector were already multiplied by
      // -1 above
      mfem::HypreParVector modified_RHS(temperature_adjoint_load_);
      modified_RHS *= dt;
      modified_RHS.Add(1.0, temperature_rate_adjoint_load_);
      modified_RHS.Add(-dt, implicit_sensitivity_temperature_start_of_step_);

      for (const auto& bc : bcs_.essentials()) {
        bc.apply(*J_T, modified_RHS, adjoint_essential);
      }

      lin_solver.SetOperator(*J_T);
      lin_solver.Mult(modified_RHS, adjoint_temperature_);

      // This multiply is technically on M transposed.  However, this matrix should be symmetric unless
      // the thermal capacity is a function of the temperature rate of change, which is thermodynamically
      // impossible, and fortunately not possible with our material interface.
      // Not doing the transpose here to avoid doing unnecessary work.
      m_mat->Mult(adjoint_temperature_, implicit_sensitivity_temperature_start_of_step_);
      implicit_sensitivity_temperature_start_of_step_ *= -1.0 / dt;
      implicit_sensitivity_temperature_start_of_step_.Add(1.0 / dt,
                                                          temperature_rate_adjoint_load_);  // already multiplied by -1
    }

    time_end_step_ = time_;
    time_ -= dt;
  }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the load for the adjoint
   * problem with respect to the parameter field for the last computed adjoint timestep
   *
   * @tparam parameter_field The index of the parameter to take a derivative with respect to
   * @return The sensitivity with respect to the parameter
   *
   * @pre `reverseAdjointTimestep` with an appropriate adjoint load must be called prior to this method.
   */
  FiniteElementDual computeTimestepSensitivity(size_t parameter_field) override
  {
    // TODO: the time is likely not being handled correctly on the reverse pass, but we don't
    //       have tests to confirm.
    auto drdparam = smith::get<DERIVATIVE>(d_residual_d_[parameter_field](time_end_step_));
    auto drdparam_mat = assemble(drdparam);

    drdparam_mat->MultTranspose(adjoint_temperature_, *parameters_[parameter_field].sensitivity);

    return *parameters_[parameter_field].sensitivity;
  }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the load for the adjoint
   * problem with respect to the shape displacement field for the last computed adjoint timestep
   *
   * @return The sensitivity with respect to the shape displacement
   *
   * @pre `reverseAdjointTimestep` with an appropriate adjoint load must be called prior to this method.
   */
  const FiniteElementDual& computeTimestepShapeSensitivity() override
  {
    auto drdshape =
        smith::get<DERIVATIVE>((*residual_)(time_end_step_, differentiate_wrt(shapeDisplacement()), temperature_,
                                            temperature_rate_, *parameters_[parameter_indices].state...));

    auto drdshape_mat = assemble(drdshape);

    drdshape_mat->MultTranspose(adjoint_temperature_, shape_displacement_dual_);

    return shapeDisplacementSensitivity();
  }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest with respect to the initial temperature
   *
   * @return The sensitivity with respect to the initial temperature
   *
   * @pre `reverseAdjointTimestep` must be called as many times as the forward solver was advanced before this is called
   */
  const std::unordered_map<std::string, const smith::FiniteElementDual&> computeInitialConditionSensitivity()
      const override
  {
    return {{"temperature", implicit_sensitivity_temperature_start_of_step_}};
  }

  /// Destroy the Thermal Solver object
  virtual ~HeatTransfer() = default;

 protected:
  /// The compile-time finite element trial space for heat transfer (H1 of order p)
  using scalar_trial = H1<order>;

  /// The compile-time finite element trial space for shape displacement (vector H1 of order 1)
  using shape_trial = H1<SHAPE_ORDER, dim>;

  /// The compile-time finite element test space for heat transfer (H1 of order p)
  using test = H1<order>;

  /// The temperature finite element state
  smith::FiniteElementState temperature_;

  /// Rate of change in temperature at the current adjoint timestep
  FiniteElementState temperature_rate_;

  /// The adjoint temperature finite element states, the multiplier on the residual for a given timestep
  smith::FiniteElementState adjoint_temperature_;

  /// The total/implicit sensitivity of the qoi with respect to the start of the previous timestep's temperature
  smith::FiniteElementDual implicit_sensitivity_temperature_start_of_step_;

  /// The downstream derivative of the qoi with repect to the primal temperature variable
  smith::FiniteElementDual temperature_adjoint_load_;

  /// The downstream derivative of the qoi with repect to the primal temperature rate variable
  smith::FiniteElementDual temperature_rate_adjoint_load_;

  /// smith::Functional that is used to calculate the residual and its derivatives
  std::unique_ptr<ShapeAwareFunctional<shape_trial, test(scalar_trial, scalar_trial, parameter_space...)>> residual_;

  /// Assembled mass matrix
  std::unique_ptr<mfem::HypreParMatrix> M_;

  /// Coefficient containing the essential boundary values
  std::shared_ptr<mfem::Coefficient> temp_bdr_coef_;

  /**
   * @brief mfem::Operator that describes the weight residual
   * and its gradient with respect to temperature
   */
  mfem_ext::StdFunctionOperator residual_with_bcs_;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  std::unique_ptr<EquationSolver> nonlin_solver_;

  /**
   * @brief the ordinary differential equation that describes
   * how to solve for the time derivative of temperature, given
   * the current temperature and source terms
   */
  mfem_ext::FirstOrderODE ode_;

  /// Assembled sparse matrix for the Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_;

  /// rows and columns of J_ that have been separated out
  /// because are associated with essential boundary conditions
  std::unique_ptr<mfem::HypreParMatrix> J_e_;

  /// The current timestep
  double dt_;

  /// The previous timestep
  double previous_dt_;

  /// @brief End of step time used in reverse mode so that the time can be decremented on reverse steps
  /// @note This time is important to save to evaluate various parameter sensitivities after each reverse step
  double time_end_step_;

  /// Predicted temperature true dofs
  mfem::Vector u_;

  /// Predicted temperature true dofs
  mfem::Vector u_predicted_;

  /// @brief Array functions computing the derivative of the residual with respect to each given parameter
  /// @note This is needed so the user can ask for a specific sensitivity at runtime as opposed to it being a
  /// template parameter.
  std::array<std::function<decltype((*residual_)(DifferentiateWRT<1>{}, 0.0, shape_displacement_, temperature_,
                                                 temperature_rate_, *parameters_[parameter_indices].state...))(double)>,
             sizeof...(parameter_indices)>
      d_residual_d_ = {[&](double TIME) {
        return (*residual_)(DifferentiateWRT<NUM_STATE_VARS + 1 + parameter_indices>{}, TIME, shape_displacement_,
                            temperature_, temperature_rate_, *parameters_[parameter_indices].state...);
      }...};
};

}  // namespace smith
