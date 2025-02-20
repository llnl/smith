
// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
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

#include "serac/physics/base_physics.hpp"
#include "serac/physics/thermomechanics_input.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

namespace thermomech {

#ifdef MFEM_USE_STRUMPACK
const serac::LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::Strumpack, .print_level = 0};
#else
const serac::LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::SuperLU, .print_level = 0};
#endif

/**
 * @brief Reasonable defaults for most thermal nonlinear solver options
 */
const serac::NonlinearSolverOptions default_nonlinear_options = {.nonlin_solver  = NonlinearSolver::Newton,
                                                                 .relative_tol   = 1.0e-4,
                                                                 .absolute_tol   = 1.0e-8,
                                                                 .max_iterations = 500,
                                                                 .print_level    = 1};

}

template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class ThermoMechanicsMonolithic;

/**
 * @brief The monolithic thermal-structural solver with operator-split options
 *
 * Uses Functional to compute action of operators
 */
template <int order, int dim, typename... parameter_space, int... parameter_indices>
class ThermoMechanicsMonolithic<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>
    : public BasePhysics {
public:
  //! @cond Doxygen_Suppress
  static constexpr int  VALUE = 0, DERIVATIVE = 1;
  static constexpr int  SHAPE = 0;
  static constexpr auto I     = Identity<dim>();
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
   * @param nonlinear_opts The options for solving the nonlinear thermomechanics residual equations
   * @param lin_opts The options for solving the linearized Jacobian thermomechanics equations
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param physics_name A name for the physics module instance
   * @param mesh_tag The tag for the mesh in the StateManager to construct the physics module on
   */
  ThermoMechanicsMonolithic(
    const NonlinearSolverOptions nonlinear_opts, const LinearSolverOptions lin_opts,
    const GeometricNonlinearities geom_nonlin,
    const std::string& physics_name, std::string mesh_tag, std::vector<std::string> parameter_names = {})
  :
  ThermoMechanicsMonolithic(
    std::make_unique<EquationSolver>(nonlinear_opts, lin_opts, StateManager::mesh(mesh_tag).GetComm()),
    geom_nonlin, physics_name, mesh_tag, parameter_names)
  {}


  /**
   * @brief Construct a new Thermal-SolidMechanics object
   *
   * @param solver The nonlinear equation solver for the heat conduction equations
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param physics_name A name for the physics module instance
   * @param mesh_tag The tag for the mesh in the StateManager to construct the physics module on
   */
  ThermoMechanicsMonolithic(std::unique_ptr<EquationSolver> solver, const GeometricNonlinearities geom_nonlin,
                            const std::string& physics_name, std::string mesh_tag, std::vector<std::string> parameter_names = {})
      : BasePhysics(physics_name, mesh_tag),
        temperature_(StateManager::newState(H1<order>{}, detail::addPrefix(physics_name, "temperature"), mesh_tag_)),
        displacement_(StateManager::newState(H1<order, dim>{}, detail::addPrefix(physics_name, "displacement"), mesh_tag_)),
        temperature_adjoint_(StateManager::newState(H1<order>{}, detail::addPrefix(physics_name, "temperature_adjoint"), mesh_tag_)),
        displacement_adjoint_(StateManager::newState(H1<order, dim>{}, detail::addPrefix(physics_name, "dispacement_adjoint"), mesh_tag_)),
        temperature_adjoint_load_(StateManager::newDual(H1<order>{}, detail::addPrefix(physics_name, "temperature_adjoint_load"),
                                                        mesh_tag_)),
        displacement_adjoint_load_(StateManager::newDual(H1<order, dim>{}, detail::addPrefix(physics_name, "displacement_adjoint_load"),
                                                         mesh_tag_)),
        bcs_displacement_(mesh_),
        block_residual_with_bcs_(temperature_.space().TrueVSize() + displacement_.space().TrueVSize()),
        nonlin_solver_(std::move(solver)),
        geom_nonlin_(geom_nonlin)
  {
    SERAC_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension, {0}, and runtime mesh dimension, {1}, mismatch", dim,
                                         mesh_.Dimension()));
    SLIC_ERROR_ROOT_IF(!nonlin_solver_,
                       "EquationSolver argument is nullptr in ThermoMechanics constructor. It is possible that it was "
                       "previously moved.");

    is_quasistatic_ = true;

    states_.push_back(&temperature_);
    states_.push_back(&displacement_);

    adjoints_.push_back(&temperature_adjoint_);
    adjoints_.push_back(&displacement_adjoint_);

    mfem::ParFiniteElementSpace* test_space_1 = &temperature_.space();
    mfem::ParFiniteElementSpace* test_space_2 = &displacement_.space();
    mfem::ParFiniteElementSpace* shape_space = &shape_displacement_.space();

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
        parameters_.emplace_back(mesh_, get<i>(types), detail::addPrefix(name_, parameter_names[i]));

        trial_spaces[i + NUM_STATE_VARS] = &(parameters_[i].state->space());
      });
    }

    residual_T_ = std::make_unique<ShapeAwareFunctional<shape_trial, scalar_test(scalar_trial, vector_trial, parameter_space...)>>(
        shape_space, test_space_1, trial_spaces);
    residual_u_ = std::make_unique<ShapeAwareFunctional<shape_trial, vector_test(scalar_trial, vector_trial, parameter_space...)>>(
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

    block_nonlinear_oper_           = std::make_unique<mfem::BlockOperator>(block_thermomech_offsets_);
    block_nonlinear_oper_transpose_ = std::make_unique<mfem::BlockOperator>(block_thermomech_offsets_);

    shape_displacement_ = 0.0;
    initializeThermoMechanicsStates();
  }

  /// @brief Destroy the ThermoMechanics Functional object
  virtual ~ThermoMechanicsMonolithic() {}

  /**
   * @brief Non virtual method to reset temperature and displacement states to zero.  This does not reset design parameters or shape.
   *
   */
  void initializeThermoMechanicsStates()
  {
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
    auto temp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(temp);

    bcs_.addEssential(temp_bdr, temp_bdr_coef_, temperature_.space());
  }

  /**
   * @brief Set the thermal flux boundary condition
   *
   * @tparam FluxType The type of the thermal flux object
   * @param flux_function A function describing the flux applied to a boundary
   * @param optional_domain The domain over which the flux is applied. If nothing is supplied the entire boundary is
   * used.
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
  void setFluxBCs(DependsOn<active_parameters...>, FluxType flux_function,
                  const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireBoundary(mesh_);

    residual_T_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, active_parameters + NUM_STATE_VARS...>{},
        [flux_function](double t, auto X, auto u, auto... params) {
          auto temp = get<VALUE>(u);
          auto n    = cross(get<DERIVATIVE>(X));

          return flux_function(X, normalize(n), t, temp, params...);
        },
        domain);
  }

  /// @overload
  template <typename FluxType>
  void setFluxBCs(FluxType flux_function, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    setFluxBCs(DependsOn<>{}, flux_function, optional_domain);
  }

  /**
   * @brief Set essential displacement boundary conditions (strongly enforced)
   *
   * @param[in] disp_bdr The boundary attributes from the mesh on which to enforce a displacement
   * @param[in] disp The prescribed boundary displacement function
   *
   * @note This method must be called prior to completeSetup()
   *
   * For the displacement function, the first argument is the input position and the second argument is the output
   * prescribed displacement.
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::function<void(const mfem::Vector&, mfem::Vector&)> disp)
  {
    // Project the coefficient onto the grid function
    auto disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, disp);

    bcs_displacement_.addEssential(disp_bdr, disp_bdr_coef_, displacement_.space());
  }

  /**
   * @brief Set essential displacement boundary conditions (strongly enforced)
   *
   * @param[in] disp_bdr The boundary attributes from the mesh on which to enforce a displacement
   * @param[in] disp The prescribed boundary displacement function
   *
   * For the displacement function, the first argument is the input position, the second argument is the time, and the
   * third argument is the output prescribed displacement.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCs(const std::set<int>&                                            disp_bdr,
                          std::function<void(const mfem::Vector&, double, mfem::Vector&)> disp)
  {
    // Project the coefficient onto the grid function
    auto disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, disp);

    bcs_displacement_.addEssential(disp_bdr, disp_bdr_coef_, displacement_.space());
  }

  /**
   * @brief Set the displacement essential boundary conditions on a single component
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp The vector function containing the set displacement values
   * @param[in] component The component to set the displacment on
   *
   * For the displacement function, the argument is the input position and the output is the value of the component of
   * the displacement.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::function<double(const mfem::Vector& x)> disp,
                          int component)
  {
    // Project the coefficient onto the grid function
    auto component_disp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(disp);

    bcs_displacement_.addEssential(disp_bdr, component_disp_bdr_coef_, displacement_.space(), component);
  }

  /**
   * @brief Set the displacement essential boundary conditions on a single component
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp The vector function containing the set displacement values
   * @param[in] component The component to set the displacment on
   *
   * For the displacement function, the argument is the input position and the output is the value of the component of
   * the displacement.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::function<double(const mfem::Vector& x, double)> disp,
                          int component)
  {
    // Project the coefficient onto the grid function
    auto component_disp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(disp);

    bcs_displacement_.addEssential(disp_bdr, component_disp_bdr_coef_, displacement_.space(), component);
  }

  /**
   * @brief Set the displacement essential boundary conditions on a set of true degrees of freedom
   *
   * @param true_dofs A set of true degrees of freedom to set the displacement on
   * @param disp The vector function containing the prescribed displacement values
   *
   * The @a true_dofs list can be determined using functions from the @a mfem::ParFiniteElementSpace class.
   *
   * @note The coefficient is required to be vector-valued. However, only the dofs specified in the @a true_dofs
   * array will be set. This means that if the @a true_dofs array only contains dofs for a specific vector component in
   * a vector-valued finite element space, only that component will be set.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCsByDofList(const mfem::Array<int>                                  true_dofs,
                                   std::function<void(const mfem::Vector&, mfem::Vector&)> disp)
  {
    auto disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, disp);

    bcs_displacement_.addEssential(true_dofs, disp_bdr_coef_, displacement_.space());
  }

  /**
   * @brief Set the displacement essential boundary conditions on a set of true degrees of freedom
   *
   * @param true_dofs A set of true degrees of freedom to set the displacement on
   * @param disp The vector function containing the prescribed displacement values
   *
   * The @a true_dofs list can be determined using functions from the @a mfem::ParFiniteElementSpace related to the
   * displacement @a serac::FiniteElementState .
   *
   * For the displacement function, the first argument is the input position, the second argument is time,
   * and the third argument is the prescribed output displacement vector.
   *
   * @note The displacement function is required to be vector-valued. However, only the dofs specified in the @a
   * true_dofs array will be set. This means that if the @a true_dofs array only contains dofs for a specific vector
   * component in a vector-valued finite element space, only that component will be set.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCsByDofList(const mfem::Array<int>                                          true_dofs,
                                   std::function<void(const mfem::Vector&, double, mfem::Vector&)> disp)
  {
    auto disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, disp);

    bcs_displacement_.addEssential(true_dofs, disp_bdr_coef_, displacement_.space());
  }

//  /**
//   * @brief Set the displacement boundary conditions on a set of nodes within a spatially-defined area
//   *
//   * @param is_node_constrained A callback function that returns true if displacement nodes at a certain position should
//   * be constrained by this boundary condition
//   * @param disp The vector function containing the prescribed displacement values
//   *
//   * The displacement function takes a spatial position as the first argument and time as the second argument. It
//   * computes the desired displacement and fills the third argument with these displacement values.
//   *
//   * @note This method searches over the entire mesh, not just the boundary nodes.
//   *
//   * @note This method must be called prior to completeSetup()
//   */
//  void setDisplacementBCs(std::function<bool(const mfem::Vector&)>                        is_node_constrained,
//                          std::function<void(const mfem::Vector&, double, mfem::Vector&)> disp)
//  {
//    auto constrained_dofs = calculateConstrainedDofs(is_node_constrained);
//
//    setDisplacementBCsByDofList(constrained_dofs, disp);
//  }
//
//  /**
//   * @brief Set the displacement boundary conditions on a set of nodes within a spatially-defined area
//   *
//   * @param is_node_constrained A callback function that returns true if displacement nodes at a certain position should
//   * be constrained by this boundary condition
//   * @param disp The vector function containing the prescribed displacement values
//   *
//   * The displacement function takes a spatial position as the first argument. It computes the desired displacement
//   * and fills the second argument with these displacement values.
//   *
//   * @note This method searches over the entire mesh, not just the boundary nodes.
//   *
//   * @note This method must be called prior to completeSetup()
//   */
//  void setDisplacementBCs(std::function<bool(const mfem::Vector&)>                is_node_constrained,
//                          std::function<void(const mfem::Vector&, mfem::Vector&)> disp)
//  {
//    auto constrained_dofs = calculateConstrainedDofs(is_node_constrained);
//
//    setDisplacementBCsByDofList(constrained_dofs, disp);
//  }
//
//  /**
//   * @brief Set the displacement boundary conditions on a set of nodes within a spatially-defined area for a single
//   * displacement vector component
//   *
//   * @param is_node_constrained A callback function that returns true if displacement nodes at a certain position should
//   * be constrained by this boundary condition
//   * @param disp The scalar function containing the prescribed component displacement values
//   * @param component The component of the displacement vector that should be set by this boundary condition. The other
//   * components of displacement are unconstrained.
//   *
//   * The displacement function takes a spatial position as the first argument and current time as the second argument.
//   * It computes the desired displacement scalar for the given component and returns that value.
//   *
//   * @note This method searches over the entire mesh, not just the boundary nodes.
//   *
//   * @note This method must be called prior to completeSetup()
//   */
//  void setDisplacementBCs(std::function<bool(const mfem::Vector&)>           is_node_constrained,
//                          std::function<double(const mfem::Vector&, double)> disp, int component)
//  {
//    auto constrained_dofs = calculateConstrainedDofs(is_node_constrained, component);
//
//    auto vector_function = [disp, component](const mfem::Vector& x, double time, mfem::Vector& displacement) {
//      displacement            = 0.0;
//      displacement(component) = disp(x, time);
//    };
//
//    setDisplacementBCsByDofList(constrained_dofs, vector_function);
//  }
//
//  /**
//   * @brief Set the displacement boundary conditions on a set of nodes within a spatially-defined area for a single
//   * displacement vector component
//   *
//   * @param is_node_constrained A callback function that returns true if displacement nodes at a certain position should
//   * be constrained by this boundary condition
//   * @param disp The scalar function containing the prescribed component displacement values
//   * @param component The component of the displacement vector that should be set by this boundary condition. The other
//   * components of displacement are unconstrained.
//   *
//   * The displacement function takes a spatial position as an argument. It computes the desired displacement scalar for
//   * the given component and returns that value.
//   *
//   * @note This method searches over the entire mesh, not just the boundary nodes.
//   *
//   * @note This method must be called prior to completeSetup()
//   */
//  void setDisplacementBCs(std::function<bool(const mfem::Vector& x)>   is_node_constrained,
//                          std::function<double(const mfem::Vector& x)> disp, int component)
//  {
//    auto constrained_dofs = calculateConstrainedDofs(is_node_constrained, component);
//
//    auto vector_function = [disp, component](const mfem::Vector& x, mfem::Vector& displacement) {
//      displacement            = 0.0;
//      displacement(component) = disp(x);
//    };
//
//    setDisplacementBCsByDofList(constrained_dofs, vector_function);
//  }

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
  void setTraction(TractionType traction_function, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    setTraction(DependsOn<>{}, traction_function, optional_domain);
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
     * public operator() method.  Must NOT be a generic lambda, or serac will not compile due to static asserts below.
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

      auto [stress, heat_accumulation, internal_heat_source, heat_flux] = material_(state, du_dX, theta, dtheta_dX, params...);

      // return serac::tuple{heat_capacity * du_dt, -1.0 * heat_flux};
      return serac::tuple{-1.0 * internal_heat_source, -1.0 * heat_flux};
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
    SolidMaterialInterface(MaterialType material, GeometricNonlinearities gn) : material_(material), geom_nonlin_(gn) {}

    /**
     * @brief Material stress response call
     */
    template <typename X, typename T, typename U, typename... Params>
    auto operator()(double /* time */, X /* x */, T temperature, U displacement, Params... params) const
    {
      typename MaterialType::State state{};

      auto [theta, dtheta_dX] = temperature;
      auto du_dX = get<DERIVATIVE>(displacement);
      auto [stress, heat_accumulation, internal_heat_source, heat_flux] = material_(state, du_dX, theta, dtheta_dX, params...);

      auto dx_dX = 0.0 * du_dX + I;
      if (geom_nonlin_ == GeometricNonlinearities::On) { dx_dX += du_dX; }

      auto flux = dot(stress, transpose(inv(dx_dX))) * det(dx_dX);
      return serac::tuple{serac::zero{}, flux};
    }

  private:
    MaterialType material_;
    GeometricNonlinearities geom_nonlin_;
  };

  template <int... active_parameters, typename MaterialType>
  void setMaterial(DependsOn<active_parameters...>, const MaterialType& material)
  {
    residual_T_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, NUM_STATE_VARS + active_parameters...>{},
                                   ThermalMaterialInterface<MaterialType>(material), mesh_);
    residual_u_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, NUM_STATE_VARS + active_parameters...>{},
                                   SolidMaterialInterface<MaterialType>(material, geom_nonlin_), mesh_);
  }

  /// @overload
  template <typename MaterialType>
  void setMaterial(const MaterialType& material)
  {
    setMaterial(DependsOn<>{}, material);
  }

  /**
   * @brief Set the thermal source function
   *
   * @tparam SourceType The type of the source function
   * @param source_function A source function for a prescribed thermal load
   * @param optional_domain The domain over which the source is applied. If nothing is supplied the entire domain is
   * used.
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
  void setSource(DependsOn<active_parameters...>, SourceType source_function,
                 const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireDomain(mesh_);

    residual_T_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, active_parameters + NUM_STATE_VARS...>{},
        [source_function](double t, auto x, auto temperature, auto... params) {
          // Get the value and the gradient from the input tuple
          auto [T, dT_dX] = temperature;

          auto source = source_function(x, t, T, dT_dX, params...);

          // Return the source and the flux as a tuple
          return serac::tuple{-1.0 * source, serac::zero{}};
        },
        domain);
  }

  /// @overload
  template <typename SourceType>
  void setSource(SourceType source_function, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    setSource(DependsOn<>{}, source_function, optional_domain);
  }

  /**
   * @brief Set the body force function
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

    residual_u_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<active_parameters + NUM_STATE_VARS...>{},
        [body_force](double t, auto x, auto... params) {
          auto bf = body_force(get<VALUE>(x), t, params...);
          return serac::tuple{-1.0 * bf, serac::zero{}};
        },
        domain);
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    addBodyForce(DependsOn<>{}, body_force, optional_domain);
  }

  /// @overload
  void completeSetup() override
  {
    // Block operator representing the nonlinear system of equations
    block_residual_with_bcs_ = mfem_ext::StdFunctionOperator(
      temperature_.space().TrueVSize() + displacement_.space().TrueVSize(),

      // A lambda representing the residual R(T, u, params...)
      // The input is the current state (T, u) in block vector form and the output is the block residual vector (r1, r2)
      [this](const mfem::Vector& u, mfem::Vector& r) {
        mfem::BlockVector block_u(const_cast<mfem::Vector&>(u), block_thermomech_offsets_);
        mfem::BlockVector block_r(r, block_thermomech_offsets_);

        auto temperature = block_u.GetBlock(0);
        auto displacement = block_u.GetBlock(1);

        auto r_1 = (*residual_T_)(time_, shape_displacement_, temperature, displacement,
                                  *parameters_[parameter_indices].state...);
        r_1.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);

        auto r_2 = (*residual_u_)(time_, shape_displacement_, temperature, displacement,
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
        auto [r1, dr1_dT] =
            (*residual_T_)(time_, shape_displacement_, differentiate_wrt(temperature), displacement,
                           *parameters_[parameter_indices].state...);
        auto [r1_again, dr1_du] = 
            (*residual_T_)(time_, shape_displacement_, temperature, differentiate_wrt(displacement),
                           *parameters_[parameter_indices].state...);

        auto [r2, dr2_dT] =
            (*residual_u_)(time_, shape_displacement_, differentiate_wrt(temperature), displacement,
                           *parameters_[parameter_indices].state...);
        auto [r2_again, dr2_du] =
            (*residual_u_)(time_, shape_displacement_, temperature, differentiate_wrt(displacement),
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
      }
    );
  }

  /// @overload
  void advanceTimestep(double dt) override
  {
    mfem::Vector zero;

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
  void setAdjointLoad(std::unordered_map<std::string, const serac::FiniteElementDual&> adjoint_loads) override
  {
    SLIC_ERROR_ROOT_IF(adjoint_loads.size() != 2,
                       "Adjoint load container is not the expected size of 2 in the thermomechanics module");
    auto temperature_adjoint_load_ptr = adjoint_loads.find("temperature");

    SLIC_ERROR_ROOT_IF(temperature_adjoint_load_ptr == adjoint_loads.end(), "Adjoint load for \"temperature\" not found.");

    temperature_adjoint_load_ = temperature_adjoint_load_ptr->second;

    auto displacement_adjoint_load_ptr = adjoint_loads.find("displacement");

    SLIC_ERROR_ROOT_IF(displacement_adjoint_load_ptr == adjoint_loads.end(), "Adjoint load for \"displacement\" not found.");

    auto displacement_ajoint_load_ = displacement_adjoint_load_ptr->second;

    // Add sign correction
    temperature_adjoint_load_ *= -1.0;
    displacement_adjoint_load_ *= -1.0;
  }

  /// @overload
  void reverseAdjointTimestep()
  {
    auto [r1, dr1_dT] =
        (*residual_T_)(time_, shape_displacement_, differentiate_wrt(temperature_), displacement_,
                       *parameters_[parameter_indices].state...);
    auto [_1, dr1_du] =
        (*residual_T_)(time_, shape_displacement_, temperature_, differentiate_wrt(displacement_),
                       *parameters_[parameter_indices].state...);

    auto [r2, dr2_dT] =
        (*residual_u_)(time_, shape_displacement_, differentiate_wrt(temperature_), displacement_,
                       *parameters_[parameter_indices].state...);
    auto [_2, dr2_du] =
        (*residual_u_)(time_, shape_displacement_, temperature_, differentiate_wrt(displacement_),
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

    auto& linear_solver = nonlin_solver_->linearSolver();

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
    time_ -= time_;
  }

  /// @overload
  FiniteElementDual& computeTimestepSensitivity(size_t parameter_field) override
  {
    SLIC_ASSERT_MSG(parameter_field < sizeof...(parameter_indices),
                    axom::fmt::format("Invalid parameter index '{}' requested for sensitivity."));

    auto dr1_dparam = serac::get<DERIVATIVE>(d_residual_T_d_[parameter_field](time_end_step_));
    auto dr2_dparam = serac::get<DERIVATIVE>(d_residual_u_d_[parameter_field](time_end_step_));

    auto dr1_dparam_mat = assemble(dr1_dparam);
    auto dr2_dparam_mat = assemble(dr2_dparam);

    auto temperature_sensitivity(*parameters_[parameter_field].sensitivity);
    auto displacement_sensitivity(*parameters_[parameter_field].sensitivity);

    dr1_dparam_mat->MultTranspose(temperature_adjoint_, temperature_sensitivity);
    dr2_dparam_mat->MultTranspose(displacement_adjoint_, displacement_sensitivity);

    add(temperature_sensitivity, displacement_sensitivity, *parameters_[parameter_field].sensitivity);

    return *parameters_[parameter_field].sensitivity;
  }

  /// @overload
  FiniteElementDual& computeTimestepShapeSensitivity() override
  {
    auto dr1_dshape = serac::get<DERIVATIVE>(
        (*residual_T_)(time_end_step_, differentiate_wrt(shape_displacement_), temperature_,
                       displacement_, *parameters_[parameter_indices].state...));
    auto dr2_dshape = serac::get<DERIVATIVE>(
        (*residual_u_)(time_end_step_, differentiate_wrt(shape_displacement_), temperature_,
                       displacement_, *parameters_[parameter_indices].state...));

    auto dr1_dshape_mat = assemble(dr1_dshape);
    auto dr2_dshape_mat = assemble(dr2_dshape);

    auto temperature_sensitivity(*shape_displacement_sensitivity_);
    auto displacement_sensitivity(*shape_displacement_sensitivity_);

    dr1_dshape_mat->MultTranspose(temperature_adjoint_, temperature_sensitivity);
    dr2_dshape_mat->MultTranspose(displacement_adjoint_, displacement_sensitivity);

    add(temperature_sensitivity, displacement_sensitivity, *shape_displacement_sensitivity_);

    return *shape_displacement_sensitivity_;
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
  const serac::FiniteElementState& temperature() const { return temperature_; };

  /**
   * @brief Get the displacement state
   *
   * @return A reference to the current displacement finite element state
   */
  const serac::FiniteElementState& displacement() const { return displacement_; };

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
  using shape_trial = serac::H1<serac::SHAPE_ORDER, dim>;

  /// The temperature finite element state
  FiniteElementState temperature_;

  /// The displacement finite element state
  FiniteElementState displacement_;

  /// The temperature adjoint finite element state
  FiniteElementState temperature_adjoint_;

  /// The displacement adjoint finite element state
  FiniteElementState displacement_adjoint_;

  /// The RHS for the temperature part of the adjoint problem, typically a downstream d(QoI)/dT
  serac::FiniteElementDual temperature_adjoint_load_;

  /// The RHS for the displacement part of the adjoint problem, typically a downstream d(QoI)/du
  serac::FiniteElementDual displacement_adjoint_load_;

  /// Additional boundary condition manager for displacement
  serac::BoundaryConditionManager bcs_displacement_;

  /// serac::Functional that is used to calculate the displacement residual and its derivatives
  std::unique_ptr<serac::ShapeAwareFunctional<shape_trial, scalar_test(scalar_trial, vector_trial, parameter_space...)>> residual_T_;

  /// serac::Functional that is used to calculate the temperature residual and its derivatives
  std::unique_ptr<serac::ShapeAwareFunctional<shape_trial, vector_test(scalar_trial, vector_trial, parameter_space...)>> residual_u_;

  /**
   * @brief The residual operator representing the PDE-based block residual operator
   * and its linearized block Jacobian.
   */
  mfem_ext::StdFunctionOperator block_residual_with_bcs_;

  /// The nonlinear solver for the block system of nonlinear residual equations
  std::unique_ptr<serac::EquationSolver> nonlin_solver_;

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

  /// @brief A flag denoting whether to compute geometric nonlinearities in the residual
  GeometricNonlinearities geom_nonlin_;

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

} // namespace serac
