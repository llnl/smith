// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file base_physics.hpp
 *
 * @brief The base interface class for a generic PDE solver
 */

#pragma once

#include <functional>
#include <memory>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mpi.h"
#include "mfem.hpp"
#include "axom/sidre.hpp"
#include "axom/fmt.hpp"
#include "axom/slic.hpp"

#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/common.hpp"

namespace smith {

class Mesh;

namespace detail {
/**
 * @brief Prepends a prefix to a target string if @p name is non-empty with an
 * underscore delimiter
 * @param[in] prefix The string to prepend
 * @param[in] target The string to prepend to
 */
std::string addPrefix(const std::string& prefix, const std::string& target);

/**
 * @brief Removes a prefix and the underscore delimiter from a target string
 * @param[in] prefix The prefix string to remove
 * @param[in] target The larger string to remove the prefix from
 */
std::string removePrefix(const std::string& prefix, const std::string& target);

}  // namespace detail

/**
 * @brief This is the abstract base class for a generic forward solver
 */
class BasePhysics {
 public:
  /**
   * @brief Empty constructor
   * @param[in] physics_name Name of the physics module instance
   * @param[in] mesh The mesh used to construct the physics on
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   * @param[in] checkpoint_to_disk A flag to save the transient states on disk instead of memory for the transient
   * adjoint solves
   */
  BasePhysics(std::string physics_name, std::shared_ptr<smith::Mesh> mesh, int cycle = 0, double time = 0.0,
              bool checkpoint_to_disk = false);

  /**
   * @brief Construct a new Base Physics object (copy constructor)
   *
   * @param other The other base physics to copy from
   */
  BasePhysics(BasePhysics&& other) = default;

  /**
   * @brief Get the current forward-solution time
   *
   * @return The current forward-solution time
   */
  virtual double time() const;

  /**
   * @brief Get the current forward-solution cycle iteration number
   *
   * @return The current forward-solution cycle iteration number
   */
  virtual int cycle() const;

  /**
   * @brief Get the maximum time reached by the forward solver
   *
   * @return The maximum time reached by the forward solver
   */
  virtual double maxTime() const;

  /**
   * @brief Get the initial time used by the forward solver
   *
   * @return The initial time used by the forward solver
   */
  virtual double minTime() const;

  /**
   * @brief The maximum cycle (timestep iteration number) reached by the forward solver
   *
   * @return The maximum cycle reached by the forward solver
   */
  virtual int maxCycle() const;

  /**
   * @brief Get the initial cycle (timestep iteration number) used by the forward solver
   *
   * @return The initial cycle used by the forward solver
   */
  virtual int minCycle() const;

  /**
   * @brief Check if the physics is setup as quasistatic
   *
   * @return true if quasistatic, false if transient
   */
  bool isQuasistatic() const { return is_quasistatic_; }

  /**
   * @brief Get a vector of the timestep sizes (i.e. \f$\Delta t\f$s) taken by the forward solver
   *
   * @return The vector of timestep sizes taken by the foward solver
   */
  virtual const std::vector<double>& timesteps() const;

  /**
   * @brief Base method to reset physics states to the initial time.  This does not reset design parameters or shape.
   *
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   */
  virtual void resetStates(int cycle = 0, double time = 0.0) = 0;

  /**
   * @brief Base method to reset physics states back to the end of time to start adjoint calculations again.  This does
   * not reset design parameters or shape.
   *
   */
  virtual void resetAdjointStates()
  {
    time_ = max_time_;
    cycle_ = max_cycle_;
  }

  /**
   * @brief Complete the setup and allocate the necessary data structures
   *
   * This finializes the underlying data structures in a solver and
   * enables it to be run through a timestepping loop.
   */
  virtual void completeSetup() = 0;

  /**
   * @brief Accessor for getting named finite element state primal solution from the physics modules
   *
   * @param state_name The name of the Finite Element State primal solution to retrieve
   * @return The named primal Finite Element State
   */
  virtual const FiniteElementState& state(const std::string& state_name) const = 0;

  /**
   * @brief Set the primal solution field values of the underlying physics solver
   */
  virtual void setState(const std::string&, const FiniteElementState&) = 0;

  /**
   * @brief Get a vector of the finite element state primal solution names
   *
   * @return The primal solution names
   */
  virtual std::vector<std::string> stateNames() const = 0;

  /**
   * @brief Accessor for getting named finite element state adjoint solution from the physics modules
   *
   * @param adjoint_name The name of the Finite Element State adjoint solution to retrieve
   * @return The named adjoint Finite Element State
   */
  virtual const FiniteElementState& adjoint(const std::string& adjoint_name) const = 0;

  /**
   * @brief Get a vector of the finite element state adjoint solution names
   *
   * @return The adjoint solution names
   */
  virtual std::vector<std::string> adjointNames() const { return {}; }

  /**
   * @brief Get a vector of the finite element state dual (reaction) solution names
   *
   * @return The dual solution names
   */
  virtual std::vector<std::string> dualNames() const { return {}; }

  /**
   * @brief Accessor for getting named finite element state dual (reaction) solution from the physics modules
   *
   * @param dual_name The name of the Finite Element State dual solution to retrieve
   * @return The named dual Finite Element State
   */
  virtual const FiniteElementDual& dual(const std::string& dual_name) const
  {
    SLIC_ERROR_ROOT(axom::fmt::format("dual '{}' requested from physics module '{}' which does not support duals",
                                      dual_name, name_));
    return *duals_[0];
  }

  /**
   * @brief Accessor for getting named finite element state dual adjoint (reaction adjoint load) from the physics
   * modules
   *
   * @param dual_name The name of the Finite Element State dual (reaction adjoint load) solution to retrieve
   * @return The named adjoint Finite Element State
   */
  virtual const FiniteElementState& dualAdjoint(const std::string& dual_name) const
  {
    SLIC_ERROR_ROOT(axom::fmt::format(
        "dualAdjoint '{}' requested from physics module '{}' which does not support duals", dual_name, name_));
    return *dual_adjoints_[0];
  }

  /**
   * @brief Accessor for getting the shape displacement field from the physics modules
   *
   * @return The shape displacement finite element state
   */
  virtual const FiniteElementState& shapeDisplacement() const;

  /**
   * @brief Accessor for getting named finite element state parameter fields from the physics modules
   *
   * @param parameter_name The name of the Finite Element State parameter to retrieve
   * @return The named parameter Finite Element State
   *
   * @note The input parameter name should not contain the base physics name. It should be identical to what
   * is in the physics module constructor argument list.
   */
  virtual const FiniteElementState& parameter(const std::string& parameter_name) const
  {
    std::string appended_name = detail::addPrefix(name_, parameter_name);

    for (auto& parameter : parameters_) {
      if (appended_name == parameter.state->name()) {
        return *parameter.state;
      }
    }

    SLIC_ERROR_ROOT(axom::fmt::format("Parameter {} requested from physics module {}, but it doesn't exist.",
                                      parameter_name, name_));

    return *states_[0];
  }

  /**
   * @brief Accessor for getting indexed finite element state parameter fields from the physics modules
   *
   * @param parameter_index The index of the Finite Element State parameter to retrieve
   * @return The indexed parameter Finite Element State
   */
  virtual const FiniteElementState& parameter(std::size_t parameter_index) const
  {
    SLIC_ERROR_ROOT_IF(
        parameter_index >= parameters_.size(),
        axom::fmt::format("Parameter index {} requested, but only {} parameters exist in physics module {}.",
                          parameter_index, parameters_.size(), name_));

    return *parameters_[parameter_index].state;
  }

  /**
   * @brief Get a vector of the finite element state parameter names
   *
   * @return The parameter names
   */
  virtual std::vector<std::string> parameterNames() const
  {
    std::vector<std::string> parameter_names;

    for (auto& parameter : parameters_) {
      parameter_names.emplace_back(detail::removePrefix(name_, parameter.state->name()));
    }

    return parameter_names;
  }

  /**
   * @brief Deep copy a parameter field into the internally-owned parameter used for simulations
   *
   * @param parameter_index the index of the parameter
   * @param parameter_state the values to use for the specified parameter
   *
   * @pre The discretization space and mesh for this finite element state must be consistent with the arguments
   * provided in the physics module constructor.
   *
   * The physics module constructs its own parameter FiniteElementState in the physics module constructor. This
   * call sets the internally-owned parameter object by value (i.e. deep copies) from the given argument.
   */
  virtual void setParameter(const size_t parameter_index, const FiniteElementState& parameter_state);

  /**
   * @brief Set the current shape displacement for the underlying mesh
   *
   * @param shape_displacement The shape displacement to copy for use in the physics module
   *
   * This updates the shape displacement field associated with the underlying mesh. Note that the input
   * FiniteElementState is deep copied into the shape displacement object owned by the StateManager.
   */
  virtual void setShapeDisplacement(const FiniteElementState& shape_displacement);

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the adjoint load with respect
   * to the parameter field (d QOI/d state * d state/d parameter).
   * @param parameter_index the index of the parameter
   *
   * @return The sensitivity of the QOI (given implicitly by the adjoint load) with respect to the parameter
   *
   * @pre completeSetup(), advanceTimestep(), and reverseAdjointTimestep() must be called prior to this method.
   */
  virtual FiniteElementDual computeTimestepSensitivity(size_t parameter_index);

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the adjoint load with respect
   * to the shape displacement field (d QOI/d state * d state/d shape displacement).
   *
   * @return The sensitivity with respect to the shape displacement
   *
   * @pre completeSetup(), advanceTimestep(), and reverseAdjointTimestep() must be called prior to this method.
   */
  virtual const FiniteElementDual& computeTimestepShapeSensitivity();

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest with respect to the initial condition fields
   *
   * @return Fields states corresponding to the sensitivities with respect to the initial condition fields
   *
   * @pre `reverseAdjointTimestep` with an appropriate adjoint load must be called prior to this method as many times as
   * the forward advance is called.
   */
  virtual const std::unordered_map<std::string, const smith::FiniteElementDual&> computeInitialConditionSensitivity()
      const
  {
    SLIC_WARNING_ROOT(axom::fmt::format("Initial condition sensitivities not enabled in physics module {}", name_));
    return {};
  }

  /**
   * @brief Advance the state variables according to the chosen time integrator
   *
   * Advance the underlying ODE with the requested time integration scheme using the previously set timestep.
   *
   * @param dt The increment of simulation time to advance the underlying physical system
   */
  virtual void advanceTimestep(double dt) = 0;

  /**
   * @brief Set the loads for the adjoint reverse timestep solve
   * @param string_to_dual An unorder map from the state field name string to the finite element adjoint/dual load
   * The adjoint load is d(qoi)/d(state)
   */
  virtual void setAdjointLoad(std::unordered_map<std::string, const smith::FiniteElementDual&> string_to_dual)
  {
    if (!string_to_dual.empty()) {
      SLIC_ERROR_ROOT(
          axom::fmt::format("Failed to setAdjointLoad.  Adjoint analysis not defined for physics module {}", name_));
    }
  }

  /**
   * @brief Set the dual loads (dirichlet values) for the adjoint reverse timestep solve
   * This must be called after setAdjointLoad
   * @param string_to_bc An unorder map from dual name string to finite element adjoint/state boundary condition
   * The adjoint bc is d(qoi)/d(dual)
   */
  virtual void setDualAdjointBcs(std::unordered_map<std::string, const smith::FiniteElementState&> string_to_bc)
  {
    if (!string_to_bc.empty()) {
      SLIC_ERROR_ROOT(
          axom::fmt::format("Failed to setDualAdjointBCs.  Adjoint analysis not defined for physics module {}", name_));
    }
  }

  /**
   * @brief Solve the adjoint reverse timestep problem
   * @pre It is expected that the forward analysis is complete and the current states are valid
   * @pre It is expected that the adjoint load has already been set in setAdjointLoad
   */
  virtual void reverseAdjointTimestep()
  {
    SLIC_ERROR_ROOT(axom::fmt::format("Adjoint analysis not defined for physics module {}", name_));
  }

  /**
   * @brief Output the current state of the PDE fields in Sidre format and optionally in Paraview format
   *  if \p paraview_output_dir is given.
   *
   * @param[in] paraview_output_dir Optional output directory for paraview visualization files
   */
  virtual void outputStateToDisk(std::optional<std::string> paraview_output_dir = {}) const;

  /**
   * @brief load checkpointed states from disk into states array
   *
   * @param cycle The cycle to retrieve state from
   */
  void loadCheckpointedStatesFromDisk(int cycle);

  /**
   * @brief Accessor for getting a single named finite element state primal solution from the physics modules at a given
   * checkpointed cycle index
   *
   * @param state_name The name of the state to retrieve (e.g. "temperature", "displacement")
   * @param cycle The cycle to retrieve state from
   * @return The named primal Finite Element State
   */
  virtual FiniteElementState loadCheckpointedState(const std::string& state_name, int cycle);

  /**
   * @brief Accessor for getting a single named finite element dual solution from the physics modules at a given
   * checkpointed cycle index
   *
   * @param state_name The name of the state to retrieve (e.g. "reaction")
   * @param cycle The cycle to retrieve state from
   * @return The named Finite Element Dual
   */
  virtual FiniteElementDual loadCheckpointedDual([[maybe_unused]] const std::string& state_name,
                                                 [[maybe_unused]] int cycle)
  {
    SLIC_ERROR_ROOT(axom::fmt::format("loadCheckpointedDual not enabled in physics module {}", name_));
    return *duals_[0];
  }

  /**
   * @brief Get a timestep increment which has been previously checkpointed at the give cycle
   * @param cycle The previous 'timestep' number where the timestep increment is requested
   * @return The timestep increment
   */
  virtual double getCheckpointedTimestep(int cycle) const;

  /**
   * @brief Initializes the Sidre structure for simulation summary data
   *
   * @param[in] datastore Sidre DataStore where data are saved
   * @param[in] t_final Final time of the simulation
   * @param[in] dt The time step
   */
  virtual void initializeSummary(axom::sidre::DataStore& datastore, const double t_final, const double dt) const;

  /**
   * @brief Saves the summary data to the Sidre Datastore
   *
   * @param[in] datastore Sidre DataStore where curves are saved
   * @param[in] t The current time of the simulation
   */
  virtual void saveSummary(axom::sidre::DataStore& datastore, const double t) const;

  /**
   * @brief Destroy the Base Solver object
   */
  virtual ~BasePhysics() = default;

  /**
   * @brief Returns a reference to the mesh object
   */
  const smith::Mesh& mesh() const;

  /**
   * @brief Returns a reference to the mfem ParMesh object
   */
  const mfem::ParMesh& mfemParMesh() const;

  /**
   * @brief Returns a reference to the mfem ParMesh object
   */
  mfem::ParMesh& mfemParMesh();

  /**
   * @brief Return the name of the physics
   */
  std::string name() const { return name_; }

 protected:
  /**
   * @brief Internally used accessor for getting the shape displacement sensitivity from the physics modules
   *
   * @return The shape displacement sensitivity dual
   */
  const FiniteElementDual& shapeDisplacementSensitivity() const;

  /**
   * @brief Create a paraview data collection for the physics package if requested
   */
  void CreateParaviewDataCollection() const;

  /**
   * @brief Update the paraview states, duals, parameters, and metadata (cycle, time) in preparation for output
   *
   * @param paraview_output_dir The directory to write the paraview output
   */
  void UpdateParaviewDataCollection(const std::string& paraview_output_dir) const;

  /**
   * @brief Protected, non-virtual method to reset physics states to zero.  This does not reset design parameters or
   * shape.
   *
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   */
  void initializeBasePhysicsStates(int cycle, double time);

  /**
   * @brief Accessor for getting all of the primal solutions from the physics modules at a given
   * checkpointed cycle index
   *
   * @param cycle The cycle to retrieve state from
   * @return A map containing the primal field names and their associated FiniteElementStates at the requested cycle
   */
  std::unordered_map<std::string, FiniteElementState> getCheckpointedStates(int cycle);

  /// @brief Name of the physics module
  std::string name_ = {};

  /**
   * @brief The primary mesh
   */
  std::shared_ptr<smith::Mesh> mesh_;

  /**
   * @brief The MPI communicator
   */
  MPI_Comm comm_;

  /**
   * @brief List of finite element primal states associated with this physics module
   */
  std::vector<const smith::FiniteElementState*> states_;

  /**
   * @brief List of finite element adjoint states associated with this physics module
   */
  std::vector<const smith::FiniteElementState*> adjoints_;

  /**
   * @brief List of finite element duals associated with this physics module
   */
  std::vector<const smith::FiniteElementDual*> duals_;

  /**
   * @brief List of adjoint finite element duals associated with this physics module
   */
  std::vector<const smith::FiniteElementState*> dual_adjoints_;

  /// @brief The information needed for the physics parameters stored as Finite Element State fields
  struct ParameterInfo {
    /**
     * @brief Construct a new Parameter Info object
     *
     * @tparam FunctionSpace The templated finite element function space used to construct the parameter field
     * @param mesh The mesh to build the new parameter on
     * @param space The templated finite element function space used to construct the parameter field
     * @param name The name of the new parameter field
     */
    template <typename FunctionSpace>
    ParameterInfo(mfem::ParMesh& mesh, FunctionSpace space, const std::string& name = "")
    {
      state = std::make_unique<FiniteElementState>(mesh, space, name);
      previous_state = std::make_unique<FiniteElementState>(mesh, space, "previous_" + name);
      sensitivity = std::make_unique<FiniteElementDual>(mesh, space, name + "_sensitivity");
      StateManager::storeState(*state);
      StateManager::storeDual(*sensitivity);
    }

    /// The finite element states representing user-defined and owned parameter fields
    std::unique_ptr<smith::FiniteElementState> state;

    /// The finite element state representing the parameter at the previous evaluation
    std::unique_ptr<smith::FiniteElementState> previous_state;

    /**
     * @brief The sensitivities (dual vectors) of the QOI encoded in the adjoint load with respect to each of the input
     * parameter fields
     * @note This quantity is also called the vector-Jacobian product during back propagation in data science.
     */
    std::unique_ptr<smith::FiniteElementDual> sensitivity;
  };

  /// @brief A vector of the parameters associated with this physics module
  std::vector<ParameterInfo> parameters_;

  /// @brief A map containing optionally in-memory checkpointed primal states for transient adjoint solvers
  mutable std::unordered_map<std::string, std::vector<smith::FiniteElementState>> checkpoint_states_;

  /**
   * @brief A container relating a checkpointed cycle and the associated finite element state fields
   *
   * @note This is only used when the disk-based checkpointing is used. It avoids thrashing the disk IO
   * by performing a file open/read/close for every separate retrieval of state checkpoint data for the
   * various primal fields.
   */
  mutable std::unordered_map<std::string, smith::FiniteElementState> cached_checkpoint_states_;

  /// @brief An optional int for disk-based checkpointing containing the cycle number of the last retrieved checkpoint
  mutable std::optional<int> cached_checkpoint_cycle_;

  /**
   *@brief Whether the simulation is time-independent
   */
  bool is_quasistatic_ = true;

  /**
   * @brief Number of significant figures to output for floating-point
   */
  static constexpr int FLOAT_PRECISION_ = 8;

  /**
   * @brief Current time for the forward pass
   */
  double time_;

  /**
   * @brief Current time step
   */
  double dt_;

  /**
   * @brief The maximum time reached for the forward solver
   */
  double max_time_;

  /**
   * @brief The time the forward solver was initialized to
   */
  double min_time_;

  /**
   * @brief A vector of the timestep sizes (i.e. \f$\Delta t\f$) taken by the forward solver
   */
  std::vector<double> timesteps_;

  /**
   * @brief Current cycle (forward pass time iteration count)
   */
  int cycle_;

  /**
   * @brief The maximum cycle (forward pass iteration count) reached by the forward solver
   */
  int max_cycle_;

  /**
   * @brief The cycle the forward solver was initialized to
   */
  int min_cycle_;

  /**
   * @brief The value of time at which the ODE solver wants to evaluate the residual
   */
  double ode_time_point_;

  /**
   * @brief MPI rank
   */
  int mpi_rank_;

  /**
   * @brief MPI size
   */
  int mpi_size_;

  /**
   * @brief DataCollection pointer for optional paraview output
   */
  mutable std::unique_ptr<mfem::ParaViewDataCollection> paraview_dc_;

  /**
   * @brief A optional map of the dual names and duals in grid function form for paraview output
   */
  mutable std::unordered_map<std::string, std::unique_ptr<mfem::ParGridFunction>> paraview_dual_grid_functions_;

  /**
   * @brief A optional view of the shape sensitivity in grid function form for paraview output
   */
  mutable std::unique_ptr<mfem::ParGridFunction> shape_sensitivity_grid_function_;

  /**
   * @brief The shape displacement field
   */
  smith::FiniteElementState shape_displacement_;

  /**
   * @brief The shape displacement field sensitivity
   */
  smith::FiniteElementDual shape_displacement_dual_;

  /**
   * @brief Boundary condition manager instance
   */
  BoundaryConditionManager bcs_;

  /// A flag denoting whether to save the state to disk or memory as needed for dynamic adjoint solves
  bool checkpoint_to_disk_;
};

}  // namespace smith
