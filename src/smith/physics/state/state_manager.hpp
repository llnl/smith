// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state_manager.hpp
 *
 * @brief This file contains the declaration of the StateManager class
 */

#pragma once

#include <optional>
#include <unordered_map>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "axom/sidre/core/MFEMSidreDataCollection.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/field_types.hpp"
#include "smith/numerics/functional/quadrature_data.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/geometry.hpp"

namespace smith {

/**
 * @brief Manages the lifetimes of FEState objects such that restarts are abstracted
 * from physics modules
 */
class StateManager {
 public:
  /**
   * @brief Initializes the StateManager with a sidre DataStore (into which state will be written/read)
   * @param[in] ds The DataStore to use
   * @param[in] output_directory The directory to output files to - cannot be empty
   */
  static void initialize(axom::sidre::DataStore& ds, const std::string& output_directory);

  /**
   * @brief Checks if StateManager has a state with the given name
   * @param[in] name A string that uniquely identifies the state
   * @return True if state exists with the given name
   */
  static bool hasState(const std::string& name) { return named_states_.find(name) != named_states_.end(); }

  /**
   * @brief Factory method for creating a new FEState object
   *
   * @tparam FunctionSpace The function space (e.g. H1<1>) to build the finite element state on
   * @param space The function space (e.g. H1<1>) to build the finite element state on
   * @param state_name The name of the new finite element state field
   * @param mesh_tag The tag for the stored mesh used to construct the finite element state
   *
   * @see FiniteElementState::FiniteElementState
   * @note If this is a restart then the options (except for the name) will be ignored
   */
  template <typename FunctionSpace>
  static FiniteElementState newState(FunctionSpace space, const std::string& state_name, const std::string& mesh_tag)
  {
    SLIC_ERROR_ROOT_IF(!ds_, "Smith's data store was not initialized - call StateManager::initialize first");
    SLIC_ERROR_ROOT_IF(!hasMesh(mesh_tag), axom::fmt::format("Mesh tag '{}' not found in the data store", mesh_tag));
    SLIC_ERROR_ROOT_IF(hasState(state_name),
                       axom::fmt::format("StateManager already contains a state named '{}'", state_name));

    FiniteElementState state(mesh(mesh_tag), space, state_name);

    storeState(state);
    return state;
  }

  /**
   * @brief Factory method for creating a new FEState object
   *
   * @param space A finite element space to copy for use in the new state
   * @param state_name The name of the new state
   * @return The constructed finite element state
   */
  static FiniteElementState newState(const mfem::ParFiniteElementSpace& space, const std::string& state_name);

  /**
   * @brief Store a pre-constructed finite element state in the state manager
   *
   * @param state The finite element state to store
   */
  static void storeState(FiniteElementState& state);

  /**
   * @brief Store a pre-constructed Quadrature Data in the state manager
   *
   * @tparam T the type to be created at each quadrature point
   * @param mesh_tag The tag for the stored mesh used to locate the datacollection
   * @param qdata The quadrature data to store
   */
  template <typename T>
  static void storeQuadratureData(const std::string& mesh_tag, std::shared_ptr<QuadratureData<T>> qdata)
  {
    SLIC_ERROR_ROOT_IF(!ds_, "Smith's data store was not initialized - call StateManager::initialize first");
    SLIC_ERROR_ROOT_IF(!hasMesh(mesh_tag),
                       axom::fmt::format("Smith's state manager does not have a mesh with given tag '{}'", mesh_tag));

    constexpr const char* qds_group_name = "quadraturedatas";

    // Get Sidre location for quadrature data inside data collection
    auto& datacoll = datacolls_.at(mesh_tag);
    axom::sidre::Group* bp_group = datacoll.GetBPGroup();  // mesh_datacoll
    // For each geometry type, use i to get both type and name from matching arrays
    for (std::size_t i = 0; i < detail::qdata_geometries.size(); ++i) {
      auto geom_type = detail::qdata_geometries[i];

      // Check if geometry type has any data
      if ((*qdata).data.find(geom_type) != (*qdata).data.end()) {
        auto geom_name = detail::qdata_geometry_names[i];

        // Get axom::Array of states in map
        auto states = (*qdata)[geom_type];

        // Get various size information
        auto num_states = static_cast<axom::IndexType>(states.size());
        SLIC_ERROR_ROOT_IF(num_states == 0, "Number of States should be more than 0 at this point.");
        auto state_size = static_cast<axom::IndexType>(sizeof(*(states.begin())));
        auto total_size = num_states * state_size;
        // Sidre treats information as an array of uint8s
        auto num_uint8s = total_size / static_cast<axom::IndexType>(sizeof(std::uint8_t));

        if (!is_restart_) {
          axom::sidre::Group* qdatas_group = bp_group->createGroup(qds_group_name);

          // Create Sidre group, store basic information, and point Sidre at the array external to Sidre
          // Note: Sidre will not own this data.
          axom::sidre::Group* geom_group = qdatas_group->createGroup(std::string(geom_name));
          geom_group->createViewScalar("num_states", num_states);
          geom_group->createViewScalar("state_size", state_size);
          geom_group->createViewScalar("total_size", total_size);

          // Tell Sidre where the external array is, how large it is (calculated above), and what is in it (faking
          // uint8)
          axom::sidre::View* states_view = geom_group->createView("states");
          states_view->setExternalDataPtr(axom::sidre::UINT8_ID, num_uint8s, states.data());
        } else {
          // Get Sidre group of where the states were stored.
          // Note: this data is not owned by Sidre and the array should have been created at this point but
          // the previous data has not been loaded yet into the array.
          SLIC_ERROR_ROOT_IF(!bp_group->hasGroup(qds_group_name),
                             axom::fmt::format("Loaded Sidre Datastore did not have group for Quadrature Datas"));
          axom::sidre::Group* qdatas_group = bp_group->getGroup(qds_group_name);
          SLIC_ERROR_ROOT_IF(
              !qdatas_group->hasGroup(std::string(geom_name)),
              axom::fmt::format("Loaded Sidre Datastore did not have group for Quadrature Data geometry type '{}'",
                                std::string(geom_name)));
          axom::sidre::Group* geom_group = qdatas_group->getGroup(std::string(geom_name));

          // Verify size correctness
          auto verify_size = [](axom::sidre::Group* group, int value, const std::string& view_name,
                                const std::string& err_msg) {
            SLIC_ERROR_IF(
                !group->hasView(view_name),
                axom::fmt::format("Loaded Sidre Datastore does not have value '{}' for Quadrature Data.", view_name));
            auto prev_value = group->getView(view_name)->getData<axom::IndexType>();
            SLIC_ERROR_IF(value != prev_value, axom::fmt::format(err_msg, value, prev_value));
          };
          verify_size(geom_group, num_states, "num_states",
                      "Current number of Quadrature Data States '{}' does not match value in restart '{}'.");
          verify_size(geom_group, state_size, "state_size",
                      "Current size of Quadrature Data State '{}' does not match value in restart '{}'.");
          verify_size(geom_group, total_size, "total_size",
                      "Current total size of Quadrature Data States '{}' does not match value in restart '{}'.");

          // Tell Sidre where the external array is
          SLIC_ERROR_ROOT_IF(!geom_group->hasView("states"),
                             "Loaded Quadrature Data geometry Sidre view did not have 'states'");
          axom::sidre::View* states_view = geom_group->getView("states");
          states_view->setExternalDataPtr(states.data());

          // TODO: swap this code for the one below after updating Axom
          // Load this set of quadrature data only
          // std::string group_name_to_load = axom::fmt::format("{0}/{1}", qds_group_name, geom_name);
          // datacoll.LoadExternalData("", group_name_to_load);
        }
      }
    }
    if (is_restart_) {
      // NOTE: This call will reload all external buffers from file stored in the DataStore
      // TODO: This should be changed to load only the current material quadrature data after
      // MFEMSidreDatacollection::LoadExternalData is enhanced to allow loading the
      // external data piecemeal. After Axom PR#1555 goes in, uncomment the loadExternalData call
      // above and remove this if block.
      datacoll.LoadExternalData();
    }
  }

  /**
   * @brief Create a shared ptr to a quadrature data buffer for the given material type
   *
   * @tparam T the type to be created at each quadrature point
   * @param mesh_tag The tag for the stored mesh used to locate the datacollection
   * @param domain The spatial domain over which to allocate the quadrature data
   * @param order The order of the discretization of the primal fields
   * @param dim The spatial dimension of the mesh
   * @param initial_state the value to be broadcast to each quadrature point
   * @return shared pointer to quadrature data buffer
   */
  template <typename T>
  static std::shared_ptr<QuadratureData<T>> newQuadratureDataBuffer(const std::string& mesh_tag, const Domain& domain,
                                                                    int order, int dim, T initial_state)
  {
    int Q = order + 1;

    std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> elems = geometry_counts(domain);
    std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> qpts_per_elem{};

    std::vector<mfem::Geometry::Type> geometries;
    if (dim == 2) {
      geometries = {mfem::Geometry::TRIANGLE, mfem::Geometry::SQUARE};
    } else {
      geometries = {mfem::Geometry::TETRAHEDRON, mfem::Geometry::CUBE};
    }

    for (auto geom : geometries) {
      qpts_per_elem[size_t(geom)] = uint32_t(num_quadrature_points(geom, Q));
    }

    auto qdata = std::make_shared<QuadratureData<T>>(elems, qpts_per_elem, initial_state);
    storeQuadratureData<T>(mesh_tag, qdata);
    return qdata;
  }

  /**
   * @brief Checks if StateManager has a dual with the given name
   * @param name A string that uniquely identifies the name
   * @return True if dual exists with the given name
   */
  static bool hasDual(const std::string& name) { return named_duals_.find(name) != named_duals_.end(); }

  /**
   * @brief Factory method for creating a new FEDual object
   *
   * @tparam FunctionSpace The function space (e.g. H1<1>) to build the finite element dual on
   * @param space The function space (e.g. H1<1>) to build the finite element dual on
   * @param dual_name The name of the new finite element dual field
   * @param mesh_tag The tag for the stored mesh used to construct the finite element state
   *
   * @see FiniteElementDual::FiniteElementDual
   * @note If this is a restart then the options (except for the name) will be ignored
   */
  template <typename FunctionSpace>
  static FiniteElementDual newDual(FunctionSpace space, const std::string& dual_name, const std::string& mesh_tag)
  {
    SLIC_ERROR_ROOT_IF(!ds_, "Smith's data store was not initialized - call StateManager::initialize first");
    SLIC_ERROR_ROOT_IF(!hasMesh(mesh_tag), axom::fmt::format("Mesh tag '{}' not found in the data store", mesh_tag));
    SLIC_ERROR_ROOT_IF(hasDual(dual_name),
                       axom::fmt::format("StateManager already contains a dual named '{}'", dual_name));

    auto dual = FiniteElementDual(mesh(mesh_tag), space, dual_name);

    storeDual(dual);
    return dual;
  }
  /**
   * @brief Factory method for creating a new FEDual object
   *
   * @param space A finite element space to copy for use in the new dual
   * @param dual_name The name of the new dual
   * @return The constructed finite element dual
   */
  static FiniteElementDual newDual(const mfem::ParFiniteElementSpace& space, const std::string& dual_name);

  /**
   * @brief Store a pre-constructed finite element dual in the state manager
   *
   * @param dual The finite element dual to store
   */
  static void storeDual(FiniteElementDual& dual);

  /**
   * @brief Updates the StateManager-owned grid function using the values from a given
   * FiniteElementState.
   *
   * This sync operation must occur prior to writing a restart file.
   *
   * @param state The state used to update the internal grid function
   */
  static void updateState(const FiniteElementState& state)
  {
    SLIC_ERROR_ROOT_IF(!hasState(state.name()),
                       axom::fmt::format("State manager does not contain state named '{}'", state.name()));

    state.fillGridFunction(*named_states_[state.name()]);
  }

  /**
   * @brief Updates the StateManager-owned grid function using the values from a given
   * FiniteElementDual.
   *
   * This sync operation must occur prior to writing a restart file.
   *
   * @param dual The dual used to update the internal grid function
   */
  static void updateDual(const FiniteElementDual& dual)
  {
    SLIC_ERROR_ROOT_IF(!hasDual(dual.name()),
                       axom::fmt::format("State manager does not contain dual named '{}'", dual.name()));

    dual.space().GetRestrictionMatrix()->MultTranspose(dual, *named_duals_[dual.name()]);
  }

  /**
   * @brief Updates the Conduit Blueprint state in the datastore and saves to a file
   * @param[in] t The current sim time
   * @param[in] cycle The current iteration number of the simulation
   * @param[in] mesh_tag A string that uniquely identifies the mesh (and accompanying fields) to save
   */
  static void save(const double t, const int cycle, const std::string& mesh_tag);

  /**
   * @brief Loads an existing DataCollection
   * @param[in] cycle_to_load What cycle to load the DataCollection from
   * @param[in] mesh_tag The mesh_tag associated with the DataCollection when it was saved
   * @return The time from specified restart cycle. Otherwise zero.
   */
  static double load(const int cycle_to_load, const std::string& mesh_tag)
  {
    // FIXME: Assumes that if one DataCollection is going to be reloaded all DataCollections will be
    is_restart_ = true;
    return newDataCollection(mesh_tag, cycle_to_load);
  }

  /**
   * @brief Resets the underlying global datacollection object
   *
   * @details After this method, the StateManager is in the same state
   * that it would be after the program started and before any
   * StateManager methods have been called. If the client wants to use
   * StateManager after a call to reset(), the initialize() method
   * must be called.
   */
  static void reset()
  {
    named_states_.clear();
    named_duals_.clear();
    shape_displacements_.clear();
    shape_displacement_duals_.clear();
    datacolls_.clear();
    output_dir_.clear();
    is_restart_ = false;
    ds_ = nullptr;
  };

  /**
   * @brief Checks if StateManager has a mesh with the given mesh_tag
   * @param[in] mesh_tag A string that uniquely identifies the mesh
   * @return True if mesh exists with the given mesh_tag
   */
  static bool hasMesh(const std::string& mesh_tag) { return datacolls_.find(mesh_tag) != datacolls_.end(); }

  /**
   * @brief Gives ownership of mesh to StateManager
   * @param[in] pmesh The mesh to register
   * @param[in] mesh_tag A string that uniquely identifies the mesh
   * @return A pointer to the stored mesh whose ownership was just passed to StateManager
   */
  static mfem::ParMesh& setMesh(std::unique_ptr<mfem::ParMesh> pmesh, const std::string& mesh_tag);

  /**
   * @brief Returns a non-owning reference to mesh held by StateManager
   * @param[in] mesh_tag A string that uniquely identifies the mesh
   * @pre A mesh identified by @a mesh_tag must be registered - either via @p load() or @p setMesh()
   */
  static mfem::ParMesh& mesh(const std::string& mesh_tag);

  /**
   * @brief Get the shape displacement finite element state
   *
   * This is the vector-valued H1 field of order 1 (linear nodal displacements) representing perturbations of the
   * underlying mesh. This is used for shape optimization problems.
   *
   * @param mesh_tag A string that uniquely identifies the mesh
   * @return The linear nodal shape displacement field
   */
  static FiniteElementState& shapeDisplacement(const std::string& mesh_tag);

  /**
   * @brief Get the shape displacement finite element dual
   *
   * This is the linear form which is dual to a vector-valued H1 field of order 1 (linear nodal displacements)
   * representing sensitivities to perturbations of the underlying mesh. This is used for shape optimization problems.
   *
   * @param mesh_tag A string that uniquely identifies the mesh
   * @return The linear nodal shape displacement dual
   */
  static FiniteElementDual& shapeDisplacementDual(const std::string& mesh_tag);

  /**
   * @brief loads the finite element states from a previously checkpointed cycle
   *
   * @param cycle_to_load
   * @param states_to_load
   */
  static void loadCheckpointedStates(int cycle_to_load, std::vector<FiniteElementState*> states_to_load);

  /**
   * @brief Get the shape displacement sensitivity finite element dual
   *
   * This is the vector-valued H1 dual of order 1 representing sensitivities of the shape displacement field of the
   * underlying mesh. This is used for shape optimization problems.
   *
   * @param mesh_tag A string that uniquely identifies the mesh
   * @return The linear shape sensitivity field
   */
  static FiniteElementDual& shapeDisplacementSensitivity(const std::string& mesh_tag);

  /**
   * @brief Returns the datacollection ID for a given mesh
   * @param[in] pmesh Pointer to a mesh (non-owning)
   * @return The collection ID corresponding to the DataCollection that owns the mesh
   * pointed to by @a pmesh.  If @a pmesh is @p nullptr then the default collection ID is returned.
   * @note A raw pointer comparison is used to identify the datacollection, i.e.,
   * @a pmesh must either been returned by either the setMesh() or mesh() method
   */
  static std::string collectionID(const mfem::ParMesh* pmesh);

  /// @brief Returns true if data was loaded into a DataCollection
  static bool isRestart() { return is_restart_; }

  /**
   * @brief Get the current cycle (iteration number) from the underlying datacollection
   *
   * @param mesh_tag The datacollection (mesh name) to query
   * @return The current forward cycle (iteration/timestep number)
   *
   * @note This will return the cycle for the last written or loaded data collection
   */
  static int cycle(std::string mesh_tag);

  /**
   * @brief Get the current simulation time from the underlying datacollection
   *
   * @param mesh_tag The datacollection (mesh name) to query
   * @return The current forward simulation time
   *
   * @note This will return the cycle for the last written or loaded data collection
   */
  static double time(std::string mesh_tag);

 private:
  /**
   * @brief Creates a new datacollection based on a registered mesh
   * @param[in] mesh_tag The mesh name used to name the new datacollection
   * @param[in] cycle_to_load What cycle to load the DataCollection from, if applicable
   * @return The time from specified restart cycle. Otherwise zero.
   */
  static double newDataCollection(const std::string& mesh_tag, const std::optional<int> cycle_to_load = {});

  /**
   * @brief Returns the name of the data collection's name for a given mesh_tag
   * @param[in] mesh_tag The mesh name used to name the new datacollection
   * @return The name of the data collection for the given mesh_tag
   */
  static std::string getCollectionName(const std::string& mesh_tag) { return mesh_tag + "_datacoll"; }

  /**
   * @brief Construct the shape displacement field for the requested mesh
   *
   * @param mesh_tag The mesh to build shape displacement field for
   */
  static void constructShapeFields(const std::string& mesh_tag);

  /**
   * @brief The datacollection instances
   * The object is constructed when the user calls StateManager::initialize.
   */
  static std::unordered_map<std::string, axom::sidre::MFEMSidreDataCollection> datacolls_;

  /// @brief A map of the shape displacement fields for each stored mesh ID
  static std::unordered_map<std::string, std::unique_ptr<FiniteElementState>> shape_displacements_;
  /// @brief A map of the shape displacement duals for each stored mesh ID
  static std::unordered_map<std::string, std::unique_ptr<FiniteElementDual>> shape_displacement_duals_;

  /**
   * @brief Whether this simulation has been restarted from another simulation
   */
  static bool is_restart_;

  /// @brief Pointer (non-owning) to the datastore
  static axom::sidre::DataStore* ds_;
  /// @brief Output directory to which all datacollections are saved
  static std::string output_dir_;

  /// @brief A collection of FiniteElementState names and their corresponding Sidre-owned grid function pointers
  static std::unordered_map<std::string, mfem::ParGridFunction*> named_states_;
  /// @brief A collection of FiniteElementDual names and their corresponding Sidre-owned grid function pointers
  static std::unordered_map<std::string, mfem::ParGridFunction*> named_duals_;
};

/// @brief Check that a mesh satisfies our required properties
void checkMesh(const mfem::ParMesh& pmesh, bool is_restart = false);

}  // namespace smith
