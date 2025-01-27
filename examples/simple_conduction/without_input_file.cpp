// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file without_input_file.cpp
 *
 * @brief A simple example of steady-state heat transfer that uses
 * the C++ API to configure the simulation
 */

// _incl_heat_transfer_header_start
#include "serac/physics/heat_transfer.hpp"
#include "serac/physics/materials/thermal_material.hpp"
// _incl_heat_transfer_header_end
// _incl_state_manager_start
#include "serac/physics/state/state_manager.hpp"
// _incl_state_manager_end
// _incl_infra_start
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
// _incl_infra_end
// _incl_mesh_start
#include "serac/mesh/mesh_utils.hpp"
// _incl_mesh_end

// _main_init_start
int main(int argc, char* argv[])
{
  /*auto [num_procs, rank] = */ serac::initialize(argc, argv);
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "without_input_file_example");
  // _main_init_end
  // _create_mesh_start
  auto mesh = serac::mesh::refineAndDistribute(serac::buildRectangleMesh(10, 10));

  std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);
  // _create_mesh_end

  // _create_module_start
  // Create a Heat Transfer class instance with Order 1 and Dimensions of 2
  constexpr int order = 1;
  constexpr int dim = 2;

  serac::HeatTransfer<order, dim> heat_transfer(
      serac::heat_transfer::default_nonlinear_options, serac::heat_transfer::default_linear_options,
      serac::heat_transfer::default_static_options, "thermal_solver", mesh_tag);
  // _create_module_end

  // _conductivity_start
  constexpr double kappa = 0.5;
  serac::heat_transfer::LinearIsotropicConductor mat(1.0, 1.0, kappa);

  serac::Domain whole_domain = serac::EntireDomain(pmesh);
  heat_transfer.setMaterial(mat, whole_domain);

  // _conductivity_end
  // _bc_start
  const std::set<int> boundary_constant_attributes = {1};
  constexpr double boundary_constant = 1.0;

  auto ebc_func = [boundary_constant](const auto&, auto) { return boundary_constant; };
  heat_transfer.setTemperatureBCs(boundary_constant_attributes, ebc_func);

  const std::set<int> boundary_function_attributes = {2, 3};
  auto boundary_function_coef = [](const auto& vec, auto) { return vec[0] * vec[0] + vec[1] - 1; };
  heat_transfer.setTemperatureBCs(boundary_function_attributes, boundary_function_coef);
  // _bc_end

  // _run_sim_start
  heat_transfer.completeSetup();
  heat_transfer.outputStateToDisk();

  heat_transfer.advanceTimestep(1.0);
  heat_transfer.outputStateToDisk();
  // _run_sim_end

  // _exit_start
  serac::exitGracefully();
}
// _exit_end
