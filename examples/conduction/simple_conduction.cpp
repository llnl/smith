// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file simple_conduction.cpp
 *
 * @brief A simple example of steady-state heat transfer that uses
 * the C++ API to configure the simulation
 */

#include <memory>
#include <set>
#include <string>

// _smith_include_header_start
#include "smith/smith.hpp"
// _smith_include_header_end

// _main_init_start
int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);
  // _main_init_end

  // _statemanager_start
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "without_input_file_example");
  // _statemanager_end

  // _create_mesh_start
  std::string mesh_tag{"mesh"};
  auto mesh = std::make_shared<smith::Mesh>(smith::buildRectangleMesh(10, 10), mesh_tag);
  // _create_mesh_end

  // _create_module_start
  // Create a Heat Transfer class instance with Order 1 and Dimensions of 2
  constexpr int order = 1;
  constexpr int dim = 2;

  smith::HeatTransfer<order, dim> heat_transfer(smith::heat_transfer::default_nonlinear_options,
                                                smith::heat_transfer::default_linear_options,
                                                smith::heat_transfer::default_static_options, "thermal_solver", mesh);
  // _create_module_end

  // _conductivity_start
  constexpr double kappa = 0.5;
  smith::heat_transfer::LinearIsotropicConductor mat(1.0, 1.0, kappa);

  heat_transfer.setMaterial(mat, mesh->entireBody());
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
  return 0;
}
// _exit_end
