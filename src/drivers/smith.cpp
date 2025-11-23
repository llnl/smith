// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file smith.cpp
 *
 * @brief Smith: nonlinear implicit thermal-structural driver
 *
 * The purpose of this code is to act as a proxy app for nonlinear implicit mechanics codes at LLNL.
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "axom/core.hpp"
#include "mfem.hpp"

#include "smith/infrastructure/about.hpp"
#include "smith/infrastructure/cli.hpp"
#include "smith/infrastructure/input.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/infrastructure/output.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/heat_transfer.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith_config.hpp"

namespace smith {

/**
 * @brief Define the input file structure for the driver code
 *
 * @param[in] inlet The inlet instance
 */
void defineInputFileSchema(axom::inlet::Inlet& inlet)
{
  // Simulation time parameters
  inlet.addDouble("t_final", "Final time for simulation.").defaultValue(1.0);
  inlet.addDouble("dt", "Time step.").defaultValue(0.25);

  // The mesh options
  auto& mesh_table = inlet.addStruct("main_mesh", "The main mesh for the problem");
  smith::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // The solid mechanics options
  auto& solid_solver_table = inlet.addStruct("solid", "Finite deformation solid mechanics module");
  SolidMechanicsInputOptions::defineInputFileSchema(solid_solver_table);

  // The heat transfer options
  auto& heat_transfer_solver_table = inlet.addStruct("heat_transfer", "Heat transfer module");
  HeatTransferInputOptions::defineInputFileSchema(heat_transfer_solver_table);

  // Verify the input file
  if (!inlet.verify()) {
    SLIC_ERROR_ROOT("Input file failed to verify.");
  }
}

}  // namespace smith

/**
 * @brief Constructs the appropriate physics object using the input file options
 *
 * @param[in] order The order of the discretization
 * @param[in] dim The spatial dimension of the mesh
 * @param[in] solid_mechanics_options Optional container of input options for SolidMechanics physics module
 * @param[in] heat_transfer_options   Optional container of input options for HeatTransfer physics module
 * @param[in] mesh_tag The mesh tag to construct the physics class on
 * @param[in] cycle The simulation timestep cycle to start the physics module at
 * @param[in] t The simulation time to start the physics module at
 *
 * @return Base class instance of the created physics class
 */
std::unique_ptr<smith::BasePhysics> createPhysics(
    int dim, int order, std::optional<smith::SolidMechanicsInputOptions> solid_mechanics_options,
    std::optional<smith::HeatTransferInputOptions> heat_transfer_options, std::string mesh_tag, int cycle, double t)
{
  std::unique_ptr<smith::BasePhysics> main_physics;

  if (solid_mechanics_options) {
    if (order == 1) {
      if (dim == 2) {
        main_physics =
            std::make_unique<smith::SolidMechanics<1, 2>>(*solid_mechanics_options, "smith", mesh_tag, cycle, t);
      } else if (dim == 3) {
        main_physics =
            std::make_unique<smith::SolidMechanics<1, 3>>(*solid_mechanics_options, "smith", mesh_tag, cycle, t);
      }
    } else if (order == 2) {
      if (dim == 2) {
        main_physics =
            std::make_unique<smith::SolidMechanics<2, 2>>(*solid_mechanics_options, "smith", mesh_tag, cycle, t);
      } else if (dim == 3) {
        main_physics =
            std::make_unique<smith::SolidMechanics<2, 3>>(*solid_mechanics_options, "smith", mesh_tag, cycle, t);
      }
    } else if (order == 3) {
      if (dim == 2) {
        main_physics =
            std::make_unique<smith::SolidMechanics<3, 2>>(*solid_mechanics_options, "smith", mesh_tag, cycle, t);
      } else if (dim == 3) {
        main_physics =
            std::make_unique<smith::SolidMechanics<3, 3>>(*solid_mechanics_options, "smith", mesh_tag, cycle, t);
      }
    }
  } else if (heat_transfer_options) {
    if (order == 1) {
      if (dim == 2) {
        main_physics = std::make_unique<smith::HeatTransfer<1, 2>>(*heat_transfer_options, "smith", mesh_tag, cycle, t);
      } else if (dim == 3) {
        main_physics = std::make_unique<smith::HeatTransfer<1, 3>>(*heat_transfer_options, "smith", mesh_tag, cycle, t);
      }
    } else if (order == 2) {
      if (dim == 2) {
        main_physics = std::make_unique<smith::HeatTransfer<2, 2>>(*heat_transfer_options, "smith", mesh_tag, cycle, t);
      } else if (dim == 3) {
        main_physics = std::make_unique<smith::HeatTransfer<2, 3>>(*heat_transfer_options, "smith", mesh_tag, cycle, t);
      }
    } else if (order == 3) {
      if (dim == 2) {
        main_physics = std::make_unique<smith::HeatTransfer<3, 2>>(*heat_transfer_options, "smith", mesh_tag, cycle, t);
      } else if (dim == 3) {
        main_physics = std::make_unique<smith::HeatTransfer<3, 3>>(*heat_transfer_options, "smith", mesh_tag, cycle, t);
      }
    }
  } else {
    SLIC_ERROR_ROOT("Neither solid or heat_transfer blocks specified in the input file.");
  }
  return main_physics;
}

/**
 * @brief Return and check correctness of the order of discretization
 *
 * @param[in] solid_mechanics_options Optional container of input options for SolidMechanics physics module
 * @param[in] heat_transfer_options   Optional container of input options for HeatTransfer physics module
 *
 * @return The order of the discretization
 */
int getOrder(std::optional<smith::SolidMechanicsInputOptions> solid_mechanics_options,
             std::optional<smith::HeatTransferInputOptions> heat_transfer_options)
{
  int order = 0;
  if (solid_mechanics_options) {
    order = solid_mechanics_options->order;
  } else if (heat_transfer_options) {
    order = heat_transfer_options->order;
  } else {
    SLIC_ERROR_ROOT("Neither solid or heat_transfer blocks specified in the input file.");
  }
  SLIC_ERROR_ROOT_IF(order < 1 || order > 3,
                     axom::fmt::format("Invalid solver order '{0}' given. Valid values are 1, 2, or 3.", order));
  return order;
}

/**
 * @brief The main smith driver code
 *
 * @param[in] argc Number of input arguments
 * @param[in] argv The vector of input arguments
 *
 * @return The return code
 */
int main(int argc, char* argv[])
{
  smith::ApplicationManager applicationManager(argc, argv);

  // Handle Command line
  std::unordered_map<std::string, std::string> cli_opts =
      smith::cli::defineAndParse(argc, argv, "Smith: a high order nonlinear thermomechanical simulation code");

  // Optionally, print about info and quit
  // TODO: add option for just version and a longer for about?
  bool print_version = cli_opts.find("version") != cli_opts.end();
  if (print_version) {
    SLIC_INFO(smith::about());
    return 0;
  }

  // Output helpful run information
  smith::printRunInfo();
  smith::cli::printGiven(cli_opts);

  // Read input file
  std::string input_file_path = "";
  auto search = cli_opts.find("input-file");
  if (search != cli_opts.end()) {
    input_file_path = search->second;
  }

  // Output directory used for all files written to the file system.
  // Example of outputted files:
  //
  // * Inlet docs + input file value file
  // * StateManager state files
  // * Summary file
  std::string output_directory = "";
  search = cli_opts.find("output-directory");
  if (search != cli_opts.end()) {
    output_directory = search->second;
  }
  axom::utilities::filesystem::makeDirsForPath(output_directory);

  search = cli_opts.find("paraview-directory");

  std::optional<std::string> paraview_output_dir = {};
  if (search != cli_opts.end()) {
    paraview_output_dir = search->second;
    axom::utilities::filesystem::makeDirsForPath(*paraview_output_dir);
  }

  // Check if a restart was requested
  std::optional<int> restart_cycle;
  if (auto cycle = cli_opts.find("restart-cycle"); cycle != cli_opts.end()) {
    restart_cycle = std::stoi(cycle->second);
  }

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Intialize MFEMSidreDataCollection
  smith::StateManager::initialize(datastore, output_directory);

  // Initialize Inlet and read input file
  auto inlet = smith::input::initialize(datastore, input_file_path);
  smith::defineInputFileSchema(inlet);

  // Optionally, create input file documentation and quit
  bool create_input_file_docs = cli_opts.find("create-input-file-docs") != cli_opts.end();
  if (create_input_file_docs) {
    std::string input_docs_path = axom::utilities::filesystem::joinPath(output_directory, "smith_input.rst");
    inlet.write(axom::inlet::SphinxWriter(input_docs_path));
    return 0;
  }

  // Optionally, print unused entries in input file and quit
  if (cli_opts.find("print-unused") != cli_opts.end()) {
    const std::vector<std::string> all_unexpected_names = inlet.unexpectedNames();
    if (all_unexpected_names.size() != 0) {
      SLIC_INFO("Printing unused entries in input file:");
      for (auto& x : all_unexpected_names) {
        SLIC_INFO("  " << x);
      }
    } else {
      SLIC_INFO("No unused entries in input file.");
    }
    return 0;
  }

  // Save input values to file
  std::string input_values_path = axom::utilities::filesystem::joinPath(output_directory, "smith_input_values.json");
  datastore.getRoot()->getGroup("input_file")->save(input_values_path, "json");

  // Initialize/set the time information
  double t = 0;
  double t_final = inlet["t_final"];
  double dt = inlet["dt"];
  int cycle = 0;

  std::string mesh_tag{"mesh"};

  // Not restarting, so we need to create the mesh and register it with the StateManager
  if (!restart_cycle) {
    // Build the mesh
    auto mesh_options = inlet["main_mesh"].get<smith::mesh::InputOptions>();
    if (const auto file_opts = std::get_if<smith::mesh::FileInputOptions>(&mesh_options.extra_options)) {
      file_opts->absolute_mesh_file_name =
          smith::input::findMeshFilePath(file_opts->relative_mesh_file_name, input_file_path);
    }
    auto mesh = smith::mesh::buildParallelMesh(mesh_options);
    smith::StateManager::setMesh(std::move(mesh), mesh_tag);
  } else {
    // If restart_cycle is non-empty, then this is a restart run and the data will be loaded here
    t = smith::StateManager::load(*restart_cycle, mesh_tag);
    cycle = *restart_cycle;
  }

  // Create nullable containers for the solid and heat transfer input file options
  std::optional<smith::SolidMechanicsInputOptions> solid_mechanics_options;
  std::optional<smith::HeatTransferInputOptions> heat_transfer_options;

  // If the blocks exist, read the appropriate input file options
  if (inlet.isUserProvided("solid")) {
    solid_mechanics_options = inlet["solid"].get<smith::SolidMechanicsInputOptions>();
  }
  if (inlet.isUserProvided("heat_transfer")) {
    heat_transfer_options = inlet["heat_transfer"].get<smith::HeatTransferInputOptions>();
  }

  // Get dimension and order of problem
  int dim = smith::StateManager::mesh(mesh_tag).Dimension();
  SLIC_ERROR_ROOT_IF(dim < 2 || dim > 3,
                     axom::fmt::format("Invalid mesh dimension '{0}' provided. Valid values are 2 or 3.", dim));
  int order = getOrder(solid_mechanics_options, heat_transfer_options);

  // Create the physics object
  auto main_physics = createPhysics(dim, order, solid_mechanics_options, heat_transfer_options, mesh_tag, cycle, t);

  // Complete the solver setup
  main_physics->completeSetup();

  main_physics->initializeSummary(datastore, t_final, dt);

  // Enter the time step loop.
  bool last_step = false;
  while (!last_step) {
    // Flush all messages held by the logger
    smith::logger::flush();

    // Compute the real timestep. This may be less than dt for the last timestep.
    double dt_real = std::min(dt, t_final - t);

    // Compute current time
    t = t + dt_real;

    // Print the timestep information
    SLIC_INFO_ROOT("step " << cycle << ", t = " << t);

    // Solve the physics module appropriately
    main_physics->advanceTimestep(dt_real);

    // Output a visualization file
    main_physics->outputStateToDisk(paraview_output_dir);

    // Save curve data to Sidre datastore to be output later
    main_physics->saveSummary(datastore, t);

    // Determine if this is the last timestep
    last_step = (t >= t_final - 1e-8 * dt);

    // Increment cycle
    cycle++;
  }

  // Output summary file (basic run info and curve data)
  smith::output::outputSummary(datastore, output_directory);

  return 0;
}
