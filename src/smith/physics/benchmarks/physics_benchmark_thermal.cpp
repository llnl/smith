// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/physics/materials/thermal_material.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/heat_transfer.hpp"
#include "smith/physics/mesh.hpp"

template <int p, int dim>
void functional_test_static()
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 1;
  int parallel_refinement = 2;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermal_functional_static_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SMITH_REPO_DIR "/data/meshes/star.mesh" : SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = std::make_shared<smith::Mesh>(filename, "default_mesh", serial_refinement, parallel_refinement);

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  smith::LinearSolverOptions linear_options = {.linear_solver = smith::LinearSolver::CG,
                                               .preconditioner = smith::Preconditioner::HypreAMG,
                                               .relative_tol = 1.0e-6,
                                               .absolute_tol = 1.0e-12,
                                               .max_iterations = 200};

  // Construct a functional-based heat transfer solver
  smith::HeatTransfer<p, dim> thermal_solver(smith::heat_transfer::default_nonlinear_options, linear_options,
                                             smith::heat_transfer::default_static_options, "thermal_functional", mesh);

  smith::tensor<double, dim, dim> cond;

  // Define an anisotropic conductor material model
  if constexpr (dim == 2) {
    cond = {{{5.0, 0.01}, {0.01, 1.0}}};
  }

  if constexpr (dim == 3) {
    cond = {{{1.5, 0.01, 0.0}, {0.01, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
  }

  smith::heat_transfer::LinearConductor<dim> mat(1.0, 1.0, cond);
  thermal_solver.setMaterial(mat, mesh->entireBody());

  // Define the function for the initial temperature and boundary condition
  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, one);
  thermal_solver.setTemperature(one);

  // Define a constant source term
  smith::heat_transfer::ConstantSource source{1.0};
  thermal_solver.setSource(source, mesh->entireBody());

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  thermal_solver.outputStateToDisk();
}

template <int p, int dim>
void functional_test_dynamic()
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 1;
  int parallel_refinement = 2;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermal_functional_dynamic_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SMITH_REPO_DIR "/data/meshes/star.mesh" : SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = std::make_shared<smith::Mesh>(filename, "default_mesh", serial_refinement, parallel_refinement);

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Construct a functional-based heat transfer solver
  smith::HeatTransfer<p, dim> thermal_solver(
      smith::heat_transfer::default_nonlinear_options, smith::heat_transfer::default_linear_options,
      smith::heat_transfer::default_timestepping_options, "thermal_functional", mesh);

  // Define an isotropic conductor material model
  smith::heat_transfer::LinearIsotropicConductor mat(1.0, 1.0, 1.0);

  thermal_solver.setMaterial(mat, mesh->entireBody());

  // Define the function for the initial temperature and boundary condition
  auto initial_temp = [](const mfem::Vector& x, double) -> double {
    if (x[0] < 0.5 || x[1] < 0.5) {
      return 1.0;
    }
    return 0.0;
  };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, initial_temp);
  thermal_solver.setTemperature(initial_temp);

  // Define a constant source term
  smith::heat_transfer::ConstantSource source{1.0};
  thermal_solver.setSource(source, mesh->entireBody());

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the time stepping
  double dt = 0.25;

  for (int i = 0; i < 4; ++i) {
    thermal_solver.outputStateToDisk();
    thermal_solver.advanceTimestep(dt);
  }

  // Output the sidre-based plot files
  thermal_solver.outputStateToDisk();
}

int main(int argc, char* argv[])
{
  smith::ApplicationManager applicationManager(argc, argv);

  // Add metadata
  SMITH_SET_METADATA("test", "thermal_functional");

  SMITH_MARK_BEGIN("2D Linear Static");
  functional_test_static<1, 2>();
  SMITH_MARK_END("2D Linear Static");

  SMITH_MARK_BEGIN("2D Quadratic Static");
  functional_test_static<2, 2>();
  SMITH_MARK_END("2D Quadratic Static");

  SMITH_MARK_BEGIN("3D Linear Static");
  functional_test_static<1, 3>();
  SMITH_MARK_END("3D Linear Static");

  SMITH_MARK_BEGIN("3D Quadratic Static");
  functional_test_static<2, 3>();
  SMITH_MARK_END("3D Quadratic Static");

  SMITH_MARK_BEGIN("2D Linear Dynamic");
  functional_test_dynamic<1, 2>();
  SMITH_MARK_END("2D Linear Dynamic");

  SMITH_MARK_BEGIN("2D Quadratic Dynamic");
  functional_test_dynamic<2, 2>();
  SMITH_MARK_END("2D Quadratic Dynamic");

  SMITH_MARK_BEGIN("3D Linear Dynamic");
  functional_test_dynamic<1, 3>();
  SMITH_MARK_END("3D Linear Dynamic");

  SMITH_MARK_BEGIN("3D Quadratic Dynamic");
  functional_test_dynamic<2, 3>();
  SMITH_MARK_END("3D Quadratic Dynamic");

  return 0;
}
