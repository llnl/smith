// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <set>
#include <string>
#include <complex>
#include <memory>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/heat_transfer.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/thermal_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/solver_config.hpp"

using namespace smith;

void functional_thermal_test_nonlinear()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 2;
  constexpr int dim = 3;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "heat_transfer_nonlinear");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";

  std::string mesh_tag{"mesh"};
  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::NewtonLineSearch,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level = 1};

  HeatTransfer<p, dim> thermal_solver(nonlinear_options, heat_transfer::default_linear_options,
                                      heat_transfer::default_static_options, "heat_transfer", mesh);

  heat_transfer::LinearIsotropicConductor mat{
      1.0,  // mass density
      9.0,  // Specific heat capacity
      21.0  // isotropic thermal conductivity
  };

  thermal_solver.setMaterial(mat, mesh->entireBody());

  // set heat source
  thermal_solver.setSource([](auto, auto, auto, auto) { return 2.0; }, mesh->entireBody());

  // clang-format off
  thermal_solver.addCustomBoundaryIntegral(smith::DependsOn<>{}, [&](auto, auto, auto temperature, auto) {
    static constexpr double radiateConstant = 5.0-7;
    static constexpr double T0              = 21.0;
    using std::pow;
    auto T = smith::get<0>(temperature);
    return radiateConstant * (pow(T, 4.0) - pow(T0, 4.0));
  }, mesh->entireBoundary());
  //  clang-format on

  // prescribe zero temperature at one end of the beam
  std::set<int> support = {1};
  auto          zero    = [](const mfem::Vector&, double) -> double { return 42.0; };
  thermal_solver.setTemperatureBCs(support, zero);

  // Finalize the data structures
  thermal_solver.completeSetup();

  thermal_solver.outputStateToDisk("thermal_nonlinear");

  // Perform the quasi-static solve
  int    num_steps = 1;
  double tmax      = 1.0;
  double dt        = tmax / num_steps;
  for (int i = 0; i < num_steps; i++) {
    thermal_solver.advanceTimestep(dt);
    thermal_solver.outputStateToDisk("thermal_nonlinear");
  }
}

TEST(HeatTransfer, robin_condition) { functional_thermal_test_nonlinear(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
