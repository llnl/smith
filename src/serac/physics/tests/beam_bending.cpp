// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <fstream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

TEST(BeamBending, TwoDimensional)
{
  constexpr int p = 1;
  constexpr int dim = 2;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "beam_bending_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-quad.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), 0, 0);

  std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  serac::LinearSolverOptions linear_options{.linear_solver = LinearSolver::GMRES,
                                            .preconditioner = Preconditioner::HypreAMG,
                                            .relative_tol = 1.0e-6,
                                            .absolute_tol = 1.0e-14,
                                            .max_iterations = 500,
                                            .print_level = 1};

#ifdef SERAC_USE_SUNDIALS
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::KINFullStep,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level = 1};
#else
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level = 1};
#endif

  SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      "solid_mechanics", mesh_tag);

  double K = 1.91666666666667;
  double G = 1.0;
  solid_mechanics::StVenantKirchhoff mat{1.0, K, G};
  Domain material_block = EntireDomain(pmesh);
  solid_solver.setMaterial(mat, material_block);

  Domain support = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));
  solid_solver.setFixedBCs(support);

  Domain top_face = Domain::ofBoundaryElements(
      pmesh, [](std::vector<vec2> vertices, int /*attr*/) { return (average(vertices)[1] > 0.99); });

  solid_solver.setTraction([](auto /*x*/, auto n, auto /*t*/) { return -0.01 * n; }, top_face);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  solid_solver.advanceTimestep(1.0);

  // Output the sidre-based plot files
  solid_solver.outputStateToDisk("paraview_output");
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
