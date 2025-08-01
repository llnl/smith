// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "mpi.h"
#include "mfem.hpp"

#include "serac/physics/solid_mechanics.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/numerics/solver_config.hpp"

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

  std::string mesh_tag{"mesh"};
  auto mesh = std::make_shared<serac::Mesh>(filename, mesh_tag, 0, 0);

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
                                      "solid_mechanics", mesh);

  double K = 1.91666666666667;
  double G = 1.0;
  solid_mechanics::StVenantKirchhoff mat{1.0, K, G};
  solid_solver.setMaterial(mat, mesh->entireBody());

  std::string support_domain_name = "support";
  mesh->addDomainOfBoundaryElements(support_domain_name, by_attr<dim>(1));
  solid_solver.setFixedBCs(mesh->domain(support_domain_name));

  std::string top_face_domain_name = "top_face";
  mesh->addDomainOfBoundaryElements(
      top_face_domain_name, [](std::vector<vec2> vertices, int /*attr*/) { return (average(vertices)[1] > 0.99); });

  solid_solver.setTraction([](auto /*x*/, auto n, auto /*t*/) { return -0.01 * n; },
                           mesh->domain(top_face_domain_name));

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
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
