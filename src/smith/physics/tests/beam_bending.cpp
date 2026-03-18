// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"

namespace smith {

TEST(BeamBending, TwoDimensional)
{
  constexpr int p = 1;
  constexpr int dim = 2;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "beam_bending_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/beam-quad.mesh";

  std::string mesh_tag{"mesh"};
  auto mesh = std::make_shared<smith::Mesh>(filename, mesh_tag, 0, 0);

  smith::LinearSolverOptions linear_options{.linear_solver = LinearSolver::GMRES,
                                            .preconditioner = Preconditioner::HypreAMG,
                                            .relative_tol = 1.0e-6,
                                            .absolute_tol = 1.0e-14,
                                            .max_iterations = 500,
                                            .print_level = 1};

  // TODO (EBC): investigate sundials usage (or deprecate sundials).  the result is not converging with these options
  // #ifdef SMITH_USE_SUNDIALS
  //   smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::KINFullStep,
  //                                                   .relative_tol = 1.0e-12,
  //                                                   .absolute_tol = 1.0e-12,
  //                                                   .max_iterations = 5000,
  //                                                   .print_level = 1};
  // #else
  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level = 1};
  // #endif

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
  // NOTE (EBC): This doesn't catch the error sundials was reporting. Sundials 7.0.0 has more robust error reporting, so
  // we could potentially catch the sundials error when (if) we upgrade.
  EXPECT_NO_THROW(solid_solver.advanceTimestep(1.0));

  // Output the sidre-based plot files
  solid_solver.outputStateToDisk("paraview_output");

  constexpr double l2_disp_regression = 19.741914557926425;
  EXPECT_NEAR(mfem::ParNormlp(solid_solver.displacement(), 2, MPI_COMM_WORLD), l2_disp_regression, 1.0e-12);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
