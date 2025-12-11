// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <memory>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"

using namespace smith;

void functional_solid_test_robin_condition()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 2;
  constexpr int dim = 3;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid_mechanics_robin_condition_test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";

  std::string mesh_tag{"mesh"};
  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level = 1};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, solid_mechanics::default_linear_options,
                                      solid_mechanics::default_quasistatic_options, "solid_mechanics", mesh);

  solid_mechanics::LinearIsotropic mat{
      1.0,  // mass density
      1.0,  // bulk modulus
      1.0   // shear modulus
  };

  solid_solver.setMaterial(mat, mesh->entireBody());

  // prescribe zero displacement in the y- and z-directions
  // at the supported end of the beam,
  mesh->addDomainOfBoundaryElements("support", by_attr<dim>(1));
  solid_solver.setFixedBCs(mesh->domain("support"), Component::Y + Component::Z);

  // apply an axial displacement at the the tip of the beam
  auto translated_in_x = [](tensor<double, dim>, double t) -> vec3 {
    tensor<double, dim> u{};
    u[0] = t;
    return u;
  };
  mesh->addDomainOfBoundaryElements("tip", by_attr<dim>(2));
  solid_solver.setDisplacementBCs(translated_in_x, mesh->domain("tip"), Component::X);

  solid_solver.addCustomBoundaryIntegral(
      DependsOn<>{},
      [](double /* t */, auto /*position*/, auto displacement, auto /*acceleration*/) {
        auto [u, du_dxi] = displacement;
        auto f = u * 3.0;
        return f;  // define a displacement-proportional traction at the support
      },
      mesh->domain("support"));

  // Finalize the data structures
  solid_solver.completeSetup();

  solid_solver.outputStateToDisk("robin_condition");

  // Perform the quasi-static solve
  int num_steps = 1;
  double tmax = 1.0;
  double dt = tmax / num_steps;
  for (int i = 0; i < num_steps; i++) {
    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk("robin_condition");
  }
}

TEST(SolidMechanics, robin_condition) { functional_solid_test_robin_condition(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
