// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <algorithm>
#include <functional>
#include <fstream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/numerics/functional/domain.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

using namespace serac;

void functional_solid_test_robin_condition()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 2;
  constexpr int dim = 3;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_robin_condition_test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  std::string mesh_tag{"mesh"};

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // _solver_params_start
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level = 1};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, solid_mechanics::default_linear_options,
                                      solid_mechanics::default_quasistatic_options, "solid_mechanics", mesh_tag);
  // _solver_params_end

  solid_mechanics::LinearIsotropic mat{
      1.0,  // mass density
      1.0,  // bulk modulus
      1.0   // shear modulus
  };

  Domain whole_domain = EntireDomain(pmesh);
  solid_solver.setMaterial(mat, whole_domain);

  // prescribe zero displacement in the y- and z-directions
  // at the supported end of the beam,
  Domain support = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));
  solid_solver.setFixedBCs(support, Component::Y + Component::Z);

  // apply an axial displacement at the the tip of the beam
  auto translated_in_x = [](tensor<double, dim>, double t) -> vec3 {
    tensor<double, dim> u{};
    u[0] = t;
    return u;
  };
  Domain tip = Domain::ofBoundaryElements(pmesh, by_attr<dim>(2));
  solid_solver.setDisplacementBCs(translated_in_x, tip, Component::X);

  solid_solver.addCustomBoundaryIntegral(
      DependsOn<>{},
      [](double /* t */, auto /*position*/, auto displacement, auto /*acceleration*/) {
        auto [u, du_dxi] = displacement;
        auto f = u * 3.0;
        return f;  // define a displacement-proportional traction at the support
      },
      support);

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
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
