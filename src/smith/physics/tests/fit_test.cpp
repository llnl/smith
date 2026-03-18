// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/fit.hpp"

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "smith/numerics/functional/functional.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith_config.hpp"

using namespace smith;

int nsamples = 1;  // because mfem doesn't take in unsigned int

int n = 0;  // index of tests used to send the output to different locations

template <typename output_space>
void stress_extrapolation_test()
{
  int serial_refinement = 2;
  int parallel_refinement = 0;

  std::string filename = SMITH_REPO_DIR "/data/meshes/notched_plate.mesh";

  constexpr int p = 2;
  constexpr int dim = 2;

  using input_space = H1<2, dim>;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid_mechanics_J2_test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string mesh_tag{"mesh"};
  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 1.0e-12,
                                           .absolute_tol = 1.0e-12,
                                           .max_iterations = 5000,
                                           .print_level = 1};

  FiniteElementState sigma_J2(mesh->mfemParMesh(), output_space{}, "sigma_J2");

  SolidMechanics<p, dim, smith::Parameters<output_space> > solid_solver(nonlinear_options, linear_options,
                                                                        solid_mechanics::default_quasistatic_options,
                                                                        "solid_mechanics", mesh, {"sigma_J2"});

  solid_mechanics::NeoHookean mat{
      1.0,    // density
      100.0,  // bulk modulus
      50.0    // shear modulus
  };

  solid_solver.setMaterial(mat, mesh->entireBody());

  // prescribe small displacement at each hole, pulling the plate apart
  mesh->addDomainOfBoundaryElements("top_hole", by_attr<dim>(2));
  auto up = [](tensor<double, dim>, double) {
    tensor<double, dim> u{};
    u[1] = 0.01;
    return u;
  };
  solid_solver.setDisplacementBCs(up, mesh->domain("top_hole"));

  mesh->addDomainOfBoundaryElements("bottom_hole", by_attr<dim>(3));
  auto down = [up](tensor<double, dim> X, double time) { return -up(X, time); };
  solid_solver.setDisplacementBCs(down, mesh->domain("bottom_hole"));

  // Finalize the data structures
  solid_solver.completeSetup();

  solid_solver.outputStateToDisk("paraview" + std::to_string(n));

  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  auto u = solid_solver.displacement();

  Empty internal_variables{};

  sigma_J2 = fit<dim, output_space(input_space)>(
      [&](double /*t*/, [[maybe_unused]] auto position, [[maybe_unused]] auto displacement_) {
        mat3 du_dx = to_3x3(get_value(get<1>(displacement_)));
        auto stress = mat(internal_variables, du_dx);
        return tuple{I2(dev(stress)), zero{}};
      },
      mesh->mfemParMesh(), u);

  solid_solver.setParameter(0, sigma_J2);

  solid_solver.outputStateToDisk("paraview" + std::to_string(n));
  n++;
}

TEST(StressExtrapolation, PiecewiseConstant2D) { stress_extrapolation_test<L2<0> >(); }
TEST(StressExtrapolation, PiecewiseLinear2D) { stress_extrapolation_test<H1<1> >(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
