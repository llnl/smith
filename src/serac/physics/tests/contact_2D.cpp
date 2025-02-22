// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics_contact.hpp"

#include <functional>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/numerics/functional/domain.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

class ContactTest
    : public testing::Test { 
};

TEST_F(ContactTest, DISABLED_Contact2D)
{
  // NOTE: p must be equal to 1 for now
  constexpr int p = 1;
  constexpr int dim = 2;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string name = "contact_beam";
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  auto mesh = mesh::refineAndDistribute(mfem::Mesh::MakeCartesian2D(6, 20, mfem::Element::QUADRILATERAL, true, 1.0, 1.0), 0, 0);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), "beam_mesh");

  LinearSolverOptions linear_options{.linear_solver = LinearSolver::Strumpack, .print_level = 0};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return;
#endif

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 1.0e-12,
                                           .absolute_tol = 1.0e-12,
                                           .max_iterations = 200,
                                           .print_level = 1};

  ContactOptions contact_options{.method = ContactMethod::SmoothMortar,
  //ContactOptions contact_options{.method = ContactMethod::SingleMortar,
                                 .enforcement = ContactEnforcement::Penalty,
                                 .jacobian = ContactJacobian::Exact};

  SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                             solid_mechanics::default_quasistatic_options, name, "beam_mesh");

  double K = 10.0;
  double G = 0.25;
  solid_mechanics::NeoHookean mat{1.0, K, G};
  Domain material_block = EntireDomain(pmesh);
  solid_solver.setMaterial(mat, material_block);

  // Pass the BC information to the solver object
  Domain support = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));
  solid_solver.setFixedBCs(support);
  auto applied_displacement = [](tensor<double, dim>, double) {
    tensor<double, dim> u{};
    u[1] = -0.15;
    return u;
  };
  auto driven_surface = Domain::ofBoundaryElements(pmesh, by_attr<dim>(3));
  solid_solver.setDisplacementBCs(applied_displacement, driven_surface);

  // Add the contact interaction
  solid_solver.addContactInteraction(0, {2}, {4}, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  // solid_solver.outputStateToDisk(paraview_name);
  printf("simulation over\n");
}

}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}


#include "tribol/physics/MortarUtils.hpp"

using Vec2 = mortar_utils::Vec2;
using Edge = mortar_utils::Edge;

TEST(Evaluate2dIntersections, WhenBIsSmaller) {
  double fixed_gap = 0.1;
  Edge edgeA;
  edgeA[0] = Vec2{1.0, 0.0};
  edgeA[1] = Vec2{0.0, 0.0};

  Edge edgeB;
  edgeB[0] = Vec2{0.2, fixed_gap};
  edgeB[1] = Vec2{0.8, fixed_gap};

  auto [xia, xib, g] = mortar_utils::compute_intersection(edgeA, edgeB,
    [](const Edge& edgeA, const Edge& edgeB) { return mortar_utils::compute_average_normal(edgeA, edgeB); });

  constexpr double tol = 1e-13;
  EXPECT_NEAR(xia[0], 0.2, tol);
  EXPECT_NEAR(xia[1], 0.8, tol);
  EXPECT_NEAR(xib[0], 1.0, tol);
  EXPECT_NEAR(xib[1], 0.0, tol);
  EXPECT_NEAR(g[0], fixed_gap, tol);
  EXPECT_NEAR(g[1], fixed_gap, tol);
}

TEST(Evaluate2dIntersections, WhenAIsSmaller) {
  double fixed_gap = 0.2;
  Edge edgeA;
  edgeA[0] = Vec2{0.9, 0.0};
  edgeA[1] = Vec2{0.3, 0.0};

  Edge edgeB;
  edgeB[0] = Vec2{0.0, fixed_gap};
  edgeB[1] = Vec2{1.0, fixed_gap};

  auto [xia, xib, g] = mortar_utils::compute_intersection(edgeA, edgeB,
    [](const Edge& edgeA, const Edge& edgeB) { return mortar_utils::compute_average_normal(edgeA, edgeB); });

  constexpr double tol = 1e-13;
  EXPECT_NEAR(xia[0], 0.0, tol);
  EXPECT_NEAR(xia[1], 1.0, tol);
  EXPECT_NEAR(xib[0], 0.9, tol);
  EXPECT_NEAR(xib[1], 0.3, tol);
  EXPECT_NEAR(g[0], fixed_gap, tol);
  EXPECT_NEAR(g[1], fixed_gap, tol);
}

TEST(Evaluate2dIntersections, WhenASlidesRightRelativelySlightly) {
  double fixed_gap = 0.2;
  double sliding = 0.9;
  double alpha = 0.3;

  Edge edgeA;
  edgeA[0] = Vec2{1.0 + alpha * sliding, 0.0};
  edgeA[1] = Vec2{0.0 + alpha * sliding, 0.0};

  Edge edgeB;
  edgeB[0] = Vec2{0.0 - (1.0-alpha) * sliding, fixed_gap};
  edgeB[1] = Vec2{1.0 - (1.0-alpha) * sliding, fixed_gap};

  auto [xia, xib, g] = mortar_utils::compute_intersection(edgeA, edgeB,
    [](const Edge& edgeA, const Edge& edgeB) { return mortar_utils::compute_average_normal(edgeA, edgeB); });

  constexpr double tol = 1e-13;
  EXPECT_NEAR(xia[0], sliding, tol);
  EXPECT_NEAR(xia[1], 1.0, tol);
  EXPECT_NEAR(xib[0], 1.0, tol);
  EXPECT_NEAR(xib[1], sliding, tol);
  EXPECT_NEAR(g[0], fixed_gap, tol);
  EXPECT_NEAR(g[1], fixed_gap, tol);
}

TEST(Evaluate2dIntersections, WhenASlideLeftRelativelySlightly) {
  double fixed_gap = -0.13;
  double sliding = 0.6;
  double alpha = 0.24;

  Edge edgeA;
  edgeA[0] = Vec2{1.0 - alpha * sliding, 0.0};
  edgeA[1] = Vec2{0.0 - alpha * sliding, 0.0};

  Edge edgeB;
  edgeB[0] = Vec2{0.0 + (1.0-alpha) * sliding, fixed_gap};
  edgeB[1] = Vec2{1.0 + (1.0-alpha) * sliding, fixed_gap};

  auto [xia, xib, g] = mortar_utils::compute_intersection(edgeA, edgeB,
    [](const Edge& edgeA, const Edge& edgeB) { return mortar_utils::compute_average_normal(edgeA, edgeB); });

  constexpr double tol = 1e-13;
  EXPECT_NEAR(xia[0], 0.0, tol);
  EXPECT_NEAR(xia[1], 1.0-sliding, tol);
  EXPECT_NEAR(xib[0], 1.0-sliding, tol);
  EXPECT_NEAR(xib[1], 0.0, tol);
  EXPECT_NEAR(g[0], fixed_gap, tol);
  EXPECT_NEAR(g[1], fixed_gap, tol);
}

TEST(Evaluate2dIntersections, WhenASlidesRightALot) {
  double fixed_gap = 0.2;
  double sliding = 1.1; // slides out of contact

  Edge edgeA;
  edgeA[0] = Vec2{1.0 + sliding, 0.0};
  edgeA[1] = Vec2{0.0 + sliding, 0.0};

  Edge edgeB;
  edgeB[0] = Vec2{0.0, fixed_gap};
  edgeB[1] = Vec2{1.0, fixed_gap};

  auto [xia, xib, g] = mortar_utils::compute_intersection(edgeA, edgeB,
    [](const Edge& edgeA, const Edge& edgeB) { return mortar_utils::compute_average_normal(edgeA, edgeB); });

  constexpr double tol = 1e-13;
  EXPECT_NEAR(xia[0], 1.0, tol);
  EXPECT_NEAR(xia[1], 1.0, tol);
  EXPECT_NEAR(xib[0], 1.0, tol);
  EXPECT_NEAR(xib[1], 1.0, tol);
  EXPECT_NEAR(g[0], fixed_gap, tol);
  EXPECT_NEAR(g[1], fixed_gap, tol);
}

TEST(Evaluate2dIntersections, WhenASlidesLeftALot) {
  double fixed_gap = 0.2;
  double sliding = -1.1; // slides out of contact

  Edge edgeA;
  edgeA[0] = Vec2{1.0 + sliding, 0.0};
  edgeA[1] = Vec2{0.0 + sliding, 0.0};

  Edge edgeB;
  edgeB[0] = Vec2{0.0, fixed_gap};
  edgeB[1] = Vec2{1.0, fixed_gap};

  auto [xia, xib, g] = mortar_utils::compute_intersection(edgeA, edgeB,
    [](const Edge& edgeA, const Edge& edgeB) { return mortar_utils::compute_average_normal(edgeA, edgeB); });

  constexpr double tol = 1e-13;
  EXPECT_NEAR(xia[0], 0.0, tol);
  EXPECT_NEAR(xia[1], 0.0, tol);
  EXPECT_NEAR(xib[0], 0.0, tol);
  EXPECT_NEAR(xib[1], 0.0, tol);
  EXPECT_NEAR(g[0], fixed_gap, tol);
  EXPECT_NEAR(g[1], fixed_gap, tol);
}