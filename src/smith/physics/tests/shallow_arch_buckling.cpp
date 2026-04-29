// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/state_manager.hpp"

namespace smith {
namespace {

constexpr double length = 10.0;
constexpr double thickness = 0.25;
constexpr double rise = 0.75;
constexpr double end_tol = 1.0e-8;

void warpToShallowArch(smith::Mesh& mesh)
{
  auto& mfem_mesh = mesh.mfemParMesh();
  for (int i = 0; i < mfem_mesh.GetNV(); ++i) {
    auto* vertex = mfem_mesh.GetVertex(i);
    const double xi = 2.0 * vertex[0] / length - 1.0;
    vertex[1] += rise * (1.0 - xi * xi);
  }

  mesh.mfemParMesh().DeleteGeometricFactors();
  auto* nodes = mesh.mfemParMesh().GetNodes();
  auto* coords = nodes->ReadWrite();
  const int vdim = nodes->VectorDim();
  const int scalar_size = nodes->Size() / vdim;

  for (int i = 0; i < scalar_size; ++i) {
    const double x = coords[i];
    const double y = coords[i + scalar_size];
    const double xi = 2.0 * x / length - 1.0;
    coords[i + scalar_size] = y + rise * (1.0 - xi * xi);
  }
}

}  // namespace

TEST(ShallowArchBuckling, NeoHookeanTractionControlled)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 1;
  constexpr int dim = 2;
  constexpr int nx = 48;
  constexpr int ny = 4;

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "shallow_arch_buckling");

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian2D(nx, ny, mfem::Element::QUADRILATERAL, true, length, thickness), "arch_mesh", 0, 0);
  warpToShallowArch(*mesh);

  mesh->addDomainOfBoundaryElements("left_end",
                                    [](std::vector<vec2> vertices, int) { return average(vertices)[0] < end_tol; });
  mesh->addDomainOfBoundaryElements(
      "right_end", [](std::vector<vec2> vertices, int) { return average(vertices)[0] > length - end_tol; });
  mesh->addDomainOfBoundaryElements("top_face", [](std::vector<vec2>, int attr) { return attr == 3; });
  EXPECT_GT(mesh->domain("top_face").total_elements(), 0);

  smith::LinearSolverOptions linear_options{.linear_solver = LinearSolver::CG,
                                            .preconditioner = Preconditioner::HypreAMG,
                                            .relative_tol = 1.0e-8,
                                            .absolute_tol = 1.0e-14,
                                            .max_iterations = 500,
                                            .print_level = 0};

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::PcgBlock,
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = 500,
                                                  .print_level = 2,
                                                  .pcg_block_len = 10,
                                                  .pcg_max_block_retries = 40};

  SolidMechanics<p, dim> solid(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                               "shallow_arch", mesh);

  solid_mechanics::NeoHookean mat{.density = 1.0, .K = 100.0, .G = 10.0};
  solid.setMaterial(mat, mesh->entireBody());
  solid.setFixedBCs(mesh->domain("left_end"));
  solid.setFixedBCs(mesh->domain("right_end"));

  constexpr double final_traction = 0.2;
  solid.setTraction([](auto, auto, double t) { return vec2{{0.0, -final_traction * t}}; }, mesh->domain("top_face"));

  solid.completeSetup();
  solid.outputStateToDisk("shallow_arch_buckling");

  constexpr int num_steps = 40;
  for (int step = 0; step < num_steps; ++step) {
    EXPECT_NO_THROW(solid.advanceTimestep(1.0 / num_steps));
    solid.outputStateToDisk("shallow_arch_buckling");
  }

  const auto diagnostics = solid.equationSolver().pcgBlockDiagnostics();
  ASSERT_TRUE(diagnostics.has_value());
  EXPECT_GT(diagnostics->num_accepted_steps, 0);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
