// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <stdexcept>
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
constexpr double thickness = 0.025;
constexpr double end_tol = 1.0e-8;
constexpr double top_tol = 1.0e-8;
std::string solver_name = "TrustRegion";
int print_level = 2;
int nonlinear_max_iterations = 300000;
int trust_subspace_option = static_cast<int>(SubSpaceOptions::NEVER);
int trust_num_leftmost = 1;
int trust_num_past_steps = 0;
bool trust_use_solve_start_direction = false;
bool trust_use_min_residual_direction = false;

NonlinearSolver selectedNonlinearSolver()
{
  if (solver_name == "NewtonLineSearch") {
    return NonlinearSolver::NewtonLineSearch;
  }
  if (solver_name == "TrustRegion") {
    return NonlinearSolver::TrustRegion;
  }

  throw std::runtime_error("Unknown --solver value '" + solver_name +
                           "'. Use NewtonLineSearch or TrustRegion.");
}

void parseCommandLine(int& argc, char** argv)
{
  int write_arg = 1;
  for (int read_arg = 1; read_arg < argc; ++read_arg) {
    const std::string arg = argv[read_arg];
    if (arg.rfind("--solver=", 0) == 0) {
      solver_name = arg.substr(std::string("--solver=").size());
    } else if (arg.rfind("--print-level=", 0) == 0) {
      print_level = std::stoi(arg.substr(std::string("--print-level=").size()));
    } else if (arg.rfind("--nonlinear-max-iterations=", 0) == 0) {
      nonlinear_max_iterations = std::stoi(arg.substr(std::string("--nonlinear-max-iterations=").size()));
    } else if (arg.rfind("--trust-subspace-option=", 0) == 0) {
      trust_subspace_option = std::stoi(arg.substr(std::string("--trust-subspace-option=").size()));
    } else if (arg.rfind("--trust-num-leftmost=", 0) == 0) {
      trust_num_leftmost = std::stoi(arg.substr(std::string("--trust-num-leftmost=").size()));
    } else if (arg.rfind("--trust-num-past-steps=", 0) == 0) {
      trust_num_past_steps = std::stoi(arg.substr(std::string("--trust-num-past-steps=").size()));
    } else if (arg.rfind("--trust-use-solve-start-direction=", 0) == 0) {
      const std::string value = arg.substr(std::string("--trust-use-solve-start-direction=").size());
      trust_use_solve_start_direction = (value == "1" || value == "true" || value == "on");
    } else if (arg.rfind("--trust-use-min-residual-direction=", 0) == 0) {
      const std::string value = arg.substr(std::string("--trust-use-min-residual-direction=").size());
      trust_use_min_residual_direction = (value == "1" || value == "true" || value == "on");
    } else {
      argv[write_arg] = argv[read_arg];
      ++write_arg;
    }
  }
  argc = write_arg;
}

}  // namespace

TEST(ShallowArchBuckling, CompressedThinBeamSnapThrough)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr int p = 1;
  constexpr int dim = 2;
  constexpr int nx = 150;
  constexpr int ny = 6;

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "shallow_arch_buckling");

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian2D(nx, ny, mfem::Element::QUADRILATERAL, true, length, thickness),
      "compressed_beam_mesh", 0, 0);

  mesh->addDomainOfBoundaryElements("left_end",
                                    [](std::vector<vec2> vertices, int) { return average(vertices)[0] < end_tol; });
  mesh->addDomainOfBoundaryElements(
      "right_end", [](std::vector<vec2> vertices, int) { return average(vertices)[0] > length - end_tol; });
  mesh->addDomainOfBoundaryElements(
      "top_face", [](std::vector<vec2> vertices, int) { return average(vertices)[1] > thickness - top_tol; });
  auto globalElementCount = [](int local_count) {
    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_count;
  };
  EXPECT_GT(globalElementCount(mesh->domain("left_end").total_elements()), 0);
  EXPECT_GT(globalElementCount(mesh->domain("right_end").total_elements()), 0);
  EXPECT_GT(globalElementCount(mesh->domain("top_face").total_elements()), 0);

  smith::LinearSolverOptions linear_options{.linear_solver = LinearSolver::CG,
                                            .preconditioner = Preconditioner::HypreJacobi,
                                            .relative_tol = 1.0e-8,
                                            .absolute_tol = 1.0e-14,
                                            .max_iterations = 10000,
                                            .print_level = 0};

  smith::NonlinearSolverOptions nonlinear_options{
      .nonlin_solver = selectedNonlinearSolver(),
      .relative_tol = 1.0e-8,
      .absolute_tol = 1.0e-10,
      .max_iterations = nonlinear_max_iterations,
      .print_level = print_level,
      .subspace_option = static_cast<SubSpaceOptions>(trust_subspace_option),
      .num_leftmost = trust_num_leftmost,
      .trust_num_past_steps = trust_num_past_steps,
      .trust_use_solve_start_direction = trust_use_solve_start_direction,
      .trust_use_min_residual_direction = trust_use_min_residual_direction};

  SolidMechanics<p, dim> solid(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                               "compressed_beam", mesh);

  solid_mechanics::NeoHookean mat{.density = 1.0, .K = 100.0, .G = 10.0};
  solid.setMaterial(mat, mesh->entireBody());
  solid.setFixedBCs(mesh->domain("left_end"));

  constexpr double final_compression = 0.2;
  constexpr double seed_down_traction = 1.0e-5;
  constexpr double final_snap_up_traction = 0.02;
  solid.setDisplacementBCs([](auto, double t) { return vec2{{-final_compression * t, 0.0}}; },
                           mesh->domain("right_end"), Component::X);
  solid.setFixedBCs(mesh->domain("right_end"), Component::Y);
  solid.setTraction(
      [](auto, auto, double t) {
        if (t < 0.5) {
          return vec2{{0.0, -seed_down_traction * (t / 0.5)}};
        }
        const double snap_ramp = (t - 0.5) / 0.5;
        return vec2{{0.0, -seed_down_traction * (1.0 - snap_ramp) + final_snap_up_traction * snap_ramp}};
      },
      mesh->domain("top_face"));

  solid.completeSetup();
  solid.outputStateToDisk("shallow_arch_buckling");

  if (rank == 0) {
    mfem::out << "Compressed thin beam snap-through run: solver = " << solver_name
              << ", trust_subspace_option = " << trust_subspace_option
              << ", trust_num_leftmost = " << trust_num_leftmost
              << ", trust_num_past_steps = " << trust_num_past_steps << '\n';
  }

  constexpr int num_steps = 5;
  for (int step = 0; step < num_steps; ++step) {
    solid.advanceTimestep(1.0 / num_steps);
    if (rank == 0) {
      mfem::out << "Load step " << step + 1 << "/" << num_steps << '\n';
    }
    solid.outputStateToDisk("shallow_arch_buckling");
  }

}

}  // namespace smith

int main(int argc, char* argv[])
{
  smith::parseCommandLine(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
