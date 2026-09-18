// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
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
int print_level = 0;
int nonlinear_max_iterations = 150;
int trust_subspace_option = static_cast<int>(SubSpaceOptions::WHEN_INDEFINITE_OR_BOUNDARY);
int trust_num_leftmost = 2;
int trust_num_previous_steps = 4;

NonlinearSolver selectedNonlinearSolver()
{
  if (solver_name == "NewtonLineSearch") {
    return NonlinearSolver::NewtonLineSearch;
  }
  if (solver_name == "TrustRegion") {
    return NonlinearSolver::TrustRegion;
  }

  throw std::runtime_error("Unknown --solver value '" + solver_name + "'. Use NewtonLineSearch or TrustRegion.");
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
    } else if (arg.rfind("--trust-num-previous-steps=", 0) == 0) {
      trust_num_previous_steps = std::stoi(arg.substr(std::string("--trust-num-previous-steps=").size()));
    } else {
      argv[write_arg] = argv[read_arg];
      ++write_arg;
    }
  }
  argc = write_arg;
}

bool globallyFinite(const FiniteElementVector& vector)
{
  int local_finite = 1;
  for (int index = 0; index < vector.Size(); ++index) {
    local_finite = local_finite && std::isfinite(vector[index]);
  }

  int global_finite = 0;
  MPI_Allreduce(&local_finite, &global_finite, 1, MPI_INT, MPI_LAND, vector.comm());
  return global_finite != 0;
}

std::pair<double, double> globalComponentExtrema(const FiniteElementState& state, int component)
{
  const auto& space = state.space();
  const mfem::ParGridFunction& grid_function = state.gridFunction();
  double local_minimum = std::numeric_limits<double>::max();
  double local_maximum = std::numeric_limits<double>::lowest();
  for (int degree_of_freedom = 0; degree_of_freedom < space.GetNDofs(); ++degree_of_freedom) {
    const double value = grid_function[space.DofToVDof(degree_of_freedom, component)];
    local_minimum = std::min(local_minimum, value);
    local_maximum = std::max(local_maximum, value);
  }

  double global_minimum = 0.0;
  double global_maximum = 0.0;
  MPI_Allreduce(&local_minimum, &global_minimum, 1, MPI_DOUBLE, MPI_MIN, state.comm());
  MPI_Allreduce(&local_maximum, &global_maximum, 1, MPI_DOUBLE, MPI_MAX, state.comm());
  return {global_minimum, global_maximum};
}

}  // namespace

TEST(ShallowArchBuckling, CompressedThinBeamSnapThrough)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 1;
  constexpr int dim = 2;
  constexpr int nx = 40;
  constexpr int ny = 2;
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
                                            .max_iterations = 100000,
                                            .print_level = 0};

  smith::NonlinearSolverOptions nonlinear_options{
      .nonlin_solver = selectedNonlinearSolver(),
      .relative_tol = 1.0e-8,
      .absolute_tol = 1.0e-10,
      .max_iterations = nonlinear_max_iterations,
      .print_level = print_level,
      .subspace_option = static_cast<SubSpaceOptions>(trust_subspace_option),
      .num_leftmost = trust_num_leftmost,
      .num_previous_steps = trust_num_previous_steps};

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

  SLIC_INFO_ROOT(
      std::format("Compressed thin beam snap-through run: solver = {}, trust_subspace_option = {}, "
                  "trust_num_leftmost = {}",
                  solver_name, trust_subspace_option, trust_num_leftmost));

  constexpr int num_steps = 5;
  for (int step = 0; step < num_steps; ++step) {
    solid.advanceTimestep(1.0 / num_steps);
    SLIC_INFO_ROOT(std::format("Load step {}/{}", step + 1, num_steps));

    EXPECT_TRUE(globallyFinite(solid.displacement()));
    EXPECT_TRUE(globallyFinite(solid.reactions()));
    const auto [minimum_vertical_displacement, maximum_vertical_displacement] =
        globalComponentExtrema(solid.displacement(), 1);
    if (step == 1) {
      EXPECT_LT(minimum_vertical_displacement, -0.5);
      EXPECT_NEAR(maximum_vertical_displacement, 0.0, 1.0e-12);
    } else if (step == 2) {
      EXPECT_NEAR(minimum_vertical_displacement, 0.0, 1.0e-12);
      EXPECT_GT(maximum_vertical_displacement, 1.3);
    } else if (step == num_steps - 1) {
      EXPECT_NEAR(minimum_vertical_displacement, 0.0, 1.0e-12);
      EXPECT_NEAR(maximum_vertical_displacement, 2.52575315, 1.0e-6);
    }
  }

  const double reaction_norm = std::sqrt(innerProduct(solid.reactions(), solid.reactions()));
  EXPECT_NEAR(reaction_norm, 0.98420158, 1.0e-6);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  smith::parseCommandLine(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
