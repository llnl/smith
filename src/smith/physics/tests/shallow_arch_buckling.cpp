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
constexpr double thickness = 0.25;
constexpr double end_tol = 1.0e-8;
constexpr double top_tol = 1.0e-8;
std::string solver_name = "TrustRegion";
int print_level = 2;
int pcg_block_len = 10;
double pcg_powell_eta = 0.005;
int nonlinear_max_iterations = 30000;

NonlinearSolver selectedNonlinearSolver()
{
  if (solver_name == "NewtonLineSearch") {
    return NonlinearSolver::NewtonLineSearch;
  }
  if (solver_name == "TrustRegion") {
    return NonlinearSolver::TrustRegion;
  }
  if (solver_name == "PcgBlock") {
    return NonlinearSolver::PcgBlock;
  }

  throw std::runtime_error("Unknown --solver value '" + solver_name +
                           "'. Use NewtonLineSearch, TrustRegion, or PcgBlock.");
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
    } else if (arg.rfind("--pcg-block-len=", 0) == 0) {
      pcg_block_len = std::stoi(arg.substr(std::string("--pcg-block-len=").size()));
    } else if (arg.rfind("--pcg-powell-eta=", 0) == 0) {
      pcg_powell_eta = std::stod(arg.substr(std::string("--pcg-powell-eta=").size()));
    } else if (arg.rfind("--nonlinear-max-iterations=", 0) == 0) {
      nonlinear_max_iterations = std::stoi(arg.substr(std::string("--nonlinear-max-iterations=").size()));
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

  constexpr int p = 1;
  constexpr int dim = 2;
  constexpr int nx = 96;
  constexpr int ny = 4;

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
  EXPECT_GT(mesh->domain("left_end").total_elements(), 0);
  EXPECT_GT(mesh->domain("right_end").total_elements(), 0);
  EXPECT_GT(mesh->domain("top_face").total_elements(), 0);

  smith::LinearSolverOptions linear_options{.linear_solver = LinearSolver::CG,
                                            .preconditioner = Preconditioner::HypreJacobi,
                                            .relative_tol = 1.0e-8,
                                            .absolute_tol = 1.0e-14,
                                            .max_iterations = 10000,
                                            .print_level = 0};

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = selectedNonlinearSolver(),
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = nonlinear_max_iterations,
                                                  .print_level = print_level,
                                                  .pcg_block_len = pcg_block_len,
                                                  .pcg_powell_eta = pcg_powell_eta,
                                                  .pcg_max_block_retries = 40};

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

  mfem::out << "Compressed thin beam snap-through run: solver = " << solver_name << '\n';

  constexpr int num_steps = 20;
  int num_converged_steps = 0;
  for (int step = 0; step < num_steps; ++step) {
    solid.advanceTimestep(1.0 / num_steps);
    const auto& nonlinear_solver = solid.equationSolver().nonlinearSolver();
    if (nonlinear_solver.GetConverged()) {
      ++num_converged_steps;
    }
    mfem::out << "Load step " << step + 1 << "/" << num_steps << ": converged = " << nonlinear_solver.GetConverged()
              << ", nonlinear iterations = " << nonlinear_solver.GetNumIterations()
              << ", final relative residual = " << nonlinear_solver.GetFinalRelNorm() << '\n';
    solid.outputStateToDisk("shallow_arch_buckling");
    if (const auto diagnostics = solid.equationSolver().pcgBlockDiagnostics()) {
      mfem::out << "  PCG diagnostics: residuals = " << diagnostics->num_residuals
                << ", hess-vecs = " << diagnostics->num_hess_vecs
                << ", preconditioner applications = " << diagnostics->num_preconds
                << ", Jacobian assemblies = " << diagnostics->num_jacobian_assembles
                << ", preconditioner updates = " << diagnostics->num_preconditioner_updates
                << ", accepted blocks = " << diagnostics->num_blocks
                << ", accepted steps = " << diagnostics->num_accepted_steps
                << ", block rejects = " << diagnostics->num_block_rejects
                << ", prefix accepts = " << diagnostics->num_prefix_accepts
                << ", momentum resets = " << diagnostics->num_momentum_resets
                << ", nonzero beta = " << diagnostics->num_nonzero_beta
                << ", zero beta = " << diagnostics->num_zero_beta
                << ", Powell restarts = " << diagnostics->num_powell_restarts
                << ", descent restarts = " << diagnostics->num_descent_restarts
                << ", negative curvature = " << diagnostics->num_negative_curvature
                << ", trust capped steps = " << diagnostics->num_trust_capped_steps
                << ", line-search backtracks = " << diagnostics->num_line_search_backtracks
                << ", final h_scale = " << diagnostics->final_h_scale
                << ", last trust ratio = " << diagnostics->last_trust_ratio << '\n';
    }
    if (!nonlinear_solver.GetConverged()) {
      throw std::runtime_error("Nonlinear solve failed to converge at load step " + std::to_string(step + 1));
    }
  }

  mfem::out << "Converged load steps: " << num_converged_steps << "/" << num_steps << '\n';
}

}  // namespace smith

int main(int argc, char* argv[])
{
  smith::parseCommandLine(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
