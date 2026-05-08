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
int pcg_block_len = 10;
double pcg_powell_eta = 0.005;
int nonlinear_max_iterations = 300000;
bool pcg_diagonal_preconditioner = false;
int trust_subspace_option = static_cast<int>(SubSpaceOptions::NEVER);
int trust_num_leftmost = 1;
int trust_num_past_steps = 0;
int trust_nonmonotone_window = 0;
bool trust_use_jacobian_operator = false;
bool trust_use_cubic_subspace = false;
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
    } else if (arg.rfind("--pcg-diagonal-preconditioner=", 0) == 0) {
      const std::string value = arg.substr(std::string("--pcg-diagonal-preconditioner=").size());
      pcg_diagonal_preconditioner = (value == "1" || value == "true" || value == "on");
    } else if (arg.rfind("--trust-subspace-option=", 0) == 0) {
      trust_subspace_option = std::stoi(arg.substr(std::string("--trust-subspace-option=").size()));
    } else if (arg.rfind("--trust-num-leftmost=", 0) == 0) {
      trust_num_leftmost = std::stoi(arg.substr(std::string("--trust-num-leftmost=").size()));
    } else if (arg.rfind("--trust-num-past-steps=", 0) == 0) {
      trust_num_past_steps = std::stoi(arg.substr(std::string("--trust-num-past-steps=").size()));
    } else if (arg.rfind("--trust-nonmonotone-window=", 0) == 0) {
      trust_nonmonotone_window = std::stoi(arg.substr(std::string("--trust-nonmonotone-window=").size()));
    } else if (arg.rfind("--trust-use-jacobian-operator=", 0) == 0) {
      const std::string value = arg.substr(std::string("--trust-use-jacobian-operator=").size());
      trust_use_jacobian_operator = (value == "1" || value == "true" || value == "on");
    } else if (arg.rfind("--trust-use-cubic-subspace=", 0) == 0) {
      const std::string value = arg.substr(std::string("--trust-use-cubic-subspace=").size());
      trust_use_cubic_subspace = (value == "1" || value == "true" || value == "on");
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
      .trust_nonmonotone_window = trust_nonmonotone_window,
      .trust_use_jacobian_operator = trust_use_jacobian_operator,
      .trust_use_cubic_subspace = trust_use_cubic_subspace,
      .subspace_option = static_cast<SubSpaceOptions>(trust_subspace_option),
      .num_leftmost = trust_num_leftmost,
      .trust_num_past_steps = trust_num_past_steps,
      .trust_use_solve_start_direction = trust_use_solve_start_direction,
      .trust_use_min_residual_direction = trust_use_min_residual_direction,
      .pcg_block_len = pcg_block_len,
      .pcg_powell_eta = pcg_powell_eta,
      .pcg_max_block_retries = 40,
      .pcg_use_jacobian_diagonal_preconditioner = pcg_diagonal_preconditioner};

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
              << ", trust_num_past_steps = " << trust_num_past_steps
              << ", trust_nonmonotone_window = " << trust_nonmonotone_window
              << ", trust_use_jacobian_operator = " << trust_use_jacobian_operator
              << ", trust_use_cubic_subspace = " << trust_use_cubic_subspace
              << ", pcg_diagonal_preconditioner = " << pcg_diagonal_preconditioner << '\n';
  }

  constexpr int num_steps = 5;
  int num_converged_steps = 0;
  for (int step = 0; step < num_steps; ++step) {
    solid.resetJacobianTimings();
    solid.advanceTimestep(1.0 / num_steps);
    const auto& nonlinear_solver = solid.equationSolver().nonlinearSolver();
    if (nonlinear_solver.GetConverged()) {
      ++num_converged_steps;
    }
    if (rank == 0) {
      mfem::out << "Load step " << step + 1 << "/" << num_steps
                << ": converged = " << nonlinear_solver.GetConverged()
                << ", nonlinear iterations = " << nonlinear_solver.GetNumIterations()
                << ", final relative residual = " << nonlinear_solver.GetFinalRelNorm() << '\n';
    }
    solid.outputStateToDisk("shallow_arch_buckling");
    if (rank == 0 && print_level >= 1) {
      if (const auto diagnostics = solid.equationSolver().pcgBlockDiagnostics()) {
        mfem::out << "  PCG diagnostics: residuals = " << diagnostics->num_residuals
                  << ", hess-vecs = " << diagnostics->num_hess_vecs
                  << ", preconditioner applications = " << diagnostics->num_preconds
                  << ", Jacobian assemblies = " << diagnostics->num_jacobian_assembles
                  << ", Jacobian operator evals = " << diagnostics->num_jacobian_operator_evals
                  << ", diagonal assemblies = " << diagnostics->num_diagonal_assembles
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
        mfem::out << "  PCG timings: residual = " << diagnostics->residual_seconds
                  << ", hess-vec = " << diagnostics->hess_vec_seconds
                  << ", operator hess-vec = " << diagnostics->jacobian_operator_hess_vec_seconds
                  << ", assembled hess-vec = " << diagnostics->assembled_hess_vec_seconds
                  << ", matrix-free hess-vec = " << diagnostics->matrix_free_hess_vec_seconds
                  << ", preconditioner = " << diagnostics->preconditioner_seconds
                  << ", Jacobian operator eval = " << diagnostics->jacobian_operator_eval_seconds
                  << ", Jacobian assembly = " << diagnostics->jacobian_assembly_seconds
                  << ", diagonal assembly = " << diagnostics->diagonal_assembly_seconds
                  << ", diagonal invert = " << diagnostics->diagonal_invert_seconds
                  << ", preconditioner update = " << diagnostics->preconditioner_update_seconds
                  << ", preconditioner setup = " << diagnostics->preconditioner_setup_seconds << '\n';
      }
      if (const auto diagnostics = solid.equationSolver().trustRegionDiagnostics()) {
        const double operator_timed_seconds =
            diagnostics->residual_seconds + diagnostics->hess_vec_seconds + diagnostics->preconditioner_seconds +
            diagnostics->jacobian_operator_eval_seconds + diagnostics->diagonal_assembly_seconds +
            diagnostics->diagonal_invert_seconds + diagnostics->jacobian_assembly_seconds +
            diagnostics->preconditioner_update_seconds;
        const double assembled_hess_vec_seconds =
            diagnostics->hess_vec_seconds - diagnostics->jacobian_operator_hess_vec_seconds;
        mfem::out << "  TrustRegion diagnostics: residuals = " << diagnostics->num_residuals
                  << ", hess-vecs = " << diagnostics->num_hess_vecs
                  << ", model hess-vecs = " << diagnostics->num_model_hess_vecs
                  << ", cauchy hess-vecs = " << diagnostics->num_cauchy_hess_vecs
                  << ", line-search hess-vecs = " << diagnostics->num_line_search_hess_vecs
                  << ", preconditioner applications = " << diagnostics->num_preconds
                  << ", Jacobian assemblies = " << diagnostics->num_jacobian_assembles
                  << ", Jacobian operator evals = " << diagnostics->num_jacobian_operator_evals
                  << ", diagonal assemblies = " << diagnostics->num_diagonal_assembles
                  << ", CG iterations = " << diagnostics->num_cg_iterations
                  << ", subspace solves = " << diagnostics->num_subspace_solves
                  << ", subspace leftmost hess-vecs = " << diagnostics->num_subspace_leftmost_hess_vecs
                  << ", subspace hess-vec batches = " << diagnostics->num_subspace_hess_vec_batches
                  << ", subspace batched hess-vecs = " << diagnostics->num_subspace_batched_hess_vecs
                  << ", subspace past-step vectors = " << diagnostics->num_subspace_past_step_vectors
                  << ", subspace past-step hess-vecs = " << diagnostics->num_subspace_past_step_hess_vecs
                  << ", quadratic subspace solves = " << diagnostics->num_quadratic_subspace_solves
                  << ", cubic subspace attempts = " << diagnostics->num_cubic_subspace_attempts
                  << ", cubic subspace uses = " << diagnostics->num_cubic_subspace_uses
                  << ", cubic subspace quadratic fallbacks = " << diagnostics->num_cubic_subspace_quadratic_fallbacks
                  << ", nonmonotone work accepts = " << diagnostics->num_nonmonotone_work_accepts
                  << ", monotone work would reject = " << diagnostics->num_monotone_work_would_reject
                  << ", preconditioner updates = " << diagnostics->num_preconditioner_updates << '\n';
        mfem::out << "  TrustRegion timings: total = " << diagnostics->total_seconds
                  << ", operator-timed = " << operator_timed_seconds << ", residual = " << diagnostics->residual_seconds
                  << ", hess-vec = " << diagnostics->hess_vec_seconds
                  << ", model hess-vec = " << diagnostics->model_hess_vec_seconds
                  << ", cauchy hess-vec = " << diagnostics->cauchy_hess_vec_seconds
                  << ", line-search hess-vec = " << diagnostics->line_search_hess_vec_seconds
                  << ", operator hess-vec = " << diagnostics->jacobian_operator_hess_vec_seconds
                  << ", assembled hess-vec = " << assembled_hess_vec_seconds
                  << ", preconditioner = " << diagnostics->preconditioner_seconds
                  << ", Jacobian operator eval = " << diagnostics->jacobian_operator_eval_seconds
                  << ", diagonal assembly = " << diagnostics->diagonal_assembly_seconds
                  << ", diagonal invert = " << diagnostics->diagonal_invert_seconds
                  << ", model solve = " << diagnostics->model_solve_seconds
                  << ", subspace = " << diagnostics->subspace_seconds
                  << ", subspace leftmost = " << diagnostics->subspace_leftmost_seconds
                  << ", subspace hess-vec batches = " << diagnostics->subspace_hess_vec_batch_seconds
                  << ", subspace filter = " << diagnostics->subspace_filter_seconds
                  << ", subspace backend = " << diagnostics->subspace_backend_seconds
                  << ", subspace project A = " << diagnostics->subspace_project_A_seconds
                  << ", subspace project gram = " << diagnostics->subspace_project_gram_seconds
                  << ", subspace project b = " << diagnostics->subspace_project_b_seconds
                  << ", subspace basis = " << diagnostics->subspace_basis_seconds
                  << ", subspace reduced A = " << diagnostics->subspace_reduced_A_seconds
                  << ", subspace dense eigensystem = " << diagnostics->subspace_dense_eigensystem_seconds
                  << ", subspace dense trust solve = " << diagnostics->subspace_dense_trust_solve_seconds
                  << ", subspace reconstruct solution = " << diagnostics->subspace_reconstruct_solution_seconds
                  << ", subspace reconstruct leftmost = " << diagnostics->subspace_reconstruct_leftmost_seconds
                  << ", subspace finalize = " << diagnostics->subspace_finalize_seconds
                  << ", cauchy point = " << diagnostics->cauchy_point_seconds
                  << ", dogleg = " << diagnostics->dogleg_seconds
                  << ", line search = " << diagnostics->line_search_seconds << ", dot = " << diagnostics->dot_seconds
                  << ", dot count = " << diagnostics->num_dot_products
                  << ", dot reductions = " << diagnostics->num_dot_reductions
                  << ", model dots = " << diagnostics->num_model_dot_products << " / " << diagnostics->model_dot_seconds
                  << ", cauchy dots = " << diagnostics->num_cauchy_dot_products << " / "
                  << diagnostics->cauchy_dot_seconds << ", dogleg dots = " << diagnostics->num_dogleg_dot_products
                  << " / " << diagnostics->dogleg_dot_seconds
                  << ", line-search dots = " << diagnostics->num_line_search_dot_products << " / "
                  << diagnostics->line_search_dot_seconds << ", setup dots = " << diagnostics->num_setup_dot_products
                  << " / " << diagnostics->setup_dot_seconds
                  << ", vector update = " << diagnostics->vector_update_seconds
                  << ", vector copy/scale = " << diagnostics->vector_copy_scale_seconds
                  << ", projection = " << diagnostics->projection_seconds
                  << ", Jacobian assembly = " << diagnostics->jacobian_assembly_seconds
                  << ", preconditioner update = " << diagnostics->preconditioner_update_seconds
                  << ", preconditioner setup = " << diagnostics->preconditioner_setup_seconds
                  << ", work objective = " << diagnostics->last_work_objective
                  << ", nonmonotone work reference = " << diagnostics->last_nonmonotone_work_reference << '\n';
      }
      const auto& jacobian_timings = solid.jacobianTimings();
      mfem::out << "  Solid Jacobian timings: legacy evals = " << jacobian_timings.legacy_jacobian_evals
                << ", legacy derivative = " << jacobian_timings.legacy_derivative_seconds
                << ", legacy sparse assembly = " << jacobian_timings.legacy_sparse_assembly_seconds
                << ", legacy EBC elimination = " << jacobian_timings.legacy_essential_elimination_seconds
                << ", operator evals = " << jacobian_timings.jacobian_operator_evals
                << ", operator assemblies = " << jacobian_timings.jacobian_operator_assemblies
                << ", operator derivative = " << jacobian_timings.jacobian_operator_derivative_seconds
                << ", operator sparse assembly = " << jacobian_timings.jacobian_operator_sparse_assembly_seconds
                << ", operator EBC elimination = " << jacobian_timings.jacobian_operator_essential_elimination_seconds
                << '\n';
    }
    if (!nonlinear_solver.GetConverged()) {
      throw std::runtime_error("Nonlinear solve failed to converge at load step " + std::to_string(step + 1));
    }
  }

  if (rank == 0) {
    mfem::out << "Converged load steps: " << num_converged_steps << "/" << num_steps << '\n';
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
