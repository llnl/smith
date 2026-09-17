// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <map>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <vector>

#include "mpi.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"

#ifdef SMITH_USE_SLEPC
#include "slepceps.h"
#endif

#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/state_manager.hpp"

namespace smith {
namespace {

enum class ProblemSize
{
  Small,
  Medium,
  Large
};

enum class WarmStartOption
{
  Default,
  Enabled,
  Disabled
};

constexpr int p = 2;
constexpr int solid_dim = 3;
constexpr int max_benchmark_elements = 200000;

std::string selected_case = "01";
std::string nonlinear_solver_name = "TrustRegion";
std::string linear_solver_name = "CG";
std::string preconditioner_name = "HypreAMG";
std::string problem_size_name = "small";
std::string deflation_coarse_mode_name = "global";
std::string deflation_smoother = "jacobi";
int sim_order = 2;
WarmStartOption warm_start_option = WarmStartOption::Default;
int print_level = 1;
int linear_print_level = 0;
constexpr int default_num_steps = 3;
int nonlinear_max_iterations = 10000;
int max_line_search_iterations = 20;
int linear_max_iterations = 840;
int trust_subspace_option = static_cast<int>(SubSpaceOptions::WHEN_INDEFINITE_OR_BOUNDARY);
int trust_num_leftmost = 1;
int trust_num_previous_steps = 5;
int trust_work_quadrature_points = 2;
int deflation_pieces = 0;
int cg_cap_min = 103;
int cg_model_stagnation_window = 2;
bool use_bsr_spmv = true;
bool assemble_bsr = false;
bool write_output = false;
bool compute_final_state_eigenpair = false;
bool write_final_state_eigenvector = false;
bool timing_summary = false;
bool timing_summary_set = false;
bool timing_breakdown = false;
int euler_extra_refinement = 0;
int euler_solution_states = 6;
int euler_coarse_load_steps = 30;
int euler_refined_load_steps = 5;
double euler_load = 0.0022916666666666667;
double euler_selector_traction = -1.0e-10;
double euler_refined_start_time = -1.0;
double euler_refined_start_traction = 0.00225;
double euler_refined_start_force = -1.0;
double shallow_arch_precompression = 0.02;
int shallow_arch_precompression_steps = 10;
int shallow_arch_load_steps = 200;
double shallow_arch_load_magnitude = 0.015;
double cg_model_energy_stagnation_reltol = 0.0010638500842686796;
double cg_forcing_rel = 1.3e-5;
double cg_cap_gamma = 0.7638340304667752;
double residual_growth_cap = 8.2;
double tr_decrease_factor = 0.39;
double tr_increase_factor = 2.0;
double tr_eta1 = 1.0e-9;
double tr_eta2 = 0.13;
double tr_eta3 = 0.55;
double tr_eta4 = 1.8;
double nonlinear_tol = 1.0e-7;
double linear_tol = 0.8e-7;
double eigenvalue_tol = 1.0e-8;
int eigenvalue_max_iterations = 1000;
int eigenvalue_count = 1;
// Performance testing convention: run challenge problems with --mesh-scale=0.5.
double mesh_scale = 1.0;
std::vector<std::string> command_line_warnings;

struct RunTiming {
  std::string name;
  double wall_time = 0.0;
  bool success = true;
  std::string error;
};

int scaled(int n) { return std::max(1, static_cast<int>(std::lround(n * mesh_scale))); }

bool selectedCaseMatches(std::initializer_list<std::string_view> names)
{
  if (selected_case == "all") return true;
  for (std::string_view name : names) {
    if (selected_case == name) return true;
  }
  return false;
}

std::string boolOption(bool enabled, std::string_view enabled_name, std::string_view disabled_name)
{
  return enabled ? std::string(enabled_name) : std::string(disabled_name);
}

std::string reproduceCommand(const std::string& name)
{
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  std::ostringstream os;
  os << "srun -n " << ranks << " ./examples/challenge_suite"
     << " --case=" << name.substr(0, 2) << " --size=" << problem_size_name << " --mesh-scale=" << mesh_scale
     << " --print-level=" << print_level << " --nonlinear-solver=" << nonlinear_solver_name
     << " --linear-solver=" << linear_solver_name << " --preconditioner=" << preconditioner_name
     << " --linear-print-level=" << linear_print_level
     << " --trust-subspace-option=" << trust_subspace_option << " --max-cg-iterations=" << linear_max_iterations
     << " --deflation-coarse-mode=" << deflation_coarse_mode_name << " --deflation-pieces=" << deflation_pieces
     << " --deflation-smoother=" << deflation_smoother << " "
     << boolOption(use_bsr_spmv, "--use-bsr-spmv", "--no-use-bsr-spmv") << " "
     << boolOption(assemble_bsr, "--assemble-bsr", "--no-assemble-bsr") << " "
     << boolOption(write_output, "--paraview", "--no-paraview") << " "
     << boolOption(compute_final_state_eigenpair, "--final-state-eigenpair", "--no-final-state-eigenpair") << " "
     << boolOption(write_final_state_eigenvector, "--write-eigenvector", "--no-write-eigenvector")
     << " --eigenvalue-tol=" << eigenvalue_tol << " --eigenvalue-max-iterations=" << eigenvalue_max_iterations
     << " --eigenvalue-count=" << eigenvalue_count;
  if (warm_start_option == WarmStartOption::Enabled) {
    os << " --use-warm-start";
  } else if (warm_start_option == WarmStartOption::Disabled) {
    os << " --no-warm-start";
  }
  if (name.starts_with("02")) {
    os << " --shallow-arch-precompression=" << shallow_arch_precompression
       << " --shallow-arch-precompression-steps=" << shallow_arch_precompression_steps
       << " --shallow-arch-load-steps=" << shallow_arch_load_steps
       << " --shallow-arch-load-magnitude=" << shallow_arch_load_magnitude;
  }
  return os.str();
}

void printProblemBanner(const std::string& name)
{
  SLIC_INFO_ROOT("");
  SLIC_INFO_ROOT("================================================================");
  SLIC_INFO_ROOT(std::format("Running challenge problem: {}", name));
  SLIC_INFO_ROOT(std::format("Reproduce with: {}", reproduceCommand(name)));
  SLIC_INFO_ROOT("================================================================");
}

void printCommandLineWarnings()
{
  for (const auto& warning : command_line_warnings) {
    SLIC_WARNING_ROOT(warning);
  }
}

bool warmStartEnabled(std::string_view name, bool enabled_by_default)
{
  if (warm_start_option == WarmStartOption::Enabled) {
    SLIC_INFO_ROOT(std::format("{} warm start = on", name));
    return true;
  }
  if (warm_start_option == WarmStartOption::Disabled) {
    if (!enabled_by_default) {
      SLIC_WARNING_ROOT(std::format("Ignoring --no-warm-start for {}: warm start already disabled", name));
    }
    SLIC_INFO_ROOT(std::format("{} warm start = off", name));
    return false;
  }
  SLIC_INFO_ROOT(std::format("{} warm start = {}", name, enabled_by_default ? "on" : "off"));
  return enabled_by_default;
}

struct RunOptionState {
  int nonlinear_max_iterations_value = nonlinear_max_iterations;
  int linear_max_iterations_value = linear_max_iterations;
  double nonlinear_tol_value = nonlinear_tol;
  double linear_tol_value = linear_tol;

  void restore() const
  {
    nonlinear_max_iterations = nonlinear_max_iterations_value;
    linear_max_iterations = linear_max_iterations_value;
    nonlinear_tol = nonlinear_tol_value;
    linear_tol = linear_tol_value;
  }
};

template <typename RunFunction>
void runTimedCase(std::vector<RunTiming>& timings, const std::string& name, RunFunction&& run_function)
{
  const RunOptionState saved_options;
  MPI_Barrier(MPI_COMM_WORLD);
  printProblemBanner(name);
  const double start_time = MPI_Wtime();
  bool success = true;
  std::string error;
  try {
    run_function();
  } catch (const std::exception& e) {
    success = false;
    error = e.what();
  }
  saved_options.restore();
  int local_failed = success ? 0 : 1;
  int global_failed = 0;
  MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (global_failed && success) {
    success = false;
    error = "case failed on another rank";
  }
  MPI_Barrier(MPI_COMM_WORLD);
  timings.push_back({name, MPI_Wtime() - start_time, success, error});
  if (!success) SLIC_INFO_ROOT(std::format("{} status = solver_failed ({})", name, error));
}

void printTimingSummary(const std::vector<RunTiming>& timings)
{
  if (!timing_summary || timings.empty()) return;

  double total_time = 0.0;
  SLIC_INFO_ROOT("");
  SLIC_INFO_ROOT("Challenge suite timing summary (max rank wall)");
  for (const auto& timing : timings) {
    double local_time = timing.wall_time;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    total_time += max_time;
    SLIC_INFO_ROOT(
        std::format("  {:32s} {:10.3f} s  {}", timing.name, max_time, timing.success ? "ok" : "solver_failed"));
  }
  SLIC_INFO_ROOT(std::format("  {:32s} {:10.3f} s", "total", total_time));
}

bool anyRunFailed(const std::vector<RunTiming>& timings)
{
  return std::any_of(timings.begin(), timings.end(), [](const RunTiming& timing) { return !timing.success; });
}

template <typename Solid>
void requireSolveConverged(Solid& solid, const std::string& message)
{
  int local_nonlinear_failed = solid.nonlinearSolveConverged() ? 0 : 1;
  int local_linear_failed = 0;
  if (solid.reportsLinearSolveConvergence()) {
    local_linear_failed = solid.linearSolveConverged() ? 0 : 1;
  }
  int global_nonlinear_failed = 0;
  int global_linear_failed = 0;
  MPI_Allreduce(&local_nonlinear_failed, &global_nonlinear_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_linear_failed, &global_linear_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (!global_nonlinear_failed && !global_linear_failed) return;

  std::string failure_message = message;
  if (global_nonlinear_failed && global_linear_failed) {
    failure_message += " (nonlinear and linear convergence failure)";
  } else if (global_nonlinear_failed) {
    failure_message += " (nonlinear convergence failure)";
  } else {
    failure_message += " (linear convergence failure)";
  }
  throw std::runtime_error(failure_message);
}

template <typename T>
T parseEnum(const std::map<std::string, T>& values, const std::string& value, const std::string& option)
{
  auto found = values.find(value);
  if (found == values.end()) {
    throw std::runtime_error(std::format("Unknown {} value '{}'", option, value));
  }
  return found->second;
}

std::filesystem::path czRepoRoot() { return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path(); }

ProblemSize selectedProblemSize()
{
  if (problem_size_name == "small") return ProblemSize::Small;
  if (problem_size_name == "medium") return ProblemSize::Medium;
  if (problem_size_name == "large") return ProblemSize::Large;
  throw std::runtime_error("Unknown --size value '" + problem_size_name + "'");
}

int parallelRefinement()
{
  switch (selectedProblemSize()) {
    case ProblemSize::Small:
      return 0;
    case ProblemSize::Medium:
      return 1;
    case ProblemSize::Large:
      return 2;
  }
  return 0;
}

NonlinearSolver selectedNonlinearSolver()
{
  if (nonlinear_solver_name == "trustregion") return NonlinearSolver::TrustRegion;
  if (nonlinear_solver_name == "newtonlinesearch") return NonlinearSolver::NewtonLineSearch;
  return parseEnum(nonlinearSolverMap, nonlinear_solver_name, "--nonlinear-solver");
}

Preconditioner selectedPreconditioner()
{
  if (preconditioner_name == "deflation") {
    command_line_warnings.push_back("B1 does not support --preconditioner=deflation; using HypreAMG");
    return Preconditioner::HypreAMG;
  }
  if (preconditioner_name == "hypreamg") return Preconditioner::HypreAMG;
  if (preconditioner_name == "hyprejacobi") return Preconditioner::HypreJacobi;
  return parseEnum(preconditionerMap, preconditioner_name, "--preconditioner");
}

LinearSolver selectedLinearSolver() { return parseEnum(linearSolverMap, linear_solver_name, "--linear-solver"); }

LinearSolverOptions linearOptions()
{
  return {.linear_solver = selectedLinearSolver(),
          .preconditioner = selectedPreconditioner(),
          .relative_tol = linear_tol,
          .absolute_tol = linear_tol,
          .max_iterations = linear_max_iterations,
          .print_level = linear_print_level};
}

LinearSolverOptions shallowArchBucklingLinearOptions()
{
  auto options = linearOptions();
  options.relative_tol = 1.0e-8;
  return options;
}

NonlinearSolverOptions nonlinearOptions()
{
  return {.nonlin_solver = selectedNonlinearSolver(),
          .relative_tol = nonlinear_tol,
          .absolute_tol = nonlinear_tol,
          .min_iterations = 0,
          .max_iterations = nonlinear_max_iterations,
          .max_line_search_iterations = max_line_search_iterations,
          .print_level = print_level,
          .subspace_option = static_cast<SubSpaceOptions>(trust_subspace_option),
          .num_leftmost = trust_num_leftmost,
          .num_previous_steps = trust_num_previous_steps,
          .cg_forcing_rel = cg_forcing_rel,
          .residual_growth_cap = residual_growth_cap,
          .tr_decrease_factor = tr_decrease_factor,
          .tr_increase_factor = tr_increase_factor,
          .tr_eta1 = tr_eta1,
          .tr_eta2 = tr_eta2,
          .tr_eta3 = tr_eta3,
          .tr_eta4 = tr_eta4};
}

NonlinearSolverOptions shallowArchBucklingNonlinearOptions()
{
  auto options = nonlinearOptions();
  options.relative_tol = 1.0e-8;
  options.absolute_tol = 1.0e-10;
  return options;
}

int globalElementCount(const Mesh& mesh)
{
  int global_elements = 0;
  int local_elements = mesh.mfemParMesh().GetNE();
  MPI_Allreduce(&local_elements, &global_elements, 1, MPI_INT, MPI_SUM, mesh.getComm());
  return global_elements;
}

void checkElementCount(const std::string& name, const Mesh& mesh)
{
  const int elements = globalElementCount(mesh);
  SLIC_INFO_ROOT(std::format("{}: global elements = {}", name, elements));
  SLIC_ERROR_ROOT_IF(
      elements >= max_benchmark_elements,
      std::format("{} has {} elements, exceeding the benchmark cap of {}", name, elements, max_benchmark_elements));
}

template <int P, int D>
void advanceTimesteps(SolidMechanics<P, D>& solid, const std::string& output_name, int steps = default_num_steps)
{
  if (write_output) {
    solid.outputStateToDisk(output_name);
  }

  for (int step = 0; step < steps; ++step) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();
    solid.advanceTimestep(1.0 / steps);
    MPI_Barrier(MPI_COMM_WORLD);
    requireSolveConverged(solid, std::format("{} nonlinear solve failed at step {}", output_name, step + 1));
    SLIC_INFO_ROOT(
        std::format("{} step {}/{} wall = {:.3f} s", output_name, step + 1, steps, MPI_Wtime() - start_time));
    if (write_output) {
      solid.outputStateToDisk(output_name);
    }
  }
}

template <int P, int D>
void advanceSingleThinBeamStep(SolidMechanics<P, D>& solid)
{
  constexpr int beam_steps = 1;
  if (write_output) {
    solid.outputStateToDisk("thin_beam_bending");
  }

  double cumulative_step_time = 0.0;
  for (int step = 0; step < beam_steps; ++step) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();
    solid.advanceTimestep(1.0 / beam_steps);
    MPI_Barrier(MPI_COMM_WORLD);
    const double step_time = MPI_Wtime() - start_time;
    cumulative_step_time += step_time;
    requireSolveConverged(solid, std::format("thin_beam_bending nonlinear solve failed at step {}", step + 1));
    SLIC_INFO_ROOT(std::format("Load step {}/{} wall = {:.3f} s  (cumulative {:.3f} s)", step + 1, beam_steps,
                               step_time, cumulative_step_time));
    if (write_output) {
      solid.outputStateToDisk("thin_beam_bending");
    }
  }
  SLIC_INFO_ROOT(std::format("Total advanceTimestep wall = {:.3f} s over {} steps", cumulative_step_time, beam_steps));
}

template <int P, int D>
void checkDisplacement(const std::string& name, const SolidMechanics<P, D>& solid)
{
  const double displacement_norm = mfem::ParNormlp(solid.displacement(), 2, MPI_COMM_WORLD);
  SLIC_INFO_ROOT(std::format("{}: final displacement l2 = {:.8e}", name, displacement_norm));
  SLIC_ERROR_ROOT_IF(!std::isfinite(displacement_norm), name + " produced a non-finite displacement norm");
  SLIC_ERROR_ROOT_IF(displacement_norm <= 0.0, name + " produced a zero displacement norm");
}

mfem::ParMesh distributeMeshContiguously(mfem::Mesh& serial_mesh)
{
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  const int elements = serial_mesh.GetNE();
  std::vector<int> partitioning(static_cast<std::size_t>(elements));
  for (int elem = 0; elem < elements; ++elem) {
    partitioning[static_cast<std::size_t>(elem)] = std::min(ranks - 1, (elem * ranks) / elements);
  }
  return mfem::ParMesh(MPI_COMM_WORLD, serial_mesh, partitioning.data());
}

void parseCommandLine(int& argc, char** argv)
{
  int write_arg = 1;
  bool use_bsr_spmv_requested = false;
  bool assemble_bsr_requested = false;
  bool deflation_smoother_requested = false;
  bool warm_start_option_requested = false;
  bool nonlinear_tol_requested = false;
  bool linear_tol_requested = false;
  bool linear_max_iterations_requested = false;
  for (int read_arg = 1; read_arg < argc; ++read_arg) {
    const std::string arg = argv[read_arg];
    if (arg.rfind("--case=", 0) == 0) {
      selected_case = arg.substr(std::string("--case=").size());
    } else if (arg.rfind("--solver=", 0) == 0) {
      nonlinear_solver_name = arg.substr(std::string("--solver=").size());
    } else if (arg.rfind("--nonlinear-solver=", 0) == 0) {
      nonlinear_solver_name = arg.substr(std::string("--nonlinear-solver=").size());
    } else if (arg.rfind("--linear-solver=", 0) == 0) {
      linear_solver_name = arg.substr(std::string("--linear-solver=").size());
    } else if (arg.rfind("--preconditioner=", 0) == 0) {
      preconditioner_name = arg.substr(std::string("--preconditioner=").size());
    } else if (arg.rfind("--size=", 0) == 0) {
      problem_size_name = arg.substr(std::string("--size=").size());
    } else if (arg.rfind("--order=", 0) == 0) {
      sim_order = std::stoi(arg.substr(std::string("--order=").size()));
    } else if (arg == "--use-warm-start") {
      if (warm_start_option_requested && warm_start_option != WarmStartOption::Enabled) {
        command_line_warnings.push_back("Both --use-warm-start and --no-warm-start passed; using last option");
      }
      warm_start_option = WarmStartOption::Enabled;
      warm_start_option_requested = true;
    } else if (arg == "--no-warm-start") {
      if (warm_start_option_requested && warm_start_option != WarmStartOption::Disabled) {
        command_line_warnings.push_back("Both --use-warm-start and --no-warm-start passed; using last option");
      }
      warm_start_option = WarmStartOption::Disabled;
      warm_start_option_requested = true;
    } else if (arg.rfind("--nonlinear-max-iterations=", 0) == 0) {
      nonlinear_max_iterations = std::stoi(arg.substr(std::string("--nonlinear-max-iterations=").size()));
    } else if (arg.rfind("--max-line-search-iterations=", 0) == 0) {
      max_line_search_iterations = std::stoi(arg.substr(std::string("--max-line-search-iterations=").size()));
    } else if (arg.rfind("--linear-max-iterations=", 0) == 0) {
      linear_max_iterations = std::stoi(arg.substr(std::string("--linear-max-iterations=").size()));
      linear_max_iterations_requested = true;
    } else if (arg.rfind("--max-cg-iterations=", 0) == 0) {
      linear_max_iterations = std::stoi(arg.substr(std::string("--max-cg-iterations=").size()));
      linear_max_iterations_requested = true;
    } else if (arg.rfind("--print-level=", 0) == 0) {
      print_level = std::stoi(arg.substr(std::string("--print-level=").size()));
    } else if (arg.rfind("--linear-print-level=", 0) == 0) {
      linear_print_level = std::stoi(arg.substr(std::string("--linear-print-level=").size()));
    } else if (arg.rfind("--trust-subspace-option=", 0) == 0) {
      trust_subspace_option = std::stoi(arg.substr(std::string("--trust-subspace-option=").size()));
    } else if (arg.rfind("--trust-num-leftmost=", 0) == 0) {
      trust_num_leftmost = std::stoi(arg.substr(std::string("--trust-num-leftmost=").size()));
    } else if (arg.rfind("--trust-num-previous-steps=", 0) == 0) {
      trust_num_previous_steps = std::stoi(arg.substr(std::string("--trust-num-previous-steps=").size()));
    } else if (arg.rfind("--trust-work-quadrature=", 0) == 0) {
      trust_work_quadrature_points = std::stoi(arg.substr(std::string("--trust-work-quadrature=").size()));
    } else if (arg.rfind("--deflation-coarse-mode=", 0) == 0) {
      deflation_coarse_mode_name = arg.substr(std::string("--deflation-coarse-mode=").size());
    } else if (arg.rfind("--deflation-pieces=", 0) == 0) {
      deflation_pieces = std::stoi(arg.substr(std::string("--deflation-pieces=").size()));
    } else if (arg.rfind("--deflation-smoother=", 0) == 0) {
      deflation_smoother_requested = true;
      deflation_smoother = arg.substr(std::string("--deflation-smoother=").size());
    } else if (arg.rfind("--cg-model-energy-stagnation-reltol=", 0) == 0) {
      cg_model_energy_stagnation_reltol =
          std::stod(arg.substr(std::string("--cg-model-energy-stagnation-reltol=").size()));
    } else if (arg.rfind("--cg-forcing-rel=", 0) == 0) {
      cg_forcing_rel = std::stod(arg.substr(std::string("--cg-forcing-rel=").size()));
    } else if (arg.rfind("--cg-cap-min=", 0) == 0) {
      cg_cap_min = std::stoi(arg.substr(std::string("--cg-cap-min=").size()));
    } else if (arg.rfind("--cg-cap-gamma=", 0) == 0) {
      cg_cap_gamma = std::stod(arg.substr(std::string("--cg-cap-gamma=").size()));
    } else if (arg.rfind("--cg-stagnation-window=", 0) == 0) {
      cg_model_stagnation_window = std::stoi(arg.substr(std::string("--cg-stagnation-window=").size()));
    } else if (arg.rfind("--nonlinear-tol=", 0) == 0) {
      nonlinear_tol = std::stod(arg.substr(std::string("--nonlinear-tol=").size()));
      nonlinear_tol_requested = true;
    } else if (arg.rfind("--linear-tol=", 0) == 0) {
      linear_tol = std::stod(arg.substr(std::string("--linear-tol=").size()));
      linear_tol_requested = true;
    } else if (arg.rfind("--residual-growth-cap=", 0) == 0) {
      residual_growth_cap = std::stod(arg.substr(std::string("--residual-growth-cap=").size()));
    } else if (arg.rfind("--tr-decrease-factor=", 0) == 0) {
      tr_decrease_factor = std::stod(arg.substr(std::string("--tr-decrease-factor=").size()));
    } else if (arg.rfind("--tr-increase-factor=", 0) == 0) {
      tr_increase_factor = std::stod(arg.substr(std::string("--tr-increase-factor=").size()));
    } else if (arg.rfind("--tr-eta1=", 0) == 0) {
      tr_eta1 = std::stod(arg.substr(std::string("--tr-eta1=").size()));
    } else if (arg.rfind("--tr-eta2=", 0) == 0) {
      tr_eta2 = std::stod(arg.substr(std::string("--tr-eta2=").size()));
    } else if (arg.rfind("--tr-eta3=", 0) == 0) {
      tr_eta3 = std::stod(arg.substr(std::string("--tr-eta3=").size()));
    } else if (arg.rfind("--tr-eta4=", 0) == 0) {
      tr_eta4 = std::stod(arg.substr(std::string("--tr-eta4=").size()));
    } else if (arg.rfind("--mesh-scale=", 0) == 0) {
      mesh_scale = std::stod(arg.substr(std::string("--mesh-scale=").size()));
    } else if (arg.rfind("--euler-extra-refinement=", 0) == 0) {
      euler_extra_refinement = std::stoi(arg.substr(std::string("--euler-extra-refinement=").size()));
    } else if (arg.rfind("--euler-load=", 0) == 0) {
      euler_load = std::stod(arg.substr(std::string("--euler-load=").size()));
    } else if (arg.rfind("--euler-selector-traction=", 0) == 0) {
      euler_selector_traction = std::stod(arg.substr(std::string("--euler-selector-traction=").size()));
    } else if (arg.rfind("--euler-solution-states=", 0) == 0) {
      euler_solution_states = std::stoi(arg.substr(std::string("--euler-solution-states=").size()));
    } else if (arg.rfind("--euler-coarse-load-steps=", 0) == 0) {
      euler_coarse_load_steps = std::stoi(arg.substr(std::string("--euler-coarse-load-steps=").size()));
    } else if (arg.rfind("--euler-refined-load-steps=", 0) == 0) {
      euler_refined_load_steps = std::stoi(arg.substr(std::string("--euler-refined-load-steps=").size()));
    } else if (arg.rfind("--euler-refined-start-time=", 0) == 0) {
      euler_refined_start_time = std::stod(arg.substr(std::string("--euler-refined-start-time=").size()));
    } else if (arg.rfind("--euler-refined-start-traction=", 0) == 0) {
      euler_refined_start_traction = std::stod(arg.substr(std::string("--euler-refined-start-traction=").size()));
    } else if (arg.rfind("--euler-refined-start-force=", 0) == 0) {
      euler_refined_start_force = std::stod(arg.substr(std::string("--euler-refined-start-force=").size()));
    } else if (arg.rfind("--shallow-arch-precompression=", 0) == 0) {
      shallow_arch_precompression = std::stod(arg.substr(std::string("--shallow-arch-precompression=").size()));
    } else if (arg.rfind("--shallow-arch-precompression-steps=", 0) == 0) {
      shallow_arch_precompression_steps =
          std::stoi(arg.substr(std::string("--shallow-arch-precompression-steps=").size()));
    } else if (arg.rfind("--shallow-arch-load-steps=", 0) == 0) {
      shallow_arch_load_steps = std::stoi(arg.substr(std::string("--shallow-arch-load-steps=").size()));
    } else if (arg.rfind("--shallow-arch-load-magnitude=", 0) == 0) {
      shallow_arch_load_magnitude = std::stod(arg.substr(std::string("--shallow-arch-load-magnitude=").size()));
    } else if (arg == "--use-bsr-spmv") {
      use_bsr_spmv_requested = true;
      use_bsr_spmv = true;
    } else if (arg == "--no-use-bsr-spmv") {
      use_bsr_spmv_requested = true;
      use_bsr_spmv = false;
    } else if (arg == "--assemble-bsr") {
      assemble_bsr_requested = true;
      assemble_bsr = true;
    } else if (arg == "--no-assemble-bsr") {
      assemble_bsr_requested = true;
      assemble_bsr = false;
    } else if (arg == "--write-output" || arg == "--paraview") {
      write_output = true;
    } else if (arg == "--no-write-output" || arg == "--no-paraview") {
      write_output = false;
    } else if (arg == "--final-state-eigenpair") {
      compute_final_state_eigenpair = true;
    } else if (arg == "--no-final-state-eigenpair") {
      compute_final_state_eigenpair = false;
    } else if (arg == "--write-eigenvector") {
      write_final_state_eigenvector = true;
    } else if (arg == "--no-write-eigenvector") {
      write_final_state_eigenvector = false;
    } else if (arg.rfind("--eigenvalue-tol=", 0) == 0) {
      eigenvalue_tol = std::stod(arg.substr(std::string("--eigenvalue-tol=").size()));
    } else if (arg.rfind("--eigenvalue-max-iterations=", 0) == 0) {
      eigenvalue_max_iterations = std::stoi(arg.substr(std::string("--eigenvalue-max-iterations=").size()));
    } else if (arg.rfind("--eigenvalue-count=", 0) == 0) {
      eigenvalue_count = std::stoi(arg.substr(std::string("--eigenvalue-count=").size()));
    } else if (arg == "--timings" || arg == "--timing-summary") {
      timing_summary = true;
      timing_summary_set = true;
    } else if (arg == "--no-timings" || arg == "--no-timing-summary") {
      timing_summary = false;
      timing_summary_set = true;
    } else if (arg == "--timing-breakdown") {
      timing_breakdown = true;
    } else {
      if (arg.rfind("--", 0) == 0) {
        command_line_warnings.push_back(std::format("Unknown challenge_suite option '{}'; likely ignored", arg));
      }
      argv[write_arg] = argv[read_arg];
      ++write_arg;
    }
  }
  argc = write_arg;

  if (shallow_arch_precompression < 0.0) {
    throw std::runtime_error("--shallow-arch-precompression must be nonnegative");
  }
  if (shallow_arch_precompression_steps <= 0 || shallow_arch_load_steps <= 0) {
    throw std::runtime_error("Shallow-arch step counts must be positive");
  }
  if (shallow_arch_load_magnitude <= 0.0) {
    throw std::runtime_error("--shallow-arch-load-magnitude must be positive");
  }
  if (eigenvalue_count <= 0) {
    throw std::runtime_error("--eigenvalue-count must be positive");
  }

  if (selectedPreconditioner() == Preconditioner::HypreAMG) {
    if (assemble_bsr_requested && assemble_bsr) {
      command_line_warnings.push_back("Ignoring --assemble-bsr with --preconditioner=HypreAMG");
    }
    if (use_bsr_spmv_requested && use_bsr_spmv) {
      command_line_warnings.push_back("Ignoring --use-bsr-spmv with --preconditioner=HypreAMG");
    }
    assemble_bsr = false;
    use_bsr_spmv = false;
  }
  if (assemble_bsr && deflation_smoother == "hypre") {
    if (deflation_smoother_requested) {
      command_line_warnings.push_back("Ignoring --deflation-smoother=hypre with --assemble-bsr; using jacobi");
    }
    deflation_smoother = "jacobi";
  }
  if (!timing_summary_set) {
    timing_summary = selected_case == "all";
  }
  const bool only_euler_selected = selected_case == "01" || selected_case == "01/euler" || selected_case == "euler";
  if (only_euler_selected && !nonlinear_tol_requested) {
    nonlinear_tol = 1.0e-11;
  }
  if (only_euler_selected && !linear_tol_requested) {
    linear_tol = 1.0e-14;
  }
  if (only_euler_selected && !linear_max_iterations_requested) {
    if (selectedPreconditioner() == Preconditioner::HypreJacobi) {
      linear_max_iterations = 60000;
    } else if (selectedPreconditioner() == Preconditioner::HypreAMG) {
      linear_max_iterations = 600;
    }
  }
  if (timing_breakdown) {
    print_level = std::max(print_level, 2);
  }
  if (write_final_state_eigenvector && !compute_final_state_eigenpair) {
    command_line_warnings.push_back("Ignoring --write-eigenvector without --final-state-eigenpair");
    write_final_state_eigenvector = false;
  }
}

struct LowestEigenpairResult {
  bool converged = false;
  bool tangent_is_symmetric = false;
  int converged_modes = 0;
  int iterations = 0;
  int convergence_reason = 0;
  long long free_dofs = 0;
  long long constrained_dofs = 0;
  double tangent_norm = std::numeric_limits<double>::quiet_NaN();
  double spectral_shift = std::numeric_limits<double>::quiet_NaN();
  struct Mode {
    double eigenvalue_real = std::numeric_limits<double>::quiet_NaN();
    double eigenvalue_imaginary = std::numeric_limits<double>::quiet_NaN();
    double absolute_residual = std::numeric_limits<double>::quiet_NaN();
    double relative_residual = std::numeric_limits<double>::quiet_NaN();
    double norm_scaled_residual = std::numeric_limits<double>::quiet_NaN();
  };
  std::vector<Mode> modes;
  mfem::Vector eigenvector;
};

#ifdef SMITH_USE_SLEPC
void checkPetscError(PetscErrorCode error, std::string_view operation)
{
  if (error != PETSC_SUCCESS) {
    throw std::runtime_error(std::format("{} failed with PETSc error {}", operation, error));
  }
}
#endif

LowestEigenpairResult computeLowestEigenpair(const mfem::HypreParMatrix& tangent,
                                             const mfem::Array<int>& essential_true_dofs)
{
#ifndef SMITH_USE_SLEPC
  static_cast<void>(tangent);
  static_cast<void>(essential_true_dofs);
  throw std::runtime_error("The final-state eigenpair diagnostic requires a build with SLEPc support.");
#else
  const MPI_Comm comm = tangent.GetComm();
  mfem::PetscParMatrix petsc_tangent(&tangent, mfem::Operator::PETSC_MATAIJ);
  Mat full_matrix = petsc_tangent;

  PetscInt local_row_start = 0;
  PetscInt local_row_end = 0;
  checkPetscError(MatGetOwnershipRange(full_matrix, &local_row_start, &local_row_end), "MatGetOwnershipRange");
  const PetscInt local_rows = local_row_end - local_row_start;
  if (local_rows != tangent.Height()) {
    throw std::runtime_error("PETSc and Hypre tangent ownership ranges do not match.");
  }

  std::vector<bool> constrained(static_cast<std::size_t>(local_rows), false);
  for (int dof : essential_true_dofs) {
    if (dof < 0 || dof >= local_rows) {
      throw std::runtime_error(std::format("Essential true degree of freedom {} is outside the local tangent.", dof));
    }
    constrained[static_cast<std::size_t>(dof)] = true;
  }

  std::vector<PetscInt> free_rows;
  free_rows.reserve(static_cast<std::size_t>(local_rows - essential_true_dofs.Size()));
  for (PetscInt local_row = 0; local_row < local_rows; ++local_row) {
    if (!constrained[static_cast<std::size_t>(local_row)]) {
      free_rows.push_back(local_row_start + local_row);
    }
  }

  PetscInt global_constrained_dofs = 0;
  PetscInt local_constrained_dofs = essential_true_dofs.Size();
  checkPetscError(MPI_Allreduce(&local_constrained_dofs, &global_constrained_dofs, 1, MPIU_INT, MPI_SUM, comm),
                  "MPI_Allreduce constrained degrees of freedom");

  IS free_index_set = nullptr;
  checkPetscError(ISCreateGeneral(comm, static_cast<PetscInt>(free_rows.size()), free_rows.data(), PETSC_COPY_VALUES,
                                  &free_index_set),
                  "ISCreateGeneral");
  Mat free_tangent = nullptr;
  checkPetscError(MatCreateSubMatrix(full_matrix, free_index_set, free_index_set, MAT_INITIAL_MATRIX, &free_tangent),
                  "MatCreateSubMatrix");

  LowestEigenpairResult result;
  PetscInt global_free_dofs = 0;
  checkPetscError(MatGetSize(free_tangent, &global_free_dofs, nullptr), "MatGetSize");
  result.free_dofs = global_free_dofs;
  result.constrained_dofs = global_constrained_dofs;
  if (global_free_dofs == 0) {
    MatDestroy(&free_tangent);
    ISDestroy(&free_index_set);
    throw std::runtime_error("The final-state tangent has no unconstrained degrees of freedom.");
  }

  Mat tangent_skew_part = nullptr;
  checkPetscError(MatTranspose(free_tangent, MAT_INITIAL_MATRIX, &tangent_skew_part), "MatTranspose");
  checkPetscError(MatAXPY(tangent_skew_part, -1.0, free_tangent, DIFFERENT_NONZERO_PATTERN), "MatAXPY");
  PetscReal tangent_norm = 0.0;
  PetscReal tangent_skew_norm = 0.0;
  checkPetscError(MatNorm(free_tangent, NORM_INFINITY, &tangent_norm), "MatNorm tangent");
  checkPetscError(MatNorm(tangent_skew_part, NORM_INFINITY, &tangent_skew_norm), "MatNorm tangent skew part");
  result.tangent_norm = tangent_norm;
  result.tangent_is_symmetric = tangent_skew_norm <= 1.0e-10 * std::max(1.0, tangent_norm);
  checkPetscError(MatDestroy(&tangent_skew_part), "MatDestroy tangent skew part");
  if (!result.tangent_is_symmetric) {
    MatDestroy(&free_tangent);
    ISDestroy(&free_index_set);
    throw std::runtime_error("The final-state tangent is nonsymmetric; the lowest real eigenvalue is not a stability "
                             "certificate for this problem.");
  }
  result.spectral_shift = 0.0;

  EPS eigensolver = nullptr;
  checkPetscError(EPSCreate(comm, &eigensolver), "EPSCreate");
  checkPetscError(EPSSetOptionsPrefix(eigensolver, "final_state_"), "EPSSetOptionsPrefix");
  checkPetscError(EPSSetOperators(eigensolver, free_tangent, nullptr), "EPSSetOperators");
  checkPetscError(EPSSetProblemType(eigensolver, EPS_HEP), "EPSSetProblemType");
  checkPetscError(EPSSetType(eigensolver, EPSJD), "EPSSetType");
  checkPetscError(EPSSetWhichEigenpairs(eigensolver, EPS_SMALLEST_REAL), "EPSSetWhichEigenpairs");
  checkPetscError(EPSSetDimensions(eigensolver, eigenvalue_count, PETSC_DETERMINE, PETSC_DETERMINE),
                  "EPSSetDimensions");
  checkPetscError(EPSSetTolerances(eigensolver, eigenvalue_tol, eigenvalue_max_iterations), "EPSSetTolerances");
  checkPetscError(EPSSetConvergenceTest(eigensolver, EPS_CONV_NORM), "EPSSetConvergenceTest");

  ST spectral_transform = nullptr;
  KSP spectral_solver = nullptr;
  PC spectral_preconditioner = nullptr;
  checkPetscError(EPSGetST(eigensolver, &spectral_transform), "EPSGetST");
  checkPetscError(STSetType(spectral_transform, STPRECOND), "STSetType");
  checkPetscError(STSetShift(spectral_transform, result.spectral_shift), "STSetShift");
  checkPetscError(STGetKSP(spectral_transform, &spectral_solver), "STGetKSP");
#ifdef PETSC_HAVE_STRUMPACK
  checkPetscError(KSPSetType(spectral_solver, KSPGMRES), "KSPSetType");
  checkPetscError(KSPSetTolerances(spectral_solver, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, 5),
                  "KSPSetTolerances");
  checkPetscError(KSPGetPC(spectral_solver, &spectral_preconditioner), "KSPGetPC");
  checkPetscError(PCSetType(spectral_preconditioner, PCLU), "PCSetType");
  checkPetscError(PCFactorSetMatSolverType(spectral_preconditioner, MATSOLVERSTRUMPACK),
                  "PCFactorSetMatSolverType");
#else
  checkPetscError(KSPSetType(spectral_solver, KSPGMRES), "KSPSetType");
  checkPetscError(KSPSetTolerances(spectral_solver, 1.0e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1000),
                  "KSPSetTolerances");
  checkPetscError(KSPGetPC(spectral_solver, &spectral_preconditioner), "KSPGetPC");
  checkPetscError(PCSetType(spectral_preconditioner, PCHYPRE), "PCSetType");
#endif
  checkPetscError(EPSSetFromOptions(eigensolver), "EPSSetFromOptions");
  checkPetscError(EPSSolve(eigensolver), "EPSSolve");

  PetscInt converged_modes = 0;
  PetscInt iterations = 0;
  EPSConvergedReason convergence_reason = EPS_CONVERGED_ITERATING;
  checkPetscError(EPSGetConverged(eigensolver, &converged_modes), "EPSGetConverged");
  checkPetscError(EPSGetIterationNumber(eigensolver, &iterations), "EPSGetIterationNumber");
  checkPetscError(EPSGetConvergedReason(eigensolver, &convergence_reason), "EPSGetConvergedReason");
  result.converged_modes = converged_modes;
  result.iterations = iterations;
  result.convergence_reason = static_cast<int>(convergence_reason);
  result.converged = converged_modes >= eigenvalue_count;

  const PetscInt modes_to_record = std::min(converged_modes, static_cast<PetscInt>(eigenvalue_count));
  if (modes_to_record > 0) {
    Vec free_eigenvector = nullptr;
    checkPetscError(MatCreateVecs(free_tangent, &free_eigenvector, nullptr), "MatCreateVecs");
    for (PetscInt mode_index = 0; mode_index < modes_to_record; ++mode_index) {
      LowestEigenpairResult::Mode mode;
      PetscScalar eigenvalue_real = 0.0;
      PetscScalar eigenvalue_imaginary = 0.0;
      checkPetscError(
          EPSGetEigenpair(eigensolver, mode_index, &eigenvalue_real, &eigenvalue_imaginary, free_eigenvector, nullptr),
          "EPSGetEigenpair");
      mode.eigenvalue_real = PetscRealPart(eigenvalue_real);
      mode.eigenvalue_imaginary = PetscRealPart(eigenvalue_imaginary);
      checkPetscError(EPSComputeError(eigensolver, mode_index, EPS_ERROR_ABSOLUTE, &mode.absolute_residual),
                      "EPSComputeError absolute");
      checkPetscError(EPSComputeError(eigensolver, mode_index, EPS_ERROR_RELATIVE, &mode.relative_residual),
                      "EPSComputeError relative");
      mode.norm_scaled_residual = mode.absolute_residual / std::max(1.0, result.tangent_norm);
      result.modes.push_back(mode);

      if (mode_index == 0) {
        PetscInt local_free_dofs = 0;
        checkPetscError(VecGetLocalSize(free_eigenvector, &local_free_dofs), "VecGetLocalSize");
        if (local_free_dofs != static_cast<PetscInt>(free_rows.size())) {
          throw std::runtime_error("Reduced eigenvector ownership does not match the local free degree-of-freedom list.");
        }
        const PetscScalar* free_values = nullptr;
        checkPetscError(VecGetArrayRead(free_eigenvector, &free_values), "VecGetArrayRead");
        result.eigenvector.SetSize(local_rows);
        result.eigenvector = 0.0;
        PetscInt free_index = 0;
        for (PetscInt local_row = 0; local_row < local_rows; ++local_row) {
          if (!constrained[static_cast<std::size_t>(local_row)]) {
            result.eigenvector[local_row] = PetscRealPart(free_values[free_index]);
            ++free_index;
          }
        }
        checkPetscError(VecRestoreArrayRead(free_eigenvector, &free_values), "VecRestoreArrayRead");
      }
    }
    checkPetscError(VecDestroy(&free_eigenvector), "VecDestroy");
  }

  checkPetscError(EPSDestroy(&eigensolver), "EPSDestroy");
  checkPetscError(MatDestroy(&free_tangent), "MatDestroy");
  checkPetscError(ISDestroy(&free_index_set), "ISDestroy");
  return result;
#endif
}

template <int order, int dim, typename... Parameters>
void runFinalStateEigenpairDiagnostic(SolidMechanics<order, dim, Parameters...>& solid, const std::string& output_name)
{
  const auto& tangent = solid.rebuildAcceptedStateTangent();
  auto result = computeLowestEigenpair(tangent, solid.essentialTrueDofs());
  const double sign_tolerance = eigenvalue_tol * std::max(1.0, result.tangent_norm);

  const int rank = solid.mfemParMesh().GetMyRank();
  if (rank == 0) {
    std::ofstream summary(output_name + "_lowest_eigenpair.csv");
    summary << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "# case " << selected_case << "\n"
            << "# nonlinear_solver " << nonlinear_solver_name << "\n"
            << "# linear_solver " << linear_solver_name << "\n"
            << "# preconditioner " << preconditioner_name << "\n"
            << "# time " << solid.time() << "\n"
            << "# cycle " << solid.cycle() << "\n"
            << "# eigensolver SLEPc\n"
            << "# tolerance " << eigenvalue_tol << "\n"
            << "# requested_modes " << eigenvalue_count << "\n"
            << "# convergence_test matrix_norm_scaled\n"
            << "# max_iterations " << eigenvalue_max_iterations << "\n"
            << "# eigensolver_method jacobi_davidson\n"
            << "# spectral_transform preconditioned\n"
            << "# spectral_shift " << result.spectral_shift << "\n"
            << "# tangent_infinity_norm " << result.tangent_norm << "\n"
            << "# sign_tolerance " << sign_tolerance << "\n"
            << "mode converged sign eigenvalue_real eigenvalue_imaginary magnitude absolute_residual "
               "norm_scaled_residual relative_residual iterations converged_modes convergence_reason free_dofs "
               "constrained_dofs tangent_is_symmetric\n";
    for (std::size_t mode_index = 0; mode_index < result.modes.size(); ++mode_index) {
      const auto& mode = result.modes[mode_index];
      const std::string sign_classification = mode.eigenvalue_real < -sign_tolerance   ? "negative"
                                              : mode.eigenvalue_real > sign_tolerance ? "positive"
                                                                                       : "neutral_within_tolerance";
      summary << mode_index << " 1 " << sign_classification << " " << mode.eigenvalue_real << " "
              << mode.eigenvalue_imaginary << " " << std::hypot(mode.eigenvalue_real, mode.eigenvalue_imaginary)
              << " " << mode.absolute_residual << " " << mode.norm_scaled_residual << " " << mode.relative_residual
              << " " << result.iterations << " " << result.converged_modes << " " << result.convergence_reason << " "
              << result.free_dofs << " " << result.constrained_dofs << " " << result.tangent_is_symmetric << "\n";
    }
  }

  for (std::size_t mode_index = 0; mode_index < result.modes.size(); ++mode_index) {
    const auto& mode = result.modes[mode_index];
    const std::string sign_classification = mode.eigenvalue_real < -sign_tolerance   ? "negative"
                                            : mode.eigenvalue_real > sign_tolerance ? "positive"
                                                                                     : "neutral_within_tolerance";
    SLIC_INFO_ROOT(std::format(
        "{} accepted-state eigenvalue {} = {:.12e} {:+.12e}i ({}), norm-scaled residual = {:.3e}", output_name,
        mode_index, mode.eigenvalue_real, mode.eigenvalue_imaginary, sign_classification, mode.norm_scaled_residual));
  }
  SLIC_INFO_ROOT(std::format("{} eigensolve iterations = {}, converged modes = {}, requested modes = {}, symmetric "
                             "tangent = {}",
                             output_name, result.iterations, result.converged_modes, eigenvalue_count,
                             result.tangent_is_symmetric));

  if (!result.converged) {
    throw std::runtime_error(std::format("{} final-state eigensolve converged {} of {} requested modes", output_name,
                                         result.converged_modes, eigenvalue_count));
  }

  if (write_final_state_eigenvector) {
    auto* displacement_space = const_cast<mfem::ParFiniteElementSpace*>(&solid.displacement().space());
    mfem::ParGridFunction eigenvector_field(displacement_space);
    eigenvector_field.SetFromTrueDofs(result.eigenvector);
    mfem::ParaViewDataCollection collection(output_name + "_lowest_eigenmode", &solid.mfemParMesh());
    collection.RegisterField("accepted_displacement", &solid.displacement().gridFunction());
    collection.RegisterField("lowest_eigenmode", &eigenvector_field);
    collection.SetCycle(solid.cycle());
    collection.SetTime(solid.time());
    collection.SetLevelsOfDetail(solid.displacement().space().GetMaxElementOrder());
    collection.SetHighOrderOutput(true);
    collection.SetDataFormat(mfem::VTKFormat::BINARY);
    collection.SetCompression(true);
    collection.Save();
  }
}

template <int order, int dim, typename... Parameters>
double averageBoundaryDisplacementComponent(const SolidMechanics<order, dim, Parameters...>& solid, Domain& domain,
                                            int component)
{
  Functional<double(H1<order, dim>)> boundary_integral({&solid.displacement().space()});
  boundary_integral.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [component](auto, auto, auto field) { return get<VALUE>(field)[component]; }, domain);

  FiniteElementState ones(solid.displacement());
  ones = 1.0;
  const double area = boundary_integral(solid.time(), ones);
  return boundary_integral(solid.time(), solid.displacement()) / area;
}

template <int order, int dim, typename... Parameters>
double sumReactionComponent(const SolidMechanics<order, dim, Parameters...>& solid, Domain& domain, int component)
{
  auto dof_list = domain.dof_list(&solid.displacement().space());
  solid.displacement().space().DofsToVDofs(component, dof_list);

  double local_reaction = 0.0;
  const auto& reactions = solid.dual("reactions");
  for (int i = 0; i < dof_list.Size(); ++i) {
    const int true_dof = solid.displacement().space().GetLocalTDofNumber(dof_list[i]);
    if (true_dof >= 0) local_reaction += reactions(true_dof);
  }
  double global_reaction = 0.0;
  MPI_Allreduce(&local_reaction, &global_reaction, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global_reaction;
}

mfem::Mesh buildShallowArchMesh(int num_elems_x, int num_elems_y, double span, double thickness, double rise)
{
  constexpr int dim = 2;
  auto mesh =
      mfem::Mesh::MakeCartesian2D(num_elems_x, num_elems_y, mfem::Element::QUADRILATERAL, true, span, thickness);

  const int num_vertices = mesh.GetNV();
  mfem::Vector vertices;
  mesh.GetVertices(vertices);
  mfem::Vector vertex(dim);

  for (int i = 0; i < num_vertices; ++i) {
    for (int d = 0; d < dim; ++d) {
      vertex(d) = vertices[d * num_vertices + i];
    }
    const double xi = vertex(0) / span - 0.5;
    const double centerline = rise * (1.0 - 4.0 * xi * xi);
    vertex(1) = centerline + vertex(1) - 0.5 * thickness;
    for (int d = 0; d < dim; ++d) {
      vertices[d * num_vertices + i] = vertex(d);
    }
  }

  mesh.SetVertices(vertices);
  return mesh;
}

double minBoundaryCoordinate(const Mesh& mesh, int component)
{
  const auto& par_mesh = mesh.mfemParMesh();
  mfem::Array<int> vertices;
  double local_min = std::numeric_limits<double>::infinity();
  for (int be = 0; be < par_mesh.GetNBE(); ++be) {
    auto* bdr_elem = par_mesh.GetBdrElement(be);
    if (!bdr_elem) continue;
    par_mesh.GetBdrElementVertices(be, vertices);
    for (int i = 0; i < vertices.Size(); ++i) {
      const double* vertex = par_mesh.GetVertex(vertices[i]);
      local_min = std::min(local_min, vertex[component]);
    }
  }
  double global_min = local_min;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, mesh.getComm());
  return global_min;
}

double maxBoundaryCoordinate(const Mesh& mesh, int component)
{
  const auto& par_mesh = mesh.mfemParMesh();
  mfem::Array<int> vertices;
  double local_max = -std::numeric_limits<double>::infinity();
  for (int be = 0; be < par_mesh.GetNBE(); ++be) {
    auto* bdr_elem = par_mesh.GetBdrElement(be);
    if (!bdr_elem) continue;
    par_mesh.GetBdrElementVertices(be, vertices);
    for (int i = 0; i < vertices.Size(); ++i) {
      const double* vertex = par_mesh.GetVertex(vertices[i]);
      local_max = std::max(local_max, vertex[component]);
    }
  }
  double global_max = local_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, mesh.getComm());
  return global_max;
}

double minBoundaryRadius(const Mesh& mesh)
{
  const auto& par_mesh = mesh.mfemParMesh();
  mfem::Array<int> vertices;
  double local_min = std::numeric_limits<double>::infinity();
  for (int be = 0; be < par_mesh.GetNBE(); ++be) {
    auto* bdr_elem = par_mesh.GetBdrElement(be);
    if (!bdr_elem) continue;
    par_mesh.GetBdrElementVertices(be, vertices);
    for (int i = 0; i < vertices.Size(); ++i) {
      const double* vertex = par_mesh.GetVertex(vertices[i]);
      local_min = std::min(local_min, std::sqrt(vertex[0] * vertex[0] + vertex[1] * vertex[1]));
    }
  }
  double global_min = local_min;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, mesh.getComm());
  return global_min;
}

double maxBoundaryRadius(const Mesh& mesh)
{
  const auto& par_mesh = mesh.mfemParMesh();
  mfem::Array<int> vertices;
  double local_max = -std::numeric_limits<double>::infinity();
  for (int be = 0; be < par_mesh.GetNBE(); ++be) {
    auto* bdr_elem = par_mesh.GetBdrElement(be);
    if (!bdr_elem) continue;
    par_mesh.GetBdrElementVertices(be, vertices);
    for (int i = 0; i < vertices.Size(); ++i) {
      const double* vertex = par_mesh.GetVertex(vertices[i]);
      local_max = std::max(local_max, std::sqrt(vertex[0] * vertex[0] + vertex[1] * vertex[1]));
    }
  }
  double global_max = local_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, mesh.getComm());
  return global_max;
}

std::pair<double, double> boundaryFaceProjectionExtents(const Mesh& mesh, int attr, const vec3& direction)
{
  const auto& par_mesh = mesh.mfemParMesh();
  mfem::Array<int> vertices;
  double local_min = std::numeric_limits<double>::infinity();
  double local_max = -std::numeric_limits<double>::infinity();
  for (int be = 0; be < par_mesh.GetNBE(); ++be) {
    auto* bdr_elem = par_mesh.GetBdrElement(be);
    if (!bdr_elem || par_mesh.GetBdrAttribute(be) != attr) continue;
    par_mesh.GetBdrElementVertices(be, vertices);
    std::vector<vec3> face_vertices;
    face_vertices.reserve(static_cast<size_t>(vertices.Size()));
    for (int i = 0; i < vertices.Size(); ++i) {
      const double* vertex = par_mesh.GetVertex(vertices[i]);
      face_vertices.push_back(vec3{vertex[0], vertex[1], vertex[2]});
    }
    const double projection = dot(average(face_vertices), direction);
    local_min = std::min(local_min, projection);
    local_max = std::max(local_max, projection);
  }
  double global_min = local_min;
  double global_max = local_max;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, mesh.getComm());
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, mesh.getComm());
  return {global_min, global_max};
}

template <int order, int dim, typename SolidSolver, typename TractionFunction>
double boundaryTractionResultant(const SolidSolver& solid, Domain& domain, TractionFunction traction_function,
                                 const vec3& direction)
{
  Functional<double(H1<order, dim>)> boundary_integral({&solid.displacement().space()});
  boundary_integral.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double time, auto X, auto) {
        const auto n = normalize(cross(get<DERIVATIVE>(X)));
        const auto traction = traction_function(get<VALUE>(X), n, time);
        double resultant = 0.0;
        for (int i = 0; i < dim; ++i) resultant += traction[i] * direction[i];
        return resultant;
      },
      domain);

  FiniteElementState ones(solid.displacement());
  ones = 1.0;
  return boundary_integral(solid.time(), ones);
}

}  // namespace
}  // namespace smith

#include "euler.hpp"
#include "shallow_arch.hpp"
#include "cylinder_crush.hpp"
#include "viscoelastic_buckling.hpp"
#include "contact_arch.hpp"
#include "sphere_into_corner.hpp"
#include "circ_in_circ.hpp"
#include "third_medium.hpp"
#include "thin_beam_bending.hpp"
#include "near_incompressible_block.hpp"
#include "sphere_penalty_contact.hpp"
#include "twisted_beam.hpp"
#include "thin_shell_bending.hpp"

int main(int argc, char** argv)
{
  smith::parseCommandLine(argc, argv);
  smith::ApplicationManager application_manager(argc, argv);
  smith::printCommandLineWarnings();

  std::vector<smith::RunTiming> timings;
  bool ran_case = false;

  if (smith::selectedCaseMatches({"01", "01/euler", "euler"})) {
    smith::runTimedCase(timings, "01/euler", smith::runEuler);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"02", "02/shallow_arch", "shallow_arch"})) {
    smith::runTimedCase(timings, "02/shallow_arch", smith::runShallowArch);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"03", "03/cylinder_crush_benchmark", "cylinder_crush_benchmark"})) {
    smith::runTimedCase(timings, "03/cylinder_crush_benchmark", smith::runCylinderCrushBenchmark);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"04", "04/viscoelastic_buckling", "viscoelastic_buckling"})) {
    smith::runTimedCase(timings, "04/viscoelastic_buckling", smith::runViscoelasticBuckling);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"06", "06/contact_arch", "contact_arch"})) {
    smith::runTimedCase(timings, "06/contact_arch", smith::runContactArch);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"07", "07/sphere_into_corner", "sphere_into_corner"})) {
    smith::runTimedCase(timings, "07/sphere_into_corner", smith::runSphereIntoCorner);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"08", "08/circ_in_circ", "circ_in_circ"})) {
    smith::runTimedCase(timings, "08/circ_in_circ", smith::runCircInCirc);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"09", "09/third_medium_c_bracket", "third_medium_c_bracket"})) {
    smith::runTimedCase(timings, "09/third_medium_c_bracket", smith::runThirdMediumCBracket);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"10", "10/thin_beam_bending", "thin_beam_bending"})) {
    smith::runTimedCase(timings, "10/thin_beam_bending", smith::runThinBeamBending);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"11", "11/near_incompressible_block", "near_incompressible_block"})) {
    smith::runTimedCase(timings, "11/near_incompressible_block", smith::runNearIncompressibleBlockCompression);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"12", "12/sphere_penalty_contact", "sphere_penalty_contact"})) {
    smith::runTimedCase(timings, "12/sphere_penalty_contact", smith::runSpherePenaltyContact);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"13", "13/twisted_beam", "twisted_beam"})) {
    smith::runTimedCase(timings, "13/twisted_beam", smith::runTwistedBeam);
    ran_case = true;
  }
  if (smith::selectedCaseMatches({"14", "14/thin_shell_bending", "thin_shell_bending"})) {
    smith::runTimedCase(timings, "14/thin_shell_bending", smith::runThinShellBending);
    ran_case = true;
  }

  if (ran_case) {
    smith::printTimingSummary(timings);
    return smith::anyRunFailed(timings) ? 1 : 0;
  }

  SLIC_ERROR_ROOT("Unknown --case value '" + smith::selected_case +
                  "'; use all, 01/euler, 02/shallow_arch, 03/cylinder_crush_benchmark, "
                  "04/viscoelastic_buckling, 06/contact_arch, 07/sphere_into_corner, "
                  "08/circ_in_circ, 09/third_medium_c_bracket, "
                  "10/thin_beam_bending, 11/near_incompressible_block, 12/sphere_penalty_contact, "
                  "13/twisted_beam, or 14/thin_shell_bending");
  return 1;
}
