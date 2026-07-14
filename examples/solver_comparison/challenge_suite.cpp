// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <algorithm>
#include <array>
#include <filesystem>
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
constexpr int default_num_steps = 3;
int nonlinear_max_iterations = 10000;
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
bool timing_summary = false;
bool timing_summary_set = false;
bool timing_breakdown = false;
double cg_model_energy_stagnation_reltol = 0.0010638500842686796;
double cg_forcing_rel = 1.2981521889723316e-05;
double cg_cap_gamma = 0.7638340304667752;
double residual_growth_cap = 8.195074738257377;
double tr_decrease_factor = 0.38624275583816037;
double tr_increase_factor = 1.9742659010008659;
double tr_eta1 = 1.0e-9;
double tr_eta2 = 0.12631113528152121;
double tr_eta3 = 0.5494445482814023;
double tr_eta4 = 1.8368324568136323;
double nonlinear_tol = 1.0e-7;
double linear_tol = 0.8e-7;
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
     << " --trust-subspace-option=" << trust_subspace_option << " --max-cg-iterations=" << linear_max_iterations
     << " --deflation-coarse-mode=" << deflation_coarse_mode_name << " --deflation-pieces=" << deflation_pieces
     << " --deflation-smoother=" << deflation_smoother << " "
     << boolOption(use_bsr_spmv, "--use-bsr-spmv", "--no-use-bsr-spmv") << " "
     << boolOption(assemble_bsr, "--assemble-bsr", "--no-assemble-bsr") << " "
     << boolOption(write_output, "--paraview", "--no-paraview");
  if (warm_start_option == WarmStartOption::Enabled) {
    os << " --use-warm-start";
  } else if (warm_start_option == WarmStartOption::Disabled) {
    os << " --no-warm-start";
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

void requireNonlinearConverged(bool local_converged, const std::string& message)
{
  int local_failed = local_converged ? 0 : 1;
  int global_failed = 0;
  MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (global_failed) throw std::runtime_error(message);
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
          .print_level = 0};
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
          .max_line_search_iterations = 20,
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
    requireNonlinearConverged(true, std::format("{} nonlinear solve failed at step {}", output_name, step + 1));
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
    requireNonlinearConverged(true, std::format("thin_beam_bending nonlinear solve failed at step {}", step + 1));
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
    } else if (arg.rfind("--linear-max-iterations=", 0) == 0) {
      linear_max_iterations = std::stoi(arg.substr(std::string("--linear-max-iterations=").size()));
    } else if (arg.rfind("--max-cg-iterations=", 0) == 0) {
      linear_max_iterations = std::stoi(arg.substr(std::string("--max-cg-iterations=").size()));
    } else if (arg.rfind("--print-level=", 0) == 0) {
      print_level = std::stoi(arg.substr(std::string("--print-level=").size()));
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
  if (timing_breakdown) {
    print_level = std::max(print_level, 2);
  }
}

template <int order, int dim>
double averageBoundaryDisplacementComponent(const SolidMechanics<order, dim>& solid, Domain& domain, int component)
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

template <int order, int dim>
double sumReactionComponent(const SolidMechanics<order, dim>& solid, Domain& domain, int component)
{
  auto dof_list = domain.dof_list(&solid.displacement().space());
  solid.displacement().space().DofsToVDofs(component, dof_list);

  double net_reaction = 0.0;
  const auto& reactions = solid.dual("reactions");
  for (int i = 0; i < dof_list.Size(); ++i) {
    net_reaction += reactions(dof_list[i]);
  }
  return net_reaction;
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

struct NeoHookeanAdditiveSplit {
  using State = Empty;

  template <typename T, int dim>
  SMITH_HOST_DEVICE auto operator()(State&, const tensor<T, dim, dim>& du_dX) const
  {
    using std::pow;
    constexpr auto I = Identity<dim>();
    auto F = I + du_dX;
    auto J = det(F);
    auto Jm13 = pow(J, -1.0 / 3.0);
    auto F_bar = Jm13 * F;
    auto Pdev = G * Jm13 * (F_bar - 1.0 / 3.0 * inner(F_bar, F_bar) * inv(transpose(F_bar)));

    using std::log1p;
    auto logJ = log1p(detApIm1(du_dX));
    auto Pvol = K * logJ * inv(transpose(F));
    return Pdev + Pvol;
  }

  double density;
  double K;
  double G;
};

}  // namespace
}  // namespace smith

#include "euler.hpp"
#include "shallow_arch.hpp"
#include "cylinder_crush.hpp"
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
                  "'; use all, 01/euler, 02/shallow_arch, 03/cylinder_crush_benchmark, 06/contact_arch, "
                  "07/sphere_into_corner, 08/circ_in_circ, 09/third_medium_c_bracket, "
                  "10/thin_beam_bending, 11/near_incompressible_block, 12/sphere_penalty_contact, "
                  "13/twisted_beam, or 14/thin_shell_bending");
  return 1;
}
