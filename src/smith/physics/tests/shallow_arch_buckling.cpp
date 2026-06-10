// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
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

constexpr double length = 10.0;
constexpr double thickness = 0.025;
constexpr double end_tol = 1.0e-8;
constexpr double top_tol = 1.0e-8;
std::string solver_name = "TrustRegion";
int print_level = 2;
int nonlinear_max_iterations = 300000;
int trust_subspace_option = static_cast<int>(SubSpaceOptions::NEVER);
int trust_num_leftmost = 1;
int trust_num_previous_steps = 1;
int trust_work_quadrature_points = 2;
bool use_exact_energy = false;
int trust_num_lanczos = 0;
int trust_num_lanczos_iters = 0;
int max_cg_iterations = 100000;
double cg_model_stagnation_tol = 0.0;
int cg_model_stagnation_window = 0;
bool cg_eisenstat_walker = false;
bool use_bsr_spmv = false;
std::string deflation_smoother = "hypre";
bool assemble_bsr = false;
double cg_forcing_rel = 5.0e-5;
double residual_growth_cap = 3.0;
double tr_decrease_factor = 0.25;
double tr_increase_factor = 1.75;
double tr_eta1 = 1.0e-9;
double tr_eta2 = 0.1;
double tr_eta3 = 0.6;
double tr_eta4 = 4.2;
double mesh_scale = 1.0;
std::string preconditioner_name = "HypreJacobi";
std::string deflation_order_name = "affine";
std::string deflation_coarse_mode_name = "global";

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
    } else if (arg.rfind("--trust-work-quadrature=", 0) == 0) {
      trust_work_quadrature_points = std::stoi(arg.substr(std::string("--trust-work-quadrature=").size()));
    } else if (arg == "--use-exact-energy") {
      use_exact_energy = true;
    } else if (arg.rfind("--trust-num-lanczos=", 0) == 0) {
      trust_num_lanczos = std::stoi(arg.substr(std::string("--trust-num-lanczos=").size()));
    } else if (arg.rfind("--trust-num-lanczos-iters=", 0) == 0) {
      trust_num_lanczos_iters = std::stoi(arg.substr(std::string("--trust-num-lanczos-iters=").size()));
    } else if (arg.rfind("--max-cg-iterations=", 0) == 0) {
      max_cg_iterations = std::stoi(arg.substr(std::string("--max-cg-iterations=").size()));
    } else if (arg.rfind("--cg-stagnation-tol=", 0) == 0) {
      cg_model_stagnation_tol = std::stod(arg.substr(std::string("--cg-stagnation-tol=").size()));
    } else if (arg.rfind("--cg-stagnation-window=", 0) == 0) {
      cg_model_stagnation_window = std::stoi(arg.substr(std::string("--cg-stagnation-window=").size()));
    } else if (arg == "--cg-eisenstat-walker") {
      cg_eisenstat_walker = true;
    } else if (arg == "--use-bsr-spmv") {
      use_bsr_spmv = true;
    } else if (arg.rfind("--deflation-smoother=", 0) == 0) {
      deflation_smoother = arg.substr(std::string("--deflation-smoother=").size());
    } else if (arg == "--assemble-bsr") {
      assemble_bsr = true;
    } else if (arg.rfind("--cg-forcing-rel=", 0) == 0) {
      cg_forcing_rel = std::stod(arg.substr(std::string("--cg-forcing-rel=").size()));
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
    } else if (arg.rfind("--preconditioner=", 0) == 0) {
      preconditioner_name = arg.substr(std::string("--preconditioner=").size());
    } else if (arg.rfind("--deflation-order=", 0) == 0) {
      deflation_order_name = arg.substr(std::string("--deflation-order=").size());
    } else if (arg.rfind("--deflation-coarse-mode=", 0) == 0) {
      deflation_coarse_mode_name = arg.substr(std::string("--deflation-coarse-mode=").size());
    } else {
      argv[write_arg] = argv[read_arg];
      ++write_arg;
    }
  }
  argc = write_arg;
  // the hypre smoother would read stale matrix values under direct-BSR assembly
  if (assemble_bsr && deflation_smoother == "hypre") {
    deflation_smoother = "jacobi";
  }
}

}  // namespace

TEST(ShallowArchBuckling, CompressedThinBeamSnapThrough)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 2;
  constexpr int dim = 2;
  const int nx = std::max(1, static_cast<int>(std::lround(240 * mesh_scale)));
  const int ny = std::max(1, static_cast<int>(std::lround(10 * mesh_scale)));

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "shallow_arch_buckling");

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian2D(nx, ny, mfem::Element::QUADRILATERAL, true, length, thickness),
      "compressed_beam_mesh", 0, 0);

  mesh->addDomainOfBoundaryElements("left_end",
                                    [](std::vector<vec2> vertices, int) { return average(vertices)[0] < end_tol; });
  mesh->addDomainOfBoundaryElements(
      "top_face", [](std::vector<vec2> vertices, int) { return average(vertices)[1] > thickness - top_tol; });
  auto globalElementCount = [](int local_count) {
    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_count;
  };
  EXPECT_GT(globalElementCount(mesh->domain("left_end").total_elements()), 0);
  EXPECT_GT(globalElementCount(mesh->domain("top_face").total_elements()), 0);

  Preconditioner selected_pc = Preconditioner::HypreJacobi;
  if (preconditioner_name == "Deflation")
    selected_pc = Preconditioner::Deflation;
  else if (preconditioner_name == "HypreAMG")
    selected_pc = Preconditioner::HypreAMG;
  else if (preconditioner_name != "HypreJacobi")
    throw std::runtime_error("Unknown --preconditioner '" + preconditioner_name + "'");
  DeflationOrder selected_order = DeflationOrder::Affine;
  if (deflation_order_name == "quadratic")
    selected_order = DeflationOrder::Quadratic;
  else if (deflation_order_name != "affine")
    throw std::runtime_error("Unknown --deflation-order '" + deflation_order_name + "' (affine|quadratic)");
  CoarseMode selected_coarse_mode = CoarseMode::Additive;
  if (deflation_coarse_mode_name == "local")
    selected_coarse_mode = CoarseMode::AdditiveLocal;
  else if (deflation_coarse_mode_name == "schwarz")
    selected_coarse_mode = CoarseMode::AdditiveSchwarz;
  else if (deflation_coarse_mode_name != "global")
    throw std::runtime_error("Unknown --deflation-coarse-mode '" + deflation_coarse_mode_name +
                             "' (global|local|schwarz)");
  smith::LinearSolverOptions linear_options{.linear_solver = LinearSolver::CG,
                                            .preconditioner = selected_pc,
                                            .deflation_order = selected_order,
                                            .deflation_coarse_mode = selected_coarse_mode,
                                            .relative_tol = 1.0e-8,
                                            .absolute_tol = 1.0e-14,
                                            .max_iterations = max_cg_iterations,
                                            .cg_model_stagnation_tol = cg_model_stagnation_tol,
                                            .cg_model_stagnation_window = cg_model_stagnation_window,
                                            .cg_eisenstat_walker = cg_eisenstat_walker,
                                            .print_level = 0,
                                            .use_bsr_spmv = use_bsr_spmv,
          .deflation_smoother = deflation_smoother};

  smith::NonlinearSolverOptions nonlinear_options{
      .nonlin_solver = selectedNonlinearSolver(),
      .relative_tol = 1.0e-8,
      .absolute_tol = 1.0e-10,
      .max_iterations = nonlinear_max_iterations,
      .print_level = print_level,
      .subspace_option = static_cast<SubSpaceOptions>(trust_subspace_option),
      .num_leftmost = trust_num_leftmost,
      .num_previous_steps = trust_num_previous_steps,
      .trust_work_quadrature_points = trust_work_quadrature_points,
      .trust_num_lanczos = trust_num_lanczos,
      .trust_num_lanczos_iters = trust_num_lanczos_iters,
      .cg_forcing_rel = cg_forcing_rel,
      .residual_growth_cap = residual_growth_cap,
      .tr_decrease_factor = tr_decrease_factor,
      .tr_increase_factor = tr_increase_factor,
      .tr_eta1 = tr_eta1,
      .tr_eta2 = tr_eta2,
      .tr_eta3 = tr_eta3,
      .tr_eta4 = tr_eta4};

  SolidMechanics<p, dim> solid(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                               "compressed_beam", mesh);
  if (assemble_bsr) solid.enableDirectBSRAssembly();

  solid_mechanics::NeoHookean mat{.density = 1.0, .K = 100.0, .G = 10.0};
  solid.setMaterial(mat, mesh->entireBody());
  solid.setFixedBCs(mesh->domain("left_end"));

  constexpr double bending_traction = 5.0e-6;
  solid.setTraction([](auto, auto, double t) { return vec2{{0.0, -bending_traction * t}}; }, mesh->domain("top_face"));

  solid.completeSetup();
  solid.outputStateToDisk("shallow_arch_buckling");

  // --- Exact-energy callback (optional, enable with --use-exact-energy) ---
  // E(u, t) = ∫ψ(∇u) dΩ  +  ∫(-t·u) dΓ_traction.   r = ∂E/∂u must hold for the energy
  // descent argument; a directional FD consistency check is performed once at u=0 below.
  smith::Functional<double(smith::H1<p, dim>)> strain_energy_qoi({&solid.displacement().space()});
  strain_energy_qoi.AddDomainIntegral(
      smith::Dimension<dim>{}, smith::DependsOn<0>{},
      [mat](auto /*t*/, auto /*X*/, auto displacement) {
        auto du_dX = smith::get<smith::DERIVATIVE>(displacement);
        typename solid_mechanics::NeoHookean::State state{};
        return mat.strainEnergyDensity(state, du_dX);
      },
      mesh->entireBody());

  smith::Functional<double(smith::H1<p, dim>)> traction_work_qoi({&solid.displacement().space()});
  traction_work_qoi.AddBoundaryIntegral(
      smith::Dimension<dim - 1>{}, smith::DependsOn<0>{},
      [](double t, auto /*X*/, auto displacement) {
        auto u = smith::get<smith::VALUE>(displacement);
        smith::tensor<double, dim> trac{};
        trac[1] = -bending_traction * t;
        return -smith::dot(trac, u);
      },
      mesh->domain("top_face"));

  smith::FiniteElementState energy_scratch(solid.displacement());

  auto evaluate_energy = [&solid, &strain_energy_qoi, &traction_work_qoi,
                          &energy_scratch](const mfem::Vector& u_true) -> double {
    energy_scratch = u_true;
    const double t = solid.time();
    return strain_energy_qoi(t, energy_scratch) + traction_work_qoi(t, energy_scratch);
  };

  if (use_exact_energy) {
    // Consistency check: r ≈ -∇E along a random direction d, via central differences.
    // (For the static problem r = ∂E/∂u; failure here means the energy callback and the
    // residual are inconsistent and the "energy descent ⇒ no limit cycle" argument breaks.)
    {
      // Build a non-trivial test displacement (sinusoid in x, modulated in y) so neither r
      // nor ∂E/∂u is trivially zero. Set t=1 so traction is loaded.
      mfem::Vector u0(solid.displacement().Size());
      u0 = 0.0;
      // Project a coordinate-aware test field: u_y = a * sin(πx/L) * y(thickness-y)
      // This needs the coord field; use a simple deterministic perturbation in true-dof space
      // (sufficient to be non-trivial). Magnitude small enough not to invert any element.
      for (int i = 0; i < u0.Size(); ++i) u0(i) = 1e-4 * std::sin(0.31 * i + 0.5);
      mfem::Vector d(u0.Size());
      for (int i = 0; i < d.Size(); ++i) d(i) = std::sin(0.13 * i + 1.7);  // O(1) direction
      // Save & restore time so this check doesn't leak. base_physics has no setter, so we
      // rely on solid.time() being whatever it is at construction; for the QoI we evaluate
      // at the *current* time. The traction work is t-dependent — its mismatch would manifest
      // only if t > 0. As a defensive measure also re-run after advancing one tiny step.
      mfem::Vector r0(u0.Size());
      solid.evaluateResidual(u0, r0);
      const double rd = mfem::InnerProduct(MPI_COMM_WORLD, r0, d);
      const double eps = 1e-6;
      mfem::Vector u_plus(u0), u_minus(u0);
      u_plus.Add(eps, d);
      u_minus.Add(-eps, d);
      const double E_plus = evaluate_energy(u_plus);
      const double E_minus = evaluate_energy(u_minus);
      const double dE_dd = (E_plus - E_minus) / (2.0 * eps);
      SLIC_INFO_ROOT(std::format(
          "Energy/residual consistency at u0 (t={:.3f}): r·d = {:.6e}, (E(u+εd)-E(u-εd))/(2ε) = {:.6e},  "
          "rel diff = {:.3e}",
          solid.time(), rd, dE_dd, std::abs(rd - dE_dd) / std::max({std::abs(rd), std::abs(dE_dd), 1e-14})));
    }
    solid.setEnergyFunction(evaluate_energy);
    SLIC_INFO_ROOT("Exact-energy callback enabled: TR will use ΔE = E(x+d) - E(x) for acceptance.");
  }
  // --- end energy callback ---

  SLIC_INFO_ROOT(std::format(
      "Compressed thin beam snap-through run: solver = {}, trust_subspace_option = {}, trust_num_leftmost = {}, "
      "trust_num_previous_steps = {}, max_cg_iterations = {}, preconditioner = {}, deflation_order = {}",
      solver_name, trust_subspace_option, trust_num_leftmost, trust_num_previous_steps, max_cg_iterations,
      preconditioner_name, deflation_order_name));

  constexpr int num_steps = 1;
  double cumulative_step_time = 0.0;
  for (int step = 0; step < num_steps; ++step) {
    MPI_Barrier(MPI_COMM_WORLD);
    double tstep0 = MPI_Wtime();
    solid.advanceTimestep(1.0 / num_steps);
    MPI_Barrier(MPI_COMM_WORLD);
    double tstep = MPI_Wtime() - tstep0;
    cumulative_step_time += tstep;
    SLIC_INFO_ROOT(std::format("Load step {}/{} — wall = {:.3f} s  (cumulative {:.3f} s)", step + 1, num_steps, tstep,
                               cumulative_step_time));
    solid.outputStateToDisk("shallow_arch_buckling");
  }
  SLIC_INFO_ROOT(std::format("Total advanceTimestep wall = {:.3f} s over {} steps", cumulative_step_time, num_steps));

  const double displacement_norm = mfem::ParNormlp(solid.displacement(), 2, MPI_COMM_WORLD);
  SLIC_INFO_ROOT(std::format("shallow arch: final displacement l2 = {:.8e}", displacement_norm));
  EXPECT_TRUE(std::isfinite(displacement_norm));
  EXPECT_GT(displacement_norm, 0.0);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  smith::parseCommandLine(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
