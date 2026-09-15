// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

template <int order>
void runEulerWithOrder()
{
  constexpr int dim = 3;
  const int nx = scaled(4);
  const int ny = scaled(7);
  const int nz = scaled(50);
  constexpr double lx = 1.0;
  constexpr double ly = 1.2;
  constexpr double lz = 30.0;
  constexpr double density = 1.0;
  constexpr double E = 10.0;
  constexpr double nu = 0.33;
  constexpr double total_time = 1.0;
  const double load = euler_load;
  const double selector_traction = euler_selector_traction;
  const int num_solution_states = std::max(2, euler_solution_states);
  const int extra_refinement = euler_extra_refinement;

  const double bulk_mod = E / (3.0 * (1.0 - 2.0 * nu));
  const double shear_mod = E / (2.0 * (1.0 + nu));

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "paper_euler_fast");

  auto mesh = std::make_shared<Mesh>(mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::HEXAHEDRON, lx, ly, lz),
                                     "paper_euler_mesh", 0, parallelRefinement() + extra_refinement);
  checkElementCount("euler", *mesh);

  SolidMechanics<order, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                   "paper_euler_fast", mesh, std::vector<std::string>{}, 0, 0.0, false,
                                   warmStartEnabled("01/euler", true));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");

  solid_mechanics::NeoHookean material{density, bulk_mod, shear_mod};
  solid.setMaterial(material, mesh->entireBody());

  mesh->addDomainOfBoundaryElements("right_surface", by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("top_surface", by_attr<dim>(6));
  mesh->addDomainOfBoundaryElements("bottom_surface", by_attr<dim>(1));

  solid.setTraction([&](auto, auto n, auto t) { return -load * t * n; }, mesh->domain("top_surface"));
  if (selector_traction != 0.0) {
    solid.setTraction([&](auto, auto n, auto) { return selector_traction * n; }, mesh->domain("right_surface"));
  }
  solid.setFixedBCs(mesh->domain("bottom_surface"));
  solid.completeSetup();

  const int rank = mesh->mfemParMesh().GetMyRank();
  std::ofstream history;
  if (rank == 0) {
    history.open("paper_euler_fast_load_displacement.csv");
    history << "# time avg_top_ux avg_top_uy avg_top_uz applied_force_z bottom_reaction_z\n";
  }
  auto write_history_snapshot = [&]() {
    const double avg_top_ux = averageBoundaryDisplacementComponent(solid, mesh->domain("top_surface"), 0);
    const double avg_top_uy = averageBoundaryDisplacementComponent(solid, mesh->domain("top_surface"), 1);
    const double avg_top_uz = averageBoundaryDisplacementComponent(solid, mesh->domain("top_surface"), 2);
    const double bottom_reaction_z = sumReactionComponent(solid, mesh->domain("bottom_surface"), 2);
    if (rank == 0) {
      history << solid.time() << " " << avg_top_ux << " " << avg_top_uy << " " << avg_top_uz << " "
              << load * solid.time() * lx * ly << " " << bottom_reaction_z << "\n";
      history.flush();
    }
  };

  std::vector<double> load_step_sizes;
  if (euler_coarse_load_steps > 0 || euler_refined_load_steps > 0 || euler_refined_start_time >= 0.0 ||
      euler_refined_start_traction >= 0.0 || euler_refined_start_force >= 0.0) {
    if (euler_coarse_load_steps <= 0 || euler_refined_load_steps <= 0) {
      throw std::runtime_error(
          "Euler refined schedule requires positive --euler-coarse-load-steps and "
          "--euler-refined-load-steps");
    }
    double refined_start_time = euler_refined_start_time;
    if (euler_refined_start_force >= 0.0) {
      refined_start_time = euler_refined_start_force / (load * lx * ly);
    } else if (euler_refined_start_traction >= 0.0) {
      refined_start_time = euler_refined_start_traction / load;
    }
    if (refined_start_time <= 0.0 || refined_start_time >= total_time) {
      throw std::runtime_error(
          std::format("Euler refined start time {} must be between 0 and {}", refined_start_time, total_time));
    }
    load_step_sizes.insert(load_step_sizes.end(), static_cast<std::size_t>(euler_coarse_load_steps),
                           refined_start_time / euler_coarse_load_steps);
    load_step_sizes.insert(load_step_sizes.end(), static_cast<std::size_t>(euler_refined_load_steps),
                           (total_time - refined_start_time) / euler_refined_load_steps);
    SLIC_INFO_ROOT(
        std::format("paper_euler_fast two-stage load schedule: {} coarse steps to time {:.8e}, "
                    "{} refined steps to time {:.8e}",
                    euler_coarse_load_steps, refined_start_time, euler_refined_load_steps, total_time));
  } else {
    load_step_sizes.insert(load_step_sizes.end(), static_cast<std::size_t>(num_solution_states - 1),
                           total_time / (num_solution_states - 1));
  }

  if (write_output) {
    solid.outputStateToDisk("paper_euler_fast");
  }
  write_history_snapshot();

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  for (std::size_t step = 0; step < load_step_sizes.size(); ++step) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double step_start = MPI_Wtime();
    solid.advanceTimestep(load_step_sizes[step]);
    MPI_Barrier(MPI_COMM_WORLD);
    requireSolveConverged(solid, std::format("paper_euler_fast nonlinear solve failed at step {}", step + 1));
    SLIC_INFO_ROOT(std::format("paper_euler_fast step {}/{} wall = {:.3f} s", step + 1, load_step_sizes.size(),
                               MPI_Wtime() - step_start));
    if (write_output) {
      solid.outputStateToDisk("paper_euler_fast");
    }
    write_history_snapshot();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  SLIC_INFO_ROOT(std::format("paper_euler_fast wall = {:.3f} s", MPI_Wtime() - t0));
  const double avg_top_uz = averageBoundaryDisplacementComponent(solid, mesh->domain("top_surface"), 2);
  const double bottom_reaction_z = sumReactionComponent(solid, mesh->domain("bottom_surface"), 2);
  const double displacement_norm = mfem::ParNormlp(solid.displacement(), 2, MPI_COMM_WORLD);
  SLIC_INFO_ROOT(std::format("paper_euler_fast avg top uz = {:.8e}", avg_top_uz));
  SLIC_INFO_ROOT(std::format("paper_euler_fast bottom reaction z = {:.8e}", bottom_reaction_z));
  SLIC_INFO_ROOT(std::format("paper_euler_fast final displacement l2 = {:.8e}", displacement_norm));
}

void runEuler()
{
  if (sim_order == 1) {
    runEulerWithOrder<1>();
  } else if (sim_order == 2) {
    runEulerWithOrder<2>();
  } else {
    throw std::runtime_error("Euler only supports --order=1 or --order=2");
  }
}
}  // namespace
}  // namespace smith
