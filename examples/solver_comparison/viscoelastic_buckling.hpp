// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/physics/materials/viscoelastic.hpp"

namespace smith {
namespace {

void runViscoelasticBuckling()
{
  constexpr int order = 2;
  constexpr int dim = 3;
  constexpr double load = 0.17;
  constexpr double max_time = 24.0;
  constexpr int num_time_steps = 60;

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "paper_viscoelastic_buckling_fast");

  const auto mesh_file = czRepoRoot() / "data/meshes/cap_hemisphere.g";
  auto mesh = std::make_shared<Mesh>(mesh_file.string(), "mesh", 0, parallelRefinement());
  checkElementCount("viscoelastic buckling", *mesh);

  mesh->addDomainOfBoundaryElements("zsymm", by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("xsymm", by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("bottom", by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("top", by_attr<dim>(4));

  FiniteElementState temperature(mesh->mfemParMesh(), L2<0>{}, "temperature");
  temperature = 300.0;

  SolidMechanics<order, dim, Parameters<L2<0>>> solid(nonlinearOptions(), linearOptions(),
                                                      solid_mechanics::default_quasistatic_options,
                                                      "paper_viscoelastic_buckling_fast", mesh, {"temperature"}, 0, 0.0,
                                                      false, warmStartEnabled("04/viscoelastic_buckling", false));
  solid.setParameter(0, temperature);

  Functional<double(H1<order, dim>)> area_computer({&solid.displacement().space()});
  area_computer.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{}, [](double, auto, auto) { return 1.0; }, mesh->domain("top"));
  const double area = area_computer(0.0, solid.displacement());
  SLIC_INFO_ROOT(std::format("viscoelastic buckling load patch area = {:.8e}", area));

  auto sawtooth_load = [=](auto, auto, double time) {
    const double rise_time = 0.25 * max_time;
    const double force_y = time < rise_time ? time / rise_time * load : 0.0;
    return vec3{0.0, -force_y / area, 0.0};
  };
  solid.setTraction(sawtooth_load, mesh->domain("top"));

  constexpr double rho = 1.0;
  constexpr double G_inf = 1.0e3;
  constexpr double G_0 = 3.0 * G_inf;
  constexpr double G_g = G_inf + G_0;
  constexpr double nu_g = 0.45;
  constexpr double K = 2.0 / 3.0 * G_g * (1.0 + nu_g) / (1.0 - 2.0 * nu_g);
  constexpr double tau_0 = 1.0;
  constexpr double eta_0 = G_0 * tau_0;
  constexpr double alpha_inf = 0.0;
  constexpr double theta_r = 300.0;
  constexpr double theta_sf = theta_r;
  constexpr double C_1 = 0.0;
  constexpr double C_2 = 50.0;

  solid_mechanics::ViscoelasticOldInterface material(rho, K, G_inf, alpha_inf, theta_sf, G_0, eta_0, theta_r, C_1, C_2);
  auto internal_states = solid.createQuadratureDataBuffer(solid_mechanics::Viscoelastic::State{}, mesh->entireBody());
  solid.setRateDependentMaterial(DependsOn<0>{}, material, mesh->entireBody(), internal_states);

  solid.setFixedBCs(mesh->domain("bottom"));
  solid.setFixedBCs(mesh->domain("xsymm"), Component::X);
  solid.setFixedBCs(mesh->domain("zsymm"), Component::Z);
  solid.completeSetup();

  const int rank = mesh->mfemParMesh().GetMyRank();
  std::ofstream history;
  if (rank == 0) {
    history.open("paper_viscoelastic_buckling_fast_force_displacement.csv");
    history << "# time displacement applied_force reaction_force\n";
  }

  auto write_history_snapshot = [&]() {
    const double force_y = sumReactionComponent(solid, mesh->domain("bottom"), 1);
    const double top_displacement_y = averageBoundaryDisplacementComponent(solid, mesh->domain("top"), 1);
    const double rise_time = 0.25 * max_time;
    const double applied_force_y = solid.time() < rise_time ? solid.time() / rise_time * load : 0.0;
    if (rank == 0) {
      history << solid.time() << " " << top_displacement_y << " " << applied_force_y << " " << force_y << "\n";
    }
  };

  if (write_output) {
    solid.outputStateToDisk("paper_viscoelastic_buckling_fast");
  }
  write_history_snapshot();

  for (int step = 0; step < num_time_steps; ++step) {
    MPI_Barrier(mesh->getComm());
    const double start_time = MPI_Wtime();
    solid.advanceTimestep(max_time / num_time_steps);
    MPI_Barrier(mesh->getComm());
    requireSolveConverged(solid,
                          std::format("paper_viscoelastic_buckling_fast nonlinear solve failed at step {}", step + 1));
    SLIC_INFO_ROOT(std::format("paper_viscoelastic_buckling_fast step {}/{} wall = {:.3f} s", step + 1, num_time_steps,
                               MPI_Wtime() - start_time));
    if (write_output) {
      solid.outputStateToDisk("paper_viscoelastic_buckling_fast");
    }
    write_history_snapshot();
  }

  const double avg_top_uy = averageBoundaryDisplacementComponent(solid, mesh->domain("top"), 1);
  const double bottom_reaction_y = sumReactionComponent(solid, mesh->domain("bottom"), 1);
  SLIC_INFO_ROOT(std::format("paper_viscoelastic_buckling_fast avg top uy = {:.8e}", avg_top_uy));
  SLIC_INFO_ROOT(std::format("paper_viscoelastic_buckling_fast bottom reaction y = {:.8e}", bottom_reaction_y));
}

}  // namespace
}  // namespace smith
