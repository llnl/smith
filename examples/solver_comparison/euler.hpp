// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

void runEuler()
{
  constexpr int order = 1;
  constexpr int dim = 3;
  constexpr int nx = 4;
  constexpr int ny = 7;
  constexpr int nz = 50;
  constexpr double lx = 1.0;
  constexpr double ly = 1.2;
  constexpr double lz = 30.0;
  constexpr double density = 1.0;
  constexpr double E = 10.0;
  constexpr double nu = 0.33;
  constexpr double load = 0.002;
  constexpr double total_time = 1.0;
  constexpr int extra_refinement = 1;

  const double bulk_mod = E / (3.0 * (1.0 - 2.0 * nu));
  const double shear_mod = E / (2.0 * (1.0 + nu));

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "paper_euler_fast");

  auto mesh = std::make_shared<Mesh>(mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::HEXAHEDRON, lx, ly, lz),
                                     "paper_euler_mesh", 0, parallelRefinement() + extra_refinement);

  SolidMechanics<order, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                   "paper_euler_fast", mesh, std::vector<std::string>{}, 0, 0.0, false,
                                   warmStartEnabled("01/euler", true));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");

  solid_mechanics::NeoHookean material{density, bulk_mod, shear_mod};
  solid.setMaterial(material, mesh->entireBody());

  mesh->addDomainOfBoundaryElements("back_surface", by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("top_surface", by_attr<dim>(6));
  mesh->addDomainOfBoundaryElements("bottom_surface", by_attr<dim>(1));

  solid.setTraction([&](auto, auto n, auto t) { return -load * t * n; }, mesh->domain("top_surface"));
  solid.setTraction([&](auto, auto n, auto) { return 1.0e-5 * n; }, mesh->domain("back_surface"));
  solid.setFixedBCs(mesh->domain("bottom_surface"));
  solid.completeSetup();

  if (write_output) {
    solid.outputStateToDisk("paper_euler_fast");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  solid.advanceTimestep(total_time);
  requireNonlinearConverged(true, "paper_euler_fast nonlinear solve failed");
  if (write_output) {
    solid.outputStateToDisk("paper_euler_fast");
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
}  // namespace
}  // namespace smith
