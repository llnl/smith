// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

void runCircInCirc()
{
  constexpr int order = 1;
  constexpr int dim = 2;
  const int circle_num_steps = 50;

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "paper_circ_in_circ_fast");

  const auto mesh_file = czRepoRoot() / "data/meshes/circ_in_circ.g";
  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(mesh_file.string()), "mesh", 0, parallelRefinement());
  SolidMechanics<order, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                   "paper_circ_in_circ_fast", mesh, std::vector<std::string>{}, 0, 0.0, false,
                                   warmStartEnabled("08/circ_in_circ", false));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");

  auto lambda = 1.0;
  auto G = 0.2;
  auto rho = 1000.0;
  double third_medium_scale = 2e-7;

  solid_mechanics::NeoHookean material1{.density = rho, .K = (3 * lambda + 2 * G) / 3, .G = G};
  mesh->addDomainOfBodyElements("ring1", by_attr<dim>(1));
  solid.setMaterial(material1, mesh->domain("ring1"));

  solid_mechanics::NeoHookean material2{.density = rho, .K = (3 * lambda + 2 * G) / 3, .G = G};
  mesh->addDomainOfBodyElements("ring2", by_attr<dim>(3));
  solid.setMaterial(material2, mesh->domain("ring2"));

  solid_mechanics::NeoHookean material3{.density = rho, .K = (3 * lambda + 2 * G) / 3, .G = G};
  mesh->addDomainOfBodyElements("center", by_attr<dim>(5));
  solid.setMaterial(material3, mesh->domain("center"));

  double jelly_K = third_medium_scale * (3 * lambda + 2 * G) / 3;
  NeoHookeanAdditiveSplit mat_jelly{.density = third_medium_scale * rho, .K = jelly_K, .G = 1.4 * jelly_K};
  mesh->addDomainOfBodyElements("jelly1", by_attr<dim>(2));
  mesh->addDomainOfBodyElements("jelly2", by_attr<dim>(4));
  solid.setMaterial(mat_jelly, mesh->domain("jelly1"));
  solid.setMaterial(mat_jelly, mesh->domain("jelly2"));

  mesh->addDomainOfBoundaryElements("bottom_surface", by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("top_surface", by_attr<dim>(1));
  solid.setFixedBCs(mesh->domain("bottom_surface"));
  solid.setTraction(
      [=](auto, auto n, double t) {
        auto trac = 0.0 * n;
        trac(1) = -1.2 * t / circle_num_steps;
        return trac;
      },
      mesh->domain("top_surface"));
  solid.completeSetup();

  if (write_output) {
    solid.outputStateToDisk("paper_circ_in_circ_fast");
  }

  for (int step = 0; step < circle_num_steps; ++step) {
    solid.advanceTimestep(1.0 / circle_num_steps);
    requireNonlinearConverged(true, std::format("paper_circ_in_circ_fast nonlinear solve failed at step {}", step + 1));
    if (write_output) {
      solid.outputStateToDisk("paper_circ_in_circ_fast");
    }
  }

  const double avg_top_uy = averageBoundaryDisplacementComponent(solid, mesh->domain("top_surface"), 1);
  SLIC_INFO_ROOT(std::format("paper_circ_in_circ_fast avg top uy = {:.8e}", avg_top_uy));
}
}  // namespace
}  // namespace smith
