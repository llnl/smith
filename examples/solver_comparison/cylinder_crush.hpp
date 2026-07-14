// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

void runCylinderCrushBenchmark()
{
  constexpr int order = 1;
  constexpr int dim = 3;
  const int num_time_steps = 16;
  const double total_compression = 0.125;

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "paper_cylinder_crush_fast");

  const auto mesh_file = czRepoRoot() / "data/meshes/thin_cylinder_crush.g";
  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(mesh_file.string()), "mesh", 0, parallelRefinement());
  SolidMechanics<order, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                   "paper_cylinder_crush_fast", mesh, std::vector<std::string>{}, 0, 0.0, false,
                                   warmStartEnabled("03/cylinder_crush_benchmark", true));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");

  const double lambda = 1.0;
  const double shear_modulus = 0.2;
  solid_mechanics::NeoHookean material{
      .density = 1.0, .K = (3.0 * lambda + 2.0 * shear_modulus) / 3.0, .G = shear_modulus};

  mesh->addDomainOfBodyElements("ring", by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("bottom", by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("top", by_attr<dim>(2));
  solid.setMaterial(material, mesh->domain("ring"));

  auto vertical_compression = [=](const tensor<double, dim>&, double time) {
    tensor<double, dim> displacement{};
    displacement[2] = -total_compression * time;
    return displacement;
  };

  solid.setFixedBCs(mesh->domain("bottom"));
  solid.setDisplacementBCs(vertical_compression, mesh->domain("top"), Component::Z);
  solid.completeSetup();

  if (write_output) {
    solid.outputStateToDisk("paper_cylinder_crush_fast");
  }

  for (int step = 0; step < num_time_steps; ++step) {
    solid.advanceTimestep(1.0 / num_time_steps);
    requireNonlinearConverged(true,
                              std::format("paper_cylinder_crush_fast nonlinear solve failed at step {}", step + 1));
    if (write_output) {
      solid.outputStateToDisk("paper_cylinder_crush_fast");
    }
  }

  const double avg_top_uz = averageBoundaryDisplacementComponent(solid, mesh->domain("top"), 2);
  const double top_reaction_z = sumReactionComponent(solid, mesh->domain("top"), 2);
  SLIC_INFO_ROOT(std::format("paper_cylinder_crush_fast avg top uz = {:.8e}", avg_top_uz));
  SLIC_INFO_ROOT(std::format("paper_cylinder_crush_fast top reaction z = {:.8e}", top_reaction_z));
}
}  // namespace
}  // namespace smith
