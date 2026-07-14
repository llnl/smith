// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

void runThirdMediumCBracket()
{
  constexpr int order = 1;
  constexpr int dim = 2;
  constexpr int third_medium_steps = 25;

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "third_medium_c_bracket");

  const auto mesh_file = czRepoRoot() / "data/meshes/c_bracket.g";
  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(mesh_file.string()), "mesh", 0, parallelRefinement());
  SolidMechanics<order, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                   "third_medium_c_bracket", mesh, std::vector<std::string>{}, 0, 0.0, false,
                                   warmStartEnabled("09/third_medium_c_bracket", false));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");

  const double lambda = 1.0;
  const double shear_modulus = 0.2;
  const double rho = 1000.0;
  const double third_medium_scale = 1.0e-6;

  solid_mechanics::NeoHookean ring_material{
      .density = rho, .K = (3.0 * lambda + 2.0 * shear_modulus) / 3.0, .G = shear_modulus};
  mesh->addDomainOfBodyElements("ring", by_attr<dim>(1));
  solid.setMaterial(ring_material, mesh->domain("ring"));

  const double jelly_K = third_medium_scale * (3.0 * lambda + 2.0 * shear_modulus) / 3.0;
  solid_mechanics::NeoHookean jelly_material{.density = third_medium_scale * rho, .K = jelly_K, .G = 0.1 * jelly_K};
  mesh->addDomainOfBodyElements("jelly", by_attr<dim>(2));
  solid.setMaterial(jelly_material, mesh->domain("jelly"));

  mesh->addDomainOfBoundaryElements("bottom", by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("right", by_attr<dim>(4));
  mesh->addDomainOfBoundaryElements("top_surface", by_attr<dim>(1));

  solid.setFixedBCs(mesh->domain("bottom"));
  solid.setDisplacementBCs([](const tensor<double, dim>&, double) { return tensor<double, dim>{}; },
                           mesh->domain("right"), Component::X);
  solid.setTraction(
      [=](auto, auto n, double time) {
        auto traction = 0.0 * n;
        traction(0) = -0.3e-3 * time / third_medium_steps;
        traction(1) = -0.3e-3 * time / third_medium_steps;
        return traction;
      },
      mesh->domain("top_surface"));
  solid.completeSetup();

  if (write_output) {
    solid.outputStateToDisk("third_medium_c_bracket");
  }

  for (int step = 0; step < third_medium_steps; ++step) {
    solid.advanceTimestep(1.0 / third_medium_steps);
    requireNonlinearConverged(true, std::format("third_medium_c_bracket nonlinear solve failed at step {}", step + 1));
    if (write_output) {
      solid.outputStateToDisk("third_medium_c_bracket");
    }
  }

  checkDisplacement("third medium c bracket", solid);
}

}  // namespace
}  // namespace smith
