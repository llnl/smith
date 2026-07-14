// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

void runShallowArch()
{
  constexpr int order = 1;
  constexpr int dim = 2;
  const double span = 12.0;
  const double thickness = 0.35;
  const double rise = 0.8;
  const double density = 1.0;
  const double E = 10.0;
  const double nu = 0.33;
  const double bulk_modulus = E / (3.0 * (1.0 - 2.0 * nu));
  const double shear_modulus = E / (2.0 * (1.0 + nu));
  const double load_magnitude = 1.5e-2;
  const int num_time_steps = 20;
  const int extra_refinement = 4;

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "paper_shallow_arch_fast");

  auto mesh = std::make_shared<Mesh>(buildShallowArchMesh(16, 3, span, thickness, rise), "mesh", 0,
                                     parallelRefinement() + extra_refinement);
  SolidMechanics<order, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                   "paper_shallow_arch_fast", mesh, std::vector<std::string>{}, 0, 0.0, false,
                                   warmStartEnabled("02/shallow_arch", true));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");

  solid_mechanics::NeoHookean material{density, bulk_modulus, shear_modulus};
  solid.setMaterial(material, mesh->entireBody());
  mesh->addDomainOfBoundaryElements("left_support", by_attr<dim>(4));
  mesh->addDomainOfBoundaryElements("right_support", by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("top_surface", by_attr<dim>(3));

  solid.setFixedBCs(mesh->domain("left_support"));
  solid.setDisplacementBCs(
      [](const tensor<double, dim>&, double) {
        tensor<double, dim> u{};
        return u;
      },
      mesh->domain("right_support"), Component::Y);
  solid.setTraction(
      [=](auto, auto, double time) {
        tensor<double, dim> traction{};
        traction[1] = -load_magnitude * time;
        return traction;
      },
      mesh->domain("top_surface"));
  solid.completeSetup();

  if (write_output) {
    solid.outputStateToDisk("paper_shallow_arch_fast");
  }

  for (int step = 0; step < num_time_steps; ++step) {
    solid.advanceTimestep(1.0 / num_time_steps);
    requireNonlinearConverged(true, std::format("paper_shallow_arch_fast nonlinear solve failed at step {}", step + 1));
    if (write_output) {
      solid.outputStateToDisk("paper_shallow_arch_fast");
    }
  }

  const double avg_crown_uy = averageBoundaryDisplacementComponent(solid, mesh->domain("top_surface"), 1);
  const double support_reaction_y = sumReactionComponent(solid, mesh->domain("left_support"), 1) +
                                    sumReactionComponent(solid, mesh->domain("right_support"), 1);
  SLIC_INFO_ROOT(std::format("paper_shallow_arch_fast avg crown uy = {:.8e}", avg_crown_uy));
  SLIC_INFO_ROOT(std::format("paper_shallow_arch_fast support reaction y = {:.8e}", support_reaction_y));
}
}  // namespace
}  // namespace smith
