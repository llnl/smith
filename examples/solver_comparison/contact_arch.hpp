// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

void runContactArch()
{
  constexpr int order = 1;
  constexpr int dim = 3;
  const int num_time_steps = 50;
  const double total_time = 0.16;
  const double load_schedule_time = 1.0;
  const double solve_abs_tol = 5.0e-6;
  const int max_linear_iterations = 30000;
  const int max_nonlinear_iterations = 5000;
  const double plane_clearance = 1.0e-4;
  const double total_support_inset = 0.05;
  const double total_plane_travel = 0.8;
  const double contact_penalty = 20.0;
  const double contact_regularization = 0.001;

  nonlinear_max_iterations = max_nonlinear_iterations;
  linear_max_iterations = max_linear_iterations;
  nonlinear_tol = solve_abs_tol;
  linear_tol = solve_abs_tol;

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "paper_contact_arch_fast");

  const auto mesh_file = czRepoRoot() / "data/meshes/half_circle_arch_3d.g";
  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(mesh_file.string()), "mesh", 0, parallelRefinement());
  SolidMechanics<order, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                   "paper_contact_arch_fast", mesh, std::vector<std::string>{}, 0, 0.0, false,
                                   warmStartEnabled("06/contact_arch", false));

  const double density = 1.0;
  const double E = 10.0;
  const double nu = 0.33;
  const double bulk_modulus = E / (3.0 * (1.0 - 2.0 * nu));
  const double shear_modulus = E / (2.0 * (1.0 + nu));
  solid_mechanics::NeoHookean material{density, bulk_modulus, shear_modulus};
  mesh->addDomainOfBodyElements("arch", by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("left_support", by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("right_support", by_attr<dim>(2));
  solid.setMaterial(material, mesh->domain("arch"));

  auto load_scale = [=](double time) { return time / load_schedule_time; };
  auto left_support_inset = [=](const vec3&, double time) {
    vec3 u{};
    u[0] = total_support_inset * load_scale(time);
    return u;
  };
  auto right_support_inset = [=](const vec3&, double time) {
    vec3 u{};
    u[0] = -total_support_inset * load_scale(time);
    return u;
  };

  solid.setDisplacementBCs(left_support_inset, mesh->domain("left_support"), Component::X);
  solid.setDisplacementBCs(right_support_inset, mesh->domain("right_support"), Component::X);
  solid.setFixedBCs(mesh->domain("left_support"), Component::Y);
  solid.setFixedBCs(mesh->domain("left_support"), Component::Z);
  solid.setFixedBCs(mesh->domain("right_support"), Component::Y);
  solid.setFixedBCs(mesh->domain("right_support"), Component::Z);

  const double arch_top_y = maxBoundaryCoordinate(*mesh, 1);
  const double arch_bottom_y = minBoundaryCoordinate(*mesh, 1);
  const double arch_outer_radius = maxBoundaryRadius(*mesh);
  const double arch_inner_radius = minBoundaryRadius(*mesh);
  const double arch_wall_thickness = arch_outer_radius - arch_inner_radius;
  const double outer_surface_radius_cutoff = arch_outer_radius - 0.25 * arch_wall_thickness;

  mesh->addDomainOfBoundaryElements("contact_surface", [=](std::vector<vec3> vertices, int) {
    const vec3 face_center = average(vertices);
    const double face_radius = std::sqrt(face_center[0] * face_center[0] + face_center[1] * face_center[1]);
    return face_radius > outer_surface_radius_cutoff;
  });
  solid.addCustomBoundaryIntegral(
      DependsOn<>{},
      [=](double time, auto X, auto displacement, auto) {
        const auto current_position = get_value(get<VALUE>(X)) + get_value(get<VALUE>(displacement));
        auto penalty_traction = 0.0 * current_position;
        const double upper_plane = arch_top_y + plane_clearance - total_plane_travel * load_scale(time);
        const double lower_plane = arch_bottom_y - plane_clearance;
        const double upper_overlap = current_position[1] - upper_plane;
        if (upper_overlap > 0.0) {
          const double regularized = upper_overlap * upper_overlap / (upper_overlap + contact_regularization);
          penalty_traction[1] = contact_penalty * regularized;
        }
        const double lower_overlap = lower_plane - current_position[1];
        if (lower_overlap > 0.0) {
          const double regularized = lower_overlap * lower_overlap / (lower_overlap + contact_regularization);
          penalty_traction[1] -= contact_penalty * regularized;
        }
        return penalty_traction;
      },
      mesh->domain("contact_surface"));
  solid.completeSetup();

  if (write_output) {
    solid.outputStateToDisk("paper_contact_arch_fast");
  }

  for (int step = 0; step < num_time_steps; ++step) {
    solid.advanceTimestep(total_time / num_time_steps);
    requireNonlinearConverged(true, std::format("paper_contact_arch_fast nonlinear solve failed at step {}", step + 1));
    if (write_output) {
      solid.outputStateToDisk("paper_contact_arch_fast");
    }
  }

  const double avg_contact_uy = averageBoundaryDisplacementComponent(solid, mesh->domain("contact_surface"), 1);
  SLIC_INFO_ROOT(std::format("paper_contact_arch_fast avg contact uy = {:.8e}", avg_contact_uy));
}
}  // namespace
}  // namespace smith
