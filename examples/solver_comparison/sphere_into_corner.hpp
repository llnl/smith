// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

void runSphereIntoCorner()
{
  constexpr int order = 1;
  constexpr int dim = 3;
  const int num_time_steps = 16;
  const double total_time = 1.0;
  const double max_patch_traction = 0.5;
  const double contact_penalty = 8.0;
  const double nominal_element_size = 0.025;
  const double load_patch_projection_margin_fraction = 0.24;
  const double contact_smoothing_band = nominal_element_size / 1000.0;

  nonlinear_tol = 3.0e-6;
  linear_tol = 1.6e-6;
  nonlinear_max_iterations *= 4;

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "paper_sphere_corner_fast");

  const auto mesh_file = czRepoRoot() / "data/meshes/hollow_sphere_octant_corner.g";
  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(mesh_file.string()), "mesh", 0, parallelRefinement());
  SolidMechanics<order, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                   "paper_sphere_corner_fast", mesh, std::vector<std::string>{}, 0, 0.0, false,
                                   warmStartEnabled("07/sphere_into_corner", false));

  const double density = 1.0;
  const double E = 6.0;
  const double nu = 0.33;
  const double bulk_modulus = E / (3.0 * (1.0 - 2.0 * nu));
  const double shear_modulus = E / (2.0 * (1.0 + nu));
  solid_mechanics::NeoHookean material{density, bulk_modulus, shear_modulus};

  mesh->addDomainOfBodyElements("sphere_shell", by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("outer_surface", by_attr<dim>(2));
  solid.setMaterial(material, mesh->domain("sphere_shell"));

  const vec3 load_direction = normalize(vec3{-1.0, -1.0, -1.0});
  const vec3 push_direction = -load_direction;
  const auto [proj_min, proj_max] = boundaryFaceProjectionExtents(*mesh, 2, push_direction);
  const double proj_threshold = proj_max - load_patch_projection_margin_fraction * (proj_max - proj_min);

  mesh->addDomainOfBoundaryElements("load_patch", [=](std::vector<vec3> vertices, int attr) {
    if (attr != 2) return false;
    return dot(average(vertices), push_direction) >= proj_threshold;
  });

  auto applied_patch_traction = [=](auto, auto, double time) {
    return 0.1 * (max_patch_traction * time / total_time) * load_direction;
  };
  solid.setTraction(applied_patch_traction, mesh->domain("load_patch"));
  solid.addCustomBoundaryIntegral(
      DependsOn<>{},
      [=](double, auto X, auto displacement, auto) {
        const auto current_position = get_value(get<VALUE>(X)) + get_value(get<VALUE>(displacement));
        auto penalty_traction = 0.0 * current_position;
        auto plane_penalty_component = [=](double signed_gap) {
          if (signed_gap >= 0.0) return 0.0;
          const double penetration = -signed_gap;
          if (penetration <= contact_smoothing_band) {
            return -0.5 * contact_penalty * penetration * penetration / contact_smoothing_band;
          }
          return contact_penalty * (signed_gap + 0.5 * contact_smoothing_band);
        };
        penalty_traction[0] = plane_penalty_component(current_position[0]);
        penalty_traction[1] = plane_penalty_component(current_position[1]);
        penalty_traction[2] = plane_penalty_component(current_position[2]);
        return penalty_traction;
      },
      mesh->domain("outer_surface"));
  solid.completeSetup();

  if (write_output) {
    solid.outputStateToDisk("paper_sphere_corner_fast");
  }

  for (int step = 0; step < num_time_steps; ++step) {
    solid.advanceTimestep(total_time / num_time_steps);
    requireNonlinearConverged(true, std::format("paper_sphere_corner_fast nonlinear solve failed at step {}", step + 1));
    if (write_output) {
      solid.outputStateToDisk("paper_sphere_corner_fast");
    }
  }

  const double drive_displacement =
      -averageBoundaryDisplacementComponent(solid, mesh->domain("load_patch"), 0) * load_direction[0];
  const double load_resultant =
      boundaryTractionResultant<order, dim>(solid, mesh->domain("load_patch"), applied_patch_traction, load_direction);
  SLIC_INFO_ROOT(std::format("paper_sphere_corner_fast drive displacement = {:.8e}", drive_displacement));
  SLIC_INFO_ROOT(std::format("paper_sphere_corner_fast load resultant = {:.8e}", load_resultant));
}
}  // namespace
}  // namespace smith
