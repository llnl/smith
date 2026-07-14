// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

constexpr double arch_length = 10.0;
constexpr double arch_thickness = 0.015;
constexpr double arch_end_tol = 1.0e-8;
constexpr double arch_top_tol = 1.0e-8;

void runThinBeamBending()
{
  constexpr int local_dim = 2;
  const int nx = std::max(1, static_cast<int>(std::lround(350 * mesh_scale)));
  const int ny = std::max(1, static_cast<int>(std::lround(12 * mesh_scale)));

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "thin_beam_bending");

  auto mesh = std::make_shared<Mesh>(
      mfem::Mesh::MakeCartesian2D(nx, ny, mfem::Element::QUADRILATERAL, true, arch_length, arch_thickness),
      "compressed_beam_mesh", 0, 0);
  checkElementCount("thin beam bending", *mesh);

  mesh->addDomainOfBoundaryElements(
      "left_end", [](std::vector<vec2> vertices, int) { return average(vertices)[0] < arch_end_tol; });
  mesh->addDomainOfBoundaryElements(
      "top_face", [](std::vector<vec2> vertices, int) { return average(vertices)[1] > arch_thickness - arch_top_tol; });

  SolidMechanics<p, local_dim> solid(shallowArchBucklingNonlinearOptions(), shallowArchBucklingLinearOptions(),
                                     solid_mechanics::default_quasistatic_options, "compressed_beam", mesh,
                                     std::vector<std::string>{}, 0, 0.0, false,
                                     warmStartEnabled("10/thin_beam_bending", true));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");

  constexpr double top_traction = 1e-8;
  solid_mechanics::NeoHookean material{.density = 1.0, .K = 100.0, .G = 10.0};
  solid.setMaterial(material, mesh->entireBody());
  solid.setFixedBCs(mesh->domain("left_end"));
  solid.setTraction([](auto, auto, double time) { return vec2{{0.0, -top_traction * time}}; },
                    mesh->domain("top_face"));
  solid.completeSetup();

  SLIC_INFO_ROOT(
      std::format("Compressed thin beam bending run: solver = {}, trust_subspace_option = {}, trust_num_leftmost = {}, "
                  "trust_num_previous_steps = {}, linear_max_iterations = {}, preconditioner = {}",
                  nonlinear_solver_name, trust_subspace_option, trust_num_leftmost, trust_num_previous_steps,
                  linear_max_iterations, preconditioner_name));

  advanceSingleThinBeamStep(solid);
  checkDisplacement("thin beam bending", solid);
}
}  // namespace
}  // namespace smith
