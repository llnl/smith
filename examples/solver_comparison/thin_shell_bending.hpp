// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

// Bending-dominated thin-shell-like cantilever: a thin rectangular plate (thin in z)
// clamped along x=0 and driven by a large transverse (z) tip deflection at x=L. The thin
// section makes the response bending- rather than stretch-dominated, and the large tip
// rotation gives geometric nonlinearity without the snap-through/branch sensitivity of the
// shallow-arch/twist cases -- a smoother, fast (~20 s) generalization probe for the solver.
void runThinShellBending()
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "thin_shell_bending");

  constexpr int nx = 100;
  constexpr int ny = 20;
  constexpr int nz = 3;
  constexpr double length = 10.0;
  constexpr double width = 2.0;
  constexpr double thickness = 0.25;
  constexpr double tip_deflection = 2.0;

  auto serial_mesh = mfem::Mesh::MakeCartesian3D(scaled(nx), scaled(ny), scaled(nz), mfem::Element::HEXAHEDRON, length,
                                                 width, thickness);
  auto mesh = std::make_shared<Mesh>(distributeMeshContiguously(serial_mesh), "thin_shell_bending_mesh");
  checkElementCount("thin shell bending", *mesh);

  mesh->addDomainOfBoundaryElements("clamped_face", by_attr<solid_dim>(5));
  mesh->addDomainOfBoundaryElements("tip_face", by_attr<solid_dim>(3));

  SolidMechanics<p, solid_dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                     "thin_shell_bending", mesh, {}, 0, 0.0, false,
                                     warmStartEnabled("14/thin_shell_bending", true));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");
  solid_mechanics::NeoHookean material{.density = 1.0, .K = 100.0, .G = 1.0};
  solid.setMaterial(material, mesh->entireBody());
  solid.setFixedBCs(mesh->domain("clamped_face"));
  solid.setDisplacementBCs(
      [=](tensor<double, solid_dim>, double time) {
        tensor<double, solid_dim> displacement{};
        displacement[2] = -tip_deflection * time;
        return displacement;
      },
      mesh->domain("tip_face"), Component::Z);

  solid.completeSetup();
  advanceTimesteps(solid, "thin_shell_bending");
  checkDisplacement("thin shell bending", solid);
}
}  // namespace
}  // namespace smith
