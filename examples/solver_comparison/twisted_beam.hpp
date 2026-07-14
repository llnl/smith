// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

void runTwistedBeam()
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "twisted_beam");

  constexpr int nx = 50;
  constexpr int ny = 5;
  constexpr int nz = 5;
  constexpr double length = 8.0;
  constexpr double width = 0.5;
  constexpr double height = 0.5;
  constexpr double twist_angle = 1.75 * M_PI;
  constexpr double axial_compression = 0.18;
  // Deliberate symmetry-breaking imperfection: a small lateral offset of the driven face
  // (2% of the beam width). The post-buckling branch is otherwise selected by roundoff,
  // which made the final answer scatter 8-13% across reduction-order changes.
  constexpr double transverse_imperfection = 1.0e-2;

  auto mesh =
      std::make_shared<Mesh>(mfem::Mesh(mfem::Mesh::MakeCartesian3D(scaled(nx), scaled(ny), scaled(nz),
                                                                    mfem::Element::HEXAHEDRON, length, width, height)),
                             "twisted_beam_mesh", 0, 0);
  checkElementCount("twisted beam", *mesh);

  mesh->addDomainOfBoundaryElements("fixed_face", by_attr<solid_dim>(5));
  mesh->addDomainOfBoundaryElements("twist_face", by_attr<solid_dim>(3));

  SolidMechanics<p, solid_dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                     "twisted_beam", mesh, {}, 0, 0.0, false,
                                     warmStartEnabled("13/twisted_beam", true));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");
  solid_mechanics::NeoHookean material{.density = 1.0, .K = 50.0, .G = 1.0};
  solid.setMaterial(material, mesh->entireBody());
  solid.setFixedBCs(mesh->domain("fixed_face"));
  solid.setDisplacementBCs(
      [=](tensor<double, solid_dim> X, double time) {
        const double theta = twist_angle * time;
        const double y0 = X[1] - 0.5 * width;
        const double z0 = X[2] - 0.5 * height;
        tensor<double, solid_dim> displacement{};
        displacement[0] = -axial_compression * time;
        displacement[1] = std::cos(theta) * y0 - std::sin(theta) * z0 - y0 + transverse_imperfection * time;
        displacement[2] = std::sin(theta) * y0 + std::cos(theta) * z0 - z0;
        return displacement;
      },
      mesh->domain("twist_face"));

  solid.completeSetup();
  advanceTimesteps(solid, "twisted_beam", 6);
  checkDisplacement("twisted beam", solid);
}
}  // namespace
}  // namespace smith
