// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

template <int P>
void runNearIncompressibleBlockCompressionT()
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "near_incompressible_block");

  constexpr int nx = 16;
  constexpr int ny = 5;
  constexpr int nz = 5;
  constexpr double length = 3.0;
  constexpr double width = 0.8;
  constexpr double height = 0.8;

  auto mesh =
      std::make_shared<Mesh>(mfem::Mesh(mfem::Mesh::MakeCartesian3D(scaled(nx), scaled(ny), scaled(nz),
                                                                    mfem::Element::HEXAHEDRON, length, width, height)),
                             "near_incompressible_block_mesh", 0, 0);
  checkElementCount("near-incompressible block", *mesh);

  mesh->addDomainOfBoundaryElements("fixed_face", by_attr<solid_dim>(5));
  mesh->addDomainOfBoundaryElements("driven_face", by_attr<solid_dim>(3));

  SolidMechanics<P, solid_dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                     "near_incompressible_block", mesh, {}, 0, 0.0, false,
                                     warmStartEnabled("11/near_incompressible_block", true));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");
  solid_mechanics::NeoHookean material{.density = 1.0, .K = 1000.0, .G = 1.0};
  solid.setMaterial(material, mesh->entireBody());
  solid.setFixedBCs(mesh->domain("fixed_face"));
  solid.setDisplacementBCs(
      [](tensor<double, solid_dim>, double time) {
        tensor<double, solid_dim> displacement{};
        displacement[0] = -0.35 * time;
        return displacement;
      },
      mesh->domain("driven_face"), Component::X);

  solid.completeSetup();
  advanceTimesteps(solid, "near_incompressible_block");
  checkDisplacement("near-incompressible block", solid);
}

void runNearIncompressibleBlockCompression()
{
  if (sim_order == 1) {
    runNearIncompressibleBlockCompressionT<1>();
  } else {
    runNearIncompressibleBlockCompressionT<2>();
  }
}
}  // namespace
}  // namespace smith
