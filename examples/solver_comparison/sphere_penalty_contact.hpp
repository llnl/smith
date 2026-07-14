// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {
namespace {

void runSpherePenaltyContact()
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "sphere_penalty_contact");

  constexpr int nx = 26;
  constexpr int ny = 26;
  constexpr int nz = 8;
  constexpr double length = 1.0;
  constexpr double width = 1.0;
  constexpr double height = 0.4;
  constexpr double sphere_radius = 0.22;
  constexpr double contact_penalty = 1.0e5;
  const vec3 sphere_center{{0.5 * length, 0.5 * width, -0.26}};

  auto mesh =
      std::make_shared<Mesh>(mfem::Mesh(mfem::Mesh::MakeCartesian3D(scaled(nx), scaled(ny), scaled(nz),
                                                                    mfem::Element::HEXAHEDRON, length, width, height)),
                             "sphere_penalty_contact_mesh", 0, 0);
  checkElementCount("sphere penalty contact", *mesh);

  mesh->addDomainOfBoundaryElements("driven_face", by_attr<solid_dim>(6));
  mesh->addDomainOfBoundaryElements("contact_face", by_attr<solid_dim>(1));

  SolidMechanics<p, solid_dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                                     "sphere_penalty_contact", mesh, {}, 0, 0.0, false,
                                     warmStartEnabled("12/sphere_penalty_contact", true));
  if (assemble_bsr) command_line_warnings.push_back("B1 does not support --assemble-bsr; ignoring");
  solid_mechanics::NeoHookean material{.density = 1.0, .K = 100.0, .G = 1.0};
  solid.setMaterial(material, mesh->entireBody());
  solid.setDisplacementBCs(
      [](tensor<double, solid_dim>, double time) {
        tensor<double, solid_dim> displacement{};
        displacement[2] = -0.06 * time;
        return displacement;
      },
      mesh->domain("driven_face"));

  solid.addCustomBoundaryIntegral(
      DependsOn<>{},
      [=](double, auto X, auto displacement, auto) {
        auto x = get<VALUE>(X) + get<VALUE>(displacement);
        auto offset = x - sphere_center;
        auto distance = norm(offset);
        auto phi = distance - sphere_radius;
        auto contact_residual = 0.0 * x;
        if (phi < 0.0) {
          contact_residual = contact_penalty * phi * offset / distance;
        }
        return contact_residual;
      },
      mesh->domain("contact_face"));

  solid.completeSetup();
  advanceTimesteps(solid, "sphere_penalty_contact");
  checkDisplacement("sphere penalty contact", solid);

  Functional<double(H1<p, solid_dim>)> contact_energy({&solid.displacement().space()});
  contact_energy.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](double, auto X, auto displacement) {
        auto x = get<VALUE>(X) + get<VALUE>(displacement);
        auto phi = norm(x - sphere_center) - sphere_radius;
        auto energy = 0.0 * phi;
        if (phi < 0.0) {
          energy = 0.5 * contact_penalty * phi * phi;
        }
        return energy;
      },
      mesh->domain("contact_face"));

  Functional<double(H1<p, solid_dim>)> active_area({&solid.displacement().space()});
  active_area.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](double, auto X, auto displacement) {
        auto x = get<VALUE>(X) + get<VALUE>(displacement);
        auto phi = norm(x - sphere_center) - sphere_radius;
        auto active = 0.0 * phi;
        if (phi < 0.0) {
          active = 1.0 + 0.0 * phi;
        }
        return active;
      },
      mesh->domain("contact_face"));

  const double energy = contact_energy(solid.time(), solid.displacement());
  const double area = active_area(solid.time(), solid.displacement());
  SLIC_INFO_ROOT(std::format("sphere penalty contact: active area = {:.8e}, contact energy = {:.8e}", area, energy));
  SLIC_ERROR_ROOT_IF(area <= 0.0, "sphere penalty contact produced no active contact area");
  SLIC_ERROR_ROOT_IF(!std::isfinite(energy), "sphere penalty contact produced non-finite contact energy");
}
}  // namespace
}  // namespace smith
