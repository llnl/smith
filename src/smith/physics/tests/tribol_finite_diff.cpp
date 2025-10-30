// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/solid_mechanics_contact.hpp"

#include <functional>
#include <iomanip>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include "gtest/gtest.h"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"

#include "smith/smith_config.hpp"

#ifndef SMITH_USE_ENZYME
#error "This file requires Enzyme to be enabled
#endif

#include "smith/numerics/functional/domain.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace smith {

class TribolFiniteDiff : public testing::TestWithParam<std::pair<ContactEnforcement, std::string>> {};

TEST_P(TribolFiniteDiff, patch)
{
  constexpr double eps = 1.0e-7;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  std::string name = "tribol_fd_" + GetParam().second;
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store

  double shift = eps * 10;
  // clang-format off
  auto pmesh = std::make_shared<smith::Mesh>(shared::MeshBuilder::Unify({
    shared::MeshBuilder::CubeMesh(1, 1, 1),
    shared::MeshBuilder::CubeMesh(1, 1, 1)
      // shift up 99.9% height of element
      .translate({0.0, 0.0, 0.999})
      // shift x and y so the element edges are not overlapping
      .translate({shift, shift, 0.0})
      // change the mesh1 boundary attribute from 1 to 7
      .updateBdrAttrib(1, 7)
      // change the mesh1 boundary attribute from 6 to 8
      .updateBdrAttrib(6, 8)
  }), "patch_mesh", 0, 0);
  // clang-format on

  ContactOptions contact_options{.method = ContactMethod::SingleMortar,
                                 .enforcement = GetParam().first,
                                 .type = ContactType::TiedNormal,
                                 .penalty = 1.0,
                                 .jacobian = ContactJacobian::Exact};
  ContactData contact_data(pmesh->mfemParMesh());
  constexpr int interaction_id = 0;
  contact_data.addContactInteraction(interaction_id, {6}, {7}, contact_options);

  mfem::Vector u(pmesh->mfemParMesh().GetNodes()->Size() + contact_data.getContactInteractions()[0].numPressureDofs());
  u = 0.0;
  mfem::Vector u_shape(pmesh->mfemParMesh().GetNodes()->Size());
  u_shape = 0.0;
  mfem::Vector f(u.Size());
  f = 0.0;
  contact_data.residualFunction(u_shape, u, f);

  double max_diff = 0.0;
  auto J_op = contact_data.mergedJacobian();
  mfem::Vector u_dot(u.Size());
  u_dot = 0.0;
  // wiggle displacement (col = j)
  for (int j{0}; j < u.Size(); ++j) {
    u_dot[j] = 1.0;
    mfem::Vector J_exact(u.Size());
    J_exact = 0.0;
    J_op->Mult(u_dot, J_exact);
    u_dot[j] = 0.0;
    u[j] += eps;
    mfem::Vector J_fd(u.Size());
    J_fd = 0.0;
    contact_data.residualFunction(u_shape, u, J_fd);
    J_fd -= f;
    J_fd /= eps;
    u[j] -= eps;
    // loop through forces (row = k)
    for (int k{0}; k < u.Size(); ++k) {
      if (J_exact[k] != 1.0 && (std::abs(J_exact[k]) > 1.0e-15 || std::abs(J_fd[k]) > 1.0e-15)) {
        auto diff = std::abs(J_exact[k] - J_fd[k]);
        if (diff > max_diff) {
          max_diff = diff;
        }
        if (diff > eps) {
          std::cout << "(" << k << ", " << j << "):  J_exact = " << std::setprecision(15) << J_exact[k]
                    << "   J_fd = " << std::setprecision(15) << J_fd[k] << "   |diff| = " << std::setprecision(15)
                    << diff << std::endl;
        }
        EXPECT_NEAR(J_exact[k], J_fd[k], eps);
      }
    }
  }
  std::cout << "Max diff = " << std::setprecision(15) << max_diff << std::endl;
}

INSTANTIATE_TEST_SUITE_P(tribol, TribolFiniteDiff,
                         testing::Values(std::make_pair(ContactEnforcement::Penalty, "penalty"),
                                         std::make_pair(ContactEnforcement::LagrangeMultiplier, "lm")));

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
