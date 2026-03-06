// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "shared/mesh/MeshBuilder.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/contact_constraint.hpp"

struct ContactConstraintFixture : public testing::Test {
  static constexpr int dim = 3;
  static constexpr int disp_order = 1;

  static constexpr double interpen = 0.1;

  using VectorSpace = smith::H1<disp_order, dim>;

  enum STATE
  {
    SHAPE,
    DISP
  };

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    smith::StateManager::initialize(datastore, "contact_constraint");

    mesh = std::make_shared<smith::Mesh>(shared::MeshBuilder::Unify({shared::MeshBuilder::CubeMesh(1, 1, 1)
                                                                         .updateBdrAttrib(3, 7)
                                                                         .updateBdrAttrib(1, 3)
                                                                         .updateBdrAttrib(4, 7)
                                                                         .updateBdrAttrib(5, 1)
                                                                         .updateBdrAttrib(6, 4),
                                                                     shared::MeshBuilder::CubeMesh(1, 1, 1)
                                                                         .translate({0.0, 0.0, 1.0 - interpen})
                                                                         .updateBdrAttrib(1, 8)
                                                                         .updateBdrAttrib(3, 7)
                                                                         .updateBdrAttrib(4, 7)
                                                                         .updateBdrAttrib(5, 1)
                                                                         .updateBdrAttrib(8, 5)}),
                                         "patch_mesh", 0, 0);

    constexpr int interaction_id = 0;
    smith::ContactOptions contact_opts{.method = smith::ContactMethod::SingleMortar,
                                       .enforcement = smith::ContactEnforcement::LagrangeMultiplier,
                                       .type = smith::ContactType::Frictionless,
                                       .jacobian = smith::ContactJacobian::Approximate};
    std::set<int> bdry_attr_surf1({4});
    std::set<int> bdry_attr_surf2({5});
    constraint = std::make_unique<smith::ContactConstraint>(interaction_id, mesh->mfemParMesh(), bdry_attr_surf1,
                                                            bdry_attr_surf2, contact_opts);

    smith::FiniteElementState shape_disp =
        smith::StateManager::newState(VectorSpace{}, "shape_displacement", mesh->tag());
    smith::FiniteElementState disp = smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    states = {shape_disp, disp};

    states[SHAPE] = 0.0;
    states[DISP] = 0.0;
  }

  const double time = 0.0;
  const double dt = 1.0;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::unique_ptr<smith::Constraint> constraint;

  std::vector<smith::FiniteElementState> states;
};

TEST_F(ContactConstraintFixture, CheckGaps)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto input_fields = getConstFieldPointers(states);

  auto gaps_tribol = constraint->evaluate(time, dt, input_fields);

  // gap should be -(tributary area * interpen).  for the unit cube mesh, tributary area should be 0.25.
  // only check nonmortar dofs
  constexpr int num_mortar_dofs = 4;
  for (int i{num_mortar_dofs}; i < gaps_tribol.Size(); ++i) {
    EXPECT_NEAR(gaps_tribol[i], -0.25 * interpen, 1.0e-12);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
