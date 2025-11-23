// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"
#include "smith/smith_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"

namespace smith {

TEST(FiniteElementVector, SetScalarFieldOver2DDomain)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermal_functional_parameterized_sensitivities");

  //  6--7--8
  //  |  |  |
  //  3--4--5
  //  |  |  |
  //  0--1--2
  auto pmesh = std::make_shared<smith::Mesh>(buildRectangleMesh(2, 2, 1.0, 1.0), "mesh", 0, 0);

  constexpr int p = 1;

  FiniteElementState u(pmesh->mfemParMesh(), H1<p, 1>{});

  pmesh->addDomainOfBoundaryElements("essential_boundary",
                                     [](std::vector<smith::vec2> x, int /*attr*/) { return average(x)[1] < 0.1; });

  mfem::FunctionCoefficient func([](const mfem::Vector& x, double) -> double { return x[0] + 1.0; });

  u = 0.0;
  u.project(func, pmesh->domain("essential_boundary"));

  EXPECT_NEAR(u[0], 1.0, 1.0e-15);
  EXPECT_NEAR(u[1], 1.5, 1.0e-15);
  EXPECT_NEAR(u[2], 2.0, 1.0e-15);
  for (int i = 3; i < 9; i++) {
    EXPECT_NEAR(u[i], 0.0, 1.0e-15);
  }
}

TEST(FiniteElementVector, SetVectorFieldOver2DDomain)
{
  constexpr int p = 1;
  constexpr int dim = 2;

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermal_functional_parameterized_sensitivities");

  //  6--7--8
  //  |  |  |
  //  3--4--5
  //  |  |  |
  //  0--1--2
  auto pmesh = std::make_shared<smith::Mesh>(buildRectangleMesh(2, 2, 1.0, 1.0), "mesh", 0, 0);

  FiniteElementState u(pmesh->mfemParMesh(), H1<p, dim>{});

  pmesh->addDomainOfBoundaryElements("essential_boundary",
                                     [](std::vector<smith::vec2> x, int /*attr*/) { return average(x)[1] < 0.1; });

  mfem::VectorFunctionCoefficient func(dim, [](const mfem::Vector& x, mfem::Vector& v) {
    v[0] = x[0] + 1.0;
    v[1] = x[0] + 2.0;
  });

  u = 0.0;
  u.project(func, pmesh->domain("essential_boundary"));

  auto vdim = u.space().GetVDim();
  auto ndofs = u.space().GetTrueVSize() / vdim;
  auto dof = [ndofs, vdim](auto node, auto component) {
    return mfem::Ordering::Map<smith::ordering>(ndofs, vdim, node, component);
  };

  EXPECT_NEAR(u[dof(0, 0)], 1.0, 1.0e-15);
  EXPECT_NEAR(u[dof(1, 0)], 1.5, 1.0e-15);
  EXPECT_NEAR(u[dof(2, 0)], 2.0, 1.0e-15);
  for (int i = 3; i < 9; i++) {
    EXPECT_NEAR(u[dof(i, 0)], 0.0, 1.0e-15);
  }

  EXPECT_NEAR(u[dof(0, 1)], 2.0, 1.0e-15);
  EXPECT_NEAR(u[dof(1, 1)], 2.5, 1.0e-15);
  EXPECT_NEAR(u[dof(2, 1)], 3.0, 1.0e-15);
  for (int i = 3; i < 9; i++) {
    EXPECT_NEAR(u[dof(i, 1)], 0.0, 1.0e-15);
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
