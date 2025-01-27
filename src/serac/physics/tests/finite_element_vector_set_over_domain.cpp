// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/finite_element_dual.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

TEST(FiniteElementVector, SetScalarFieldOver2DDomain)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_parameterized_sensitivities");

  //  6--7--8
  //  |  |  |
  //  3--4--5
  //  |  |  |
  //  0--1--2
  auto mesh = buildRectangleMesh(2, 2, 1.0, 1.0);

  auto pmesh = mesh::refineAndDistribute(std::move(mesh), 0, 0);

  constexpr int p = 1;

  FiniteElementState u(*pmesh, H1<p, 1>{});

  Domain essential_boundary =
      Domain::ofBoundaryElements(*pmesh, [](std::vector<serac::vec2> x, int /*attr*/) { return average(x)[1] < 0.1; });

  mfem::FunctionCoefficient func([](const mfem::Vector& x, double) -> double { return x[0] + 1.0; });

  u = 0.0;
  u.project(func, essential_boundary);

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
  serac::StateManager::initialize(datastore, "thermal_functional_parameterized_sensitivities");

  //  6--7--8
  //  |  |  |
  //  3--4--5
  //  |  |  |
  //  0--1--2
  auto mesh = buildRectangleMesh(2, 2, 1.0, 1.0);

  auto pmesh = mesh::refineAndDistribute(std::move(mesh), 0, 0);

  FiniteElementState u(*pmesh, H1<p, dim>{});

  Domain essential_boundary =
      Domain::ofBoundaryElements(*pmesh, [](std::vector<serac::vec2> x, int /*attr*/) { return average(x)[1] < 0.1; });

  mfem::VectorFunctionCoefficient func(dim, [](const mfem::Vector& x, mfem::Vector& v) {
    v[0] = x[0] + 1.0;
    v[1] = x[0] + 2.0;
  });

  u = 0.0;
  u.project(func, essential_boundary);

  auto vdim = u.space().GetVDim();
  auto ndofs = u.space().GetTrueVSize() / vdim;
  auto dof = [ndofs, vdim](auto node, auto component) {
    return mfem::Ordering::Map<serac::ordering>(ndofs, vdim, node, component);
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

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
