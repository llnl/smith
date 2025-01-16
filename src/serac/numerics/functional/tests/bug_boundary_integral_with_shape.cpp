// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;
using namespace serac::profiling;

double t = 0.0;

int num_procs, myid;

void test(int dof)
{
  static constexpr int dim{2};

  using shape_space = H1<1, dim>;

  //  6--7--8
  //  |  |  |
  //  3--4--5
  //  |  |  |
  //  0--1--2
  auto undeformed_mesh = buildRectangleMesh(2, 2, 1.0, 1.0);

  auto deformed_mesh = buildRectangleMesh(2, 2, 1.0, 1.0);

  mfem::Vector vertex_coordinates;
  deformed_mesh.GetVertices(vertex_coordinates);
  vertex_coordinates[dof] += 0.25;  // nudge the top-middle vertex off-center
  deformed_mesh.SetVertices(vertex_coordinates);

  deformed_mesh.Save("deformed_" + std::to_string(dof) + ".mesh");

  auto undeformed_pmesh = mesh::refineAndDistribute(std::move(undeformed_mesh), 0, 0);
  auto deformed_pmesh = mesh::refineAndDistribute(std::move(deformed_mesh), 0, 0);

  auto [fes, fec] = generateParFiniteElementSpace<shape_space>(undeformed_pmesh.get());

  serac::ShapeAwareFunctional<shape_space, double()> saf_qoi(fes.get(), {});
  saf_qoi.AddBoundaryIntegral(
      serac::Dimension<1>{}, serac::DependsOn<>{}, [](auto...) { return 1.0; }, *undeformed_pmesh);

  serac::Functional<double(shape_space)> qoi({fes.get()});
  qoi.AddBoundaryIntegral(
      serac::Dimension<1>{}, serac::DependsOn<0>{}, [](auto...) { return 1.0; }, *deformed_pmesh);

  std::unique_ptr<mfem::HypreParVector> u(fes->NewTrueDofVector());
  *u = 0.0;
  (*u)[dof] = 0.25;
  std::cout << "(ShapeAwareFunctional) perimeter of undeformed mesh + shape: " << saf_qoi(t, *u) << std::endl;
  std::cout << "(          Functional) perimeter of deformed mesh: " << qoi(t, *u) << std::endl;
}

TEST(QoI, BoundaryIntegralWithTangentialShapeDisplacements)
{
  //  6--7--8       6---7-8
  //  |  |  |       |  /  |
  //  3--4--5       3--4--5
  //  |  |  |       |  |  |
  //  0--1--2       0--1--2
  test(7);
}

TEST(QoI, BoundaryIntegralWithNormalShapeDisplacements)
{
  //  6--7--8       6--7--8
  //  |  |  |       |  |   \ 
  //  3--4--5       3--4----5
  //  |  |  |       |  |   /
  //  0--1--2       0--1--2
  test(5);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
