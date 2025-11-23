// Copyright Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "axom/slic/core/SimpleLogger.hpp"
#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/shape_aware_functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/infrastructure/application_manager.hpp"

#include "smith/numerics/functional/tests/check_gradient.hpp"

using namespace smith;

double t = 0.0;

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

  smith::ShapeAwareFunctional<shape_space, double()> saf_qoi(fes.get(), {});
  saf_qoi.AddBoundaryIntegral(
      smith::Dimension<1>{}, smith::DependsOn<>{}, [](auto...) { return 1.0; }, *undeformed_pmesh);

  smith::Functional<double(shape_space)> qoi({fes.get()});
  qoi.AddBoundaryIntegral(smith::Dimension<1>{}, smith::DependsOn<0>{}, [](auto...) { return 1.0; }, *deformed_pmesh);

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
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
