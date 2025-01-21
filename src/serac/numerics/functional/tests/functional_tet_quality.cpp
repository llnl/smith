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

TEST(QoI, TetrahedronQuality)
{
  static constexpr int dim{3};

  double displacement_to_regular_tetrahedron[4][3] = {
      {0., 0., 0.}, {0.122462, 0., 0.}, {0.561231, -0.0279194, 0.}, {0.561231, 0.324027, -0.0835136}};

  tensor<double, 3, 3> regular_tet_correction = {
      {{1.00000, -0.577350, -0.408248}, {0, 1.15470, -0.408248}, {0, 0, 1.22474}}};

  auto mu = [](auto J) {
    using std::pow;
    return tr(dot(J, J)) / (3 * pow(serac::det(J), 2. / 3.)) - 1.0;
  };

  using shape_space = H1<1, dim>;

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/onetet.mesh";
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), 0, 0);

  auto [fes, fec] = generateParFiniteElementSpace<shape_space>(mesh.get());

  Domain whole_domain = EntireDomain(*mesh);

  // Define the shape-aware QOI objects
  serac::ShapeAwareFunctional<shape_space, double()> saf_qoi(fes.get(), {});

  // Note that the integral does not have a shape parameter field. The transformations are handled under the hood
  // so the user only sees the modified x = X + p input arguments
  saf_qoi.AddDomainIntegral(
      serac::Dimension<3>{}, serac::DependsOn<>{},
      [=](double /*t*/, auto position) {
        auto [x, dx_dxi] = position;
        return mu(dot(regular_tet_correction, dx_dxi));
      },
      whole_domain);

  serac::Functional<double(shape_space)> qoi({fes.get()});

  qoi.AddDomainIntegral(
      serac::Dimension<3>{}, serac::DependsOn<0>{},
      [=](double /*t*/, auto position, auto displacement) {
        auto [X, dX_dxi] = position;  // <--- the position derivative term is w.r.t. xi, not X!
        auto [u, du_dX] = displacement;

        // x := X + u,
        // so, dx/dxi = dX/dxi + du/dxi
        //            = dX/dxi + du/dX * dX/dxi
        //            = (I + du/dX) * dX/dxi
        // auto dx_dxi = dot(I + du_dX, dX_dxi);
        auto dx_dxi = dX_dxi + dot(du_dX, dX_dxi);
        return mu(dot(regular_tet_correction, dx_dxi));
      },
      whole_domain);

  std::unique_ptr<mfem::HypreParVector> u(fes->NewTrueDofVector());
  *u = 0.0;
  std::cout << "(ShapeAwareFunctional) mu(J) for right tetrahedron: " << saf_qoi(t, *u) << std::endl;
  std::cout << "(          Functional) mu(J) for right tetrahedron: " << qoi(t, *u) << std::endl;

  // apply a displacement to make the domain into a regular tetrahedron
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      (*u)[i + j * 4] = displacement_to_regular_tetrahedron[i][j];
    }
  }

  std::cout << "(ShapeAwareFunctional) mu(J) for regular tetrahedron: " << saf_qoi(t, *u) << std::endl;
  std::cout << "(          Functional) mu(J) for regular tetrahedron: " << qoi(t, *u) << std::endl;
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
