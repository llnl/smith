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

template <typename T, int m, int n>
struct mat {
  T data[m][n];

  mat(const T (&new_data)[m][n])
  {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        data[i][j] = new_data[i][j];
      }
    }
  }
};

template <typename T1, typename T2>
auto axiSymmetricDisplacementGradient(const T1& r, const T2& displacement)
{
  auto [u, du_dx] = displacement;
  using scalar_t = decltype(u[0] / r + du_dx[0][0]);
  scalar_t z{};
  return serac::tensor<scalar_t, 3, 3>{
      {{du_dx[0][0], du_dx[0][1], z}, {du_dx[1][0], du_dx[1][1], z}, {z, z, u[0] / r}}};
}

TEST(QoI, BoundaryIntegralWithTangentialShapeDisplacements)
{
  static constexpr int dim{2};

  using space = H1<1, dim>;

  //  6--7--8
  //  |  |  |
  //  3--4--5
  //  |  |  |
  //  0--1--2
  auto pmesh = mesh::refineAndDistribute(buildRectangleMesh(2, 2, 1.0, 1.0));

  auto [fes, fec] = generateParFiniteElementSpace<space>(pmesh.get());

  serac::Functional<space(space)> residual(fes.get(), {fes.get()});
  residual.AddDomainIntegral(
      serac::Dimension<2>{}, serac::DependsOn<0>{},
      [&](auto /*t*/, auto position, auto displacement) {
        auto r = serac::get<0>(position)[0];
        auto du_dx_prime = axiSymmetricDisplacementGradient(r, displacement);
        auto stress = du_dx_prime * 3;

        using source_type = decltype(stress[2][2] * r);
        serac::tensor<source_type, 2> source = {stress[2][2] * 6.28, 0.0};
        serac::tensor<source_type, 2, 2> flux = {{{stress[0][0], stress[0][1]}, {stress[1][0], stress[1][1]}}};

        return serac::tuple{source, flux * 6.28 * r};
      },
      *pmesh);
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
