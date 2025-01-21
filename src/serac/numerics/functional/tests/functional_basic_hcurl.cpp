// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;
using namespace serac::profiling;

template <int p>
void hcurl_test_2D()
{
  constexpr int dim = 2;
  using test_space = Hcurl<p>;
  using trial_space = Hcurl<p>;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/patch2D.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  auto fec = mfem::ND_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Construct the new functional object using the specified test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  auto d00 = make_tensor<dim, dim>([](int i, int j) { return i + j * j - 1; });
  auto d01 = make_tensor<dim>([](int i) { return i * i + 3; });
  auto d10 = make_tensor<dim>([](int i) { return 3 * i - 2; });
  auto d11 = 1.0;

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto /*x*/, auto vector_potential) {
        auto [A, curl_A] = vector_potential;
        auto source = dot(d00, A) + d01 * curl_A;
        auto flux = dot(d10, A) + d11 * curl_A;
        return serac::tuple{source, flux};
      },
      *mesh);

  check_gradient(residual, t, U);
}

template <int p>
void hcurl_test_3D()
{
  constexpr int dim = 3;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/patch3D.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  // Create standard MFEM bilinear and linear forms on H1
  auto fec = mfem::ND_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space = Hcurl<p>;
  using trial_space = Hcurl<p>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  auto d00 = make_tensor<dim, dim>([](int i, int j) { return i + j * j - 1; });
  auto d01 = make_tensor<dim, dim>([](int i, int j) { return i * i - j + 3; });
  auto d10 = make_tensor<dim, dim>([](int i, int j) { return 3 * i + j - 2; });
  auto d11 = make_tensor<dim, dim>([](int i, int j) { return i * i + j + 2; });

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto /*x*/, auto vector_potential) {
        auto [A, curl_A] = vector_potential;
        auto source = dot(d00, A) + dot(d01, curl_A);
        auto flux = dot(d10, A) + dot(d11, curl_A);
        return serac::tuple{source, flux};
      },
      *mesh);

  check_gradient(residual, t, U);
}

TEST(basic, hcurl_test_2D_linear) { hcurl_test_2D<1>(); }

TEST(basic, hcurl_test_3D_linear) { hcurl_test_3D<1>(); }

int main(int argc, char* argv[])
{
  int num_procs, myid;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
