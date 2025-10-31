// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "gtest/gtest.h"

#include "axom/slic/core/SimpleLogger.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"

using namespace smith;

template <int p>
void hcurl_test_2D()
{
  constexpr int dim = 2;
  using test_space = Hcurl<p>;
  using trial_space = Hcurl<p>;

  std::string meshfile = SMITH_REPO_DIR "/data/meshes/patch2D.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  auto fec = mfem::ND_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  int seed = 7;
  U.Randomize(seed);

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
        return smith::tuple{source, flux};
      },
      *mesh);

  check_gradient(residual, t, U);
}

template <int p>
void hcurl_test_3D()
{
  constexpr int dim = 3;

  std::string meshfile = SMITH_REPO_DIR "/data/meshes/patch3D.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  // Create standard MFEM bilinear and linear forms on H1
  auto fec = mfem::ND_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  int seed = 8;
  U.Randomize(seed);

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
        return smith::tuple{source, flux};
      },
      *mesh);

  check_gradient(residual, t, U);
}

TEST(basic, hcurl_test_2D_linear) { hcurl_test_2D<1>(); }

TEST(basic, hcurl_test_3D_linear) { hcurl_test_3D<1>(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
