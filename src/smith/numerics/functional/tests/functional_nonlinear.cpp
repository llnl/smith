// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/profiling.hpp"
#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"
#include "smith/infrastructure/accelerator.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/geometry.hpp"
#include "smith/numerics/functional/tuple.hpp"

using namespace smith;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

static constexpr double a = 1.7;
static constexpr double b = 2.1;
template <int dim>
struct hcurl_qfunction {
  template <typename x_t, typename vector_potential_t>
  SMITH_HOST_DEVICE auto operator()(x_t x, vector_potential_t vector_potential) const
  {
    auto [A, curl_A] = vector_potential;
    auto J_term = a * A - tensor<double, dim>{10 * x[0] * x[1], -5 * (x[0] - x[1]) * x[1]};
    auto H_term = b * curl_A;
    return smith::tuple{J_term, H_term};
  }
};

// this test sets up a toy "thermal" problem where the residual includes contributions
// from a temperature-dependent source term and a temperature-gradient-dependent flux
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, H1<p> test, H1<p> trial, Dimension<dim>)
{
  std::string postfix = profiling::concat("_H1<", p, ">");
  smith::profiling::initialize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space = decltype(test);
  using trial_space = decltype(trial);

  // Create standard MFEM bilinear and linear forms on H1
  auto [fespace, fec] = smith::generateParFiniteElementSpace<test_space>(&mesh);

  // Set a random state to evaluate the residual
  mfem::Vector U(fespace->TrueVSize());
  int seed = 3;
  U.Randomize(seed);

  // Define the types for the test and trial spaces using the function arguments
  using test_space = decltype(test);
  using trial_space = decltype(trial);

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(fespace.get(), {fespace.get()});

  Domain dom = EntireDomain(mesh);
  Domain bdr = EntireBoundary(mesh);

  // Add the total domain residual term to the functional
  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dX] = temperature;
        auto source = a * u * u - (100 * X[0] * X[1]);
        auto flux = b * du_dX;
        return smith::tuple{source, flux};
      },
      dom);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto u = get<0>(temperature);
        return X[0] + X[1] - cos(u);
      },
      bdr);

  double t = 0.0;
  check_gradient(residual, t, U);

  smith::profiling::finalize();
}

template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, H1<p, dim> test, H1<p, dim> trial, Dimension<dim>)
{
  std::string postfix = profiling::concat("_H1<", p, ",", dim, ">");
  smith::profiling::initialize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space = decltype(test);
  using trial_space = decltype(trial);

  // Create standard MFEM bilinear and linear forms on H1
  auto [fespace, fec] = smith::generateParFiniteElementSpace<test_space>(&mesh);

  // Set a random state to evaluate the residual
  mfem::Vector U(fespace->TrueVSize());
  int seed = 4;
  U.Randomize(seed);

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(fespace.get(), {fespace.get()});

  Domain dom = EntireDomain(mesh);
  Domain bdr = EntireBoundary(mesh);

  // Add the total domain residual term to the functional
  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto /*x*/, auto displacement) {
        // get the value and the gradient from the input tuple
        auto [u, du_dx] = displacement;
        auto source = a * u * u[0];
        auto flux = b * du_dx;
        return smith::tuple{source, flux};
      },
      dom);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto displacement) {
        auto [X, dX_dxi] = position;
        std::cout << X << " " << dX_dxi << std::endl;
        auto u = get<0>(displacement);
        auto n = normalize(cross(dX_dxi));
        return (X[0] + X[1] - cos(u[0])) * n;
      },
      bdr);

  double t = 0.0;
  check_gradient(residual, t, U);

  smith::profiling::finalize();
}

TEST(Thermal, 2DLinear) { functional_test(*mesh2D, H1<1>{}, H1<1>{}, Dimension<2>{}); }
TEST(Thermal, 2DQuadratic) { functional_test(*mesh2D, H1<2>{}, H1<2>{}, Dimension<2>{}); }
TEST(Thermal, 2DCubic) { functional_test(*mesh2D, H1<3>{}, H1<3>{}, Dimension<2>{}); }

TEST(Thermal, 3DLinear) { functional_test(*mesh3D, H1<1>{}, H1<1>{}, Dimension<3>{}); }
TEST(Thermal, 3DQuadratic) { functional_test(*mesh3D, H1<2>{}, H1<2>{}, Dimension<3>{}); }
TEST(Thermal, 3DCubic) { functional_test(*mesh3D, H1<3>{}, H1<3>{}, Dimension<3>{}); }

TEST(Elasticity, 2DLinear) { functional_test(*mesh2D, H1<1, 2>{}, H1<1, 2>{}, Dimension<2>{}); }
TEST(Elasticity, 2DQuadratic) { functional_test(*mesh2D, H1<2, 2>{}, H1<2, 2>{}, Dimension<2>{}); }
TEST(Elasticity, 2DCubic) { functional_test(*mesh2D, H1<3, 2>{}, H1<3, 2>{}, Dimension<2>{}); }

TEST(Elasticity, 3DLinear) { functional_test(*mesh3D, H1<1, 3>{}, H1<1, 3>{}, Dimension<3>{}); }
TEST(Elasticity, 3DQuadratic) { functional_test(*mesh3D, H1<2, 3>{}, H1<2, 3>{}, Dimension<3>{}); }
TEST(Elasticity, 3DCubic) { functional_test(*mesh3D, H1<3, 3>{}, H1<3, 3>{}, Dimension<3>{}); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);

  int serial_refinement = 1;
  int parallel_refinement = 0;

  std::string meshfile2D = SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  std::string meshfile3D = SMITH_REPO_DIR "/data/meshes/patch3D_tets_and_hexes.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);
  return RUN_ALL_TESTS();
}
