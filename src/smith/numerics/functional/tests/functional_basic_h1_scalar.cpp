// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <string>

#include "gtest/gtest.h"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/tuple.hpp"

using namespace smith;

#ifdef SMITH_USE_CUDA_KERNEL_EVALUATION
constexpr auto exec_space = smith::ExecutionSpace::GPU;
#else
constexpr auto exec_space = smith::ExecutionSpace::CPU;
#endif

template <int dim>
struct TestThermalModelOne {
  template <typename P, typename Temp>
  SMITH_HOST_DEVICE auto operator()(double, [[maybe_unused]] P position, [[maybe_unused]] Temp temperature) const
  {
    double d00 = 1.0;
    constexpr static auto d01 = 1.0 * make_tensor<dim>([](int i) { return i; });
    constexpr static auto d10 = 1.0 * make_tensor<dim>([](int i) { return 2 * i * i; });
    constexpr static auto d11 = 1.0 * make_tensor<dim, dim>([](int i, int j) { return i + j * (j + 1) + 1; });
    auto [X, dX_dxi] = position;
    auto [u, du_dx] = temperature;
    auto source = d00 * u + dot(d01, du_dx) - 0.0 * (100 * X[0] * X[1]);
    auto flux = d10 * u + dot(d11, du_dx);

    return smith::tuple{source, flux};
  }
};

struct TestThermalModelTwo {
  template <typename PositionType, typename TempType>
  SMITH_HOST_DEVICE auto operator()(double, PositionType position, TempType temperature) const
  {
    auto [X, dX_dxi] = position;
    auto [u, du_dxi] = temperature;
    return X[0] + X[1] - cos(u);
  }
};

template <int ptest, int ptrial, int dim>
void thermal_test_impl(std::unique_ptr<mfem::ParMesh>& mesh)
{
  // Define the types for the test and trial spaces using the function arguments
  using test_space = H1<ptest>;
  using trial_space = H1<ptrial>;

  // Create standard MFEM bilinear and linear forms on H1
  auto [test_fespace, test_fec] = smith::generateParFiniteElementSpace<test_space>(mesh.get());
  auto [trial_fespace, trial_fec] = smith::generateParFiniteElementSpace<trial_space>(mesh.get());

  mfem::Vector U(trial_fespace->TrueVSize());

  mfem::ParGridFunction U_gf(trial_fespace.get());
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);
  U_gf.GetTrueDofs(U);

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space), exec_space> residual(test_fespace.get(), {trial_fespace.get()});

  Domain dom = EntireDomain(*mesh);
  Domain bdr = EntireBoundary(*mesh);

  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, TestThermalModelOne<dim>{}, dom);

  residual.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0>{}, TestThermalModelTwo{}, bdr);

  double t = 0.0;
  check_gradient(residual, t, U);
}

template <int ptest, int ptrial>
void thermal_test(std::string meshfile)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR + meshfile), 1);

  if (mesh->Dimension() == 2) {
    thermal_test_impl<ptest, ptrial, 2>(mesh);
  }

  if (mesh->Dimension() == 3) {
    thermal_test_impl<ptest, ptrial, 3>(mesh);
  }
}

TEST(basic, thermal_hexes) { thermal_test<1, 1>("/data/meshes/patch3D_hexes.mesh"); }
#ifndef SMITH_USE_CUDA_KERNEL_EVALUATION
TEST(basic, thermal_tris) { thermal_test<1, 1>("/data/meshes/patch2D_tris.mesh"); }
TEST(basic, thermal_tets) { thermal_test<1, 1>("/data/meshes/patch3D_tets.mesh"); }
TEST(basic, thermal_quads) { thermal_test<1, 1>("/data/meshes/patch2D_quads.mesh"); }
TEST(basic, thermal_tris_and_quads) { thermal_test<1, 1>("/data/meshes/patch2D_tris_and_quads.mesh"); }

TEST(basic, thermal_tets_and_hexes) { thermal_test<1, 1>("/data/meshes/patch3D_tets_and_hexes.mesh"); }
TEST(mixed, thermal_tris_and_quads) { thermal_test<2, 1>("/data/meshes/patch2D_tris_and_quads.mesh"); }
TEST(mixed, thermal_tets_and_hexes) { thermal_test<2, 1>("/data/meshes/patch3D_tets_and_hexes.mesh"); }
#endif

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
