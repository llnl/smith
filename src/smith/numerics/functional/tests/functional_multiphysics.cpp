// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <set>
#include <string>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/tuple.hpp"

using namespace smith;

TEST(FunctionalMultiphysics, NonlinearThermalTest3D)
{
  int serial_refinement = 1;
  int parallel_refinement = 0;

  constexpr auto p = 3;

  std::string meshfile = SMITH_REPO_DIR "/data/meshes/patch3D_tets_and_hexes.mesh";
  auto mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile), serial_refinement, parallel_refinement);

  // Define the types for the test and trial spaces using the function arguments
  using test_space = H1<p>;
  using trial_space = H1<p>;

  // Create standard MFEM bilinear and linear forms on H1
  auto [fespace, fec] = smith::generateParFiniteElementSpace<test_space>(mesh3D.get());

  mfem::Vector U(fespace->TrueVSize());
  mfem::Vector dU_dt(fespace->TrueVSize());
  int seed = 1;
  U.Randomize(seed);
  dU_dt.Randomize(seed + 1);

  double cp = 1.0;
  double rho = 1.0;
  double kappa = 1.0;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space, trial_space)> residual(fespace.get(), {fespace.get(), fespace.get()});

  Domain dom = EntireDomain(*mesh3D);
  Domain bdr = EntireBoundary(*mesh3D);

  residual.AddVolumeIntegral(
      DependsOn<0, 1>{},
      [=](double /*t*/, auto position, auto temperature, auto dtemperature_dt) {
        auto [X, dX_dxi] = position;
        auto [u, du_dX] = temperature;
        auto [du_dt, unused] = dtemperature_dt;
        auto source = rho * cp * du_dt * du_dt - (100 * X[0] * X[1]);
        auto flux = kappa * du_dX;
        return smith::tuple{source, flux};
      },
      dom);

  residual.AddSurfaceIntegral(
      DependsOn<0, 1>{},
      [=](double /*t*/, auto position, auto temperature, auto dtemperature_dt) {
        auto [X, dX_dxi] = position;
        auto [u, _0] = temperature;
        auto [du_dt, _1] = dtemperature_dt;
        return X[0] + X[1] - cos(u) * du_dt;
      },
      bdr);

  double t = 0.0;
  mfem::Vector r = residual(t, U, dU_dt);

  check_gradient(residual, t, U, dU_dt);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
