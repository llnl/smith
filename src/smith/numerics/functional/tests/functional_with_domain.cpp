// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "gtest/gtest.h"

#include "smith/infrastructure/application_manager.hpp"
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

template <int dim, bool apply_mask = false>
struct TestThermalIntegratorOne {
  template <typename Position, typename Temperature>
  SMITH_HOST_DEVICE auto operator()(double, Position position, Temperature temperature) const
  {
    constexpr static double d00 = 1.0;
    constexpr static auto d01 = 1.0 * make_tensor<dim>([](int i) { return i; });
    constexpr static auto d10 = 1.0 * make_tensor<dim>([](int i) { return 2 * i * i; });
    constexpr static auto d11 = 1.0 * make_tensor<dim, dim>([](int i, int j) { return i + j * (j + 1) + 1; });

    auto [X, dX_dxi] = position;
    auto [u, du_dx] = temperature;

    double mask = 1.0;
    if constexpr (apply_mask) {
      mask = (X[0] < 4.0);
    }

    auto source = mask * (d00 * u + dot(d01, du_dx) - 0.0 * (100 * X[0] * X[1]));
    auto flux = mask * (d10 * u + dot(d11, du_dx));
    return smith::tuple{source, flux};
  }
};

template <bool apply_mask = false>
struct TestThermalIntegratorTwo {
  template <typename Position, typename Temperature>
  SMITH_HOST_DEVICE auto operator()(double, Position position, Temperature temperature) const
  {
    auto [X, dX_dxi] = position;
    auto [u, du_dxi] = temperature;

    double mask = 1.0;
    if constexpr (apply_mask) {
      mask = (X[1] >= 0.99);
    }

    return mask * (X[0] + X[1] - cos(u));
  }
};

struct TrivialIntegrator {
  template <typename UnusedType>
  SMITH_HOST_DEVICE auto operator()(double /*t*/, UnusedType /*position*/) const
  {
    return 1.0;
  }
};

template <int ptest, int ptrial, int dim>
void whole_mesh_comparison_test_impl(std::unique_ptr<mfem::ParMesh>& mesh)
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
  Functional<test_space(trial_space)> residual(test_fespace.get(), {trial_fespace.get()});
  Functional<test_space(trial_space)> residual_comparison(test_fespace.get(), {trial_fespace.get()});

  auto everything = [](std::vector<tensor<double, dim>>, int /* attr */) { return true; };
  auto on_left = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] < 0.5; };
  auto on_right = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] >= 0.5; };
  auto on_bottom = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[1] < 0.5; };
  auto on_top = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[1] >= 0.5; };

  Domain whole_mesh = Domain::ofElements(*mesh, everything);
  Domain left = Domain::ofElements(*mesh, on_left);
  Domain right = Domain::ofElements(*mesh, on_right);

  Domain whole_boundary = Domain::ofBoundaryElements(*mesh, everything);
  Domain bottom_boundary = Domain::ofBoundaryElements(*mesh, on_bottom);
  Domain top_boundary = Domain::ofBoundaryElements(*mesh, on_top);

  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, TestThermalIntegratorOne<dim>{}, left);

  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, TestThermalIntegratorOne<dim>{}, right);

  residual.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0>{}, TestThermalIntegratorTwo{}, bottom_boundary);

  residual.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0>{}, TestThermalIntegratorTwo{}, top_boundary);

  double t = 0.0;
  check_gradient(residual, t, U);

  auto r0 = residual(t, U);

  //////////////

  residual_comparison.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, TestThermalIntegratorOne<dim>{}, whole_mesh);

  residual_comparison.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0>{}, TestThermalIntegratorTwo{},
                                          whole_boundary);

  auto r1 = residual_comparison(t, U);

  EXPECT_LT(r0.DistanceTo(r1.GetData()), 1.0e-14);
}

template <int ptest, int ptrial>
void whole_mesh_comparison_test(std::string meshfile)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR + meshfile), 1);

  if (mesh->Dimension() == 2) {
    whole_mesh_comparison_test_impl<ptest, ptrial, 2>(mesh);
  }

  if (mesh->Dimension() == 3) {
    whole_mesh_comparison_test_impl<ptest, ptrial, 3>(mesh);
  }
}

TEST(basic, whole_mesh_comparison_tris) { whole_mesh_comparison_test<1, 1>("/data/meshes/patch2D_tris.mesh"); }
TEST(basic, whole_mesh_comparison_quads) { whole_mesh_comparison_test<1, 1>("/data/meshes/patch2D_quads.mesh"); }
TEST(basic, whole_mesh_comparison_tris_and_quads)
{
  whole_mesh_comparison_test<1, 1>("/data/meshes/patch2D_tris_and_quads.mesh");
}

TEST(basic, whole_mesh_comparison_tets) { whole_mesh_comparison_test<1, 1>("/data/meshes/patch3D_tets.mesh"); }
TEST(basic, whole_mesh_comparison_hexes) { whole_mesh_comparison_test<1, 1>("/data/meshes/patch3D_hexes.mesh"); }
TEST(basic, whole_mesh_comparison_tets_and_hexes)
{
  whole_mesh_comparison_test<1, 1>("/data/meshes/patch3D_tets_and_hexes.mesh");
}

TEST(mixed, whole_mesh_comparison_tris_and_quads)
{
  whole_mesh_comparison_test<2, 1>("/data/meshes/patch2D_tris_and_quads.mesh");
}
TEST(mixed, whole_mesh_comparison_tets_and_hexes)
{
  whole_mesh_comparison_test<2, 1>("/data/meshes/patch3D_tets_and_hexes.mesh");
}

template <int ptest, int ptrial, int dim>
void partial_mesh_comparison_test_impl(std::unique_ptr<mfem::ParMesh>& mesh)
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
  Functional<test_space(trial_space)> residual(test_fespace.get(), {trial_fespace.get()});
  Functional<test_space(trial_space)> residual_comparison(test_fespace.get(), {trial_fespace.get()});

  auto on_left = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] < 4.0; };
  Domain left = Domain::ofElements(*mesh, on_left);

  auto on_top = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[1] >= 0.99; };
  Domain top_boundary = Domain::ofBoundaryElements(*mesh, on_top);

  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, TestThermalIntegratorOne<dim>{}, left);

  residual.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0>{}, TestThermalIntegratorTwo{}, top_boundary);

  double t = 0.0;
  check_gradient(residual, t, U);

  auto r0 = residual(t, U);

  //////////////

  Domain d = EntireDomain(*mesh);
  residual_comparison.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, TestThermalIntegratorOne<dim, true>{}, d);

  Domain bdr = EntireBoundary(*mesh);
  residual_comparison.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0>{}, TestThermalIntegratorTwo<true>{}, bdr);

  auto r1 = residual_comparison(t, U);

  EXPECT_LT(r0.DistanceTo(r1.GetData()), 1.0e-14);
}

template <int ptest, int ptrial>
void partial_mesh_comparison_test(std::string meshfile)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR + meshfile), 1);

  if (mesh->Dimension() == 2) {
    partial_mesh_comparison_test_impl<ptest, ptrial, 2>(mesh);
  }

  if (mesh->Dimension() == 3) {
    partial_mesh_comparison_test_impl<ptest, ptrial, 3>(mesh);
  }
}

TEST(basic, partial_mesh_comparison_tris) { partial_mesh_comparison_test<1, 1>("/data/meshes/beam-tri.mesh"); }
TEST(basic, partial_mesh_comparison_quads) { partial_mesh_comparison_test<1, 1>("/data/meshes/beam-quad.mesh"); }

TEST(basic, partial_mesh_comparison_tets) { partial_mesh_comparison_test<1, 1>("/data/meshes/beam-tet.mesh"); }
TEST(basic, partial_mesh_comparison_hexes) { partial_mesh_comparison_test<1, 1>("/data/meshes/beam-hex.mesh"); }

TEST(qoi, partial_boundary)
{
  constexpr auto dim = 3;
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR "/data/meshes/beam-hex.mesh"), 1);

  // Define the types for the test and trial spaces using the function arguments
  using test_space = double;
  using trial_space = H1<1>;

  auto [trial_fespace, trial_fec] = smith::generateParFiniteElementSpace<trial_space>(mesh.get());

  mfem::Vector U(trial_fespace->TrueVSize());

  mfem::ParGridFunction U_gf(trial_fespace.get());
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);
  U_gf.GetTrueDofs(U);

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> qoi({trial_fespace.get()});

  auto on_top = [](std::vector<tensor<double, 3>> X, int /* attr */) { return average(X)[1] >= 0.99; };
  Domain top_boundary = Domain::ofBoundaryElements(*mesh, on_top);

  qoi.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn</*nothing*/>{}, TrivialIntegrator{}, top_boundary);

  double time = 0.0;

  auto area = qoi(time, U);

  EXPECT_NEAR(area, 8.0, 1.0e-14);
}

TEST(qoi, partial_domain)
{
  constexpr auto dim = 3;
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR "/data/meshes/beam-hex.mesh"), 1);

  // Define the types for the test and trial spaces using the function arguments
  using test_space = double;
  using trial_space = H1<1>;

  auto [trial_fespace, trial_fec] = smith::generateParFiniteElementSpace<trial_space>(mesh.get());

  mfem::Vector U(trial_fespace->TrueVSize());

  mfem::ParGridFunction U_gf(trial_fespace.get());
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);
  U_gf.GetTrueDofs(U);

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> qoi({trial_fespace.get()});

  auto on_left = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] < 4.0; };
  Domain left = Domain::ofElements(*mesh, on_left);

  qoi.AddDomainIntegral(Dimension<dim>{}, DependsOn</*nothing*/>{}, TrivialIntegrator{}, left);

  double time = 0.0;
  auto volume = qoi(time, U);

  EXPECT_NEAR(volume, 4.0, 1.0e-14);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
