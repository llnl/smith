// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/shape_aware_functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"

using namespace smith;

namespace {

template <typename space>
auto makeSpace(mfem::ParMesh& mesh)
{
  return generateParFiniteElementSpace<space>(&mesh);
}

TEST(FunctionalBoundarySourceFlux, ParametricDerivativeUsesPhysicalBoundaryMeasure)
{
  constexpr double LX = 3.0;
  constexpr double LY = 2.0;
  constexpr double SOURCE_VALUE = 2.0;
  constexpr double FLUX_VALUE = 3.0;

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, LX, LY);
  serial_mesh.EnsureNodes();
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mesh.EnsureNodes();

  using space = H1<1>;
  auto [fespace, fec] = makeSpace<space>(mesh);
  Functional<space(space)> residual(fespace.get(), {fespace.get()});
  Domain boundary = EntireBoundary(mesh);

  residual.AddBoundarySourceFluxIntegral(
      Dimension<1>{}, DependsOn<0>{},
      [=](double, auto, auto) { return smith::tuple{SOURCE_VALUE, FLUX_VALUE}; }, boundary);

  mfem::Vector U(fespace->TrueVSize());
  U = 0.0;
  mfem::Vector actual = residual(0.0, U);
  mfem::Vector expected(actual.Size());
  expected = 0.0;

  for (int e = 0; e < mesh.GetNBE(); ++e) {
    mfem::Array<int> dofs;
    fespace->GetBdrElementDofs(e, dofs);
    ASSERT_EQ(dofs.Size(), 2);

    auto* transformation = mesh.GetBdrElementTransformation(e);
    const mfem::IntegrationPoint midpoint(0.5);
    transformation->SetIntPoint(&midpoint);
    const double boundary_measure = transformation->Weight();

    expected[dofs[0]] += boundary_measure * (0.5 * SOURCE_VALUE - FLUX_VALUE);
    expected[dofs[1]] += boundary_measure * (0.5 * SOURCE_VALUE + FLUX_VALUE);
  }

  EXPECT_NEAR(actual.DistanceTo(expected.GetData()), 0.0, 1.0e-13);

  Functional<space(space)> flux_only_residual(fespace.get(), {fespace.get()});
  flux_only_residual.AddBoundarySourceFluxIntegral(
      Dimension<1>{}, DependsOn<0>{}, [=](double, auto, auto) { return smith::tuple{zero{}, FLUX_VALUE}; }, boundary);
  actual = flux_only_residual(0.0, U);
  expected = 0.0;
  for (int e = 0; e < mesh.GetNBE(); ++e) {
    mfem::Array<int> dofs;
    fespace->GetBdrElementDofs(e, dofs);
    auto* transformation = mesh.GetBdrElementTransformation(e);
    const mfem::IntegrationPoint midpoint(0.5);
    transformation->SetIntPoint(&midpoint);
    const double boundary_measure = transformation->Weight();
    expected[dofs[0]] -= boundary_measure * FLUX_VALUE;
    expected[dofs[1]] += boundary_measure * FLUX_VALUE;
  }
  EXPECT_NEAR(actual.DistanceTo(expected.GetData()), 0.0, 1.0e-13);
}

template <typename space, int dim>
void checkSourceOnlyCompatibility(const std::string& mesh_file)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR + mesh_file), 0);
  using test_space = space;
  auto [fespace, fec] = makeSpace<test_space>(*mesh);
  Domain boundary = EntireBoundary(*mesh);

  Functional<test_space(test_space)> legacy(fespace.get(), {fespace.get()});
  legacy.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0>{},
                             [](double, auto position, auto) {
                               const auto& X = get<VALUE>(position);
                               return 1.0 + X[0] * X[0];
                             },
                             boundary);

  Functional<test_space(test_space)> source_flux(fespace.get(), {fespace.get()});
  source_flux.AddBoundarySourceFluxIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [](double, auto position, auto) {
        const auto& X = get<VALUE>(position);
        return smith::tuple{1.0 + X[0] * X[0], zero{}};
      },
      boundary);

  mfem::Vector U(fespace->TrueVSize());
  U = 0.0;
  mfem::Vector expected = legacy(0.0, U);
  mfem::Vector actual = source_flux(0.0, U);
  EXPECT_NEAR(actual.DistanceTo(expected.GetData()), 0.0, 1.0e-13);
}

TEST(FunctionalBoundarySourceFlux, SourceOnlyH1Compatibility2D)
{
  checkSourceOnlyCompatibility<H1<2>, 2>("/data/meshes/patch2D_tris_and_quads.mesh");
}

TEST(FunctionalBoundarySourceFlux, SourceOnlyL2Compatibility3D)
{
  checkSourceOnlyCompatibility<L2<2>, 3>("/data/meshes/patch3D_tets_and_hexes.mesh");
}

template <typename space, int dim>
void checkScalarGradient(const std::string& mesh_file)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR + mesh_file), 0);
  auto [fespace, fec] = makeSpace<space>(*mesh);

  mfem::Vector U(fespace->TrueVSize());
  U.Randomize(19);

  Functional<space(space)> residual(fespace.get(), {fespace.get()});
  Domain boundary = EntireBoundary(*mesh);
  residual.AddBoundarySourceFluxIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [](double, auto position, auto field) {
        const auto& X = get<VALUE>(position);
        const auto& u = get<VALUE>(field);
        const auto& du_dxi = get<DERIVATIVE>(field);
        auto source = u * u + X[0];
        auto flux = du_dxi * (1.0 + u);
        return smith::tuple{source, flux};
      },
      boundary);

  check_gradient(residual, 0.0, U);
}

TEST(FunctionalBoundarySourceFlux, ScalarH1Gradient2D)
{
  checkScalarGradient<H1<2>, 2>("/data/meshes/patch2D_tris_and_quads.mesh");
}

TEST(FunctionalBoundarySourceFlux, ScalarL2Gradient3D)
{
  checkScalarGradient<L2<2>, 3>("/data/meshes/patch3D_tets_and_hexes.mesh");
}

template <int dim>
void checkVectorGradient(const std::string& mesh_file)
{
  using space = H1<2, dim + 1>;
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR + mesh_file), 0);
  auto [fespace, fec] = makeSpace<space>(*mesh);

  mfem::Vector U(fespace->TrueVSize());
  U.Randomize(23);

  Functional<space(space)> residual(fespace.get(), {fespace.get()});
  Domain boundary = EntireBoundary(*mesh);
  residual.AddBoundarySourceFluxIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [](double, auto, auto field) {
        const auto& u = get<VALUE>(field);
        const auto& du_dxi = get<DERIVATIVE>(field);
        auto source = u * (1.0 + u[0]);
        auto flux = du_dxi * (1.0 + u[0]);
        return smith::tuple{source, flux};
      },
      boundary);

  check_gradient(residual, 0.0, U);
}

TEST(FunctionalBoundarySourceFlux, VectorH1Gradient2D)
{
  checkVectorGradient<2>("/data/meshes/patch2D_tris_and_quads.mesh");
}

TEST(FunctionalBoundarySourceFlux, VectorH1Gradient3D)
{
  checkVectorGradient<3>("/data/meshes/patch3D_tets_and_hexes.mesh");
}

template <int dim>
void checkShapeGradient(const std::string& mesh_file)
{
  using shape_space = H1<1, dim>;
  using test_space = H1<1>;
  using trial_space = H1<1>;

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR + mesh_file), 0);
  auto [shape_fespace, shape_fec] = makeSpace<shape_space>(*mesh);
  auto [test_fespace, test_fec] = makeSpace<test_space>(*mesh);

  mfem::Vector shape(shape_fespace->TrueVSize());
  shape.Randomize(29);
  shape *= 0.01;
  mfem::Vector direction(shape_fespace->TrueVSize());
  direction.Randomize(31);
  mfem::Vector U(test_fespace->TrueVSize());
  U.Randomize(37);

  ShapeAwareFunctional<shape_space, test_space(trial_space)> residual(shape_fespace.get(), test_fespace.get(),
                                                                      {test_fespace.get()});
  Domain boundary = EntireBoundary(*mesh);
  residual.AddBoundarySourceFluxIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [](double, auto position, auto field) {
        const auto& X = get<VALUE>(position);
        const auto& u = get<VALUE>(field);
        const auto& du_dxi = get<DERIVATIVE>(field);
        return smith::tuple{u * (1.0 + X[0]), du_dxi * (1.0 + X[0])};
      },
      boundary);

  auto [value, gradient] = residual(0.0, differentiate_wrt(shape), U);
  mfem::Vector derivative = gradient(direction);

  constexpr double EPSILON = 1.0e-5;
  mfem::Vector shape_plus(shape);
  shape_plus.Add(EPSILON, direction);
  mfem::Vector shape_minus(shape);
  shape_minus.Add(-EPSILON, direction);
  mfem::Vector finite_difference = residual(0.0, shape_plus, U);
  finite_difference -= residual(0.0, shape_minus, U);
  finite_difference /= 2.0 * EPSILON;

  mfem::Vector error(derivative);
  error -= finite_difference;
  const double scale = std::max(1.0, derivative.Norml2());
  EXPECT_LT(error.Norml2() / scale, 2.0e-7);
}

TEST(FunctionalBoundarySourceFlux, ShapeGradient2D)
{
  checkShapeGradient<2>("/data/meshes/patch2D_tris_and_quads.mesh");
}

TEST(FunctionalBoundarySourceFlux, ShapeGradient3D)
{
  checkShapeGradient<3>("/data/meshes/patch3D_tets_and_hexes.mesh");
}

}  // namespace

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager application_manager(argc, argv);
  return RUN_ALL_TESTS();
}
