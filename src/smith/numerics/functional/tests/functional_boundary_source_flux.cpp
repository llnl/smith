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

#include "smith/mesh_utils/mesh_utils_base.hpp"
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

TEST(FunctionalBoundarySourceFlux, SurfacePatchTest)
{
  constexpr int dim = 3;
  std::string filename = std::string(SMITH_REPO_DIR) + "/data/meshes/" + "patch3D_hexes.mesh";
  auto serial_mesh = buildMeshFromFile(filename);

  
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), 1, 0);
  mesh->EnsureNodes();
  mesh->Print();

  // Jiggle the non-boundary node on the face so that the element
  // shape mappings are nonlinear. This makes the integrand non-trivial.
  // Although the physical stress is uniform on the surface, the
  // *components* of the stress in the curvilinear bases are not.
  double face_node_coordinates[dim];
  constexpr int face_node_index = 64;
  mesh->GetNode(face_node_index, face_node_coordinates);
  face_node_coordinates[1] += 0.15;
  face_node_coordinates[2] -= 0.03;
  mesh->SetNode(face_node_index, face_node_coordinates);
  mesh->NodesUpdated();

  using space = H1<1, dim>;
  auto [fespace, fec] = generateParFiniteElementSpace<space>(mesh.get());
  Functional<space(space)> residual(fespace.get(), {fespace.get()});
  // Attribute 1 of this mesh is on the yz plane
  Domain boundary = Domain::ofBoundaryElements(*mesh, by_attr<dim>(1));

  // Surface elasticity:
  // Green-St Venant law expressed in curvilinear coordinates.
  residual.AddBoundarySourceFluxIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [](double, auto position, auto field) {
        double E = 2.4;
        double nu = 0.25;
        double mu = 0.5*E/(1.0 + nu);
        double lam_prime = E*nu/(1.0 - nu*nu); // second Lame modulus for plane stress
        auto dX_dxi = get<DERIVATIVE>(position);
        auto du_dxi = get<DERIVATIVE>(field);
        // covariant components of Green's strain
        auto epsilon = 0.5*(transpose(du_dxi) * dX_dxi + transpose(dX_dxi) * du_dxi);
        auto g0 = dot(transpose(dX_dxi), dX_dxi); // metric tensor
        auto g0_dual = inv(g0);  // dual metric tensor
        auto tr_epsilon = inner(g0_dual, epsilon);
        auto sigma = 2*mu*(g0_dual * epsilon * g0_dual) + lam_prime * tr_epsilon * g0_dual;
        mfem::out << "sigma = " << sigma << std::endl;
        return smith::tuple{zero{}, dX_dxi * sigma};
      }, 
      boundary);

  tensor<double, dim, dim> H{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                              {0.198344644470483, 0.060514559793513, 0.084137393813728},
                              {0.011544253485023, 0.060942846497753, 0.186383473579596}}};
  mfem::DenseMatrix disp_grad(&H[0][0], dim, dim);
  mfem::Vector U(fespace->TrueVSize());
  mfem::ParGridFunction U_gf(fespace.get());
  mfem::VectorFunctionCoefficient c(dim, [&disp_grad](const mfem::Vector& X, mfem::Vector& u) { disp_grad.Mult(X, u); });
  U_gf.ProjectCoefficient(c);
  U_gf.GetTrueDofs(U);

  mfem::Vector F(fespace->TrueVSize());
  F = residual(0.0, U);
  mfem::out << "Residual\n";
  F.Print();
  mfem::ParGridFunction F_gf(fespace.get());
  F_gf.SetFromTrueDofs(F);
  double fy = F_gf(fespace->DofToVDof(64, 1));
  double fz = F_gf(fespace->DofToVDof(64, 2));
  mfem::out << axom::fmt::format("fy = {:.6e}\n", fy);
  mfem::out << axom::fmt::format("fz = {:.6e}\n", fz);
  ASSERT_LT(std::abs(fy), 1e-15);
  ASSERT_LT(std::abs(fz), 1e-15);
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
  auto [fespace, fec] = generateParFiniteElementSpace<space>(&mesh);
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
  auto [fespace, fec] = generateParFiniteElementSpace<test_space>(mesh.get());
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
  auto [fespace, fec] = generateParFiniteElementSpace<space>(mesh.get());

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
  auto [fespace, fec] = generateParFiniteElementSpace<space>(mesh.get());

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
  auto [shape_fespace, shape_fec] = generateParFiniteElementSpace<shape_space>(mesh.get());
  auto [test_fespace, test_fec] = generateParFiniteElementSpace<test_space>(mesh.get());

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
