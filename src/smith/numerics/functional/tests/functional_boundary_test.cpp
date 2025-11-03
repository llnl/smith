// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>
#include <cmath>
#include <string>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"
#include "smith/infrastructure/mpi_fstream.hpp"
#include "smith/infrastructure/application_manager.hpp"

using namespace smith;

int refinements = 0;
constexpr bool verbose = true;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

double relative_error_frobenius_norm(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B)
{
  if (A.Height() != B.Height()) return false;
  if (A.Width() != B.Width()) return false;

  double fnorm_A = 0.0;
  double fnorm_A_minus_B = 0.0;

  for (int r = 0; r < A.Height(); r++) {
    auto columns = A.GetRowColumns(r);
    for (int j = 0; j < A.RowSize(r); j++) {
      int c = columns[j];
      fnorm_A += A(r, c) * A(r, c);
      fnorm_A_minus_B += (A(r, c) - B(r, c)) * (A(r, c) - B(r, c));
    }
  }

  return sqrt(fnorm_A_minus_B / fnorm_A);
}

bool operator==(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B)
{
  return relative_error_frobenius_norm(A, B) < 1.0e-15;
}

bool operator!=(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B) { return !(A == B); }

template <int p, int dim>
void boundary_test(mfem::ParMesh& mesh, H1<p> test, H1<p> trial, Dimension<dim>)
{
  double rho = 1.75;
  using test_space = decltype(test);
  using trial_space = decltype(trial);

  auto [fespace, fec] = smith::generateParFiniteElementSpace<test_space>(&mesh);

  mfem::ParLinearForm f(fespace.get());
  mfem::FunctionCoefficient scalar_function([&](const mfem::Vector& coords) { return coords(0) * coords(1); });
  mfem::VectorFunctionCoefficient vector_function(dim, [&](const mfem::Vector& coords, mfem::Vector& output) {
    output = 0.0;
    output[0] = sin(coords[0]);
    output[1] = coords[0] * coords[1];
  });

  f.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(scalar_function, 2, 0));
  f.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(vector_function, 2, 0));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  mfem::ParBilinearForm B(fespace.get());
  mfem::ConstantCoefficient density(rho);
  B.AddBoundaryIntegrator(new mfem::BoundaryMassIntegrator(density));
  B.Assemble(0);

  B.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(B.ParallelAssemble());

  mfem::Vector U(fespace->TrueVSize());
  int seed = 9;
  U.Randomize(seed);

  Functional<test_space(trial_space)> residual(fespace.get(), {fespace.get()});

  Domain bdr = EntireBoundary(mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [&](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, unused] = temperature;

        auto n = normalize(cross(dX_dxi));

        tensor<double, dim> b{sin(X[0]), X[0] * X[1]};
        return X[0] * X[1] + dot(b, n) + rho * u;
      },
      bdr);

  // mfem::Vector r1 = (*J) * U + (*F);
  mfem::Vector r1(U.Size());
  J->Mult(U, r1);
  r1 += (*F);
  double t = 0.0;
  mfem::Vector r2 = residual(t, U);

  check_gradient(residual, t, U);

  mfem::Vector diff(r1.Size());
  subtract(r1, r2, diff);

  if (verbose) {
    mpi::out << "sum(r1):  " << r1.Sum() << std::endl;
    mpi::out << "sum(r2):  " << r2.Sum() << std::endl;
    mpi::out << "||r1||: " << r1.Norml2() << std::endl;
    mpi::out << "||r2||: " << r2.Norml2() << std::endl;
    mpi::out << "||r1-r2||/||r1||: " << diff.Norml2() / r1.Norml2() << std::endl;
  }

  if (r1.Norml2() < 1.0e-15) {
    EXPECT_NEAR(0., diff.Norml2(), 1.e-12);
  } else {
    EXPECT_NEAR(0., diff.Norml2() / r1.Norml2(), 1.e-12);
  }
}

template <int p, int dim>
void boundary_test(mfem::ParMesh& mesh, L2<p> test, L2<p> trial, Dimension<dim>)
{
  double rho = 1.75;
  using test_space = decltype(test);
  using trial_space = decltype(trial);

  auto [fespace, fec] = smith::generateParFiniteElementSpace<test_space>(&mesh);

  mfem::ParLinearForm f(fespace.get());
  mfem::FunctionCoefficient scalar_function([&](const mfem::Vector& coords) { return coords(0) * coords(1); });
  f.AddBdrFaceIntegrator(new mfem::BoundaryLFIntegrator(scalar_function, 2, 0));

  // mfem is missing the implementation of BoundaryNormalLFIntegrator for L2
  // mfem::VectorFunctionCoefficient vector_function(dim, [&](const mfem::Vector& coords, mfem::Vector & output) {
  //  output[0] = sin(coords[0]);
  //  output[1] = coords[0] * coords[1];
  //});
  // f.AddBdrFaceIntegrator(new mfem::BoundaryNormalLFIntegrator(vector_function));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  mfem::ParBilinearForm B(fespace.get());
  mfem::ConstantCoefficient density(rho);
  B.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(density));
  B.Assemble(0);
  B.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(B.ParallelAssemble());

  mfem::ParGridFunction u_global(fespace.get());
  int seed = 1;
  u_global.Randomize(seed);
  mfem::FunctionCoefficient xfunc([](mfem::Vector x) { return x[0]; });
  u_global.ProjectCoefficient(xfunc);

  mfem::Vector U(fespace->TrueVSize());
  u_global.GetTrueDofs(U);

  Functional<test_space(trial_space)> residual(fespace.get(), {fespace.get()});

  Domain bdr = EntireBoundary(mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [&](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, unused] = temperature;

        // mfem is missing the integrator to compute this term
        // tensor<double,dim> b{sin(x[0]), x[0] * x[1]};
        return X[0] * X[1] + /* dot(b, n) +*/ rho * u;
      },
      bdr);

  // mfem::Vector r1 = (*J) * U + (*F);
  mfem::Vector r1(U.Size());
  J->Mult(U, r1);
  r1 += (*F);
  double t = 0.0;
  mfem::Vector r2 = residual(t, U);

  mfem::Vector diff(r1.Size());
  subtract(r1, r2, diff);

  if (verbose) {
    mpi::out << "sum(r1):  " << r1.Sum() << std::endl;
    mpi::out << "sum(r2):  " << r2.Sum() << std::endl;
    mpi::out << "||r1||: " << r1.Norml2() << std::endl;
    mpi::out << "||r2||: " << r2.Norml2() << std::endl;
    mpi::out << "||r1-r2||/||r1||: " << diff.Norml2() / r1.Norml2() << std::endl;
  }

  if (r1.Norml2() < 1.0e-15) {
    EXPECT_NEAR(0., diff.Norml2(), 1.e-12);
  } else {
    EXPECT_NEAR(0., diff.Norml2() / r1.Norml2(), 1.e-12);
  }
}

TEST(FunctionalBoundary, 2DLinear) { boundary_test(*mesh2D, H1<1>{}, H1<1>{}, Dimension<2>{}); }
TEST(FunctionalBoundary, 2DQuadratic) { boundary_test(*mesh2D, H1<2>{}, H1<2>{}, Dimension<2>{}); }

TEST(FunctionalBoundary, 3DLinear) { boundary_test(*mesh3D, H1<1>{}, H1<1>{}, Dimension<3>{}); }
TEST(FunctionalBoundary, 3DQuadratic) { boundary_test(*mesh3D, H1<2>{}, H1<2>{}, Dimension<3>{}); }

TEST(boundaryL2, 2DLinear) { boundary_test(*mesh2D, L2<1>{}, L2<1>{}, Dimension<2>{}); }
TEST(boundaryL2, 2DQuadratic) { boundary_test(*mesh2D, L2<2>{}, L2<2>{}, Dimension<2>{}); }

TEST(boundaryL2, 3DLinear) { boundary_test(*mesh3D, L2<1>{}, L2<1>{}, Dimension<3>{}); }
TEST(boundaryL2, 3DQuadratic) { boundary_test(*mesh3D, L2<2>{}, L2<2>{}, Dimension<3>{}); }
TEST(boundaryL2, 3DCubic) { boundary_test(*mesh3D, L2<3>{}, L2<3>{}, Dimension<3>{}); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);

  int serial_refinement = 1;
  int parallel_refinement = 0;

  std::string meshfile2D = SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  std::string meshfile3D = SMITH_REPO_DIR "/data/meshes/patch3D_tets_and_hexes.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  // by default, mfem::Meshes aren't completely initialized on construction,
  // so we have to manually initialize some of it
  mesh2D->EnsureNodes();
  mesh3D->EnsureNodes();

  return RUN_ALL_TESTS();
}
