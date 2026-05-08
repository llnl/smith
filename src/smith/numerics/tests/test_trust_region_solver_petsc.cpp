// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/trust_region_solver.hpp"
#include "smith/infrastructure/profiling.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_vector.hpp"
#include "smith/numerics/petsc_solvers.hpp"

const std::string MESHTAG = "mesh";

static constexpr int scalar_field_order = 1;

struct MeshFixture : public testing::Test {
  void SetUp()
  {
    smith::StateManager::initialize(datastore_, "solver_test");

    auto mfem_shape = mfem::Element::QUADRILATERAL;

    double length = 0.5;
    double width = 2.0;
    auto meshtmp =
        smith::mesh::refineAndDistribute(mfem::Mesh::MakeCartesian2D(2, 1, mfem_shape, true, length, width), 0, 0);
    mesh_ = &smith::StateManager::setMesh(std::move(meshtmp), MESHTAG);
  }

  axom::sidre::DataStore datastore_;
  mfem::ParMesh* mesh_;
};

std::vector<mfem::Vector> applyLinearOperator(const Mat& A, const std::vector<const mfem::Vector*>& states)
{
  std::vector<mfem::Vector> Astates;
  for (auto s : states) {
    Astates.emplace_back(*s);
  }

  int local_rows(states[0]->Size());
  int global_rows(smith::globalSize(*states[0], PETSC_COMM_WORLD));

  Vec x;
  Vec y;

  VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &x);
  VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &y);

  PetscInt iStart, iEnd;
  VecGetOwnershipRange(x, &iStart, &iEnd);

  std::vector<int> col_indices;
  col_indices.reserve(static_cast<size_t>(local_rows));
  for (int i = iStart; i < iEnd; ++i) {
    col_indices.push_back(i);
  }

  size_t num_cols = states.size();
  for (size_t c = 0; c < num_cols; ++c) {
    VecSetValues(x, local_rows, &col_indices[0], &(*states[c])[0], INSERT_VALUES);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    MatMult(A, x, y);
    VecGetValues(y, local_rows, &col_indices[0], &Astates[c][0]);
  }

  VecDestroy(&x);
  VecDestroy(&y);

  return Astates;
}

auto createDiagonalTestMatrix(mfem::Vector& x)
{
  const int local_rows = x.Size();
  const int global_rows = smith::globalSize(x, PETSC_COMM_WORLD);

  Vec b;
  VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &b);

  PetscInt iStart, iEnd;
  VecGetOwnershipRange(b, &iStart, &iEnd);
  VecDestroy(&b);

  std::vector<int> col_indices;
  col_indices.reserve(static_cast<size_t>(local_rows));
  for (int i = iStart; i < iEnd; ++i) {
    col_indices.push_back(i);
  }

  std::vector<int> row_offsets(static_cast<size_t>(local_rows) + 1);
  for (int i = 0; i < local_rows + 1; ++i) {
    row_offsets[static_cast<size_t>(i)] = i;
  }

  Mat A;
  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, local_rows, local_rows, global_rows, global_rows, &row_offsets[0],
                            &col_indices[0], &x[0], &A);

  return A;
}

void expectNearVector(const mfem::Vector& a, const mfem::Vector& b, double tol)
{
  ASSERT_EQ(a.Size(), b.Size());
  for (int i = 0; i < a.Size(); ++i) {
    EXPECT_NEAR(a[i], b[i], tol);
  }
}

TEST_F(MeshFixture, PetscSubspaceSolveHitsTrustRegionBoundary)
{
  SMITH_MARK_FUNCTION;

  auto u1 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u1", MESHTAG);
  auto u2 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u2", MESHTAG);
  auto u3 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u3", MESHTAG);
  auto a = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "a", MESHTAG);
  auto b = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "b", MESHTAG);

  u1 = 1.0;
  for (int i = 0; i < u2.Size(); ++i) {
    u2[i] = i + 2;
    u3[i] = i * i - 15.0;
    a[i] = 2 * i + 0.01 * i * i + 1.25;
    b[i] = -i + 0.02 * i * i + 0.1;
  }
  std::vector<const mfem::Vector*> states = {&u1, &u2, &u3};

  auto A_parallel = createDiagonalTestMatrix(a);
  std::vector<mfem::Vector> Astates = applyLinearOperator(A_parallel, states);

  std::vector<const mfem::Vector*> AstatePtrs;
  for (size_t i = 0; i < Astates.size(); ++i) {
    AstatePtrs.push_back(&Astates[i]);
  }

  double delta = 0.001;
  auto [sol, leftvecs, leftvals, energy] = smith::solveSubspaceProblemPetsc(states, AstatePtrs, b, delta, 1);

  EXPECT_NEAR(sol.Norml2(), delta, 1e-12);
  EXPECT_FALSE(leftvecs.empty());
  EXPECT_EQ(leftvals.size(), 1);
  EXPECT_LT(energy, 0.0);

  MatDestroy(&A_parallel);
}

TEST_F(MeshFixture, MfemSubspaceSolveMatchesPetsc)
{
  SMITH_MARK_FUNCTION;

  auto u1 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u1", MESHTAG);
  auto u2 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u2", MESHTAG);
  auto u3 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u3", MESHTAG);
  auto a = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "a", MESHTAG);
  auto b = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "b", MESHTAG);

  u1 = 1.0;
  for (int i = 0; i < u2.Size(); ++i) {
    u2[i] = i + 2;
    u3[i] = i * i - 15.0;
    a[i] = 2 * i + 0.01 * i * i + 1.25;
    b[i] = -i + 0.02 * i * i + 0.1;
  }

  std::vector<const mfem::Vector*> states = {&u1, &u2, &u3, &u2};
  auto A_parallel = createDiagonalTestMatrix(a);
  std::vector<mfem::Vector> Astates = applyLinearOperator(A_parallel, states);

  std::vector<const mfem::Vector*> AstatePtrs;
  for (size_t i = 0; i < Astates.size(); ++i) {
    AstatePtrs.push_back(&Astates[i]);
  }

  auto [petsc_sol, petsc_leftvecs, petsc_leftvals, petsc_energy] =
      smith::solveSubspaceProblemPetsc(states, AstatePtrs, b, 0.001, 2);
  auto [mfem_sol, mfem_leftvecs, mfem_leftvals, mfem_energy] =
      smith::solveSubspaceProblemMfem(states, AstatePtrs, b, 0.001, 2);

  expectNearVector(mfem_sol, petsc_sol, 1e-10);
  ASSERT_EQ(mfem_leftvecs.size(), petsc_leftvecs.size());
  ASSERT_EQ(mfem_leftvals.size(), petsc_leftvals.size());
  for (size_t i = 0; i < mfem_leftvecs.size(); ++i) {
    const double same = smith::innerProduct(*mfem_leftvecs[i], *petsc_leftvecs[i], MPI_COMM_WORLD);
    mfem::Vector neg(*petsc_leftvecs[i]);
    neg *= -1.0;
    const double flipped = smith::innerProduct(*mfem_leftvecs[i], neg, MPI_COMM_WORLD);
    if (std::abs(flipped) > std::abs(same)) {
      expectNearVector(*mfem_leftvecs[i], neg, 1e-9);
    } else {
      expectNearVector(*mfem_leftvecs[i], *petsc_leftvecs[i], 1e-9);
    }
    EXPECT_NEAR(mfem_leftvals[i], petsc_leftvals[i], 1e-10);
  }
  EXPECT_NEAR(mfem_energy, petsc_energy, 1e-12);

  MatDestroy(&A_parallel);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
