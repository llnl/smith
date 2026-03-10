// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
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

// auto createDiagonalTestMatrix(smith::FiniteElementState& x)
auto createDiagonalTestMatrix(mfem::Vector& x)
{
  const int local_rows = x.Size();
  mfem::Vector one = x;
  one = 1.0;
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

TEST_F(MeshFixture, QR)
{
  SMITH_MARK_FUNCTION;

  auto u1 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u1", MESHTAG);
  auto u2 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u2", MESHTAG);
  auto u3 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u3", MESHTAG);
  auto u4 = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "u4", MESHTAG);
  auto a = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "a", MESHTAG);
  auto b = smith::StateManager::newState(smith::H1<scalar_field_order, 1>{}, "b", MESHTAG);

  u1 = 1.0;
  for (int i = 0; i < u2.Size(); ++i) {
    u2[i] = i + 2;
    u3[i] = i * i - 15.0;
    u4[i] = -i + 0.1 * i * i * i - 1.0;
    a[i] = 2 * i + 0.01 * i * i + 1.25;
    b[i] = -i + 0.02 * i * i + 0.1;
  }
  std::vector<const mfem::Vector*> states = {&u1, &u2, &u3};  //,u4};

  auto A_parallel = createDiagonalTestMatrix(a);
  std::vector<mfem::Vector> Astates = applyLinearOperator(A_parallel, states);

  std::vector<const mfem::Vector*> AstatePtrs;
  for (size_t i = 0; i < Astates.size(); ++i) {
    AstatePtrs.push_back(&Astates[i]);
  }

  double delta = 0.001;
  auto [sol, leftvecs, leftvals, energy] = smith::solveSubspaceProblem(states, AstatePtrs, b, delta, 1);

  smith::FiniteElementState smith_sol(b);
  smith_sol = sol;

  EXPECT_NEAR(std::sqrt(smith::innerProduct(smith_sol, smith_sol)), delta, 1e-12);

  MatDestroy(&A_parallel);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
