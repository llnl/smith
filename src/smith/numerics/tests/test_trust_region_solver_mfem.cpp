// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/trust_region_solver.hpp"

namespace {

constexpr int test_size = 5;
constexpr double test_delta = 1.0e-3;

std::vector<mfem::Vector> applyDiagonalOperator(const mfem::Vector& diag, const std::vector<const mfem::Vector*>& states)
{
  std::vector<mfem::Vector> out;
  out.reserve(states.size());
  for (const auto* state : states) {
    out.emplace_back(state->Size());
    for (int i = 0; i < state->Size(); ++i) {
      out.back()[i] = diag[i] * (*state)[i];
    }
  }
  return out;
}

void expectNearVector(const mfem::Vector& a, const mfem::Vector& b, double tol)
{
  ASSERT_EQ(a.Size(), b.Size());
  for (int i = 0; i < a.Size(); ++i) {
    EXPECT_NEAR(a[i], b[i], tol);
  }
}

std::vector<const mfem::Vector*> toPointers(const std::vector<mfem::Vector>& vectors)
{
  std::vector<const mfem::Vector*> ptrs;
  ptrs.reserve(vectors.size());
  for (const auto& v : vectors) {
    ptrs.push_back(&v);
  }
  return ptrs;
}

struct DiagonalSubspaceFixture {
  DiagonalSubspaceFixture(int size)
      : u1(size),
        u2(size),
        u3(size),
        diag(size),
        b(size)
  {
    u1 = 1.0;
    for (int i = 0; i < size; ++i) {
      u2[i] = i + 2.0;
      u3[i] = i * i - 15.0;
      diag[i] = 2.0 * i + 0.01 * i * i + 1.25;
      b[i] = -i + 0.02 * i * i + 0.1;
    }
  }

  mfem::Vector u1;
  mfem::Vector u2;
  mfem::Vector u3;
  mfem::Vector diag;
  mfem::Vector b;
};

}  // namespace

TEST(TrustRegionSubspaceMfem, RemoveDependentDirectionsDropsDuplicatesAndZero)
{
  mfem::Vector d1(4);
  mfem::Vector d2(4);
  mfem::Vector d3(4);
  mfem::Vector hd1(4);
  mfem::Vector hd2(4);
  mfem::Vector hd3(4);

  d1 = 0.0;
  d2 = 0.0;
  d3 = 0.0;
  hd1 = 0.0;
  hd2 = 0.0;
  hd3 = 0.0;

  d1[0] = 1.0;
  d1[1] = 2.0;
  d2 = d1;
  d2 *= 3.0;

  hd1[0] = 2.0;
  hd1[1] = 5.0;
  hd2 = hd1;
  hd2 *= 3.0;

  std::vector<const mfem::Vector*> dirs = {&d1, &d2, &d3};
  std::vector<const mfem::Vector*> hdirs = {&hd1, &hd2, &hd3};

  auto [dirs_new, hdirs_new] = smith::removeDependentDirections(dirs, hdirs);

  ASSERT_EQ(dirs_new.size(), 1);
  ASSERT_EQ(hdirs_new.size(), 1);
  expectNearVector(*dirs_new[0], d1, 0.0);
  expectNearVector(*hdirs_new[0], hd1, 0.0);
}

TEST(TrustRegionSubspaceMfem, RemoveDependentDirectionTriplesKeepsHistoryAligned)
{
  mfem::Vector d1(3);
  mfem::Vector d2(3);
  mfem::Vector d3(3);
  mfem::Vector hd1(3);
  mfem::Vector hd2(3);
  mfem::Vector hd3(3);
  mfem::Vector old_hd1(3);
  mfem::Vector old_hd2(3);
  mfem::Vector old_hd3(3);

  d1 = 0.0;
  d2 = 0.0;
  d3 = 0.0;
  hd1 = 0.0;
  hd2 = 0.0;
  hd3 = 0.0;
  old_hd1 = 0.0;
  old_hd2 = 0.0;
  old_hd3 = 0.0;

  d1[0] = 1.0;
  d2 = d1;
  d2 *= 2.0;
  d3[2] = 1.0;
  hd1[0] = 3.0;
  hd2[0] = 6.0;
  hd3[2] = 4.0;
  old_hd1[0] = 2.0;
  old_hd2[0] = 4.0;
  old_hd3[2] = 5.0;

  std::vector<const mfem::Vector*> dirs = {&d1, &d2, &d3};
  std::vector<const mfem::Vector*> hdirs = {&hd1, &hd2, &hd3};
  std::vector<const mfem::Vector*> old_hdirs = {&old_hd1, &old_hd2, &old_hd3};

  auto [dirs_new, hdirs_new, old_hdirs_new] = smith::removeDependentDirectionTriples(dirs, hdirs, old_hdirs);

  ASSERT_EQ(dirs_new.size(), 2);
  expectNearVector(*dirs_new[0], d1, 0.0);
  expectNearVector(*hdirs_new[0], hd1, 0.0);
  expectNearVector(*old_hdirs_new[0], old_hd1, 0.0);
  expectNearVector(*dirs_new[1], d3, 0.0);
  expectNearVector(*hdirs_new[1], hd3, 0.0);
  expectNearVector(*old_hdirs_new[1], old_hd3, 0.0);
}

TEST(TrustRegionSubspaceMfem, SolveHitsTrustRegionBoundary)
{
  DiagonalSubspaceFixture fixture(test_size);

  const std::vector<const mfem::Vector*> states = {&fixture.u1, &fixture.u2, &fixture.u3};
  const auto astates = applyDiagonalOperator(fixture.diag, states);
  const auto astate_ptrs = toPointers(astates);

  auto [sol, leftvecs, leftvals, energy] =
      smith::solveSubspaceProblemMfem(states, astate_ptrs, fixture.b, test_delta, 1);

  EXPECT_NEAR(sol.Norml2(), test_delta, 1.0e-12);
  EXPECT_FALSE(leftvecs.empty());
  EXPECT_EQ(leftvals.size(), 1);
  EXPECT_LT(energy, 0.0);
}

TEST(TrustRegionSubspaceMfem, GenericSolveUsesMfemBackend)
{
  DiagonalSubspaceFixture fixture(test_size);

  const std::vector<const mfem::Vector*> states = {&fixture.u1, &fixture.u2, &fixture.u3, &fixture.u2};
  const auto astates = applyDiagonalOperator(fixture.diag, states);
  const auto astate_ptrs = toPointers(astates);

  auto [generic_sol, generic_leftvecs, generic_leftvals, generic_energy] =
      smith::solveSubspaceProblem(states, astate_ptrs, fixture.b, test_delta, 2);
  auto [mfem_sol, mfem_leftvecs, mfem_leftvals, mfem_energy] =
      smith::solveSubspaceProblemMfem(states, astate_ptrs, fixture.b, test_delta, 2);

  expectNearVector(generic_sol, mfem_sol, 1.0e-12);
  ASSERT_EQ(generic_leftvecs.size(), mfem_leftvecs.size());
  ASSERT_EQ(generic_leftvals.size(), mfem_leftvals.size());
  for (size_t i = 0; i < generic_leftvecs.size(); ++i) {
    const double same = smith::innerProduct(*generic_leftvecs[i], *mfem_leftvecs[i], MPI_COMM_WORLD);
    mfem::Vector neg(*mfem_leftvecs[i]);
    neg *= -1.0;
    const double flipped = smith::innerProduct(*generic_leftvecs[i], neg, MPI_COMM_WORLD);
    if (std::abs(flipped) > std::abs(same)) {
      expectNearVector(*generic_leftvecs[i], neg, 1.0e-10);
    } else {
      expectNearVector(*generic_leftvecs[i], *mfem_leftvecs[i], 1.0e-10);
    }
    EXPECT_NEAR(generic_leftvals[i], mfem_leftvals[i], 1.0e-12);
  }
  EXPECT_NEAR(generic_energy, mfem_energy, 1.0e-12);
}

TEST(TrustRegionSubspaceMfem, SolveHandlesZeroDirection)
{
  mfem::Vector u1(4);
  mfem::Vector u2(4);
  mfem::Vector zero(4);
  mfem::Vector diag(4);
  mfem::Vector b(4);

  zero = 0.0;
  for (int i = 0; i < 4; ++i) {
    u1[i] = 1.0 + i;
    u2[i] = 0.25 * i - 0.5;
    diag[i] = 1.0 + i;
    b[i] = 0.5 - 0.1 * i;
  }

  const std::vector<const mfem::Vector*> states = {&u1, &zero, &u2};
  const auto astates = applyDiagonalOperator(diag, states);
  const auto astate_ptrs = toPointers(astates);

  auto [sol, leftvecs, leftvals, energy] = smith::solveSubspaceProblemMfem(states, astate_ptrs, b, 0.25, 1);

  EXPECT_LE(sol.Norml2(), 0.25 + 1.0e-12);
  EXPECT_FALSE(leftvecs.empty());
  EXPECT_EQ(leftvals.size(), 1);
  EXPECT_LT(energy, 0.0);
}

TEST(TrustRegionCubicSubspaceMfem, ZeroCubicMatchesInteriorQuadraticSolve)
{
  mfem::DenseMatrix A(2);
  A = 0.0;
  A(0, 0) = 4.0;
  A(1, 1) = 2.0;

  mfem::Vector b(2);
  b[0] = 2.0;
  b[1] = -1.0;

  std::vector<mfem::DenseMatrix> cubic(2, mfem::DenseMatrix(2));
  for (auto& matrix : cubic) {
    matrix = 0.0;
  }

  auto [x, energy] = smith::solveDenseCubicTrustRegionProblemMfem(A, b, cubic, 10.0);

  EXPECT_NEAR(x[0], 0.5, 1.0e-10);
  EXPECT_NEAR(x[1], -0.5, 1.0e-10);
  EXPECT_NEAR(energy, -0.75, 1.0e-10);
}

TEST(TrustRegionCubicSubspaceMfem, CubicTermChangesOneDimensionalMinimizer)
{
  mfem::DenseMatrix A(1);
  A(0, 0) = 1.0;

  mfem::Vector b(1);
  b[0] = 1.0;

  std::vector<mfem::DenseMatrix> cubic(1, mfem::DenseMatrix(1));
  cubic[0](0, 0) = 6.0;

  auto [x, energy] = smith::solveDenseCubicTrustRegionProblemMfem(A, b, cubic, 1.0);

  const double expected = (-1.0 + std::sqrt(13.0)) / 6.0;
  EXPECT_NEAR(x[0], expected, 2.0e-3);
  EXPECT_NEAR(energy, 0.5 * expected * expected - expected + expected * expected * expected, 5.0e-6);
}

TEST(TrustRegionCubicSubspaceMfem, RespectsTrustRegionBoundary)
{
  mfem::DenseMatrix A(1);
  A(0, 0) = 1.0;

  mfem::Vector b(1);
  b[0] = 10.0;

  std::vector<mfem::DenseMatrix> cubic(1, mfem::DenseMatrix(1));
  cubic[0] = 0.0;

  auto [x, energy] = smith::solveDenseCubicTrustRegionProblemMfem(A, b, cubic, 0.25);

  EXPECT_NEAR(x.Norml2(), 0.25, 1.0e-12);
  EXPECT_NEAR(x[0], 0.25, 1.0e-12);
  EXPECT_NEAR(energy, 0.5 * 0.25 * 0.25 - 10.0 * 0.25, 1.0e-12);
}

TEST(TrustRegionCubicSubspaceMfem, HistoryProjectedSubspaceSolveRuns)
{
  mfem::Vector e1(2);
  mfem::Vector e2(2);
  e1 = 0.0;
  e2 = 0.0;
  e1[0] = 1.0;
  e2[1] = 1.0;

  mfem::Vector h1(2);
  mfem::Vector h2(2);
  mfem::Vector old_h1(2);
  mfem::Vector old_h2(2);
  h1 = 0.0;
  h2 = 0.0;
  old_h1 = 0.0;
  old_h2 = 0.0;
  h1[0] = 2.0;
  h2[1] = 3.0;
  old_h1[0] = 1.0;
  old_h2[1] = 3.0;

  mfem::Vector previous_step(2);
  previous_step = 0.0;
  previous_step[0] = 1.0;

  mfem::Vector b(2);
  b[0] = 1.0;
  b[1] = 0.25;

  std::vector<const mfem::Vector*> directions = {&e1, &e2};
  std::vector<const mfem::Vector*> h_directions = {&h1, &h2};
  std::vector<const mfem::Vector*> old_h_directions = {&old_h1, &old_h2};

  auto [x, leftvecs, leftvals, energy] =
      smith::solveCubicSubspaceProblemMfem(directions, h_directions, old_h_directions, previous_step, b, 0.5, 1);

  EXPECT_LE(x.Norml2(), 0.5 + 1.0e-12);
  EXPECT_FALSE(leftvecs.empty());
  EXPECT_EQ(leftvals.size(), 1);
  EXPECT_LT(energy, 0.0);
}

TEST(TrustRegionCubicSubspaceMfem, FallsBackToQuadraticWhenCubicPredictionDoesNotImprove)
{
  mfem::Vector e1(1);
  mfem::Vector h1(1);
  mfem::Vector old_h1(1);
  mfem::Vector previous_step(1);
  mfem::Vector b(1);

  e1[0] = 1.0;
  h1[0] = 1.0;
  old_h1[0] = 1.0;
  previous_step[0] = 1.0;
  b[0] = 1.0;

  std::vector<const mfem::Vector*> directions = {&e1};
  std::vector<const mfem::Vector*> h_directions = {&h1};
  std::vector<const mfem::Vector*> old_h_directions = {&old_h1};

  auto [cubic_x, cubic_leftvecs, cubic_leftvals, cubic_energy] =
      smith::solveCubicSubspaceProblemMfem(directions, h_directions, old_h_directions, previous_step, b, 1.0, 1);
  auto [quadratic_x, quadratic_leftvecs, quadratic_leftvals, quadratic_energy] =
      smith::solveSubspaceProblemMfem(directions, h_directions, b, 1.0, 1);

  expectNearVector(cubic_x, quadratic_x, 1.0e-12);
  EXPECT_EQ(cubic_leftvecs.size(), quadratic_leftvecs.size());
  EXPECT_EQ(cubic_leftvals.size(), quadratic_leftvals.size());
  EXPECT_NEAR(cubic_energy, quadratic_energy, 1.0e-12);
}

TEST(TrustRegionCubicSubspaceMfem, PreviousStepSecantIsExactForCompatibleCubic)
{
  mfem::Vector e1(2);
  mfem::Vector e2(2);
  e1 = 0.0;
  e2 = 0.0;
  e1[0] = 1.0;
  e2[1] = 1.0;

  mfem::Vector h1(2);
  mfem::Vector h2(2);
  mfem::Vector old_h1(2);
  mfem::Vector old_h2(2);
  h1 = 0.0;
  h2 = 0.0;
  old_h1 = 0.0;
  old_h2 = 0.0;
  h1[0] = 1.0;
  h2[1] = 1.0;
  old_h1[0] = 7.0;
  old_h2[1] = 1.0;

  mfem::Vector previous_step(2);
  previous_step = 0.0;
  previous_step[0] = 1.0;

  mfem::Vector b(2);
  b = 0.0;
  b[0] = 0.1;

  std::vector<const mfem::Vector*> directions = {&e1, &e2};
  std::vector<const mfem::Vector*> h_directions = {&h1, &h2};
  std::vector<const mfem::Vector*> old_h_directions = {&old_h1, &old_h2};

  bool used_cubic = false;
  auto [x, leftvecs, leftvals, energy] =
      smith::solveCubicSubspaceProblemMfem(directions, h_directions, old_h_directions, previous_step, b, 1.0, 1,
                                           &used_cubic);

  mfem::DenseMatrix A(2);
  A = 0.0;
  A(0, 0) = 1.0;
  A(1, 1) = 1.0;
  std::vector<mfem::DenseMatrix> cubic(2, mfem::DenseMatrix(2));
  cubic[0] = 0.0;
  cubic[1] = 0.0;
  cubic[0](0, 0) = -6.0;
  auto [expected_x, expected_energy] = smith::solveDenseCubicTrustRegionProblemMfem(A, b, cubic, 1.0);

  EXPECT_TRUE(used_cubic);
  expectNearVector(x, expected_x, 1.0e-12);
  EXPECT_NEAR(energy, expected_energy, 1.0e-12);
  EXPECT_FALSE(leftvecs.empty());
  EXPECT_EQ(leftvals.size(), 1);
}

TEST(TrustRegionCubicSubspaceMfem, PreviousStepSecantIsExactForRotatedCompatibleCubic)
{
  mfem::Vector e1(2);
  mfem::Vector e2(2);
  e1 = 0.0;
  e2 = 0.0;
  e1[0] = 1.0;
  e2[1] = 1.0;

  constexpr double lambda = -6.0;
  mfem::Vector previous_step(2);
  previous_step[0] = 1.0;
  previous_step[1] = 1.0;
  mfem::Vector u(previous_step);
  u /= u.Norml2();

  mfem::DenseMatrix delta_h(2);
  delta_h = 0.0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      delta_h(i, j) = lambda * previous_step.Norml2() * u[i] * u[j];
    }
  }

  mfem::Vector h1(e1);
  mfem::Vector h2(e2);
  mfem::Vector old_h1(e1);
  mfem::Vector old_h2(e2);
  for (int i = 0; i < 2; ++i) {
    old_h1[i] -= delta_h(i, 0);
    old_h2[i] -= delta_h(i, 1);
  }

  mfem::Vector b(2);
  b[0] = 0.1 * u[0];
  b[1] = 0.1 * u[1];

  std::vector<const mfem::Vector*> directions = {&e1, &e2};
  std::vector<const mfem::Vector*> h_directions = {&h1, &h2};
  std::vector<const mfem::Vector*> old_h_directions = {&old_h1, &old_h2};

  bool used_cubic = false;
  auto [x, leftvecs, leftvals, energy] =
      smith::solveCubicSubspaceProblemMfem(directions, h_directions, old_h_directions, previous_step, b, 1.0, 1,
                                           &used_cubic);

  mfem::DenseMatrix A(2);
  A = 0.0;
  A(0, 0) = 1.0;
  A(1, 1) = 1.0;
  std::vector<mfem::DenseMatrix> cubic(2, mfem::DenseMatrix(2));
  cubic[0] = 0.0;
  cubic[1] = 0.0;
  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        cubic[size_t(k)](i, j) = lambda * u[k] * u[i] * u[j];
      }
    }
  }
  auto [expected_x, expected_energy] = smith::solveDenseCubicTrustRegionProblemMfem(A, b, cubic, 1.0);

  EXPECT_TRUE(used_cubic);
  expectNearVector(x, expected_x, 1.0e-12);
  EXPECT_NEAR(energy, expected_energy, 1.0e-12);
  EXPECT_FALSE(leftvecs.empty());
  EXPECT_EQ(leftvals.size(), 1);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
