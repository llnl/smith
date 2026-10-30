// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/trust_region_subspace_cache.hpp"
#include "smith/numerics/trust_region_solver.hpp"

namespace {

constexpr int test_size = 5;
constexpr double test_delta = 1.0e-3;

std::vector<mfem::Vector> applyDiagonalOperator(const mfem::Vector& diag,
                                                const std::vector<const mfem::Vector*>& states)
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
  DiagonalSubspaceFixture(int size) : u1(size), u2(size), u3(size), diag(size), b(size)
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

TEST(TrustRegionSubspaceMfem, SolveHitsTrustRegionBoundary)
{
  DiagonalSubspaceFixture fixture(test_size);

  const std::vector<const mfem::Vector*> states = {&fixture.u1, &fixture.u2, &fixture.u3};
  const auto astates = applyDiagonalOperator(fixture.diag, states);
  const auto astate_ptrs = toPointers(astates);

  auto [sol, leftvecs, leftvals, energy] = smith::solveSubspaceProblem(states, astate_ptrs, fixture.b, test_delta, 1);

  EXPECT_NEAR(sol.Norml2(), test_delta, 1.0e-12);
  EXPECT_FALSE(leftvecs.empty());
  EXPECT_EQ(leftvals.size(), 1);
  EXPECT_LT(energy, 0.0);
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

  auto [sol, leftvecs, leftvals, energy] = smith::solveSubspaceProblem(states, astate_ptrs, b, 0.25, 1);

  EXPECT_LE(sol.Norml2(), 0.25 + 1.0e-12);
  EXPECT_FALSE(leftvecs.empty());
  EXPECT_EQ(leftvals.size(), 1);
  EXPECT_LT(energy, 0.0);
}

TEST(TrustRegionSubspaceMfem, SolveIndefiniteHardCaseUsesShiftedNewtonPoint)
{
  mfem::Vector e0(2);
  mfem::Vector e1(2);
  mfem::Vector Ae0(2);
  mfem::Vector Ae1(2);
  mfem::Vector b(2);

  e0 = 0.0;
  e1 = 0.0;
  Ae0 = 0.0;
  Ae1 = 0.0;
  b = 0.0;

  e0[0] = 1.0;
  e1[1] = 1.0;
  Ae0[0] = -1.0;
  Ae1[1] = 2.0;
  b[1] = 1.0;

  const std::vector<const mfem::Vector*> states = {&e0, &e1};
  const std::vector<const mfem::Vector*> astates = {&Ae0, &Ae1};

  auto [sol, leftvecs, leftvals, energy] = smith::solveSubspaceProblem(states, astates, b, 1.0, 1);

  EXPECT_NEAR(sol.Norml2(), 1.0, 1.0e-12);
  EXPECT_NEAR(std::abs(sol[0]), std::sqrt(8.0 / 9.0), 1.0e-10);
  EXPECT_NEAR(sol[1], 1.0 / 3.0, 1.0e-10);
  EXPECT_EQ(leftvecs.size(), 1);
  EXPECT_EQ(leftvals.size(), 1);
  EXPECT_NEAR(leftvals[0], -1.0, 1.0e-12);
  EXPECT_NEAR(energy, -2.0 / 3.0, 1.0e-10);
}

TEST(TrustRegionSubspaceMfem, SolveThrowsOnNanProjection)
{
  mfem::Vector state(2);
  mfem::Vector astate(2);
  mfem::Vector b(2);

  state = 1.0;
  astate = 1.0;
  b = 0.0;
  astate[1] = std::numeric_limits<double>::quiet_NaN();

  const std::vector<const mfem::Vector*> states = {&state};
  const std::vector<const mfem::Vector*> astates = {&astate};

  EXPECT_THROW(smith::solveSubspaceProblem(states, astates, b, 1.0, 1), smith::TrustRegionException);
}

TEST(TrustRegionSubspaceMfem, SolveRejectsNonfiniteSingularResult)
{
  mfem::Vector first_direction(2);
  mfem::Vector second_direction(2);
  mfem::Vector zero_action(2);
  mfem::Vector b(2);

  first_direction = 0.0;
  second_direction = 0.0;
  zero_action = 0.0;
  b = 0.0;
  first_direction[0] = 1.0;
  second_direction[1] = 1.0;
  b[0] = 1.0;

  const std::vector<const mfem::Vector*> states = {&first_direction, &second_direction};
  const std::vector<const mfem::Vector*> astates = {&zero_action, &zero_action};

  EXPECT_THROW(smith::solveSubspaceProblem(states, astates, b, 1.0, 1), smith::TrustRegionException);
}

TEST(TrustRegionSubspaceMfem, ProjectionUsesSymmetricOperatorPart)
{
  mfem::Vector first_direction(2);
  mfem::Vector second_direction(2);
  mfem::Vector first_action(2);
  mfem::Vector second_action(2);
  mfem::Vector b(2);

  first_direction = 0.0;
  second_direction = 0.0;
  first_action = 0.0;
  second_action = 0.0;
  b = 0.0;

  first_direction[0] = 1.0;
  second_direction[1] = 2.0;
  first_action[0] = 2.0;
  second_action[0] = 8.0;
  second_action[1] = 12.0;

  const std::vector<const mfem::Vector*> states = {&first_direction, &second_direction};
  const std::vector<const mfem::Vector*> astates = {&first_action, &second_action};

  smith::TrustRegionSubspaceCache cache;
  cache.prepare(states, astates, b, 1);

  ASSERT_EQ(cache.projected_hessian.Height(), 2);
  ASSERT_EQ(cache.projected_hessian.Width(), 2);
  EXPECT_NEAR(cache.projected_hessian(0, 0), 2.0, 1.0e-12);
  EXPECT_NEAR(cache.projected_hessian(0, 1), 2.0, 1.0e-12);
  EXPECT_NEAR(cache.projected_hessian(1, 0), 2.0, 1.0e-12);
  EXPECT_NEAR(cache.projected_hessian(1, 1), 6.0, 1.0e-12);
}

TEST(TrustRegionSubspaceMfem, PreparedCacheSupportsMultipleRadii)
{
  mfem::Vector first_direction(2);
  mfem::Vector second_direction(2);
  mfem::Vector first_action(2);
  mfem::Vector second_action(2);
  mfem::Vector b(2);

  first_direction = 0.0;
  second_direction = 0.0;
  first_action = 0.0;
  second_action = 0.0;
  first_direction[0] = 1.0;
  second_direction[1] = 1.0;
  first_action[0] = 1.0;
  second_action[1] = 2.0;
  b[0] = 1.0;
  b[1] = 2.0;

  const std::vector<const mfem::Vector*> states = {&first_direction, &second_direction};
  const std::vector<const mfem::Vector*> astates = {&first_action, &second_action};

  smith::TrustRegionSubspaceCache cache;
  cache.prepare(states, astates, b, 1);

  const mfem::DenseMatrix projected_hessian(cache.projected_hessian);
  const mfem::Vector projected_rhs(cache.projected_rhs);
  const mfem::Vector eigenvalues(cache.eigenvalues);
  const mfem::DenseMatrix eigenvectors(cache.eigenvectors);
  const mfem::Vector eigen_rhs(cache.eigen_rhs);

  constexpr double boundary_radius = 0.5;
  auto [boundary_solution, boundary_leftmosts, boundary_leftvals, boundary_energy] = cache.solve(boundary_radius);
  auto [interior_solution, interior_leftmosts, interior_leftvals, interior_energy] = cache.solve(2.0);
  auto [repeated_solution, repeated_leftmosts, repeated_leftvals, repeated_energy] = cache.solve(boundary_radius);

  EXPECT_EQ(boundary_leftmosts.size(), 1);
  EXPECT_EQ(boundary_leftvals.size(), 1);
  EXPECT_EQ(interior_leftmosts.size(), 1);
  EXPECT_EQ(interior_leftvals.size(), 1);
  EXPECT_EQ(repeated_leftmosts.size(), 1);
  EXPECT_EQ(repeated_leftvals.size(), 1);
  EXPECT_NEAR(boundary_solution.Norml2(), boundary_radius, 1.0e-12);
  EXPECT_NEAR(repeated_solution.Norml2(), boundary_radius, 1.0e-12);
  EXPECT_NEAR(interior_solution[0], 1.0, 1.0e-12);
  EXPECT_NEAR(interior_solution[1], 1.0, 1.0e-12);
  EXPECT_LT(interior_solution.Norml2(), 2.0);
  EXPECT_LT(boundary_energy, 0.0);
  EXPECT_LT(interior_energy, boundary_energy);
  EXPECT_NEAR(repeated_energy, boundary_energy, 1.0e-14);
  for (int index = 0; index < repeated_solution.Size(); ++index) {
    EXPECT_NEAR(repeated_solution[index], boundary_solution[index], 1.0e-14);
  }

  for (int row = 0; row < cache.projected_hessian.Height(); ++row) {
    EXPECT_DOUBLE_EQ(cache.projected_rhs[row], projected_rhs[row]);
    EXPECT_DOUBLE_EQ(cache.eigenvalues[row], eigenvalues[row]);
    EXPECT_DOUBLE_EQ(cache.eigen_rhs[row], eigen_rhs[row]);
    for (int column = 0; column < cache.projected_hessian.Width(); ++column) {
      EXPECT_DOUBLE_EQ(cache.projected_hessian(row, column), projected_hessian(row, column));
      EXPECT_DOUBLE_EQ(cache.eigenvectors(row, column), eigenvectors(row, column));
    }
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
