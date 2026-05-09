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

  mfem::Vector workspace(2000);
  auto [sol, leftvecs, leftvals, energy] =
      smith::solveSubspaceProblemMfem(states, astate_ptrs, fixture.b, test_delta, 1, workspace);

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

  mfem::Vector workspace(2000);
  auto [generic_sol, generic_leftvecs, generic_leftvals, generic_energy] =
      smith::solveSubspaceProblem(states, astate_ptrs, fixture.b, test_delta, 2, workspace);
  auto [mfem_sol, mfem_leftvecs, mfem_leftvals, mfem_energy] =
      smith::solveSubspaceProblemMfem(states, astate_ptrs, fixture.b, test_delta, 2, workspace);

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

  mfem::Vector workspace(2000);
  auto [sol, leftvecs, leftvals, energy] = smith::solveSubspaceProblemMfem(states, astate_ptrs, b, 0.25, 1, workspace);

  EXPECT_LE(sol.Norml2(), 0.25 + 1.0e-12);
  EXPECT_FALSE(leftvecs.empty());
  EXPECT_EQ(leftvals.size(), 1);
  EXPECT_LT(energy, 0.0);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
