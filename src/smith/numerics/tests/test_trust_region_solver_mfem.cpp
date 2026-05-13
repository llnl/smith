// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

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

  auto [sol, leftvecs, leftvals, energy] = smith::solveSubspaceProblemMfem(states, astate_ptrs, b, 0.25, 1);

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
