// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "mfem.hpp"
#include "petsc.h"

#include "smith/infrastructure/application_manager.hpp"

#ifndef MFEM_USE_SLEPC
#error This examples requires that MFEM is build with MFEM_USE_SLEPC=YES
#endif

TEST(PETSC_AND_SLEPC, CanComputeSmallestEigenvalueAndEigenvectors)
{
  int world_rank;
  MPI_Comm comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_split(MPI_COMM_WORLD, (0 != world_rank) ? MPI_UNDEFINED : 0, 0, &comm);
  if (world_rank != 0) return;

  Mat A; /* problem matrix */
  int N = 6;
  unsigned int M = 3;
  MatCreate(comm, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatSetFromOptions(A);

  int Istart, Iend;
  MatGetOwnershipRange(A, &Istart, &Iend);
  for (int i = Istart; i < Iend; i++) {
    MatSetValue(A, i, i, i - 2, INSERT_VALUES);
  }

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  mfem::Vector v, w;
  v.SetSize(N);
  w.SetSize(N);

  mfem::SlepcEigenSolver eig(comm);
  eig.SetNumModes(static_cast<int>(M));
  eig.SetWhichEigenpairs(mfem::SlepcEigenSolver::SMALLEST_REAL);
  // eig.SetWhichEigenpairs(mfem::SlepcEigenSolver::LARGEST_MAGNITUDE);
  eig.SetOperator(A);
  eig.Solve();

  EXPECT_GE(eig.GetNumConverged(), M);

  for (unsigned int i = 0; i < M; ++i) {
    eig.GetEigenvector(i, v);
    double eval;
    eig.GetEigenvalue(i, eval);
    EXPECT_NEAR(eval, static_cast<double>(i) - 2.0, 1e-14);
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
