// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/infrastructure/application_manager.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/profiling.hpp"

#include "problems/Problems.hpp"
#include "solvers/Solvers.hpp"
#include "utilities.hpp"

using namespace serac;
using namespace serac::profiling;

using namespace mfem;
/* convex quadratic-programming problem
 *
 * min_u (u^T u) / 2
 *  s.t.  u - u_l >= 0
 *
 *  solution u = max{0, ul} (component-wise)
 */
class QPTestProblem : public ParOptProblem {
 protected:
  Vector ul_;
  HypreParMatrix* dgdu_;
  HypreParMatrix* d2Edu2_;

 public:
  QPTestProblem(int n);
  double E(const Vector& u, int& eval_err);

  void DdE(const Vector& u, Vector& gradE);

  HypreParMatrix* DddE(const Vector& u);

  void g(const Vector& u, Vector& gu, int& eval_err);

  HypreParMatrix* Ddg(const Vector&);

  virtual ~QPTestProblem();
};

TEST(InteriorPointMethod, QuadraticProgramming)
{
  int n = 30;
  double outerSolveTol = 1.e-8;
  double linSolveTol = 1.e-10;
  int maxiter = 40;

  QPTestProblem opt_problem(n);

  ParInteriorPointSolver solver(&opt_problem);
  GMRESSolver linSolver(MPI_COMM_WORLD);
  linSolver.SetRelTol(linSolveTol);
  linSolver.SetMaxIter(1000);
  linSolver.SetPrintLevel(2);
  solver.SetLinearSolver(linSolver);

  int dimx = opt_problem.GetDimU();
  Vector x0(dimx);
  x0 = 0.0;
  Vector xf(dimx);
  xf = 0.0;

  solver.SetTol(outerSolveTol);
  solver.SetMaxIter(maxiter);

  solver.Mult(x0, xf);

  EXPECT_TRUE(solver.GetConverged());
}

TEST(HomotopyMethod, QuadraticProgramming)
{
  int n = 30;
  double outerSolveTol = 1.e-8;
  double linSolveTol = 1.e-10;
  int maxiter = 40;

  QPTestProblem opt_problem(n);
  OptNLMCProblem nlmc_problem(&opt_problem);

  HomotopySolver solver(&nlmc_problem);
  GMRESSolver linSolver(MPI_COMM_WORLD);
  linSolver.SetRelTol(linSolveTol);
  linSolver.SetMaxIter(1000);
  linSolver.SetPrintLevel(2);
  solver.SetLinearSolver(linSolver);

  int dimx = nlmc_problem.GetDimx();
  int dimy = nlmc_problem.GetDimy();
  Vector x0(dimx);
  x0 = 0.0;
  Vector y0(dimy);
  y0 = 0.0;
  Vector xf(dimx);
  xf = 0.0;
  Vector yf(dimy);
  yf = 0.0;

  solver.SetTol(outerSolveTol);
  solver.SetMaxIter(maxiter);

  solver.Mult(x0, y0, xf, yf);

  EXPECT_TRUE(solver.GetConverged());
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);

  return RUN_ALL_TESTS();
}

// Ex1Problem
QPTestProblem::QPTestProblem(int n) : ParOptProblem(), dgdu_(nullptr), d2Edu2_(nullptr)
{
  MFEM_VERIFY(n >= 1, "QPTestProblem::QPTestProblem -- problem must have nontrivial size");

  // generate parallel partition
  int nprocs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();

  HYPRE_BigInt dofOffsets[2];
  HYPRE_BigInt constraintOffsets[2];
  if (n >= nprocs) {
    dofOffsets[0] = HYPRE_BigInt((myid * n) / nprocs);
    dofOffsets[1] = HYPRE_BigInt(((myid + 1) * n) / nprocs);
  } else {
    if (myid < n) {
      dofOffsets[0] = myid;
      dofOffsets[1] = myid + 1;
    } else {
      dofOffsets[0] = n;
      dofOffsets[1] = n;
    }
  }
  constraintOffsets[0] = dofOffsets[0];
  constraintOffsets[1] = dofOffsets[1];
  Init(dofOffsets, constraintOffsets);

  Vector temp(dimU);
  temp = 1.0;
  d2Edu2_ = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);
  dgdu_ = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);

  // random entries in [-1, 1]
  ul_.SetSize(dimM);
  ul_.Randomize(myid);
  ul_ *= 2.0;
  ul_ -= 1.0;
}

double QPTestProblem::E(const Vector& u, int& eval_err)
{
  eval_err = 0;
  double Eeval = 0.5 * InnerProduct(MPI_COMM_WORLD, u, u);
  return Eeval;
}

void QPTestProblem::DdE(const Vector& u, Vector& gradE) { gradE.Set(1.0, u); }

HypreParMatrix* QPTestProblem::DddE(const Vector& /*u*/) { return d2Edu2_; }

void QPTestProblem::g(const Vector& u, Vector& gu, int& eval_err)
{
  eval_err = 0;
  gu = 0.0;
  gu.Set(1.0, u);
  gu.Add(-1.0, ul_);
}

HypreParMatrix* QPTestProblem::Ddg(const Vector& /*u*/) { return dgdu_; }

QPTestProblem::~QPTestProblem()
{
  delete d2Edu2_;
  delete dgdu_;
}
